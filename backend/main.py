import os
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Dict
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReviewRadarAPI")

# File & Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDING_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "embedding_classifier.joblib")
TFIDF_PIPELINE_PATH = os.path.join(MODELS_DIR, "tfidf_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Global ML objects
encoder = None
embedding_classifier = None
tfidf_pipeline = None
model_load_errors: Dict[str, str] = {}
encoder_backend = None  # "sentence-transformers" or "onnx"

class MiniLMONNXEncoder:
    """
    Lightweight, high-speed ONNX runtime encoder for all-MiniLM-L6-v2.
    Produces identical 384-dimensional normalized dense embeddings
    WITHOUT requiring heavy PyTorch downloads (runs on ~20MB onnxruntime).
    """
    def __init__(self, onnx_path: str, tokenizer_path: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=256)
        self.tokenizer.enable_truncation(max_length=256)
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_names = [i.name for i in self.session.get_inputs()]

    def encode(self, texts: List[str], normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        encs = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feed = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if 'token_type_ids' in self.input_names:
            feed['token_type_ids'] = np.array([e.type_ids for e in encs], dtype=np.int64)

        outputs = self.session.run(None, feed)
        token_embeddings = outputs[0]  # shape: [batch_size, seq_len, 384]

        # Mean Pooling - take attention mask into account for correct averaging
        mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        if normalize_embeddings:
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            return mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)
        return mean_pooled


def load_encoder_model():
    """
    Load the sentence encoder once on startup.
    Tries sentence_transformers first; falls back to lightweight ONNX runtime
    which avoids the multi-gigabyte PyTorch dependency.
    """
    global encoder, encoder_backend
    
    # Priority 1: sentence_transformers (if available)
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Attempting to load sentence-transformers/all-MiniLM-L6-v2 via PyTorch...")
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        encoder_backend = "sentence-transformers"
        logger.info("SentenceTransformer encoder loaded successfully via PyTorch.")
        return
    except Exception as exc:
        logger.info(f"PyTorch sentence-transformers not available ({exc}). Checking lightweight ONNX...")

    # Priority 2: ONNX Runtime (fast, lightweight, ~22MB model, already pre-installed)
    try:
        import huggingface_hub
        logger.info("Resolving all-MiniLM-L6-v2 ONNX model files...")
        onnx_file = huggingface_hub.hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="onnx/model_quint8_avx2.onnx"
        )
        tokenizer_file = huggingface_hub.hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="tokenizer.json"
        )
        encoder = MiniLMONNXEncoder(onnx_file, tokenizer_file)
        encoder_backend = "onnx-runtime"
        logger.info("Sentence encoder successfully loaded via lightweight ONNX runtime (0 PyTorch needed)!")
    except Exception as exc:
        err_msg = f"Failed to load sentence encoder: {str(exc)}"
        logger.error(err_msg)
        model_load_errors["encoder"] = err_msg
        encoder = None


def load_joblib_models():
    """Load trained classifiers exported from Google Colab into models/."""
    global embedding_classifier, tfidf_pipeline

    model_load_errors.pop("embedding_classifier", None)
    model_load_errors.pop("tfidf_pipeline", None)

    # 1. Main model: embedding_classifier.joblib
    if os.path.exists(EMBEDDING_CLASSIFIER_PATH):
        try:
            embedding_classifier = joblib.load(EMBEDDING_CLASSIFIER_PATH)
            logger.info("embedding_classifier.joblib loaded successfully.")
        except Exception as exc:
            err_msg = f"Failed to load embedding_classifier.joblib: {str(exc)}"
            logger.error(err_msg)
            model_load_errors["embedding_classifier"] = err_msg
            embedding_classifier = None
    else:
        err_msg = f"embedding_classifier.joblib not found at {EMBEDDING_CLASSIFIER_PATH}"
        logger.warning(err_msg)
        model_load_errors["embedding_classifier"] = err_msg
        embedding_classifier = None

    # 2. Baseline / Optional model: tfidf_pipeline.joblib
    if os.path.exists(TFIDF_PIPELINE_PATH):
        try:
            tfidf_pipeline = joblib.load(TFIDF_PIPELINE_PATH)
            logger.info("tfidf_pipeline.joblib loaded successfully.")
        except Exception as exc:
            err_msg = f"Failed to load tfidf_pipeline.joblib: {str(exc)}"
            logger.error(err_msg)
            model_load_errors["tfidf_pipeline"] = err_msg
            tfidf_pipeline = None
    else:
        err_msg = f"tfidf_pipeline.joblib not found at {TFIDF_PIPELINE_PATH}"
        logger.warning(err_msg)
        model_load_errors["tfidf_pipeline"] = err_msg
        tfidf_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models once
    load_joblib_models()
    load_encoder_model()
    yield


def _ensure_models_loaded():
    """Ensure models are initialized even if lifespan was bypassed in a test runner."""
    if embedding_classifier is None and tfidf_pipeline is None and not model_load_errors:
        load_joblib_models()
        load_encoder_model()


app = FastAPI(
    title="ReviewRadar API",
    description="Sentiment classification backend utilizing TF-IDF and all-MiniLM-L6-v2 embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS specifically for local React development address (Vite default: port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class SinglePredictRequest(BaseModel):
    review_text: Optional[str] = Field(None, description="The customer review text to classify")
    text: Optional[str] = Field(None, description="Alternative field for review text")
    model: Optional[str] = Field("minilm", description="Model choice: 'minilm' (default) or 'tfidf'")


class PredictionResult(BaseModel):
    review_text: str
    sentiment: str
    confidence: Optional[float] = None
    confidence_percentage: Optional[str] = None
    model_used: str


class BatchPredictRequest(BaseModel):
    reviews: Optional[List[str]] = Field(None, description="List of review texts to classify")
    model: Optional[str] = Field("minilm", description="Model choice: 'minilm' (default) or 'tfidf'")


def _predict_single(text: str, model_choice: str) -> PredictionResult:
    """Internal helper to classify a single cleaned review text."""
    _ensure_models_loaded()
    normalized_choice = (model_choice or "minilm").strip().lower()

    if normalized_choice in ("minilm", "embedding", "embedding_classifier", "transformer"):
        # MiniLM Embeddings + Logistic Regression
        if encoder is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MiniLM sentence encoder is not loaded."
            )
        if embedding_classifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model file 'embedding_classifier.joblib' is missing from the models/ directory."
            )

        # Generate dense embeddings (matching Google Colab training settings: normalize_embeddings=True)
        emb = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
        probas = embedding_classifier.predict_proba(emb)[0]
        classes = embedding_classifier.classes_
        best_idx = int(np.argmax(probas))
        predicted_sentiment = str(classes[best_idx])
        confidence_val = float(probas[best_idx])
        model_name = "all-MiniLM-L6-v2 + Logistic Regression"

    elif normalized_choice in ("tfidf", "tfidf_lr", "baseline"):
        # TF-IDF Pipeline
        if tfidf_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model file 'tfidf_pipeline.joblib' is missing from the models/ directory."
            )

        probas = tfidf_pipeline.predict_proba([text])[0]
        classes = tfidf_pipeline.classes_
        best_idx = int(np.argmax(probas))
        predicted_sentiment = str(classes[best_idx])
        confidence_val = float(probas[best_idx])
        model_name = "TF-IDF + Logistic Regression"

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model '{model_choice}'. Choose 'minilm' or 'tfidf'."
        )

    conf_pct = f"{round(confidence_val * 100, 1)}%" if confidence_val is not None else "N/A"
    return PredictionResult(
        review_text=text,
        sentiment=predicted_sentiment,
        confidence=round(confidence_val, 4),
        confidence_percentage=conf_pct,
        model_used=model_name
    )


@app.get("/api/health")
def health_check():
    """Confirm the backend is running and report status of models."""
    _ensure_models_loaded()
    return {
        "status": "healthy",
        "message": "ReviewRadar API is up and running",
        "models": {
            "embedding_classifier_loaded": embedding_classifier is not None,
            "tfidf_pipeline_loaded": tfidf_pipeline is not None,
            "encoder_loaded": encoder is not None,
            "encoder_backend": encoder_backend
        },
        "errors": model_load_errors if model_load_errors else None
    }


@app.get("/api/metrics")
def get_model_metrics():
    """
    Read model metrics from models/metrics.json for display.
    Never invents or hardcodes metrics.
    """
    if not os.path.exists(METRICS_PATH):
        return {
            "available": False,
            "message": "metrics.json not found in models/ directory. Run evaluation in Google Colab and place metrics.json into the models/ folder."
        }
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "available": True,
            "data": data
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read models/metrics.json: {str(exc)}"
        )


@app.post("/api/predict", response_model=PredictionResult)
def predict_sentiment(payload: SinglePredictRequest):
    """
    Accepts JSON with review_text and returns predicted sentiment and confidence.
    Returns clear errors if model files are missing or input text is empty.
    """
    raw_text = payload.review_text if payload.review_text is not None else payload.text
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty. Please provide a valid review string."
        )

    return _predict_single(raw_text.strip(), payload.model)


@app.post("/api/predict-batch", response_model=List[PredictionResult])
def predict_batch_sentiment(payload: Union[BatchPredictRequest, List[str]]):
    """
    Accepts a list of review texts (or a BatchPredictRequest object)
    and returns predictions for each review.
    """
    _ensure_models_loaded()
    if isinstance(payload, list):
        reviews_list = payload
        model_choice = "minilm"
    else:
        reviews_list = payload.reviews or []
        model_choice = payload.model or "minilm"

    if not reviews_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The list of reviews cannot be empty."
        )

    # Process batch efficiently
    normalized_choice = (model_choice or "minilm").strip().lower()
    is_minilm = normalized_choice in ("minilm", "embedding", "embedding_classifier", "transformer")

    if is_minilm:
        if encoder is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MiniLM sentence encoder is not loaded."
            )
        if embedding_classifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model file 'embedding_classifier.joblib' is missing from the models/ directory."
            )

        # Batch encode
        cleaned_texts = [t.strip() if isinstance(t, str) and t.strip() else "" for t in reviews_list]
        non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]

        if not non_empty_indices:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All provided reviews are empty. Please provide at least one valid review string."
            )

        results: List[Optional[PredictionResult]] = [None] * len(reviews_list)
        
        # Fill in empty text results
        for i, t in enumerate(cleaned_texts):
            if not t:
                results[i] = PredictionResult(
                    review_text=reviews_list[i] if isinstance(reviews_list[i], str) else "",
                    sentiment="Neutral",
                    confidence=0.0,
                    confidence_percentage="0.0%",
                    model_used="N/A (empty review)"
                )

        if non_empty_indices:
            valid_texts = [cleaned_texts[i] for i in non_empty_indices]
            embeddings = encoder.encode(valid_texts, normalize_embeddings=True, show_progress_bar=False)
            all_probas = embedding_classifier.predict_proba(embeddings)
            classes = embedding_classifier.classes_

            for idx_in_valid, orig_idx in enumerate(non_empty_indices):
                probas = all_probas[idx_in_valid]
                best_idx = int(np.argmax(probas))
                conf = float(probas[best_idx])
                results[orig_idx] = PredictionResult(
                    review_text=valid_texts[idx_in_valid],
                    sentiment=str(classes[best_idx]),
                    confidence=round(conf, 4),
                    confidence_percentage=f"{round(conf * 100, 1)}%",
                    model_used="all-MiniLM-L6-v2 + Logistic Regression"
                )

        return results

    elif normalized_choice in ("tfidf", "tfidf_lr", "baseline"):
        if tfidf_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model file 'tfidf_pipeline.joblib' is missing from the models/ directory."
            )

        cleaned_texts = [t.strip() if isinstance(t, str) and t.strip() else "" for t in reviews_list]
        non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]

        if not non_empty_indices:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All provided reviews are empty. Please provide at least one valid review string."
            )

        results = [None] * len(reviews_list)
        for i, t in enumerate(cleaned_texts):
            if not t:
                results[i] = PredictionResult(
                    review_text=reviews_list[i] if isinstance(reviews_list[i], str) else "",
                    sentiment="Neutral",
                    confidence=0.0,
                    confidence_percentage="0.0%",
                    model_used="N/A (empty review)"
                )

        if non_empty_indices:
            valid_texts = [cleaned_texts[i] for i in non_empty_indices]
            all_probas = tfidf_pipeline.predict_proba(valid_texts)
            classes = tfidf_pipeline.classes_

            for idx_in_valid, orig_idx in enumerate(non_empty_indices):
                probas = all_probas[idx_in_valid]
                best_idx = int(np.argmax(probas))
                conf = float(probas[best_idx])
                results[orig_idx] = PredictionResult(
                    review_text=valid_texts[idx_in_valid],
                    sentiment=str(classes[best_idx]),
                    confidence=round(conf, 4),
                    confidence_percentage=f"{round(conf * 100, 1)}%",
                    model_used="TF-IDF + Logistic Regression"
                )

        return results

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model '{model_choice}'. Choose 'minilm' or 'tfidf'."
        )
