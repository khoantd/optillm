"""ModernBERT-based approach classifier with singleton model cache."""

import logging
import threading
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)

MAX_LENGTH = 1024
ML_APPROACHES: List[str] = [
    "none",
    "mcts",
    "bon",
    "moa",
    "rto",
    "z3",
    "self_consistency",
    "pvg",
    "rstar",
    "cot_reflection",
    "plansearch",
    "leap",
    "re2",
]
BASE_MODEL = "answerdotai/ModernBERT-large"
OPTILLM_MODEL_NAME = "codelion/optillm-modernbert-large"

_load_lock = threading.Lock()
_cached_model = None
_cached_tokenizer = None
_cached_device = None
_load_failed = False


def _torch():
    import torch
    return torch


def _torch_nn():
    import torch.nn as nn
    return nn


def _torch_F():
    import torch.nn.functional as F
    return F


def _build_classifier_class():
    nn = _torch_nn()
    torch = _torch()

    class OptILMClassifier(nn.Module):
        def __init__(self, base_model, num_labels: int):
            super().__init__()
            self.base_model = base_model
            self.effort_encoder = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(base_model.config.hidden_size + 64, num_labels)

        def forward(self, input_ids, attention_mask, effort):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            effort_encoded = self.effort_encoder(effort.unsqueeze(1))
            combined_input = torch.cat((pooled_output, effort_encoded), dim=1)
            return self.classifier(combined_input)

    return OptILMClassifier


def _get_device():
    torch = _torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_optillm_model():
    """Load classifier once per process (thread-safe singleton)."""
    global _cached_model, _cached_tokenizer, _cached_device, _load_failed

    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer, _cached_device

    with _load_lock:
        if _cached_model is not None and _cached_tokenizer is not None:
            return _cached_model, _cached_tokenizer, _cached_device
        if _load_failed:
            raise RuntimeError("Approach router model failed to load previously")

        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_model
            from transformers import AutoModel, AutoTokenizer

            device = _get_device()
            logger.info("Loading approach router model: %s", OPTILLM_MODEL_NAME)
            OptILMClassifier = _build_classifier_class()
            base_model = AutoModel.from_pretrained(BASE_MODEL)
            model = OptILMClassifier(base_model, num_labels=len(ML_APPROACHES))
            model.to(device)
            safetensors_path = hf_hub_download(
                repo_id=OPTILLM_MODEL_NAME, filename="model.safetensors"
            )
            load_model(model, safetensors_path)
            tokenizer = AutoTokenizer.from_pretrained(OPTILLM_MODEL_NAME)
            model.eval()
            _cached_model = model
            _cached_tokenizer = tokenizer
            _cached_device = device
            logger.info("Approach router model loaded on %s", device)
            return model, tokenizer, device
        except Exception as exc:
            _load_failed = True
            logger.error("Failed to load approach router model: %s", exc)
            raise


def reset_model_cache_for_testing() -> None:
    """Clear singleton cache (tests only)."""
    global _cached_model, _cached_tokenizer, _cached_device, _load_failed
    _cached_model = None
    _cached_tokenizer = None
    _cached_device = None
    _load_failed = False


def format_router_input(system_prompt: str, initial_query: str) -> str:
    """
    Build classifier text from server parse_conversation() output.

    initial_query already uses ``User:`` / ``Assistant:`` tags per turn; do not
    prepend another ``User:`` prefix (that duplicated tags and skewed routing).
    """
    sp = (system_prompt or "").strip()
    iq = (initial_query or "").strip()
    if sp and iq:
        return f"{sp}\n\n{iq}"
    return sp or iq


def preprocess_input(tokenizer, system_prompt: str, initial_query: str):
    combined_input = format_router_input(system_prompt, initial_query)
    encoding = tokenizer(
        combined_input,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


def _predict_probabilities(model, input_ids, attention_mask, device, effort: float = 0.7):
    torch = _torch()
    F = _torch_F()
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        effort_tensor = torch.tensor([effort], dtype=torch.float).to(device)
        logits = model(input_ids, attention_mask=attention_mask, effort=effort_tensor)
        return F.softmax(logits, dim=1)[0]


def predict_approach_topk(
    model,
    input_ids,
    attention_mask,
    device,
    effort: float = 0.7,
    top_k: int = 3,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Return (best approach, best confidence, top-k [(approach, confidence), ...])."""
    probabilities = _predict_probabilities(model, input_ids, attention_mask, device, effort)
    k = min(top_k, len(ML_APPROACHES))
    values, indices = probabilities.topk(k)
    top_scores = [
        (ML_APPROACHES[indices[i].item()], values[i].item()) for i in range(k)
    ]
    approach, confidence = top_scores[0]
    return approach, confidence, top_scores


def predict_approach(model, input_ids, attention_mask, device, effort: float = 0.7) -> Tuple[str, float]:
    approach, confidence, _ = predict_approach_topk(
        model, input_ids, attention_mask, device, effort=effort, top_k=1
    )
    return approach, confidence
