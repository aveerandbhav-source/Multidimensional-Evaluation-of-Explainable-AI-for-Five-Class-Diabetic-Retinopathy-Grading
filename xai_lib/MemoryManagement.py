import os
import gc
import random
import time
import warnings
warnings.filterwarnings("ignore")

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from scipy.stats import spearmanr
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from skimage.segmentation import slic
import shap

from PIL import Image as PILImage
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch.utils.data import Dataset, DataLoader

import ctypes
import resource

# Module-level state (previously stored on `self` in an unused class instance).
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None
    
try:
    from quantus.helpers.model.pytorch_model import PyTorchModel as _QuantusPyTorchModel
except Exception:
    _QuantusPyTorchModel = None


def clear_quantus_caches():
    """Evict quantus's internal lru_cache of PyTorchModel wrappers (see note above)."""
    if _QuantusPyTorchModel is None:
        return
    for _method_name in ("_get_last_softmax_layer_index", "_get_model_with_linear_top"):
        _method = getattr(_QuantusPyTorchModel, _method_name, None)
        if _method is not None and hasattr(_method, "cache_clear"):
            _method.cache_clear()


def clear_memory():
    """Reclaim python and cuda memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if _libc is not None:
        _libc.malloc_trim(0)


def log_rss(tag=""):
    """Log resident set size (the number SLURM's OOM killer actually watches)."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux
    msg = f"[rss{' ' + tag if tag else ''}] peak RSS so far = {rss_mb:.0f} MB"
    if torch.cuda.is_available():
        msg += f"  |  GPU allocated = {torch.cuda.memory_allocated()/1e6:.0f} MB"
    print(msg)


def unload_model(model=None, engine=None, engine_cache=None):
    """Fully detach a model + its AttributionEngine from memory before loading the next one."""
    if engine is not None and engine_cache is not None:
        engine_cache.pop(id(model), None)
        # Drop GradCAM hooks / explainer state held by the engine
        engine._gradcam = None
        engine.shap_explainer = None
        del engine
    if model is not None:
        model.to("cpu")
        del model
    clear_memory()


def gpu_mem_report(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[mem{' ' + tag if tag else ''}] allocated={alloc:.2f} GB  reserved={reserved:.2f} GB")


def load_model(model_name, weight_path, device):
    """Load the model given the model weights"""
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, 5))
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, 5))
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)