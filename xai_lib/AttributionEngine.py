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

import cv2

from shap import PartitionExplainer

import quantus
from quantus import PointingGame, MaxSensitivity
import glob
from PIL import Image

from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from quantus import norm_func, perturb_func, similarity_func
from datetime import datetime

import ctypes
import resource

class AttributionEngine:
    """
    Unified wrapper for generating XAI attribution maps for the moedls. Wraps the XAI techniques behind one interface, dealing with Grad-CAM, LIME and SHAP.
    
    Two types of attribution maps are produced depending on normalisation. The first is MinMax normalisation which is used for faithfulness evaluations, the second 
    clips out negative attributes before normalisaitn and can be used for localisation evaluations.

    The engine also genrates visualisations for all XAI techniques.

    Aknowledgements: https://github.com/SivaSubramaniyenM/ecs-xai/blob/main/src/ecs_metric.py
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    CLASS_NAMES = ["No DR","Mild DR","Moderate DR","Severe DR", "Proliferative DR"]
    SEED = 42

    def __init__(self, model, device, lime_samples=1000, img_size=224, shap_max_evals=500,
                 shap_batch_size=50, predict_chunk_size=32):
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.lime_samples    = lime_samples     # How many perturbed versions of the image LIME generates to fit its local surrogate model
        self.img_size        = img_size
        self.shap_max_evals  = shap_max_evals   # How many masked evaluations shap.Explainer's Partition explainer is allowed per image
        self.shap_batch_size = shap_batch_size  # Batch size used when shap.Explainer evaluates masked images through the model
        self.predict_chunk_size = predict_chunk_size
        self.class_names     = self.CLASS_NAMES
        self.seed = self.SEED

        self._gradcam = None

        self._norm_transform = transforms.Normalize(mean=self.IMAGENET_MEAN.tolist(), std=self.IMAGENET_STD.tolist()) # Stores the standard ImageNet normalization
        self.shap_explainer = None  # Built lazily on first compute_shap() call, once we know the image shape
        self.lime_explainer = lime_image.LimeImageExplainer(random_state=np.random.RandomState(self.seed))

    def find_gradcam_target_layer(self):
      """The target convolutional layer needed for Grad-CAM to generate their heatmaps"""
      if hasattr(self.model, "layer4"):
          target_layer = self.model.layer4[-1]
      elif hasattr(self.model, "features"):
          target_layer = self.model.features[-1]
      else:
          conv_layers = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
          if not conv_layers:
              raise ValueError("Could not locate a Conv2d layer in the model for Grad-CAM.")
          target_layer = conv_layers[-1]
      return target_layer

    def get_gradcam(self):
        """build a Grad-CAM object"""
        if self._gradcam is None:
            target_layer = self.find_gradcam_target_layer()
            self._gradcam = GradCAM(model=self.model, target_layers=[target_layer])
        return self._gradcam

    def unnormalize_to_uint8(self, img_tensor):
        """Unormalizes the image for LIME and SHAP (since they need the original image to create their superpixels)"""
        img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = img * self.IMAGENET_STD + self.IMAGENET_MEAN
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)

    def uint8_to_model_input(self, raw_uint8):
        """Opposite of unnormalize_to_uint8: uint8 H,W,C becomes a normalized (1,3,H,W) tensor on device."""
        t=torch.from_numpy(raw_uint8).float().div(255.0).permute(2,0,1)
        return self._norm_transform(t).unsqueeze(0).to(self.device)

    @staticmethod
    def normalize_map(m):
        """Min max normalisation"""
        m = np.nan_to_num(m.astype(np.float32))  
        mn, mx = m.min(), m.max()
        if mx - mn > 1e-12:
            m = (m - mn) / (mx - mn)
        else:
            m = np.zeros_like(m)
        return m
    
    @staticmethod
    def localization_normalize_map(m):
        """Clipping out zero attribution and Min max normalisation"""
        m = np.nan_to_num(m.astype(np.float32))  
        m = np.clip(m, 0, None)
        mn, mx = m.min(), m.max()
        if mx - mn > 1e-12:
            m = (m - mn) / (mx - mn)
        else:
            m = np.zeros_like(m)
        return m

    def resize_map(self, m):
        """Grad-CAM generally produces a smaller image, so the image is resized back to a normal viewing standard (224 x 224 pixels)"""
        if m.shape[0] != self.img_size or m.shape[1] != self.img_size:
            m = cv2.resize(m.astype(np.float32), (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return m

    def compute_gradcam(self, img, target_class):
        """Computes and returns the Grad-CAM heatmap"""
        cam = self.get_gradcam()
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=img, targets=targets)[0]
        heatmap = self.normalize_map(grayscale_cam)
        return self.resize_map(heatmap)

    def compute_localization_gradcam(self, img, target_class):
        """Computes and returns the Grad-CAM heatmap for localization evaluation"""
        cam = self.get_gradcam()
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=img, targets=targets)[0]
        heatmap = self.localization_normalize_map(grayscale_cam)
        return self.resize_map(heatmap)

    def show_grad_cam(self, img, raw_img_uint8, target_class):
        """Displays the GradCAM heatmap"""
        heatmap = self.compute_gradcam(img, target_class)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_img_uint8)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(raw_img_uint8)
        axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    def _predict_in_chunks(self, numpy_images):
        """Forward-pass a batch of perturbed images in small chunks."""
        outputs = [] 
        n = numpy_images.shape[0]
        for start in range(0, n, self.predict_chunk_size):
            chunk = numpy_images[start:start + self.predict_chunk_size]
            tensor = torch.from_numpy(chunk).float().to(self.device)
            tensor = tensor / 255.0
            tensor = tensor.permute(0, 3, 1, 2)
            tensor = self._norm_transform(tensor)
            with torch.no_grad():
                logits = self.model(tensor)
            outputs.append(logits.cpu().numpy())
            del tensor, logits, chunk
        return np.concatenate(outputs, axis=0)
    
    def _lime_predict_fn(self, numpy_images):
        return self._predict_in_chunks(numpy_images)

    def compute_lime(self, raw_img_uint8, target_class):
        """Generating the LIME explanation"""
        self.lime_explainer.random_state.seed(self.seed)
        explanation = self.lime_explainer.explain_instance(
            raw_img_uint8.astype('double'),
            self._lime_predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=self.lime_samples,
            random_seed=self.seed,
        )

        if target_class not in explanation.local_exp:
            raise ValueError(
                f"LIME did not explain class {target_class} "
                f"(only explained {list(explanation.local_exp.keys())}); increase top_labels."
            )
        return explanation
    

    def get_lime_heatmap(self, explanation, target_class):
        """Generating the LIME heatmap"""
        dict_heatmap = dict(explanation.local_exp[target_class])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        heatmap = self.normalize_map(heatmap)
        return self.resize_map(heatmap)

    def get_lime_localization_heatmap(self, explanation, target_class):
        """Generating the LIME heatmap using localisation normalisation"""
        dict_heatmap = dict(explanation.local_exp[target_class])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        heatmap = self.localization_normalize_map(heatmap)
        return self.resize_map(heatmap)

    def show_lime(self, raw_img_uint8, target_class, num_features=5, positive_only=True):
        """Displays the LIME maps."""
        explanation = self.compute_lime(raw_img_uint8, target_class)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(f"LIME Detailed Subpixel Diagnostics: {type(self.model).__name__} (Class {target_class})",fontsize=14, weight='bold', y=1.02)

        # Original image
        axes[0].imshow(raw_img_uint8)
        axes[0].set_title("Original Image Structure")
        axes[0].axis("off")

        # Positive-only masking, using the method's own parameters
        temp1, mask1 = explanation.get_image_and_mask(
            target_class, positive_only=positive_only, num_features=num_features, hide_rest=True
        )
        axes[1].imshow(mark_boundaries(temp1 / 255.0, mask1))
        axes[1].set_title(f"Top {num_features} {'Positive' if positive_only else 'Pos/Neg'} Features")
        axes[1].axis("off")

        # Pros & cons boundary (fixed at 10 features, both signs, rest shown)
        temp2, mask2 = explanation.get_image_and_mask(
            target_class, positive_only=False, num_features=10, hide_rest=False
        )
        axes[2].imshow(mark_boundaries(temp2 / 255.0, mask2))
        axes[2].set_title("Top 10 Pros/Cons Boundary")
        axes[2].axis("off")

        # Symmetrical continuous weight heatmap
        dict_heatmap = dict(explanation.local_exp[target_class])
        heatmap_matrix = np.vectorize(dict_heatmap.get)(explanation.segments)

        max_val = np.abs(heatmap_matrix).max() if np.abs(heatmap_matrix).max() > 0 else 1.0
        im = axes[3].imshow(heatmap_matrix, cmap='RdBu', vmin=-max_val, vmax=max_val)
        axes[3].set_title("Symmetrical Weight Distribution")
        axes[3].axis("off")

        fig.colorbar(im, ax=axes[3], orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def close_lime_explanation(self, explanation):
        explanation.segments = None
        explanation.local_exp = None
        explanation.local_pred = None

    def _shap_predict_fn(self, numpy_images):
        return self._predict_in_chunks(numpy_images)

    def _build_shap_explainer(self, image_shape):
        """Builds the SHAP explainer object."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        masker_blur = shap.maskers.Image("blur(32,32)", image_shape)
        self.shap_explainer = shap.Explainer(self._shap_predict_fn, masker_blur, output_names=self.class_names)

    def compute_shap(self, raw_img_uint8, target_class=None, all_classes=True):
        """Return the SHAP values for the given image."""

        if self.shap_explainer is None:
            self._build_shap_explainer(raw_img_uint8.shape)

        images_batch = raw_img_uint8[np.newaxis, ...]
        n_classes = len(self.class_names) if self.class_names else 5
        outputs = list(range(n_classes)) if all_classes else [target_class]

        return self.shap_explainer(images_batch,max_evals=self.shap_max_evals,batch_size=self.shap_batch_size,outputs=outputs,)

    def get_shap_heatmap(self, shap_values, class_position=0):
        """Generating the SHAP heatmaps"""
        raw_shap = shap_values.values[0, :, :, :, class_position]
        sv_signed = raw_shap.sum(axis=-1)
        heatmap = self.normalize_map(sv_signed)
        return self.resize_map(heatmap), self.resize_map(sv_signed)

    def get_shap_localization_heatmap(self, shap_values, class_position=0):
        """Generating the SHAP heatmaps for localisation"""
        raw_shap = shap_values.values[0, :, :, :, class_position]
        sv_signed = raw_shap.sum(axis=-1)
        heatmap = self.localization_normalize_map(sv_signed)
        return self.resize_map(heatmap), self.resize_map(sv_signed)

    def show_shap(self, raw_img_uint8, target_class=None, all_classes=True):
        """Visualizes SHAP values using SHAP's own built-in image_plot."""
        shap_values = self.compute_shap(raw_img_uint8, target_class=target_class, all_classes=all_classes)
        shap.image_plot(shap_values)
        
    def compute_all(self, img_tensor, label):
        """Runs all the XAI techniques at once, returns a dictionary with the results."""
        self.model.eval()

        raw_img_uint8 = self.unnormalize_to_uint8(img_tensor)
        img = self.uint8_to_model_input(raw_img_uint8)

        with torch.no_grad():
            pred = self.model(img).argmax(dim=1).item()

        gradcam_map = self.compute_gradcam(img, pred)

        lime_explanation = self.compute_lime(raw_img_uint8, pred)
        lime_map = self.get_lime_heatmap(lime_explanation, pred)
        self.close_lime_explanation(lime_explanation)
        del lime_explanation

        shap_values = self.compute_shap(raw_img_uint8, target_class=pred, all_classes=False)
        shap_map, sv_signed = self.get_shap_heatmap(shap_values, class_position=0)
        del shap_values

        del img, raw_img_uint8
        return {'lime': lime_map, 'shap': shap_map, 'visual_shap': sv_signed, 'gradcam': gradcam_map, 'pred': pred}

    def localization_compute_all(self, img_tensor, label=None):
        """Runs all the XAI techniques at once with localisation normalisation, returns a dictionary with the results."""
        self.model.eval()

        raw_img_uint8 = self.unnormalize_to_uint8(img_tensor)
        img = self.uint8_to_model_input(raw_img_uint8)

        with torch.no_grad():
            pred = self.model(img).argmax(dim=1).item()

        gradcam_map = self.compute_localization_gradcam(img, pred)

        lime_explanation = self.compute_lime(raw_img_uint8, pred)
        lime_map = self.get_lime_localization_heatmap(lime_explanation, pred)
        self.close_lime_explanation(lime_explanation)
        del lime_explanation

        shap_values = self.compute_shap(raw_img_uint8, target_class=pred, all_classes=False)
        shap_map, sv_signed = self.get_shap_localization_heatmap(shap_values, class_position=0)
        del shap_values

        del img, raw_img_uint8
        return {'lime': lime_map, 'shap': shap_map, 'visual_shap': sv_signed, 'gradcam': gradcam_map, 'pred': pred}
        