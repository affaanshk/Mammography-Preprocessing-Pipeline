# mask_blending.py
import cv2
import numpy as np

def apply_clean(img_gray, pred_mask):
    """
    Suppresses artifact and noise zones using mathematical distance 
    transforms and Gaussian blurring to seamlessly blend pixels.
    """
    # Create binary mask of to-remove regions (pectoral or noise)
    rem = (pred_mask == 1) | (pred_mask == 2)
    rem = rem.astype(np.uint8) * 255
    
    # Feather using distance transform and gaussian blur
    dist = cv2.distanceTransform(255 - rem, cv2.DIST_L2, 5)
    maxd = dist.max() if dist.max() > 0 else 1.0
    alpha = np.clip(dist / (0.05 * maxd + 1e-8), 0, 1) 
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=5)
    
    # Blend: keep original where alpha ~ 1, suppress where alpha ~ 0
    cleaned = (img_gray.astype(np.float32) * alpha + (1 - alpha) * 0).astype(np.uint8)
    return cleaned

def smooth_and_feather_mask(img_gray, threshold_val=15, kernel_size=7, feather_radius=11):
    """
    Generates an estimated mask from dark background pixel distributions, 
    applies morphologic cleaning, and maps structural feathering values.
    """
    mask = (img_gray < threshold_val).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    feather = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (feather_radius | 1, feather_radius | 1), 0)
    feather = np.clip(feather, 0.0, 1.0)
    return feather

def apply_mask_blend(img_gray, feather_mask):
    """
    Blends foreground tissue structures flawlessly into uniform 
    background areas using feather weight maps.
    """
    fg = img_gray.astype(np.float32)
    bg_val = np.median(img_gray[0:10, 0:10])
    bg = np.ones_like(fg) * bg_val
    
    blended = (fg * (1 - feather_mask) + bg * feather_mask).astype(np.uint8)
    return blended