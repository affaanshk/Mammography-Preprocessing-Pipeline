# normalization.py
import cv2
import numpy as np

def read_gray(path, target_size=(256, 256)):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_NEAREST)
    return im

def binarize_mask(mask, label_val):
    return (mask == label_val).astype(np.uint8)

def normalize_intensity(img_array):
    img_float = img_array.astype(np.float32)
    min_val = img_float.min()
    max_val = img_float.max()
    
    normalized = (img_float - min_val) / (max_val - min_val + 1e-8)
    return (normalized * 255).astype(np.uint8)