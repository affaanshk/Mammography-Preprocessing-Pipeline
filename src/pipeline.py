# pipeline.py
import os
import cv2
import numpy as np
from normalization import normalize_intensity
from pectoral_suppression import PectoralRemover
from mask_blending import apply_clean, smooth_and_feather_mask, apply_mask_blend

class MammographyPreprocessingPipeline:
    def __init__(self, weights_path, device=None):
        self.remover = PectoralRemover(model_path=weights_path, device=device)

    def preprocess_image(self, image_path):
        """
        Processes a single digital mammogram file.
        """
        raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise FileNotFoundError(f"Failed to resolve image array path: {image_path}")
            
        # 1. Predict and isolate boundaries
        pred_mask = self.remover.predict_mask(raw_img)
        # ✅ FIXED: Unpacking corrected to match the return signature
        cleaned_tissue, pectoral_mask = self.remover.remove_pectoral(raw_img, pred_mask)
        
        # 2. Extract artifact suppressions via distance maps
        artifact_suppressed = apply_clean(cleaned_tissue, pred_mask)
        
        # 3. Create feather mask configurations
        feather_mask = smooth_and_feather_mask(artifact_suppressed)
        
        # 4. Blend background matrices and normalize intensity fields
        final_blended_output = apply_mask_blend(artifact_suppressed, feather_mask)
        final_normalized_output = normalize_intensity(final_blended_output)
        
        return final_normalized_output

def process_directory(input_dir, output_dir, pipeline_instance):
    """
    Processes an entire directory structure iteratively using an instantiated pipeline.
    """
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        save_root = os.path.join(output_dir, rel_root)
        os.makedirs(save_root, exist_ok=True)
        
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                in_path = os.path.join(root, f)
                out_path = os.path.join(save_root, f)
                
                try:
                    cleaned_img = pipeline_instance.preprocess_image(in_path)
                    cv2.imwrite(out_path, cleaned_img)
                except Exception as e:
                    print(f"⚠️ Error processing file {in_path}: {str(e)}")