# utils.py
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def show_comparison(title, orig_p, v3_p, v4_p, v4f_p):
    imgs = []
    for p in [orig_p, v3_p, v4_p, v4f_p]:
        img = cv2.imread(p, 0)
        imgs.append(img)
        
    labels = ["Original", "Cleaned v3", "Cleaned v4", "Feathered v4 (Final)"]
    plt.figure(figsize=(16, 5))
    for i, (im, lbl) in enumerate(zip(imgs, labels)):
        plt.subplot(1, 4, i + 1)
        if im is not None:
            plt.imshow(im, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def safe_compare(title, img_path, v3_path, v4_path):
    orig = cv2.imread(img_path, 0)
    clean_v3 = cv2.imread(v3_path, 0)
    clean_v4 = cv2.imread(v4_path, 0)
    
    if orig is None:
        print("❌ Original image not found or unreadable:", img_path)
        return
        
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(orig, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    if clean_v3 is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(clean_v3, cmap='gray')
        plt.title('Cleaned v3')
        plt.axis('off')
        
    if clean_v4 is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(clean_v4, cmap='gray')
        plt.title('Cleaned v4')
        plt.axis('off')
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()

def visual_preview_fixed(original_root, cleaned_root, num_samples=6):
    pairs = []
    for root, _, files in os.walk(original_root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                rel_path = os.path.relpath(os.path.join(root, f), original_root)
                cleaned_path = os.path.join(cleaned_root, rel_path)
                if os.path.exists(cleaned_path):
                    pairs.append((os.path.join(root, f), cleaned_path))
                    
    if not pairs:
        print("Check your file systems — no matching image paths found.")
        return
        
    samples = random.sample(pairs, min(num_samples, len(pairs)))
    fig, axes = plt.subplots(len(samples), 2, figsize=(8, len(samples) * 4))
    
    if len(samples) == 1:
        axes = np.expand_dims(axes, 0)
        
    for i, (orig_path, clean_path) in enumerate(samples):
        img_o = cv2.imread(orig_path, 0)
        img_c = cv2.imread(clean_path, 0)
        if img_o is None or img_c is None:
            continue
        axes[i, 0].imshow(img_o, cmap='gray')
        axes[i, 0].set_title(f"Original\n{os.path.basename(orig_path)}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(img_c, cmap='gray')
        axes[i, 1].set_title("Cleaned Output")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()