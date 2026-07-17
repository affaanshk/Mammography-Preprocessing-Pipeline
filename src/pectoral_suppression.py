# pectoral_suppression.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2

class UNetResNet18(nn.Module):
    """
    U-Net structural architecture mapping a ResNet18 encoder backbone.
    Required by PyTorch for parsing state parameters during execution.
    """
    def __init__(self, n_classes=3):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256 + 256, 128, 2, 2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 2, 2), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 64, 2, 2), nn.ReLU(inplace=True))
        self.up0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.final = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d4 = self.up4(x4)
        d3 = self.up3(torch.cat([d4, x3], 1))
        d2 = self.up2(torch.cat([d3, x2], 1))
        d1 = self.up1(torch.cat([d2, x1], 1))
        d0 = self.up0(d1)
        return self.final(d0)

class PectoralRemover:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetResNet18(n_classes=3).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _postprocess_mask(self, mask, threshold=0.5):
        mask_bin = (mask > threshold).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
        mask_bin = cv2.medianBlur(mask_bin, 5)
        return mask_bin.astype(np.uint8)

    def predict_mask(self, img_gray):
        orig_h, orig_w = img_gray.shape[:2]
        img_resized = cv2.resize(img_gray, (256, 256))
        
        tensor = torch.tensor(img_resized / 255.).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 3, 1, 1).to(self.device)
        
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred = np.argmax(probs, axis=0).astype(np.uint8)
            
        pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return pred_full

    def remove_pectoral(self, img_gray, pred_mask):
        mask = (pred_mask == 1).astype(np.uint8)
        mask_smoothed = self._postprocess_mask(mask)
        
        cleaned = img_gray.copy()
        cleaned[mask_smoothed > 0] = 0
        return cleaned, mask_smoothed

    def remove_pectoral_from_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        mask = self.predict_mask(img)
        # ✅ FIXED: Corrected unpacking bug as recommended by advisor
        cleaned, final_mask = self.remove_pectoral(img, mask)
        return img, final_mask, cleaned