from __future__ import annotations

import os
from typing import Dict

import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as T


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):   # ✅ changed to 2 classes
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class QualityClassifier:
    def __init__(self, weights_path: str = 'models/quality_classifier.pt', device: str | None = None):

        # ✅ only 2 classes now
        self.class_names = ['fresh', 'rotten']

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model = SmallCNN(num_classes=len(self.class_names)).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((160, 160)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.is_trained = False
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.is_trained = True

    # ✅ ADD THIS (missing in your code)
    def preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = self.transform(img_rgb)
        return img.unsqueeze(0).to(self.device)

    # ✅ FIXED FUNCTION
    def predict_proba(self, img_bgr):

        # 🔁 Fallback if model not trained
        if not self.is_trained:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            brightness = np.mean(hsv[:, :, 2])
            saturation = np.mean(hsv[:, :, 1])

            if brightness > 100 and saturation > 90:
                return {'fresh': 0.9, 'rotten': 0.1}
            else:
                return {'fresh': 0.2, 'rotten': 0.8}

        # ✅ Normal model prediction
        img = self.preprocess(img_bgr)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {
            'fresh': float(probs[0]),
            'rotten': float(probs[1])
        }