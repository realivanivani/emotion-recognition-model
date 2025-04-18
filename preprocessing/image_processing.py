import cv2
import numpy as np

def load_image(image_path):
    """Load image with OpenCV"""
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(img, target_size=(224, 224)):
    """Resize and normalize image"""
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    # Normalize based on ImageNet stats if using pretrained models
    img[..., 0] -= 0.485
    img[..., 1] -= 0.456
    img[..., 2] -= 0.406
    img[..., 0] /= 0.229
    img[..., 1] /= 0.224
    img[..., 2] /= 0.225
    return img
