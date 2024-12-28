import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

valid_transforms = A.Compose([
    A.Resize(256, 256),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
 
])

# Create a dummy image
img = np.zeros((512, 512, 3), dtype=np.uint8)
transformed_img=valid_transforms(image=img)['image']
print(transformed_img.shape)
