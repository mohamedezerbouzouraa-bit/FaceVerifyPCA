"""
Image loading and preprocessing module
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

class ImageLoader:
    """Handles image loading and preprocessing"""
    
    def __init__(self, image_size=(100, 100)):
        self.image_size = image_size
        
    def load_image(self, image_path):
        """Load and preprocess a single image"""
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found")
            return None
            
        try:
            # Load and convert to grayscale
            img = Image.open(image_path).convert('L')
            # Resize
            img = img.resize(self.image_size)
            # Normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array.flatten()
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def load_from_folder(self, folder_path, image_names=None):
        """Load multiple images from a folder"""
        images = []
        loaded_names = []
        
        if image_names is None:
            # Load all images in folder
            image_names = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Loading {len(image_names)} images from {folder_path}")
        
        for img_name in tqdm(image_names, desc="Loading images"):
            img_path = os.path.join(folder_path, img_name)
            img_vector = self.load_image(img_path)
            
            if img_vector is not None:
                images.append(img_vector)
                loaded_names.append(img_name)
        
        if not images:
            return None, []
            
        return np.array(images), loaded_names
    
    def save_processed(self, images, names, output_dir):
        """Save processed images"""
        os.makedirs(output_dir, exist_ok=True)
        for img_array, name in zip(images, names):
            # Reshape and denormalize
            img_reshaped = (img_array.reshape(self.image_size) * 255).astype(np.uint8)
            img = Image.fromarray(img_reshaped)
            img.save(os.path.join(output_dir, f"processed_{name}"))
