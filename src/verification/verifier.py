"""
Face verification module
"""

import os
import numpy as np
from .threshold import ThresholdCalculator

class FaceVerifier:
    """Verifies faces against trained model"""
    
    def __init__(self, pca_model, train_images, train_names):
        self.pca_model = pca_model
        self.train_images = train_images
        self.train_names = train_names
        self.threshold = None
        self.threshold_calculator = ThresholdCalculator()
        
        # Project training images
        self.train_projections = self.pca_model.project(train_images)
        
    def compute_threshold(self):
        """Compute verification threshold from training data"""
        self.threshold = self.threshold_calculator.compute_adaptive_threshold(
            self.train_projections
        )
        return self.threshold
    
    def verify_image(self, image_path):
        """Verify a single image"""
        from ..preprocessing.image_loader import ImageLoader
        
        # Load and preprocess image
        loader = ImageLoader()
        img_vector = loader.load_image(image_path)
        
        if img_vector is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Project onto PCA space
        test_projection = self.pca_model.project(img_vector.reshape(1, -1))[0]
        
        # Compute distances to training set
        distances = np.linalg.norm(self.train_projections - test_projection, axis=1)
        min_distance = np.min(distances)
        closest_idx = np.argmin(distances)
        
        # Make verification decision
        verified = min_distance < self.threshold if self.threshold is not None else False
        
        # Reconstruct image
        reconstructed = self.pca_model.reconstruct(test_projection.reshape(1, -1))[0]
        
        return {
            'filename': os.path.basename(image_path),
            'distance': float(min_distance),
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'verified': verified,
            'closest_match': self.train_names[closest_idx],
            'closest_index': int(closest_idx),
            'projection': test_projection,
            'reconstructed': reconstructed
        }
    
    def verify_batch(self, image_paths):
        """Verify multiple images"""
        results = []
        for path in image_paths:
            try:
                result = self.verify_image(path)
                results.append(result)
            except Exception as e:
                print(f"Error verifying {path}: {e}")
        return results
