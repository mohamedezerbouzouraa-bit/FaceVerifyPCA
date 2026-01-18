"""
PCA core implementation using SVD
"""

import numpy as np

class PCAFaceModel:
    """PCA model for face recognition"""
    
    def __init__(self, variance_threshold=0.95, min_components=2, max_components=50):
        self.variance_threshold = variance_threshold
        self.min_components = min_components
        self.max_components = max_components
        
        # Model parameters
        self.mean_face = None
        self.components = None
        self.singular_values = None
        self.cumulative_variance = None
        
    def train(self, X):
        """Train PCA model on face data"""
        # Compute mean face
        self.mean_face = np.mean(X, axis=0)
        
        # Center data
        X_centered = X - self.mean_face
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Calculate variance explained
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        total_variance = np.sum(explained_variance)
        variance_ratio = explained_variance / total_variance
        self.cumulative_variance = np.cumsum(variance_ratio)
        
        # Determine number of components
        n_components = np.argmax(self.cumulative_variance >= self.variance_threshold) + 1
        n_components = max(self.min_components, min(n_components, self.max_components, Vt.shape[0]))
        
        # Store components
        self.components = Vt[:n_components]
        self.singular_values = S[:n_components]
        
        print(f"PCA Model Trained:")
        print(f"  - Training samples: {X.shape[0]}")
        print(f"  - Features per sample: {X.shape[1]}")
        print(f"  - Components kept: {n_components}")
        print(f"  - Variance explained: {self.cumulative_variance[n_components-1]*100:.1f}%")
        
        return self
    
    def project(self, X):
        """Project data onto PCA space"""
        if self.mean_face is None or self.components is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_centered = X - self.mean_face
        return np.dot(X_centered, self.components.T)
    
    def reconstruct(self, projections):
        """Reconstruct data from PCA space"""
        return np.dot(projections, self.components) + self.mean_face
    
    def get_model_info(self):
        """Get model information"""
        return {
            'num_components': self.components.shape[0],
            'mean_face_shape': self.mean_face.shape,
            'variance_explained': self.cumulative_variance[self.components.shape[0]-1],
            'singular_values': self.singular_values
        }
