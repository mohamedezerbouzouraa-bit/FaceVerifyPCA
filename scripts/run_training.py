

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_loader import ImageLoader
from src.pca.pca_core import PCAFaceModel
import pickle
from config import settings

def train_model():
    print("Training PCA Face Model")
    print("=" * 50)
    
    # Load images
    loader = ImageLoader(settings.IMAGE_SIZE)
    images, names = loader.load_from_folder(
        settings.RAW_IMAGES_DIR,
        image_names=["m1.jpg", "m2.jpg", "m3.jpg", "m4.jpg",
                    "m6.jpg", "m7.jpg", "m8.jpg", "m9.jpg",
                    "m10.jpg", "m11.jpg", "m12.jpg"]
    )
    
    if images is None:
        print("No images loaded. Exiting.")
        return
    
    pca_model = PCAFaceModel(
        variance_threshold=settings.PCA_VARIANCE_THRESHOLD,
        min_components=settings.MIN_COMPONENTS,
        max_components=settings.MAX_COMPONENTS
    )
    pca_model.train(images)
    
    # Saving model
    model_dir = os.path.join(settings.BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "pca_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': pca_model,
            'train_images': images,
            'train_names': names,
            'image_size': settings.IMAGE_SIZE
        }, f)
    
    print(f"\nModel saved to: {model_path}")
    print("Training complete!")

if __name__ == "__main__":
    train_model()
