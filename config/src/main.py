#!/usr/bin/env python3
"""
Main entry point for FaceVerifyPCA system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_loader import ImageLoader
from src.pca.pca_core import PCAFaceModel
from src.verification.verifier import FaceVerifier
from src.visualization.plotter import ResultPlotter
from config import settings

def main():
    """Main execution function"""
    print("=" * 60)
    print("FaceVerifyPCA - PCA-Based Face Verification System")
    print("=" * 60)
    
    # Load configuration
    config = {
        'image_size': settings.IMAGE_SIZE,
        'pca_variance': settings.PCA_VARIANCE_THRESHOLD,
        'data_dir': settings.RAW_IMAGES_DIR
    }
    
    # Step 1: Load images
    print("\n[1/4] Loading training images...")
    loader = ImageLoader(config['image_size'])
    train_images, train_names = loader.load_from_folder(
        folder_path=config['data_dir'],
        image_names=["m1.jpg", "m2.jpg", "m3.jpg", "m4.jpg",
                    "m6.jpg", "m7.jpg", "m8.jpg", "m9.jpg",
                    "m10.jpg", "m11.jpg", "m12.jpg"]
    )
    
    if train_images is None:
        print("Error: No training images loaded!")
        return
    
    # Step 2: Train PCA model
    print("\n[2/4] Training PCA model...")
    pca_model = PCAFaceModel(variance_threshold=config['pca_variance'])
    pca_model.train(train_images)
    
    # Step 3: Create verifier
    print("\n[3/4] Setting up verifier...")
    verifier = FaceVerifier(pca_model, train_images, train_names)
    verifier.compute_threshold()
    
    # Step 4: Test images
    print("\n[4/4] Testing verification...")
    test_images = ["m5.jpg", "mx.jpg"]
    
    results = []
    for test_img in test_images:
        test_path = os.path.join(config['data_dir'], test_img)
        if os.path.exists(test_path):
            result = verifier.verify_image(test_path)
            results.append(result)
            
            # Display result
            status = "✅ VERIFIED" if result['verified'] else "❌ REJECTED"
            print(f"\n{test_img}: {status}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Threshold: {result['threshold']:.4f}")
    
    # Step 5: Visualize results
    print("\nGenerating visualizations...")
    plotter = ResultPlotter()
    plotter.plot_results(results, save_dir=settings.RESULTS_DIR)
    
    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print(f"Results saved to: {settings.RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
