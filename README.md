FaceVerifyPCA 🎭
A PCA-based facial verification system using eigenface decomposition for identity authentication.

📌 Quick Start
python
# 1. Place training images (m1.jpg, m2.jpg, etc.) in folder
# 2. Run the script
python face_verify_pca.py
# 3. System trains on known faces, verifies test images
🚀 Features
✅ PCA/SVD-based face recognition

✅ Automatic threshold calculation

✅ Visual reconstruction comparisons

✅ Multiple distance metrics

✅ Batch image processing

📚 How It Works
Training: PCA extracts eigenfaces from known images

Projection: Test images mapped to face space

Verification: Distance comparison with adaptive threshold
