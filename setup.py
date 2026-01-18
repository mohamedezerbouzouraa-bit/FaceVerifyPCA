from setuptools import setup, find_packages

setup(
    name="faceverifypca",
    version="1.0.0",
    description="PCA-based facial verification system",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)
