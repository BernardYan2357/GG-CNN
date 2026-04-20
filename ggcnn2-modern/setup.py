"""
setup.py for ggcnn2-modern package.
Supports Python 3.10+ with modern PyTorch (2.0+).
"""

from setuptools import find_packages, setup

setup(
    name="ggcnn2-modern",
    version="2.0.0",
    description="Modern GGCNN2 implementation for Python 3.10+ and PyTorch 2.0+",
    author="GGCNN2 Modern",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "imageio>=2.28.0",
        "imageio[tifffile]",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
