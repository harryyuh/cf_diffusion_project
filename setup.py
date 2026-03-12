from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="cf_diffusion",
    version="0.1.0",
    description="Counterfactual thickness editing on MorphoMNIST with VAE + conditional diffusion",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=9.0.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
    ],
)
