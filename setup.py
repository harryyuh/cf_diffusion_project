from setuptools import setup

setup(
    name="cf-diffusion-celeba",
    version="0.1.0",
    description="Conditional VAE + diffusion on CelebA (cf-diffusion-project style)",
    packages=["data", "models", "utils", "improved_diffusion", "training", "evaluation", "inference"],
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "Pillow",
    ],
)
