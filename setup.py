from setuptools import setup, find_packages

setup(
    name="minicv",
    version="1.0.0",
    description="A lightweight OpenCV-inspired image processing library built on NumPy.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
    ],
)
