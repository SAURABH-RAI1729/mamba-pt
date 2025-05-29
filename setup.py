"""
Setup script for MAMBA-PT
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mamba-pt",
    version="0.1.0",
    author="Saurabh Rai",
    author_email="saurabhraigr1729@gmail.com",
    description="MAMBA-based architecture for particle tracking in High Energy Physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SAURABH-RAI1729/mamba-pt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6b0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mamba-trackingbert-train=train:main",
            "mamba-trackingbert-evaluate=evaluate:main",
            "mamba-trackingbert-preprocess=scripts.preprocess_data:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
