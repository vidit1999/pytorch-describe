import os
from setuptools import setup

email = os.environ.get("USER_EMAIL", "")
description = "Describe PyTorch model in PyTorch way"

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except:
    long_description = description


setup(
    name = "torchdescribe",
    version = "1.0.0",
    description = description,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/vidit1999/pytorch-describe",
    author = "Vidit Sarkar",
    author_email = email,
    python_requires = ">=3.6",
    classifiers = [
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license = "MIT",
    license_files = ["LICENSE",],
    packages = ["torchdescribe",],
    install_requires = ["torch",],
)
