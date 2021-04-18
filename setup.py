from setuptools import setup

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
    python_requires = ">=3.6",
    license = "MIT",
    license_files = ["LICENSE",],
    packages = ["torchdescribe",],
    install_requires = ["torch",],
)
