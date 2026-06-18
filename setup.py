from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fattummy",
    version="0.1.0",
    author="FatTummy Engineering",
    author_email="your.email@example.com",
    description="A declarative, ultra-minimalist ML framework for zero-boilerplate hardware-agnostic inference and training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fattummy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        # Dependencies are managed dynamically by FatTummy's installer.py!
        # But we can list the absolute bare minimum here if needed.
    ],
)
