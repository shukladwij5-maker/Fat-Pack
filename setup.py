from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fattummy",
    version="0.2.6",
    author="Origin-Labs",
    author_email="Shukladwij5@gmail.com",
    description="A declarative, ultra-minimalist ML framework for zero-boilerplate hardware-agnostic inference and training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shukladwij5-maker/fattummy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[],
    extras_require={
        "data": ["datasets", "huggingface_hub"],
        "hf": ["transformers", "torch"],
        "native": ["torch"],
        "train": ["torch"],
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "gemini": ["google-genai"],
        "all": [
            "datasets",
            "huggingface_hub",
            "transformers",
            "torch",
            "openai",
            "anthropic",
            "google-genai",
        ],
    },
)
