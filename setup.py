"""Full-TS Cognitive Architecture - Setup."""
from setuptools import setup, find_packages

setup(
    name="full-ts-cognitive-architecture",
    version="0.1.0",
    description="Full-scale Thinking System cognitive architecture (GOAT-TS)",
    author="Full-TS",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "sentence-transformers>=2.2.0",
        "networkx>=3.2.0",
        "neo4j>=5.15.0",
        "faiss-cpu>=1.7.4",
        "fairlearn>=0.9.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "celery[redis]>=5.3.0",
        "redis>=5.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.23.0"],
        "finetune": ["peft>=0.7.0", "trl>=0.7.0", "bitsandbytes>=0.41.0"],
    },
    entry_points={
        "console_scripts": [
            "full-ts=src.main:main",
        ],
    },
)
