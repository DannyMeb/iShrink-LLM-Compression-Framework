from setuptools import setup, find_packages

setup(
    name="llm_pruning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if not line.startswith("#") and not line.startswith("-e")
    ],
    extras_require={
        'dev': [
            'black>=23.3.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'pytest>=7.3.1'
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0'
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="LLM Pruning using Reinforcement Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm_pruning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)