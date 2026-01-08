from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quant-portfolio-optimization",
    version="1.0.0",
    author="Quantitative Research Team",
    author_email="research@example.com",
    description="Quantitative portfolio optimization and backtesting system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/quant-portfolio-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.18",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.2.0",
        "cvxpy>=1.3.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-portfolio=main:main",
        ],
    },
)
