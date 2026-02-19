from setuptools import setup, find_packages

setup(
    name="data-quality-framework",
    version="0.1.0",
    author="LAM CHIT WUI",
    author_email="sgclam6@liverpool.ac.uk",
    description="Adaptive ML Framework for Data Quality Assessment",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "pyspark>=3.5.0",
        "pandas>=2.2.0",
        "numpy>=2.1.0",
        "scikit-learn>=1.5.0",
        "streamlit>=1.41.0",
        "plotly>=5.24.0",
        "joblib>=1.3.0",
        "pyarrow>=18.0.0",
        "openpyxl>=3.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.12",
    ],
)
