from setuptools import setup, find_packages

setup(
    name="tabularforge",
    version="1.0.0",
    description="A tabular foundation model framework (SCM synthetic data + masked cell modeling + zero-shot imputation).",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={"console_scripts": ["tabularforge=tabularforge.cli.main:main"]},
)
