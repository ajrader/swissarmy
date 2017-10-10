from setuptools import setup

setup(
  name="swissarmy",
  version="0.0.1",
  author="A.J. Rader",
  author_email="arader@dmcinsurnace.com",
  description=("Python Exploratory data analysis (EDA) toolkit"),

  keywords="pandas exploratory data analysis",
  install_requires = [
    "matplotlib>=1.5.1",
    "pandas>=0.17.1",
    "numpy>=1.10.4",
    "scipy>=0.17.0",
    "scikit-learn>=0.18",
  ],
  packages=['swissarmy']
)