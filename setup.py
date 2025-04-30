from setuptools import setup, find_packages

setup(
    name="AutoBTE",
    version="0.1.0",
    description="A package for automating Boltzmann Transport Equation calculations.",
    author="Daehong Kim",
    author_email="adgjl7777777@naver.com",
    url="https://github.com/adgjl7777777/AutoBTE",
    packages=find_packages(),
    install_requires=[
        "BoltzTraP2>=24.9.0",
        "chgnet>=0.4.0",
        "matplotlib>=3.9.0",
        "MDAnalysis>=2.8.0",
        "pyFFTW>=0.15.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)