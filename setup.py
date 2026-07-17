#This file is for project management
#to run the setup file - pip install -e .

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-PROJECT-1",
    version="0.1",
    author="Mide",
    packages= find_packages(),
    install_requires= requirements,
)