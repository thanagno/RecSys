"""
Setup file defining the packages attributes and properties
"""
from pathlib import Path
from typing import List
from setuptools import find_packages, setup


def read_requirements() -> List[str]:
    """Returns requirements.txt parsed to a list"""
    requirements_file_path = Path(__file__).parent / 'requirements.txt'
    required_packages = []
    if requirements_file_path.is_file():
        with open(requirements_file_path, 'r') as req:
            required_packages = req.read().splitlines()

    return required_packages


setup(
    name='recommender',
    package_dir={'': 'src'},
    packages=find_packages(
        "src",
        exclude=("test", "docs", "data", "notebooks", "archive", "loggers")
    ),
    version='0.1.0',
    description='Collaborative filtering user based recommendations',
    author='George Chalkiopoulos',
    install_requires=read_requirements(),  # Include the dependencies from requirements.txt
)
