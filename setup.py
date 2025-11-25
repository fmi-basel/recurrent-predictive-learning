from setuptools import find_packages, setup


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="recurrent-predictive-learning",
    version="0.1.0",
    install_requires=requirements,
    packages=find_packages(),
)
