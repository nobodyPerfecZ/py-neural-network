from setuptools import setup, find_packages

# Load the requirements from requirements.txt
with open("requirements.txt", "r") as file:
    req = file.read().splitlines()

# Load the README from README.md
with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="PyNeuralNet",
    version="0.0.1",
    author="Dennis J.",
    url="https://github.com/nobodyPerfecZ/py-neural-network",
    python_requires=">=3.10",
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
        ]
    ),
    install_requires=req,
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)