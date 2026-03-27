"""Python setup.py for lbp-neural-cbf package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("lbp-neural-cbf", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    requirements = []
    for line in read(path).split("\n"):
        line = line.strip()
        if line.startswith("-r "):
            # Recursively read referenced requirements files
            ref_path = line.split()[1]
            requirements.extend(read_requirements(ref_path))
        elif not line.startswith(('"', "#", "-", "git+")):
            if line:  # Skip empty lines
                requirements.append(line)
    return requirements


setup(
    name="lbp_neural_cbf",
    version=read("lbp_neural_cbf", "VERSION"),
    description="lbp-neural-cbf created by nikovert, Zinoex",
    url="https://github.com/Zinoex/scalable-verification-of-neural-control-barrier-functions-using-linear-bound-propagation/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="nikovert, Zinoex",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"test": read_requirements("requirements-test.txt")},
)
