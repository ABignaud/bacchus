#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BACterial CHromosome USeful tools for genomics study. (BACCHUS)
"""

from setuptools import setup, find_packages
import codecs

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Visualization",
    "Operating System :: Unix",
]

name = "bacchus"

MAJOR = 1
MINOR = 0
MAINTENANCE = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MAINTENANCE)

LICENSE = "BSD3"
URL = "https://github.com/ABignaud/bacchus"

DESCRIPTION = __doc__.strip("\n")

with codecs.open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().splitlines()

with open("version.py", "w") as f:
    f.write("__version__ = '{}'\n".format(VERSION))


setup(
    name=name,
    author="amaury.bignaud@pasteur.fr",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    version=VERSION,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    url=URL,
    packages=find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
    long_description_content_type="text/markdown",
    install_requires=REQUIREMENTS,
)
