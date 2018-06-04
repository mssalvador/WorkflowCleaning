#!/usr/bin/env python3

import sys
from setuptools import setup, find_packages


def setup_package():
    setup(
        name = 'dabaiworkflows',

        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version = '0.0.3',

        install_requires = open('requirements.txt').read().split(),

        packages = find_packages(),
        url = 'https://github.com/mssalvador/ReadData',
        license = 'Apache License, Version 2.0',
        author = 'Michael Salvador Svanholm, Sidsel Sørensen',
        author_email = 'michael.salvador.svanholm@visma.com, sidsel.sorensen@visma.com',
        description = 'This is a short test project for the Dabai-team @ Visma',

        long_description = open('./README.rst').read()
)


if __name__ == "__main__":
    setup_package()
