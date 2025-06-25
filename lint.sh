#!/bin/bash

# Lint

# Pyuprade
pyupgrade `find src -name "*.py"`

# isort
isort src

# black
black src

# pep8
autopep8 src

# flake
flake8 src
