# phoNy: phonology in spaCy!
[![ci status](https://github.com/direct-phonology/phoNy/actions/workflows/ci.yml/badge.svg)](https://github.com/direct-phonology/phoNy/actions/workflows/ci.yml)
[![pypi version](https://img.shields.io/pypi/v/spacy-phony.svg?style=flat)](https://pypi.org/project/spacy-phony]/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

phoNy is a spaCy extension that adds pipeline components, models, and features for working with phonology.
## installation
requires spacy v3.
```sh
pip install spacy-phony
```
## usage
this package currently provides a single pipeline component, the `Phonemizer`, which performs grapheme-to-phoneme conversion. documentation is coming soon!

## developing
after cloning the repository:
```sh
pip install -e ".[dev]"
pre-commit install
```

## testing
to run tests:
```sh
python -m unittest
```
you can also generate a [coverage report](https://coverage.readthedocs.io/en/latest/):
```sh
coverage run --source=src -m unittest discover
coverage report
```
## building
clear out any previously built packages before building:
```sh
rm -rf dist/*
```
build a source archive and distribution for a release:
```sh
python -m build
```
publish the release on [test PyPI](https://test.pypi.org/) (useful for making sure everything worked):
```sh
python -m twine upload --repository testpypi dist/*
```
if everything looks ok, upload to the real PyPI:
```sh
python -m twine upload dist/*
```
## license
code is licensed under the [MIT license](LICENSE).
