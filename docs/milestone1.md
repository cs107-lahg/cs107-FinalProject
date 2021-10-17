# Milestone 1 Documentation

## Introduction

## Background

## How to use <AAD>

## Organization
### Directory structure and modules

```
<Project Name>
├── README.md               Main project README
├── requirements.txt        Package dependencies
├── setup.py                setup function for package
│
├── docs
│   ├── README.md           README for docs
│   └── milestone[i].md    Documentation for each milestone
│
├── src:                    Package source files
│   ├── AD.py               Main constructor
│   ├── AD_helper.py        Helper functions
│   ├── AD_forward.py       Forward mode
│   └── AD_reverse.py       Reverse mode
│
└── tests                   Package test scripts
    ├── run_tests.py       script that runs all tests
    └── test_[i].py        tests function [i]

```
### What modules do you plan on including? What is their basic functions.
The modules we will include are
* numpy: numpy supports computation for multi-dimensional arrays and matricies
* math: math supports common mathematical objects and functions required for automatic differentiation
* setuptools: this allows us to setup our pacakge and facilitates easy installation and  distribution

### Where will your test suite live? Will you use TravisCI? CodeCov?

* Our test suite will live in the `tests/` directory, which will contain scripts that tests each function in our class
* TravisCI and CodeCov will be used to test our package and check code coverage.

### How will you distribute your package (e.g. PyPI)?

* We will use both Github and PyPI to distribute our package, this allows the user to install our package using both `pip` or building straight from source.
* We will also consider distribution using Anaconda since that is a popular numerical programming package distribution channel.

### How will you package your software? Will you  use a framework? If so which one and why? If not, why not?

* We will not use a framework for our software because it is simple enough to not use a framework -- a framework will overcomplicate the design.
* We will use `setup_tools` library to help us with package development.
