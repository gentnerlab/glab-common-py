Gentner Lab Common
====================

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/gentnerlab/glab-common-py/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

shared code for common lab functions, analyses, etc

currently implemented functions:

* utils.load_mat - improved loading of *.mat files so that matlab structs are maintained as python dicts
* utils.load_rDAT - imports rDAT files (from old 'c' behavior scripts) into a numpy.recarray, ignoring header rows and playwav errors
