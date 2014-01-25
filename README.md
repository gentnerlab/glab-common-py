gentnerlab-shared-py
====================

shared code for common lab functions, analyses, etc

currently implemented functions:

* utils.load_mat - improved loading of *.mat files so that matlab structs are maintained as python dicts
* utils.load_rDAT - imports rDAT files (from old 'c' behavior scripts) into a numpy.recarray, ignoring header rows and playwav errors
