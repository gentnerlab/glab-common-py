#!/usr/bin/env python

from distutils.core import setup

setup(
    name='glab-common-py',
    version='0.0.1',
    description='shared code for common lab functions, analyses, etc',
    author='Justin Kiggins',
    author_email='justin.kiggins@gmail.com',
    packages=['glab_common',
              'kk_pipeline',
              ],
    entry_points={
        'console_scripts': [
            's2mat_to_kwd = kk_pipeline.s2mat_to_kwd:main',
            'make_s2mat_list = kk_pipeline.make_s2mat_list:main'
        ],
    },
)
