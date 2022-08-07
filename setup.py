import os
import codecs

from setuptools import setup

base_dir = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()

setup(
    name='CouDALFISh',
    version='2019.1',
    packages=[''],
    url='https://github.com/david-kamensky/CouDALFISh',
    license='GNU LGPLv3',
    author='D. Kamensky',
    author_email='',
    description="DAL-based IMGA for FEniCS",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
