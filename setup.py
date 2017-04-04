import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "field_toolkit",
    version = "0.0.1",
    author = "Christopher 'ckt' Tomaszewski",
    author_email = "christomaszewski@gmail.com",
    description = ("A library for working with scalar and vector fields"),
    license = "BSD",
    keywords = "scalar vector field reconstruction",
    url = "https://github.com/christomaszewski/field_toolkit.git",
    packages=['field_toolkit', 'tests', 'examples'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)