from setuptools import setup, find_packages
from codecs     import open
from os         import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

if path.isfile(path.join(here, "smili/libmfista.so")):
    errmsg="Apparently, you have not compiled C/Fortran libraries with make."
    errmsg+=" Please install this library by 'make install' not by 'python setup.py install'"
    raise RuntimeError(errmsg)

setup(
    name="smili",
    version = "0.0.1",
    description = "SMILI: Sparse Modeling Imaging library for Interferometry",
    long_description = long_description,
    url = "https://smili.github.io/smili",
    author = "Kazunori Akiyama",
    author_email = "kakiyama@mit.edu",
    license = "GPL ver 3.0",
    keywords = "radio astronomy, interferometry, imaging, event horizon telescope",
    packages = find_packages(exclude=["doc*", "test*"]),
    package_data={'smili': ['*.so','imaging/*.so']},
    install_requires = [
        "future","numpy","scipy","matplotlib","pandas",
        "scikit-image","astropy<=2.0.14","tqdm","theano","pyds9"]
)
