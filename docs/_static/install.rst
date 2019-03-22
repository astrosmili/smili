============
Installation
============

Requirements
===============

SMILI consists of python modules and Fortran/C internal libraries called from python modules.
Here, we summarize required python packages and external packages for SMILI.

You will also need `ds9`_ for some functions such as setting imaging regions
(CLEAN box) interatively.

.. _ds9: http://ds9.si.edu/site/Home.html

Python Packages and Modules
===========================
SMILI has been tested and developed in Python 2.7 environments provided
by the `Anaconda`_ package. We recommend using Anaconda for SMILI.

.. _Anaconda: https://www.continuum.io/anaconda-overview

All of mandatory packages will be automatically installed during installation.
There are some optional packages that can be used for SMILI.

 - ehtim: https://github.com/achael/eht-imaging
 - ehtplot: https://github.com/chanchikwan/ehtplot

External Libraries (For MacPort users, Ubuntu/Debian users)
===========================================================

Fortran/C internal libraries of SMILI use following external libraries.
This path has been tested for

- Mac OS X 10.012/10.13 with MacPort's gcc 8
- Ubuntu 2016LTS & 2018LTS.


1) OpenBLAS
  We use OpenBLAS, which is the fastest library among publicly-available BLAS implementations.
  Our recommendation is to build up `OpenBLAS`_ by yourself with a compile option USE_OPENMP=1 and use it for our library.
  The option USE_OPENMP=1 enables OpenBLAS to perform paralleled multi-threads calculations, which will accelerate our library.

  .. _OpenBLAS: https://github.com/xianyi/OpenBLAS

  In most of cases, you can compile this library by

  .. code-block:: Bash

    # Clone the current repository
    git clone https://github.com/xianyi/OpenBLAS

    # Compile and install
    cd OpenBLAS
    make USE_OPENMP=1
    make PREFIX="Your install directory; such as /usr/local or $HOME/local" install

  Note that for MacOSX, USE_OPENMP=1 option does not work and should be omitted.
  You may need superuser to install OpenBLAS (i.e. to run the last command).

  SMILI uses **pkg-config** to find appropriate compiler flags for OpenBLAS.
  Once the library is installed, you can check if the package configuration file
  can be accessible. You can type

  .. code-block:: Bash

    pkg-config --debug openblas

  If you can see the correct package configuration file in the output (should be
  $PREFIX/lib/pkgconfig/openblas.pc), you are all set with OpenBLAS. Otherwise,
  you need to set **PKG_CONFIG_PATH** to your pkgconfig directory by, for instance

  .. code-block:: Bash

    export PKG_CONFIG_PATH="Your prefix for OpenBLAS such as /usr/local"/lib/pkgconfig:$PKG_CONFIG_PATH

  Then you can check by ``pkg-config --debug openblas'' if the path is correct.

  Some Other Tips:
    If you are using Ubuntu, RedHat and its variant, the default OpenBLAS package,
    which is installable with `apt-get/aptitude` or `yum`, seems compiled **without**
    this option (USE_OPENMP=1), so we recommend compiling OpenBLAS by yourself.

    If you are using macOS, unfortunately, this option is not available so far.
    You may use a package available in a popular package system (e.g. MacPort, Fink, Homebrew).

2) FFTW3
  We use FFTW3, which is one of the fastest library among publicly-available FFT library.
  For non-Ubuntu users, our recommendation is to build up `FFTW3`_ by yourself.

    .. _FFTW3: http://www.fftw.org

  In most of cases, you can compile this library by

  .. code-block:: Bash

    # Download the library (in case of version 3.3.7)
    wget http://www.fftw.org/fftw-3.3.7.tar.gz # you should check the latest version
    tar xzvf fftw-3.3.7.tar.gz
    cd fftw-3.3.7

    # Compile and install
    ./configure --prefix="install directory; such as /usr/local, $HOME/local" --enable-openmp --enable-threads --enable-shared
    make
    make install

  You may need superuser to install FFTW3 (i.e. to run the last command).

  SMILI uses **pkg-config** to find appropriate compiler flags for FFTW3.
  Once the library is installed, you can check if the package configuration file
  can be accessible. You can type

  .. code-block:: Bash

    pkg-config --debug fftw3

  If you can see the correct package configuration file in the output (should be
  $PREFIX/lib/pkgconfig/fftw3.pc), you are all set with OpenBLAS. Otherwise,
  you need to set **PKG_CONFIG_PATH** to your pkgconfig directory by, for instance

  .. code-block:: Bash

    export PKG_CONFIG_PATH="Your prefix such as /usr/local"/lib/pkgconfig:$PKG_CONFIG_PATH

  Then you can check by ``pkg-config --debug fftw3'' if the path is correct.

  Some Other Tips:
    If you are using Ubuntu, the default fftw3 package,
    which is installable with `apt-get/aptitude` seems compiled **with**
    the option for Openmp (--enable-openmp). So, you don't need to install it
    by yourself.


External Libraries (for homebrew users in MacOS)
===========================================================
1) pyenv and Anaconda installation: Since Anaconda conflicts with Homebrew, it should be installed via pyenv.

  .. code-block:: Bash

    brew install pyenv
    export PATH=$HOME/.pyenv/shims:$PATH

Then install Anaconda.

  .. code-block:: Bash

    pyenv install -l | grep anaconda2
    pyenv install anaconda2-X   # select anaconda 2 version
    pyenv global anaconda2-X
    python --version # check versions

2) OPENBLAS installation: It is mostly same to the above one, but you will need to install gcc.

  .. code-block:: Bash

    # Clone the current repository
    git clone https://github.com/xianyi/OpenBLAS

    # Install gcc49
    brew install gcc49
    sudo ln -sf /usr/local/bin/gcc-4.9 /usr/bin/gcc
    sudo ln -sf /usr/local/bin/g++-4.9 /usr/bin/g++

    # Install OPENBLAS
    make USE_OPENMP=1 CC=gcc
    make PREFIX=/usr/local install

3) FFTW3 installation: No net change from the above one.

  .. code-block:: Bash

    # Download the source code
    wget http://www.fftw.org/fftw-3.3.X.tar.gz
    tar xzvf fftw-3.3.X.tar.gz
    cd fftw-3.X.7

    # Install
    ./configure prefix="/usr/local" --enable-openmp --enable-threads --enable-shared
    make
    make install

Downloading SMILI
=================
You can download the code from github.

.. code-block:: Bash

  # Clone the repository
  git clone https://github.com/astrosmili/smili

Installing SMILI
================

For compiling the whole library, you need to work in your SMILI directory.

.. code-block:: Bash

  cd (Your SMILI Directory)

Generate Makefiles with `./configure`.
If you have correct paths to package-config files for OpenBLAS and FFTW3,
you would not need any options.

.. code-block:: Bash

  ./configure

If you don't have paths to these files, then you need to specify them manually
prior to type ./configure

.. code-block:: Bash

  # Example for OpenBLAS
  export OPENBLAS_LIBS="-LYOURPREFIX/lib -lopenblas"
  export OPENBLAS_CFLAGS="-IYOURPREFIX/include"

  # Example for FFTW3
  export FFTW3_LIBS="-LYOURPREFIX/lib -lfftw3"
  export FFTW3_CFLAGS="-IYOURPREFIX/include"

Make and compile the library.
The internal C/Fortran Library will be compiled into python modules,
and then the whole python modules will be added to the package list of
your Python environment.

.. code-block:: Bash

  make install

If you can load following modules in your python interpretator,
SMILI is probably installed successfully.

.. code-block:: Python

  # import SMILI
  from smili import imdata, uvdata, imaging

**(IMPORTANT NOTE; 2018/04/26)**
Previously, you needed to type autoconf before ./configure command.
This is no longer necessary.

**(IMPORTANT NOTE; 2018/01/04)**
Previously, you needed to add a PYTHONPATH to your SMILI Directory.
This is no longer required, because the `make` command will run setup.py and install
SMILI into the package list of your Python environment.


Updating SMILI
==============

**We strongly recommend cleaning up the entire library before updating.**

.. code-block:: Bash

  cd (Your SMILI Directory)
  make uninstall

Then, you can update the repository with `git pull`.

.. code-block:: Bash

  git pull

Now, the repository has updated. You can follow the above section `Installing SMILI`_ for recompiling your SMILI.
