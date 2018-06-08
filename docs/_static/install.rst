============
Installation
============

Requirements
===============

Smili consists of python modules and Fortran/C internal libraries called from python modules.
Here, we summarize required python packages and external packages for Smili.

You will also need `ds9`_ for compiling the library.

.. _ds9: http://ds9.si.edu/site/Home.html

Python Packages and Modules
---------------------------
Smili has been tested and developped in Python 2.7 environments provided by the `Anaconda`_ package. We recommend using Anaconda for Smili.
Smili and related functions will use **future**, **numpy**, **scipy**, **matplotlib**, **pandas**, **astropy**, **xarray**, **pyds9**, **tqdm**, **sympy**, **theano**, **pymc3**.

.. _Anaconda: https://www.continuum.io/anaconda-overview

You can install above packages with conda and/or pip as follows
(see the official website of `pyds9`_ for its installation).

.. code-block:: Bash

  # if you have conda
  conda install future xarray tqdm sympy theano pymc3
  # You may use pip, if you do not have or want to use conda
  pip install future xarray tqdm sympy theano pymc3

  # to install pyds9, you can use pip command.
  pip install git+https://github.com/ericmandel/pyds9.git#egg=pyds9

.. _xarray: http://xarray.pydata.org/en/stable/
.. _pyds9: https://github.com/ericmandel/pyds9


External Libraries
------------------

Fortran/C internal libraries of Smili use following external libraries.

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

  You may need superuser to install OpenBLAS (i.e. to run the last command).

  Smili uses **pkg-config** to find appropriate compiler flags for OpenBLAS.
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

  Smili uses **pkg-config** to find appropriate compiler flags for FFTW3.
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

3) LAPACK
  LAPACK does not have a big impact on computational costs of imaging.
  The default LAPACK package in your Linux/OS X package system would be acceptable for Spareselab.
  Of course, you may build up `LAPACK`_ by yourself. If you build up LAPACK by yourself,
  please do not forget adding **``-fPIC''** flag to the configuration variables
  **CFLAGS**, **OPTS**, **NOOPT**, **LOADEROPTS** in make.inc. I (Kazu Akiyama)
  usually add ``-fPIC -O3 -march=native'' for Linux and ``-fPIC -O3 -march=core2'' for macOS.

  .. _LAPACK: https://github.com/Reference-LAPACK/lapack-release

  Unfortunately, Lapack does not have a pkg-config file, which
  may cause some problems if you put lapack in an unusual place.
  It would be useful to make and put lapack.pc in a directory specified by
  **PKG_CONFIG_PATH** to avoid potential problems for compiling Smili.
  Smili tries to find lapack.pc in your PKG_CONFIG_PATH. If it does not find,
  it will check if a lapack function can run with -llapack correctly.

  Following is a sample for lapack.pc

  .. code-block:: Plain

    libdir=<YOUR LAPACK DIRECTORY>

    Name: LAPACK
    Description: FORTRAN reference implementation of LAPACK Linear Algebra PACKage
    Version: @LAPACK_VERSION@
    URL: http://www.netlib.org/lapack/
    Libs: -L${libdir} -llapack
    Requires.private: openblas
    Cflags:


Download, Install and Update
============================

Downloading Smili
---------------------
You can download the code from github.

.. code-block:: Bash

  # Clone the repository
  git clone https://github.com/astrosmili/smili

Installing Smili
--------------------

For compiling the whole library, you need to work in your Smili directory.

.. code-block:: Bash

  cd (Your Smili Directory)

Generate Makefiles with `./configure`.
If you have correct paths to package-config files for OpenBLAS, FFTW3 and
path to library or package-config file for LAPACK, you would not need any options.

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

  # Example for LPACK
  export LAPACK_LIBS="-LYOURPREFIX/lib -llapack"

Make and compile the library.
The internal C/Fortran Library will be compiled into python modules,
and then the whole python modules will be added to the package list of
your Python environment.

.. code-block:: Bash

  make install

If you can load following modules in your python interpretator,
Smili is probably installed successfully.

.. code-block:: Python

  # import smili
  from smili import imdata, uvdata, imaging

**(IMPORTANT NOTE; 2018/04/26)**
Previously, you needed to type autoconf before ./configure command.
This is no longer necessary.

**(IMPORTANT NOTE; 2018/01/04)**
Previously, you needed to add a PYTHONPATH to your Smili Directory.
This is no longer required, because the `make` command will run setup.py and install
smili into the package list of your Python environment.


Updating Smili
------------------

**We strongly recommend cleaning up the entire library before updating.**

.. code-block:: Bash

  cd (Your Smili Directory)
  make uninstall

Then, you can update the repository with `git pull`.

.. code-block:: Bash

  git pull

Now, the repository has updated. You can follow the above section `Installing Smili`_ for recompiling your Smili.
