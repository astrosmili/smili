============
Installation
============

Introduction
===============

SMILI consists of python interfaces and Fortran/C internal libraries called from
python interfaces. Here, we summarize required python packages and external packages
for SMILI.

This installation path has been tested for

- macOS 10.12/10.13/10.14 with MacPort's GCC 8
- macOS 10.14 with homebrew's gcc
- Ubuntu 2016LTS & 2018LTS.


Python Environments and Packages
================================
**SMILI has been transferred to Python 3 after verion 0.1.0**.
This version has been tested and developed in `pyenv`_. In particular, we use
Python 3.7 environments provided by the `Anaconda`_ package.
We recommend using pyenv not to affect any system-related python environments.

.. _pyenv: https://github.com/pyenv/pyenv

.. _Anaconda: https://www.continuum.io/anaconda-overview

All of mandatory python packages will be automatically installed during installation.
There are some optional packages that may be used for SMILI.

 - ehtim: https://github.com/achael/eht-imaging
 - ehtplot: https://github.com/chanchikwan/ehtplot


Preparation
===========================================================


1) Ubuntu 2016LTS / 2018LTS users

  .. code-block:: Bash

    # Ubuntu users
    sudo apt-get install build-essential pkg-config git

2) macOS: MacPorts Users

  .. code-block:: Bash

    # macOS: MacPorts users
    sudo port install gcc8 pkgconfig

    # check the installed gcc
    port select --list gcc

    # select installed gcc as your default compilers
    sudo port select --set gcc mp-gcc8

    # please make sure that you can use them.
    #   If you can't, please make sure that your MacPorts bin directory
    #   (in default, /opt/local/bin) is set before /usr/bin in $PATH
    #   by typing "echo $PATH"
    ls -l `which gcc`
    ls -l `which g++`
    ls -l `which gfortran`

3) macOS: Homebrew Users

  .. code-block:: Bash

    # macOS: homebrew
    brew install pkg-config
    brew install gcc
    brew install libomp

    # your homebrew PREFIX directory
    HBPREFIX="your homebrew PREFIX; in default /usr/local"

    # here I assume that you have installed gcc9
    #   This usually works very good to pick up all of commands installed with gcc9
    \ls $HBPREFIX/bin/*g*-9

    #   if the above commands work OK, you can create symbolic links
    #   of commands without "-9"
    FILES=`\ls $HBPREFIX/bin/*g*-9`
    for FILE in $FILES; do ln -s $FILE ${FILE/-9/}; done

    # check if your symbolic links work OK with which command
    # if everything works OK, you will see that everything is in
    # the $HBPREFIX/bin directory.
    which gcc g++ gfortran

    # If it doesn't, please make sure that your homebrew's bin directory
    # (in default, /usr/local/bin) is set before /usr/bin in $PATH
    # by typing "echo $PATH"

External Libraries
===========================================================
Fortran/C internal libraries of SMILI use following external libraries.

You will also need `ds9`_ for some functions such as setting imaging regions
(CLEAN box) interatively.

.. _ds9: http://ds9.si.edu/site/Home.html

Please make sure that you have **pkg-config** and gcc in your system.
You can install them from your OS's package system for LINUX and MacPortsfor
macOS.

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
    #   macOS MacPorts users may not use USE_OPENMP=1 option, and need to omit it.
    cd OpenBLAS
    make USE_OPENMP=1 CC=gcc FC=gfortran
    make PREFIX="Your prefix, e.g. $HOME/local" install

  Note that for macOS MacPorts, USE_OPENMP=1 option does not work and should be omitted.
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

    export PKG_CONFIG_PATH="Your prefix, e.g. $HOME/local"/lib/pkgconfig:$PKG_CONFIG_PATH

  Then you can check by ``pkg-config --debug openblas'' if the path is correct.

2) FFTW3
  We use FFTW3, which is one of the fastest library among publicly-available FFT library.
  For non-Ubuntu users, our recommendation is to build up `FFTW3`_ by yourself.

    .. _FFTW3: http://www.fftw.org

  In most of cases, you can compile this library by

  .. code-block:: Bash

    # Download the library (in case of version 3.3.X)
    wget http://www.fftw.org/fftw-3.3.X.tar.gz # you should check the latest version
    tar xzvf fftw-3.3.X.tar.gz
    cd fftw-3.3.X

    # Compile and install
    ./configure --prefix="Your prefix, e.g. $HOME/local" --enable-openmp --enable-threads --enable-shared
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

    export PKG_CONFIG_PATH="Your prefix, e.g. $HOME/local"/lib/pkgconfig:$PKG_CONFIG_PATH

  Then you can check by ``pkg-config --debug fftw3'' if the path is correct.

3) FINUFFT
  Flaton Institue Non-uniform fast Fourier transform library (FINUFFT) is a
  key library of SMILI.

  .. code-block:: Bash

    # Download the directory
    PREFIX="Your prefix, e.g. $HOME/local"
    cd $PREFIX
    git clone https://github.com/flatironinstitute/finufft
    cd finufft


  Then, you need to create a make.inc file. This should be something like this.
  See also https://finufft.readthedocs.io/en/latest/install.html.


  .. code-block:: Bash

    # Compilers
    CXX=g++
    CC=gcc
    FC=gfortran

    # (compile flags for use with GCC are as in linux makefile)
    CFLAGS +=

    # If you're getting warning messages of the form:
    #    ld: warning: object file (lib-static/libfinufft.a(finufft1d.o)) was built for
    #    newer OSX version (10.13) than being linked (10.9)
    # Then you can uncomment the following two lines with the older version number
    # (in this example -mmacosx-version-min=10.9)
    #
    #CFLAGS += "-mmacosx-version-min=<OLDER OSX VERSION NUMBER>"

    # if you are macOS homebrew users, uncomment this.
    # (assuming that /usr/local is your homebrew's PREFIX)
    #CFLAGS += -I src -I/usr/local/include
    #LIBS += -L/usr/local/lib

    # if you are macOS MacPorts users, uncomment this.
    # (assuming that /opt/local is your MacPorts' PREFIX)
    #CFLAGS += -I src -I/opt/local/include
    #LIBS += -L/opt/local/lib

    # Your FFTW3's installation PREFIX
    CFLAGS += -I$HOME/local/include
    LIBS += -L$HOME/local/lib

    # You can keep them
    FFLAGS   = $(CFLAGS)
    CXXFLAGS = $(CFLAGS) -DNEED_EXTERN_C

    # OpenMP with GCC on OSX needs following...
    OMPFLAGS = -fopenmp
    OMPLIBS = -lomp
    # since fftw3_omp doesn't work in OSX, you need to uncomment this
    #FFTWOMPSUFFIX=threads

  Once you finished editing the make.inc file, you can compile the library.

  .. code-block:: Bash

    # compile the library
    make lib

  In one of your PKG_CONFIG_PATH directory, please put this pkg-config file
  **finufft.pc** like this

  .. code-block:: Bash

    # This is an example pkg-config file. Here is an brief instruction.
    # (1) Please change finufftdir depending on your install directory.
    # (2) please change its filename to finufft.sample.pc and
    #     copy to a directory specified in $PKG_CONFIG_PATH
    finufftdir=$(HOME)/local/finufft
    libdir=${finufftdir}/lib-static
    includedir=${finufftdir}/src

    Name: FINUFFT
    Description: Flatiron Institute Nonuniform Fast Fourier Transform libraries
    Version: github
    Libs: -L${libdir} -lfinufft
    Cflags: -I${includedir}

  Once you locate the above finufft.pc file,
  you can check by ``pkg-config --debug finufft'' if the path is correct.

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

  # Example for FINUFFT
  export FINUFFT_LIBS="-LYOURPREFIX/lib -lfftw3"
  export FINUFFT_CFLAGS="-IYOURPREFIX/include"

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
