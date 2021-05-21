# We will use ubuntu 18.04 as the base image
FROM ubuntu:20.04

# use bash to setup
SHELL ["/bin/bash", "-c"]

# apt update and install commands
RUN apt-get update -y
RUN apt-get install -y sudo
RUN apt-get install -y wget
RUN apt-get install -y build-essential
RUN apt-get install -y gfortran
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y pkg-config
RUN apt-get install -y git
RUN apt-get clean

# add smili user
RUN adduser --disabled-password --gecos '' smiliuser
RUN echo 'smiliuser:smiligroup' | chpasswd

# set up sudo
RUN gpasswd -a smiliuser sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# switch user
USER smiliuser
ENV HOME /home/smiliuser

# create a local dir
RUN mkdir $HOME/local
RUN mkdir $HOME/local/lib
RUN mkdir $HOME/local/bin

# install pyenv
RUN cd $HOME/local && \
    git clone https://github.com/pyenv/pyenv && \
    cd $HOME && \
    ln -s $HOME/local/pyenv .pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN echo 'export PYENV_ROOT=$HOME/.pyenv' >> ~/.bashrc && \
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# create a path to $HOME/local
ENV LD_LIBRARY_PATH $HOME/local/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH $HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV PATH $HOME/local/bin:$PATH
RUN echo 'export PATH=$HOME/local/bin:$PATH' >> ~/.bashrc  && \
    echo 'export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc  && \
    echo 'export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc


# miniconda3
RUN pyenv install miniconda3-latest && \
    pyenv global  miniconda3-latest

# OpenBLAS
RUN cd $HOME/local && \
    git clone https://github.com/xianyi/OpenBLAS && \
    cd OpenBLAS && \
    make USE_OPENMP=1 CC=gcc FC=gfortran && \
    make PREFIX=$HOME/local install

# FFTW3
RUN cd $HOME/local && \
    wget http://www.fftw.org/fftw-3.3.9.tar.gz && \
    tar xzvf fftw-3.3.9.tar.gz && \
    cd fftw-3.3.9 && \
    ./configure --prefix=$HOME/local --enable-openmp --enable-threads --enable-shared --enable-float --enable-type-prefix && \
    make && \
    make install && \
    make clean && \
    ./configure --prefix=$HOME/local --enable-openmp --enable-threads --enable-shared --enable-type-prefix && \
    make && \
    make install

# FINUFFT
RUN cd $HOME/local && \
    git clone https://github.com/flatironinstitute/finufft && \
    cd finufft && \
    echo 'CXX = g++'                             >> make.inc && \
    echo 'CC = gcc'                              >> make.inc && \
    echo 'FC = gfortran'                         >> make.inc && \
    echo 'CFLAGS += -I$(HOME)/local/include'     >> make.inc && \
    echo 'LIBS += -L$(HOME)/local/lib'           >> make.inc && \
    echo 'FLAGS = $(CFLAGS)'                     >> make.inc && \
    echo 'CXXFLAGS = $(CFLAGS) -DNEED_EXTERN_C'  >> make.inc && \
    echo 'OMPFLAGS = -fopenmp'                   >> make.inc && \
    echo 'OMPLIBS = -lgomp'                      >> make.inc && \
    make lib && \
    echo 'finufftdir=$(HOME)/local/finufft'                                            >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo 'libdir=${finufftdir}/lib-static'                                             >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo 'includedir=${finufftdir}/src'                                                >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo ''                                                                            >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo 'Name: FINUFFT'                                                               >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo 'Description: Flatiron Institute Nonuniform Fast Fourier Transform libraries' >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo 'Version: github'                                                             >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo 'Libs: -L${libdir} -lfinufft'                                                 >> $HOME/local/lib/pkgconfig/finufft.pc && \
    echo 'Cflags: -I${includedir}'                                                     >> $HOME/local/lib/pkgconfig/finufft.pc

# python packages
RUN conda install ipython jupyter numpy scipy matplotlib pandas astropy seaborn h5py xarray # basic scientific python
RUN conda install ephem scikit-image scikit-learn # extras utils
RUN conda install tqdm # extras utils

# ehtim
RUN cd $HOME/local && \
    git clone https://github.com/achael/eht-imaging && \
    cd eht-imaging && \
    conda install -c conda-forge pynfft && \
    pip install .

# ehtplot
RUN cd $HOME/local && \
    git clone https://github.com/liamedeiros/ehtplot && \
    cd ehtplot && \
    python setup.py install

# smili
RUN cd $HOME/local && \
    git clone https://github.com/astrosmili/smili && \
    cd smili && \
    ./configure && \
    make install
