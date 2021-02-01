#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
#-------------------------------------------------------------------------------
AC_PREREQ([2.69])
AC_INIT([SMILI], [0.0.1], [kakiyama@mit.edu])
AC_CONFIG_SRCDIR([src/fortlib/param.f90])


#-------------------------------------------------------------------------------
# Checks for programs.
#-------------------------------------------------------------------------------
# C Compiler
: ${CFLAGS=""}
AC_PROG_CC

# Fortran Compiler
: ${FCFLAGS=""}
AC_PROG_FC

# CPP Compiler
#: ${CXXFLAGS=""}
#AC_PROG_CXX

# Python
AC_CHECK_PROGS([PYTHON],
               [python],
               [python])

# f2py in numpy
AC_CHECK_PROGS([F2PY],
               [f2py],
               [f2py])

AC_LANG([Fortran])

#-------------------------------------------------------------------------------
# PKG CONFIG
#-------------------------------------------------------------------------------
PKG_PROG_PKG_CONFIG
PKG_CONFIG="$PKG_CONFIG --static"
PKG_CHECK_MODULES(OPENBLAS, openblas)
AC_MSG_RESULT("CFLAGS: $OPENBLAS_CFLAGS")
AC_MSG_RESULT("LIBS:   $OPENBLAS_LIBS")

#-------------------------------------------------------------------------------
# Check FFTW3
#-------------------------------------------------------------------------------
PKG_CHECK_MODULES(FFTW3, fftw3)
AC_MSG_RESULT("CFLAGS: $FFTW3_CFLAGS")
AC_MSG_RESULT("LIBS:   $FFTW3_LIBS")

AC_LANG_PUSH(C)
CFLAGS_tmp=$CFLAGS
CFLAGS="$FFTW3_CFLAGS"
AC_CHECK_LIB(fftw3_omp $FFTW3_LIBS -fopenmp, fftw_init_threads,
  [FFTW3TH_LIBS="-lfftw3_omp"],
  [AC_CHECK_LIB(fftw3_threads $FFTW3_LIBS -lpthread, fftw_init_threads,
    [FFTW3TH_LIBS="-lfftw3_threads"],
    [AC_MSG_ERROR([fftw3_threads nor fftw3_omp was not found.])]
  )]
)
AC_SUBST([FFTW3TH_LIBS])
CFLAGS=$CFLAGS_tmp
AC_LANG_POP(C)


#-------------------------------------------------------------------------------
# Check Lapack
#-------------------------------------------------------------------------------
#AC_LANG_PUSH(Fortran)
#PKG_CHECK_MODULES(LAPACK, lapack,
#  [],
#  [AC_CHECK_LIB(lapack, dgesv,
#               [LAPACK_LIBS="-llapack"],
#               [AC_MSG_ERROR([lapack was not found.])])])
#AC_SUBST([LAPACK_LIBS])
#AC_LANG_POP(Fortran)


#-------------------------------------------------------------------------------
# Check FINUFFT
#-------------------------------------------------------------------------------
PKG_CHECK_MODULES(FINUFFT, finufft)
AC_MSG_RESULT("CFLAGS: $FINUFFT_CFLAGS")
AC_MSG_RESULT("LIBS:   $FINUFFT_LIBS")


#-------------------------------------------------------------------------------
# Check CPP libraries.
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Checks for header files.
#-------------------------------------------------------------------------------
# AC_CHECK_HEADERS([limits.h stdlib.h string.h sys/time.h])


#-------------------------------------------------------------------------------
# Checks for typedefs, structures, and compiler characteristics.
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Checks for library functions.
#-------------------------------------------------------------------------------
#AC_FUNC_MALLOC
#AC_CHECK_FUNCS([clock_gettime floor gettimeofday memset pow sqrt])


#-------------------------------------------------------------------------------
# Command Options
#-------------------------------------------------------------------------------
#  debug options
AC_ARG_ENABLE([dbg],
              [AC_HELP_STRING([--enable-dbg],
                              [Compile source files with DEBUG options])],
              [dbg_flag="True"],
              [dbg_flag="False"])
AC_SUBST([dbg_flag])


#-------------------------------------------------------------------------------
# Makefiles
AC_CONFIG_FILES([makefile
                 src/lbfgsb/makefile
                 src/wrapfinufft/makefile
                 src/fortlib/makefile])
AC_OUTPUT
