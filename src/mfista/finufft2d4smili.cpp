#include "utils.h"

extern "C" {
#include "finufft_c.h"
}

#include "finufft.h"

// Shiro Ikeda did the following because we need this... 2018/5/28

int finufft2d1_c(int nj,FLT* xj,FLT* yj,FLT _Complex* cj,int iflag, FLT eps,int ms,int mt, FLT _Complex* fk)
//int finufft2d1_c(int nj,double* xj,double* yj,dcomplex* cj,int iflag, double eps,int ms,int mt, dcomplex* fk)
{
  nufft_opts opts; finufft_default_opts(opts);

  opts.fftw = FFTW_ESTIMATE;
  return finufft2d1((INT)nj,xj,yj,(CPX *)cj,iflag,eps,(INT)ms,(INT)mt,(CPX *)fk,opts);
}

int finufft2d2_c(int nj,FLT* xj,FLT* yj,FLT _Complex* cj,int iflag, FLT eps,int ms,int mt, FLT _Complex* fk)
//int finufft2d2_c(int nj,double* xj,double* yj,dcomplex* cj,int iflag, double eps,int ms,int mt, dcomplex* fk)
{
  nufft_opts opts; finufft_default_opts(opts);

  opts.fftw = FFTW_ESTIMATE;
  return finufft2d2((INT)nj,xj,yj,(CPX *)cj,iflag,eps,(INT)ms,(INT)mt,(CPX *)fk,opts);
}

