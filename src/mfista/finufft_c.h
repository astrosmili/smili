#include <complex.h>

// this fails:
//  int finufft1d1_c(int nj,double* xj,std::complex<double>* cj,int iflag, double eps,int ms, std::complex<double>* fk);

int finufft2d1_c(int nj,double* xj,double* yj,double _Complex* cj,int iflag, double eps,int ms, int mt, double _Complex* fk);
int finufft2d2_c(int nj,double* xj,double* yj,double _Complex* cj,int iflag, double eps,int ms, int mt, double _Complex* fk);

