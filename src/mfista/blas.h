#ifndef BLAS_INTERFACE_HEADER
#define BLAS_INTERFACE_HEADER

#ifdef __cplusplus 
extern "C"{
#endif

//Structs
typedef struct complex_Tag{
  float r;
  float i;
} complex;

typedef struct doublecomplex_Tag{
  double r;
  double i;
} doublecomplex;


//int xrebla_(char *srname, int *info);

//Level1

//AXPY
void saxpy_(int *n, float         *alpha, float         *x, int *incx, float         *y, int *incy);
void daxpy_(int *n, double        *alpha, double        *x, int *incx, double        *y, int *incy);
void caxpy_(int *n, complex       *alpha, complex       *x, int *incx, complex       *y, int *incy);
void zaxpy_(int *n, doublecomplex *alpha, doublecomplex *x, int *incx, doublecomplex *y, int *incy);

//SUM
float   sasum_(int *n, float         *x, int *incx);
float  scasum_(int *n, complex       *x, int *incx);
double  dasum_(int *n, double        *x, int *incx);
double dzasum_(int *n, doublecomplex *x, int *incx);

//COPY
void scopy_(int *n, float  *x, int *incx, float  *y, int *incy);
void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
void ccopy_(int *n, float  *x, int *incx, float  *y, int *incy);
void zcopy_(int *n, double *x, int *incx, double *y, int *incy);

//DOT
float  sdot_(int *n, float  *x, int *incx, float  *y, int *incy);
double ddot_(int *n, double *x, int *incx, double *y, int *incy);

//DOTC
complex       cdotc_(int *n, complex       *x, int *incx, complex       *y, int *incy);
doublecomplex zdotc_(int *n, doublecomplex *x, int *incx, doublecomplex *y, int *incy);

//DOTU
complex       cdotu_(int *n, complex       *x, int *incx, complex       *y, int *incy);
doublecomplex zdotu_(int *n, doublecomplex *x, int *incx, doublecomplex *y, int *incy);

//NRM2
float   snrm2_(int *n, float         *x, int *incx);
double  dnrm2_(int *n, double        *x, int *incx);
float  scnrm2_(int *n, complex       *x, int *incx);
double dznrm2_(int *n, doublecomplex *x, int *incx);

//ROT
void  srot_(int *n, float         *x, int *incx, float         *y, int *incy, float  *c, float  *s);
void  drot_(int *n, double        *x, int *incx, double        *y, int *incy, double *c, double *s);
void csrot_(int *n, complex       *x, int *incx, complex       *y, int *incy, float  *c, float  *s);
void zdrot_(int *n, doublecomplex *x, int *incx, doublecomplex *y, int *incy, double *c, double *s);

//ROTG
void srotg_(float         *a,float         *b, float  *c, float  *s);
void drotg_(double        *a,double        *b, double *c, double *s);
void crotg_(complex       *a,complex       *b, float  *c, float  *s);
void zrotg_(doublecomplex *a,doublecomplex *b, double *c, double *s);

//Stub
//ROTMG
//ROTM


//SCAL
void  sscal_(int *n,  float         *a, float          *x, int *incx);
void  dscal_(int *n,  double        *a, double         *x, int *incx);
void  cscal_(int *n,  complex       *a, complex        *x, int *incx);
void  zscal_(int *n,  doublecomplex *a, doublecomplex  *x, int *incx);
void csscal_(int *n,  float         *a, complex        *x, int *incx);
void zdscal_(int *n,  double        *a, doublecomplex  *x, int *incx);

//SWAP
void sswap_(int *n, float         *x, int *incx, float         *y, int *incy);
void dswap_(int *n, double        *x, int *incx, double        *y, int *incy);
void cswap_(int *n, complex       *x, int *incx, complex       *y, int *incy);
void zswap_(int *n, doublecomplex *x, int *incx, doublecomplex *y, int *incy);

//IAMAX
int isamax_(int *n, float         *x, int *incx);
int idamax_(int *n, double        *x, int *incx);
int icamax_(int *n, complex       *x, int *incx);
int izamax_(int *n, doublecomplex *x, int *incx);

//IAMIN
int isamin_(int *n, float         *x, int *incx);
int idamin_(int *n, double        *x, int *incx);
int icamin_(int *n, complex       *x, int *incx);
int izamin_(int *n, doublecomplex *x, int *incx);

//IMAX
int ismax_(int *n, float  *x, int *incx);
int idmax_(int *n, double *x, int *incx);

//IMIN
int ismin_(int *n, float  *x, int *incx);
int idmin_(int *n, double *x, int *incx);

//Level2

//GBMV
void sgbmv_(char *trans, int *m, int *n, int *kl, int *ku, 
            float         *alpha, float         *A, int *ldA, float         *x, int *incx,
            float         *beta , float         *y, int *incy);
void dgbmv_(char *trans, int *m, int *n, int *kl, int *ku, 
            double        *alpha, double        *A, int *ldA, double        *x, int *incx, 
            double        *beta , double        *y, int *incy);
void cgbmv_(char *trans, int *m, int *n, int *kl, int *ku, 
            complex       *alpha, complex       *A, int *ldA, complex       *x, int *incx, 
            complex       *beta , complex       *y, int *incy);
void zgbmv_(char *trans, int *m, int *n, int *kl, int *ku, 
            doublecomplex *alpha, doublecomplex *A, int *ldA, doublecomplex *x, int *incx, 
            doublecomplex *beta , doublecomplex *y, int *incy);

//GEMV
void sgemv_(char *trans, int *m, int *n, 
            float         *alpha, float         *A, int *ldA, float         *x, int *incx,
            float         *beta , float         *y, int *incy);
void dgemv_(char *trans, int *m, int *n, 
            double        *alpha, double        *A, int *ldA, double        *x, int *incx, 
            double        *beta , double        *y, int *incy);
void cgemv_(char *trans, int *m, int *n, 
            complex       *alpha, complex       *A, int *ldA, complex       *x, int *incx, 
            complex       *beta , complex       *y, int *incy);
void zgemv_(char *trans, int *m, int *n, 
            doublecomplex *alpha, doublecomplex *A, int *ldA, doublecomplex *x, int *incx, 
            doublecomplex *beta , doublecomplex *y, int *incy);

//GER
void sger_(int *m, int *n, float  *alpha,  float *x, int *incx,  float *y, int *incy,  float *A, int *ldA);
void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *A, int *ldA);

//GERC
void cgerc_(int *m, int *n, complex       *alpha, complex       *x, int *incx,
            complex       *y, int *incy, complex       *A, int *ldA);
void zgerc_(int *m, int *n, doublecomplex *alpha, doublecomplex *x, int *incx, 
            doublecomplex *y, int *incy, doublecomplex *A, int *ldA);

//GREU
void cgeru_(int *m, int *n, complex       *alpha, complex       *x, int *incx,
            complex       *y, int *incy, complex       *A, int *ldA);
void zgeru_(int *m, int *n, doublecomplex *alpha, doublecomplex *x, int *incx, 
            doublecomplex *y, int *incy, doublecomplex *A, int *ldA);

//HBMV
void chbmv_(char *uplo, int *n, int *k, complex       *alpha, complex       *A, int *ldA,
            complex       *x, int *incx, complex       *beta, complex       *y, int *incy);
void zhbmv_(char *uplo, int *n, int *k, doublecomplex *alpha, doublecomplex *A, int *ldA,
            doublecomplex *x, int *incx, doublecomplex *beta, doublecomplex *y, int *incy);

//HEMV
void chemv_(char *uplo, int *n, complex       *alpha, complex       *A, int *ldA,
            complex       *x, int *incx, complex       *beta, complex       *y, int *incy);
void zhemv_(char *uplo, int *n, doublecomplex *alpha, doublecomplex *A, int *ldA,
            doublecomplex *x, int *incx, doublecomplex *beta, doublecomplex *y, int *incy);

//HER
void cher_(char *uplo, int *n, float  *alpha, complex       *x, int *incx, complex       *A, int *ldA);
void zher_(char *uplo, int *n, double *alpha, doublecomplex *x, int *incx, doublecomplex *A, int *ldA);

//Stub
//HER2

//HPMV
void chpmv_(char *uplo, int *n, complex       *alpha, complex       *A,
            complex       *x, int *incx, complex       *beta, complex       *y, int *incy);
void zhpmv_(char *uplo, int *n, doublecomplex *alpha, doublecomplex *A,
            doublecomplex *x, int *incx, doublecomplex *beta, doublecomplex *y, int *incy);

//HPR
void chpr_ (char *uplo, int *n, float   *alpha, complex       *x, int *incx, complex       *A);
void zhpr_ (char *uplo, int *n, double  *alpha, doublecomplex *x, int *incx, doublecomplex *A);

//Stub
//HPR2

//SBMV
void ssbmv_(char *uplo, int *n, int *k, float  *alpha, float  *A, int *ldA,
            float  *x, int *incx, float  *beta, float  *y, int *incy);
void dsbmv_(char *uplo, int *n, int *k, double *alpha, double *A, int *ldA,
            double *x, int *incx, double *beta, double *y, int *incy);

//SPMV
void sspmv_(char *uplo, int *n, float  *alpha, float  *A, float  *x, int *incx, float  *beta, float  *y, int *incy);
void dspmv_(char *uplo, int *n, double *alpha, double *A, double *x, int *incx, double *beta, double *y, int *incy);

//SPR
void sspr_(char *uplo, int *n, float  *alpha, float  *x, int *incx, float  *A);
void dspr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *A);

//Stub
//SPR2

//SYMV
void ssymv_(char *uplo, int *n, float  *alpha, float  *A, int *ldA,
            float  *x, int *incx, float  *beta, float  *y, int *incy);
void dsymv_(char *uplo, int *n, double *alpha, double *A, int *ldA,
            double *x, int *incx, double *beta, double *y, int *incy);

//SYR
void ssyr_(char *uplo, int *n, float  *alpha, float  *x, int *incx, float  *A, int *ldA);
void dsyr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *A, int *ldA);

//Stub
//SYR2

//TBMV
void stbmv_(char *uplo, char *trans, char *diag, int *n, int *k, float         *A, int *ldA, float         *x, int *incx);
void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k, double        *A, int *ldA, double        *x, int *incx);
void ctbmv_(char *uplo, char *trans, char *diag, int *n, int *k, complex       *A, int *ldA, complex       *x, int *incx);
void ztbmv_(char *uplo, char *trans, char *diag, int *n, int *k, doublecomplex *A, int *ldA, doublecomplex *x, int *incx);

//TBSV
void stbsv_(char *uplo, char *trans, char *diag, int *n, int *k, float         *A, int *ldA, float         *x, int *incx);
void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k, double        *A, int *ldA, double        *x, int *incx);
void ctbsv_(char *uplo, char *trans, char *diag, int *n, int *k, complex       *A, int *ldA, complex       *x, int *incx);
void ztbsv_(char *uplo, char *trans, char *diag, int *n, int *k, doublecomplex *A, int *ldA, doublecomplex *x, int *incx);

//TPMV
void stpmv_(char *uplo, char *trans, char *diag, int *n, float         *A, float         *x, int *incx);
void dtpmv_(char *uplo, char *trans, char *diag, int *n, double        *A, double        *x, int *incx);
void ctpmv_(char *uplo, char *trans, char *diag, int *n, complex       *A, complex       *x, int *incx);
void ztpmv_(char *uplo, char *trans, char *diag, int *n, doublecomplex *A, doublecomplex *x, int *incx);

//TPSV
void stpsv_(char *uplo, char *trans, char *diag, int *n, float         *A, float         *x, int *incx);
void dtpsv_(char *uplo, char *trans, char *diag, int *n, double        *A, double        *x, int *incx);
void ctpsv_(char *uplo, char *trans, char *diag, int *n, complex       *A, complex       *x, int *incx);
void ztpsv_(char *uplo, char *trans, char *diag, int *n, doublecomplex *A, doublecomplex *x, int *incx);

//TRSV
void strsv_(char *uplo, char *trans, char *diag, int *n, float         *A, int *ldA, float         *x, int *incx);
void dtrsv_(char *uplo, char *trans, char *diag, int *n, double        *A, int *ldA, double        *x, int *incx);
void ctrsv_(char *uplo, char *trans, char *diag, int *n, complex       *A, int *ldA, complex       *x, int *incx);
void ztrsv_(char *uplo, char *trans, char *diag, int *n, doublecomplex *A, int *ldA, doublecomplex *x, int *incx);

//TRMV
void strmv_(char *uplo, char *trans, char *diag, int *n, float         *A, int *ldA, float         *x, int *incx);
void dtrmv_(char *uplo, char *trans, char *diag, int *n, double        *A, int *ldA, double        *x, int *incx);
void ctrmv_(char *uplo, char *trans, char *diag, int *n, complex       *A, int *ldA, complex       *x, int *incx);
void ztrmv_(char *uplo, char *trans, char *diag, int *n, doublecomplex *A, int *ldA, doublecomplex *x, int *incx);

//Level3

//GEMM
void sgemm_(char *transa, char *transb, int *m, int *n, int *k,
            float         *alpha, float         *A, int *ldA, float         *B, int *ldB, 
            float         *beta , float         *C, int *ldC);
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, 
            double        *alpha, double        *A, int *ldA, double        *B, int *ldB, 
            double        *beta , double        *C, int *ldC);
void cgemm_(char *transa, char *transb, int *m, int *n, int *k, 
            complex       *alpha, complex       *A, int *ldA, complex       *B, int *ldB, 
            complex       *beta , complex       *C, int *ldC);
void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
            doublecomplex *alpha, doublecomplex *A, int *ldA, doublecomplex *B, int *ldB, 
            doublecomplex *beta , doublecomplex *C, int *ldC);

//HEMM
void chemm_(char *side, char *uplo, int *m, int *n, complex       *alpha, complex       *A, int *ldA,
            complex       *B, int *ldB, complex       *beta, complex       *C, int *ldC);
void zhemm_(char *side, char *uplo, int *m, int *n, doublecomplex *alpha, doublecomplex *A, int *ldA,
            doublecomplex *B, int *ldB, doublecomplex *beta, doublecomplex *C, int *ldC);

//HERK
void cherk_(char *uplo, char *trans, int *n, int *k, float  *alpha, complex       *A, int *ldA,
            float  *beta , complex       *C, int *ldC);
void zherk_(char *uplo, char *trans, int *n, int *k, double *aplha, doublecomplex *A, int *ldA,
            double *beta , doublecomplex *C, int *ldC);

//HERK2
void cher2k_(char *uplo, char *trans, int *n, int *k, complex       *alpha, complex       *A, int *ldA,
             complex       *B, int *ldB,float  *beta, complex       *C, int *ldC);
void zher2k_(char *uplo, char *trans, int *n, int *k, doublecomplex *alpha, doublecomplex *A, int *ldA,
             doublecomplex *B, int *ldB,double *beta, doublecomplex *C, int *ldC);

//SYMM
void ssymm_(char *side, char *uplo, int *m, int *n, 
            float         *alpha, float         *A, int *ldA, float         *B, int *ldB, 
            float         *beta , float         *C, int *ldC);
void dsymm_(char *side, char *uplo, int *m, int *n, 
            double        *alpha, double        *A, int *ldA, double        *B, int *ldB, 
            double        *beta , double        *C, int *ldC);
void csymm_(char *side, char *uplo, int *m, int *n, 
            complex       *alpha, complex       *A, int *ldA, complex       *B, int *ldB, 
            complex       *beta , complex       *C, int *ldC);
void zsymm_(char *side, char *uplo, int *m, int *n, 
            doublecomplex *alpha, doublecomplex *A, int *ldA, doublecomplex *B, int *ldB, 
            doublecomplex *beta , doublecomplex *C, int *ldC);

//SYRK
void ssyrk_(char *uplo, char *trans, int *n, int *k, float  *alpha, float         *A, int *ldA,
            float  *beta , float         *C, int *ldC);
void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, double        *A, int *ldA,
            double *beta , double        *C, int *ldC);
void csyrk_(char *uplo, char *trans, int *n, int *k, float  *alpha, complex       *A, int *ldA,
            float  *beta , complex       *C, int *ldC);
void zsyrk_(char *uplo, char *trans, int *n, int *k, double *aplha, doublecomplex *A, int *ldA,
            double *beta , doublecomplex *C, int *ldC);

//SYR2K
void ssyr2k_(char *uplo, char *trans, int *n, int *k, float  *alpha, float         *A, int *ldA, float         *B, int *ldB,
            float  *beta , float         *C, int *ldC);
void dsyr2k_(char *uplo, char *trans, int *n, int *k, double *alpha, double        *A, int *ldA, double        *B, int *ldB,
            double *beta , double        *C, int *ldC);
void csyr2k_(char *uplo, char *trans, int *n, int *k, float  *alpha, complex       *A, int *ldA, complex       *B, int *ldB,
            float  *beta , complex       *C, int *ldC);
void zsyr2k_(char *uplo, char *trans, int *n, int *k, double *aplha, doublecomplex *A, int *ldA, doublecomplex *B, int *ldB,
            double *beta , doublecomplex *C, int *ldC);

//TRMM
void strmm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            float         *alpha, float         *A, int *ldA, float         *B, int *ldB);
void dtrmm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            double        *alpha, double        *A, int *ldA, double        *B, int *ldB);
void ctrmm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            complex       *alpha, complex       *A, int *ldA, complex       *B, int *ldB);
void ztrmm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            doublecomplex *alpha, doublecomplex *A, int *ldA, doublecomplex *B, int *ldB);

//TRSM
void strsm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            float         *alpha, float         *A, int *ldA, float         *B, int *ldB);
void dtrsm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            double        *alpha, double        *A, int *ldA, double        *B, int *ldB);
void ctrsm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            complex       *alpha, complex       *A, int *ldA, complex       *B, int *ldB);
void ztrsm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, 
            doublecomplex *alpha, doublecomplex *A, int *ldA, doublecomplex *B, int *ldB);

#ifdef __cplusplus 
}
#endif

#endif
