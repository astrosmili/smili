/*
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista.h'. An optimization algorithm for imaging of
   interferometry. The idea of the algorithm was from the following
   two papers,

   Beck and Teboulle (2009) SIAM J. Imaging Sciences,
   Beck and Teboulle (2009) IEEE trans. on Image Processing


   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include "blas.h"
/* #include "lapack.h" */
#include <complex.h>
#include <fftw3.h>

#define MAXITER   50000
#define MINITER   100
#define FGPITER   100
#define TD        50
#define ETA       1.1
#define EPS       1.0e-5

/* result format */

struct RESULT{
  int M;
  int N;
  int NX;
  int NY;
  int N_active;
  int maxiter;
  int ITER;
  int nonneg;
  double lambda_l1;
  double lambda_tv;
  double lambda_tsv;
  double sq_error;
  double mean_sq_error;
  double l1cost;
  double tvcost;
  double tsvcost;
  double looe_m;
  double looe_std;
  double Hessian_positive;
  double finalcost;
  double comp_time;
  double *residual;
  double Lip_const;
};

struct IO_FNAMES{
  unsigned int fft;
  char *fft_fname;
  char *v_fname;
  char *A_fname;
  char *in_fname;
  char *out_fname;
};

/* memory allocation of matrix and vectors */

extern int    *alloc_int_vector(int length);
extern float  *alloc_f_vector(int length);
extern double *alloc_vector(int length);
extern double *alloc_matrix(int height, int width);
extern void    clear_matrix(double *matrix, int height, int width);

/* file in-out */

extern FILE* fopenr(char* fn);
extern FILE* fopenw(char* fn);
extern int read_int_vector(char *fname, int length, int *vector);
extern int read_f_vector(char *fname, int length, float *vector);
extern int read_V_vector(char *fname, int length, double *vector);
extern unsigned long read_A_matrix(char *fname, int height, int width,
				   double *matrix);
extern int write_X_vector(char *fname, int length, double *vector);

/* simple matrix operations */

extern void transpose_matrix(double *matrix, int origheight, int origwidth);

extern double calc_Q_part(int *N, double *xvec1, double *xvec2,
			  double c, double *AyAz, double *buffxvec1);
/* soft thresholding */

extern void soft_threshold(double *vec, int length, double eta, double *nvec);

extern void soft_threshold_box(double *vec, int length, double eta,
			       double *nvec, int box_flag, float *cl_box);

extern void soft_threshold_nonneg(double *vec, int length, double eta,
				  double *nvec);

extern void soft_threshold_nonneg_box(double *vec, int length, double eta,
				      double *nvec, int box_flag, float *cl_box);

/* Some routines for TV */

extern double TV(int NX, int NY, double *xvec);

/* Some routines for TSV */

extern double TSV(int NX, int NY, double *xvec);

extern void d_TSV(int NX, int NY, double *xvec, double *dvec);

/* subroutines for mfista_L1_TV */

extern void FGP_L1(int *N, int NX, int NY,
		   double *bvec, double lambda_l1, double lambda_tv, int ITER,
		   double *pmat, double *qmat, double *rmat, double *smat,
		   double *npmat, double *nqmat, double *xvec);

extern void FGP_L1_box(int *N, int NX, int NY,
		       double *bvec, double lambda_l1, double lambda_tv, int ITER,
		       double *pmat, double *qmat, double *rmat, double *smat,
		       double *npmat, double *nqmat, double *xvec,
		       int box_flag, float *cl_box);

extern void FGP_nonneg(int *N, int NX, int NY,
		       double *bvec, double lambda_tv, int ITER,
		       double *pmat, double *qmat, double *rmat, double *smat,
		       double *npmat, double *nqmat, double *xvec);

extern void FGP_nonneg_box(int *N, int NX, int NY,
			   double *bvec, double lambda_tv, int ITER,
			   double *pmat, double *qmat, double *rmat, double *smat,
			   double *npmat, double *nqmat, double *xvec,
			   int box_flag, float *cl_box);

/* subroutines for mfista_L1_TSV_fftw */

extern void idx2mat(int M, int NX, int NY,
		    int *u_idx, int *v_idx, double *y_r, double *y_i, double *noise_stdev,
		    fftw_complex *yf, double *mk);

/* for mfista_imaging_dft */

extern void mfista_imaging_core(double *y, double *A,
				int *M, int *N, int NX, int NY, int maxiter, double eps,
				double lambda_l1, double lambda_tv, double lambda_tsv,
				double cinit, double *xinit, double *xout,
				int nonneg_flag, int looe_flag,
				int box_flag, float *cl_box,
				struct RESULT *mfista_result);

/* for mfista_imaging_fft */

extern void mfista_imaging_core_fft(int *u_idx, int *v_idx,
				    double *y_r, double *y_i, double *noise_stdev,
				    int M, int NX, int NY, int maxiter, double eps,
				    double lambda_l1, double lambda_tv, double lambda_tsv,
				    double cinit, double *xinit, double *xout,
				    int nonneg_flag, unsigned int fftw_plan_flag,
				    int box_flag, float *cl_box,
				    struct RESULT *mfista_result);

/* looe */

extern double compute_LOOE_L1(int *M, int *N, double lambda_l1,
			      double *yvec, double *Amat, double *xvec, double *yAx,
			      double *looe_m, double *looe_std);

extern double compute_LOOE_L1_TSV(int *M, int *N, int NX, int NY,
				  double lambda_l1, double lambda_tsv,
				  double *yvec, double *Amat, double *xvec, double *yAx,
				  double *looe_m, double *looe_std);

/* output */

extern void show_io_fnames(FILE *fid, char *fname, struct IO_FNAMES *mfista_io);

extern void show_result(FILE *fid, char *fname, struct RESULT *mfista_result);

extern void get_current_time(struct timespec *t);
