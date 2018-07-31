/*
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista_lib.c'. An optimization algorithm for imaging
   of interferometry. The idea of the algorithm was from the following
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
#include "mfista.h"

void fft_half2full(int NX, int NY, fftw_complex *FT_h, fftw_complex *FT)
{
  int i, j, NX_h, NY_h;

  /* set parameters */

  NX_h = (int)floor(((double)NX)/2) + 1;
  NY_h = (int)floor(((double)NY)/2) + 1;

  /* main */

  for(i=0; i<NX; ++i) for(j=0; j<NY_h; ++j) FT[NY*i+j] = FT_h[NY_h*i+j];

  for(i=1; i<NX_h-1; ++i) for(j=1; j<NY_h-1; ++j) FT[NY*(NX-i)+(NY-j)] = conj(FT_h[NY_h*i+j]);

  for(i=NX_h; i<NX; ++i)  for(j=1; j<NY_h-1; ++j) FT[NY*(NX-i)+(NY-j)] = conj(FT_h[NY_h*i+j]);

  for(j=1; j<NY_h-1; ++j){
    FT[(NY-j)]             = conj(FT_h[j]);
    FT[NY*(NX_h-1)+(NY-j)] = conj(FT_h[NY_h*(NX_h-1) + j]);
  }
}

void fft_full2half(int NX, int NY, fftw_complex *FT, fftw_complex *FT_h)
{
  int i, j, NY_h;

  NY_h = (int)floor(((double)NY)/2)+1;

  for(i=0; i<NX; ++i) for(j=0; j<NY_h; ++j) FT_h[NY_h*i + j] = FT[NY*i + j];

}

void full2half(int NX, int NY, double *mk, double *mk_h)
{
  int i, j, NY_h;

  NY_h = (int)floor(((double)NY)/2)+1;

  for(i=0; i<NX; ++i) for(j=0; j<NY_h; ++j) mk_h[NY_h*i + j] = mk[NY*i + j];

}

void idx2mat(int M, int NX, int NY,
	     int *u_idx, int *v_idx, double *y_r, double *y_i, double *noise_stdev,
	     fftw_complex *yf, double *mk)
{
  int i, j;

  /* main */

  for(i=0; i<NX; ++i) for(j=0; j<NY; ++j){
      mk[NY*i + j] = 0.0;
      yf[NY*i + j] = 0.0 + 0.0*I;
    }

  for(i=0; i<M; ++i){
    mk[NY*(u_idx[i]) + (v_idx[i])] = 1/noise_stdev[i];
    yf[NY*(u_idx[i]) + (v_idx[i])] = (y_r[i] + y_i[i]*I)/noise_stdev[i];
  }

}

void idx2mat_h(int M, int NX, int NY,
	       int *u_idx, int *v_idx, double *y_r, double *y_i, double *noise_stdev,
	       fftw_complex *yf_h, double *mask_h)
{
  double *mk;
  fftw_complex *yf;

  /* allocate vectors */

  mk = alloc_vector(NX*NY);
  yf = (fftw_complex*)  fftw_malloc(NX*NY*sizeof(fftw_complex));

  /* main */

  idx2mat(M, NX, NY, u_idx, v_idx, y_r, y_i, noise_stdev, yf, mk);

  full2half(NX, NY, mk, mask_h);
  fft_full2half(NX, NY, yf, yf_h);

  /* free vectors */

  free(mk);
  fftw_free(yf);

}

void calc_yAx_fft(int NX, int NY, fftw_complex *y_fft_h, double *mask, fftw_complex *yAx_h)
{
  int i, NY_h;
  double sqrtNN;

  /* set parameters */

  NY_h   = (int)floor(((double)NY)/2) + 1;
  //sqrtNN = sqrt((double)(NX*NY));

  /* main */

  //for(i=0;i<NX*NY_h;++i) yAx_h[i] = y_fft_h[i] - mask[i]*yAx_h[i]/sqrtNN;
  for(i=0;i<NX*NY_h;++i) yAx_h[i] = y_fft_h[i] - mask[i]*yAx_h[i];

}

double norm_fftw_complex(int N, fftw_complex *vec)
{
  int i;
  fftw_complex tmp_norm = 0.0 + 0.0*I;

  for(i=0; i<N; ++i) tmp_norm += vec[i]*conj(vec[i]);

  return(creal(tmp_norm));
}

double calc_F_part_fft(int *N, int NX, int NY,
		       fftw_complex *yf_h, double *mask_h,
		       fftw_plan *fftwplan, double *xvec,
		       fftw_complex *yAx_fh, double *x4f, fftw_complex *yAx_f)
{
  double result;
  int inc = 1;

  dcopy_(N, xvec, &inc, x4f, &inc);
  fftw_execute(*fftwplan);

  calc_yAx_fft(NX, NY, yf_h, mask_h, yAx_fh);
  fft_half2full(NX, NY, yAx_fh, yAx_f);

  result = norm_fftw_complex(*N, yAx_f);

  return(result/4);
}

void dF_dx_fft(int *N, int NX, int NY,
	       fftw_complex *yAx_fh, double *mask_h, double *xvec,
	       fftw_plan *ifftwplan, double *x4f, double *dFdx)
{
  int i, inc = 1, NY_h;
  //double sqNN = sqrt((double)(NX*NY));
  double NN = (double)(NX*NY);

  NY_h = (int)floor(((double)NY)/2) + 1;

  //for(i=0;i<NX*NY_h;++i) yAx_fh[i] *= (mask_h[i]/(2*sqNN));
  for(i=0;i<NX*NY_h;++i) yAx_fh[i] *= (mask_h[i]/(2*NN));

  fftw_execute(*ifftwplan);

  dcopy_(N, x4f, &inc, dFdx, &inc);
}

/* TV */

int mfista_L1_TV_core_fft(int NX, int NY, int maxiter, double eps,
			  fftw_complex *yf, double *mask,
			  double lambda_l1, double lambda_tv,
			  double *cinit, double *xinit, double *xout,
			  int nonneg_flag, unsigned int fftw_plan_flag,
			  int box_flag, float *cl_box)
{
  int i, iter, inc = 1, NN, NY_h;
  double *mask_h, *zvec, *xtmp, *xnew, *dfdx, *ones,
    *rmat, *smat, *npmat, *nqmat, *x4f,
    Qcore, Fval, Qval, c, costtmp, *cost, *pmat, *qmat,
    mu=1, munew, alpha = 1, tmpa, l1cost, tvcost;
  fftw_complex *yf_h, *yAx_fh, *yAx_f;
  fftw_plan fftwplan, ifftwplan;

  /* set parameters */

  NN = NX*NY;
  NY_h = ((int)floor(((double)NY)/2)+1);

  printf("computing image with MFISTA.\n");

  printf("stop if iter = %d, or Delta_cost < %e\n", maxiter, eps);

  /* allocate variables */

  cost  = alloc_vector(maxiter);
  dfdx  = alloc_vector(NN);
  xnew  = alloc_vector(NN);
  xtmp  = alloc_vector(NN);
  zvec  = alloc_vector(NN);
  x4f   = alloc_vector(NN);
  mask_h = alloc_vector(NX*NY_h);

  /* preparation for TV */

  ones  = alloc_vector(NN);
  for(i = 0; i < NN; ++i) ones[i]=1;

  pmat = alloc_matrix(NX-1,NY);
  qmat = alloc_matrix(NX,NY-1);

  npmat = alloc_matrix(NX-1,NY);
  nqmat = alloc_matrix(NX,NY-1);

  rmat = alloc_matrix(NX-1,NY);
  smat = alloc_matrix(NX,NY-1);

  /* fftw malloc */

  yAx_f  = (fftw_complex*)  fftw_malloc(NN*sizeof(fftw_complex));
  yAx_fh = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yf_h   = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));

  /* preparation for fftw */

  full2half(NX, NY, mask, mask_h);
  fft_full2half(NX, NY, yf, yf_h);

#ifdef PTHREAD
  int omp_num = THREAD_NUM;
  printf("Run mfista with %d threads.\n",omp_num);

  if(fftw_init_threads()==0)
    printf("Could not initialize multi threads for fftw3.\n");
#endif

  fftwplan  = fftw_plan_dft_r2c_2d( NX, NY, x4f, yAx_fh, fftw_plan_flag);
  ifftwplan = fftw_plan_dft_c2r_2d( NX, NY, yAx_fh, x4f, fftw_plan_flag);

  /* initialize xvec */

  dcopy_(&NN, xinit, &inc, xout, &inc);
  dcopy_(&NN, xinit, &inc, zvec, &inc);

  c = *cinit;

  /* main */

  costtmp = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h,
			    &fftwplan, xout, yAx_fh, x4f, yAx_f);

  l1cost = dasum_(&NN, xout, &inc);
  costtmp += lambda_l1*l1cost;

  tvcost = TV(NX, NY, xout);
  costtmp += lambda_tv*tvcost;

  for(iter=0; iter<maxiter; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0) printf("%d cost = %f, c = %f \n",(iter+1), cost[iter], c);

    Qcore = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h,
			    &fftwplan, zvec, yAx_fh, x4f, yAx_f);

    dF_dx_fft(&NN, NX, NY, yAx_fh, mask_h, zvec, &ifftwplan, x4f, dfdx);

    for(i=0; i<maxiter; i++){

      if(nonneg_flag == 1){
	dcopy_(&NN, ones, &inc, xtmp, &inc);
	tmpa = -lambda_l1/c;
	dscal_(&NN, &tmpa, xtmp, &inc);
	tmpa = 1/c;
	daxpy_(&NN, &tmpa, dfdx, &inc, xtmp, &inc);
	daxpy_(&NN, &alpha, zvec, &inc, xtmp, &inc);

	FGP_nonneg_box(&NN, NX, NY, xtmp, lambda_tv/c, FGPITER,
		       pmat, qmat, rmat, smat, npmat, nqmat, xnew,
		       box_flag, cl_box);
      }
      else{
	dcopy_(&NN, dfdx, &inc, xtmp, &inc);
	tmpa = 1/c;
	dscal_(&NN, &tmpa, xtmp, &inc);
	daxpy_(&NN, &alpha, zvec, &inc, xtmp, &inc);

	FGP_L1_box(&NN, NX, NY, xtmp, lambda_l1/c, lambda_tv/c, FGPITER,
		   pmat, qmat, rmat, smat, npmat, nqmat, xnew,
		   box_flag, cl_box);
      }

      Fval = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h,
			     &fftwplan, xnew, yAx_fh, x4f, yAx_f);

      Qval = calc_Q_part(&NN, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1 + sqrt(1 + 4*mu*mu))/2;

    l1cost = dasum_(&NN, xnew, &inc);
    Fval += lambda_l1*l1cost;

    tvcost = TV(NX, NY, xnew);
    Fval += lambda_tv*tvcost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(&NN, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);

      dcopy_(&NN, xnew, &inc, xout, &inc);
    }
    else{
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(&NN, &tmpa, zvec, &inc);

      tmpa = mu/munew;
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);

      /* another stopping rule */
      if((iter>1) && (dasum_(&NN, xout, &inc) == 0)){
	printf("x becomes a 0 vector.\n");
	break;
      }
    }

    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])< eps )) break;

    mu = munew;
  }
  if(iter == maxiter){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);

  printf("\n");

  *cinit = c;

  /* clear memory */

  free(ones);

  free(npmat);
  free(nqmat);
  free(pmat);
  free(qmat);
  free(rmat);
  free(smat);

  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(zvec);
  free(x4f);
  free(mask_h);

  fftw_free(yAx_f);
  fftw_free(yAx_fh);
  fftw_free(yf_h);

  fftw_destroy_plan(fftwplan);
  fftw_destroy_plan(ifftwplan);

#ifdef PTHREAD
  fftw_cleanup_threads();
#else
  fftw_cleanup();
#endif


  return(iter+1);
}

/* TSV */

int mfista_L1_TSV_core_fft(int NX, int NY, int maxiter, double eps,
			   fftw_complex *yf, double *mask,
			   double lambda_l1, double lambda_tsv,
			   double *cinit, double *xinit, double *xout,
			   int nonneg_flag, unsigned int fftw_plan_flag,
			   int box_flag, float *cl_box)
{
  void (*soft_th_box)(double *vector, int length, double eta, double *newvec,
		      int box_flag, float *cl_box);
  int NN, i, iter, inc = 1, NY_h;
  double *mask_h, *xtmp, *xnew, *zvec, *dfdx, *x4f,
    Qcore, Fval, Qval, c, cinv, tmpa, l1cost, tsvcost, costtmp, *cost,
    mu=1, munew, alpha = 1, beta = -lambda_tsv;
  fftw_complex *yf_h, *yAx_fh, *yAx_f;
  fftw_plan fftwplan, ifftwplan;

  /* set parameters */

  NN = NX*NY;
  NY_h = ((int)floor(((double)NY)/2)+1);

  printf("computing image with MFISTA.\n");

  printf("stop if iter = %d, or Delta_cost < %e\n", maxiter, eps);

  /* allocate variables */

  cost   = alloc_vector(maxiter);
  dfdx   = alloc_vector(NN);
  xnew   = alloc_vector(NN);
  xtmp   = alloc_vector(NN);
  zvec   = alloc_vector(NN);
  x4f    = alloc_vector(NN);
  mask_h = alloc_vector(NX*NY_h);

  /* fftw malloc */

  yAx_f  = (fftw_complex*) fftw_malloc(NN*sizeof(fftw_complex));
  yAx_fh = (fftw_complex*) fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yf_h   = (fftw_complex*) fftw_malloc(NX*NY_h*sizeof(fftw_complex));

  /* preparation for fftw */

  full2half(NX, NY, mask, mask_h);
  fft_full2half(NX, NY, yf, yf_h);

#ifdef PTHREAD
  int omp_num = THREAD_NUM;
  printf("Run mfista with %d threads.\n",omp_num);

  if(fftw_init_threads()==0)
    printf("Could not initialize multi threads for fftw3.\n");
#endif

  fftwplan  = fftw_plan_dft_r2c_2d( NX, NY, x4f, yAx_fh, fftw_plan_flag);
  ifftwplan = fftw_plan_dft_c2r_2d( NX, NY, yAx_fh, x4f, fftw_plan_flag);

  /* initialization */

  if(nonneg_flag == 0)
    soft_th_box =soft_threshold_box;
  else if(nonneg_flag == 1)
    soft_th_box =soft_threshold_nonneg_box;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return(0);
  }

  dcopy_(&NN, xinit, &inc, xout, &inc);
  dcopy_(&NN, xinit, &inc, zvec, &inc);

  c = *cinit;

  /* main */

  costtmp = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h,
			    &fftwplan, xout, yAx_fh, x4f, yAx_f);

  l1cost = dasum_(&NN, xout, &inc);
  costtmp += lambda_l1*l1cost;

  if( lambda_tsv > 0 ){
    tsvcost = TSV(NX, NY, xout);
    costtmp += lambda_tsv*tsvcost;
  }

  for(iter = 0; iter < maxiter; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f, c = %f \n",(iter+1), cost[iter], c);

    Qcore = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h,
			    &fftwplan, zvec, yAx_fh, x4f, yAx_f);

    dF_dx_fft(&NN, NX, NY, yAx_fh, mask_h, zvec, &ifftwplan, x4f, dfdx);

    if( lambda_tsv > 0.0 ){
      tsvcost = TSV(NX, NY, zvec);
      Qcore += lambda_tsv*tsvcost;

      d_TSV(NX, NY, zvec, x4f);
      dscal_(&NN, &beta, x4f, &inc);
      daxpy_(&NN, &alpha, x4f, &inc, dfdx, &inc);
    }

    for( i = 0; i < maxiter; i++){
      dcopy_(&NN, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(&NN, &cinv, xtmp, &inc);
      daxpy_(&NN, &alpha, zvec, &inc, xtmp, &inc);
      soft_th_box(xtmp, NN, lambda_l1/c, xnew, box_flag, cl_box);

      Fval = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h,
			     &fftwplan, xnew, yAx_fh, x4f, yAx_f);

      if( lambda_tsv > 0.0 ){
	tsvcost = TSV(NX, NY, xnew);
	Fval += lambda_tsv*tsvcost;
      }

      Qval = calc_Q_part(&NN, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(&NN, xnew, &inc);
    Fval += lambda_l1*l1cost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(&NN, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);

      dcopy_(&NN, xnew, &inc, xout, &inc);

    }
    else{
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(&NN, &tmpa, zvec, &inc);

      tmpa = mu/munew;
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);

      /* another stopping rule */
      if((iter>1) && (dasum_(&NN, xout, &inc) == 0)) break;
    }

    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])< eps )) break;

    mu = munew;
  }
  if(iter == maxiter){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);

  printf("\n");

  *cinit = c;

  /* free */

  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(zvec);
  free(x4f);
  free(mask_h);

  fftw_free(yAx_f);
  fftw_free(yAx_fh);
  fftw_free(yf_h);

  fftw_destroy_plan(fftwplan);
  fftw_destroy_plan(ifftwplan);

#ifdef PTHREAD
  fftw_cleanup_threads();
#else
  fftw_cleanup();
#endif

  return(iter+1);
}

/* results */

void calc_result_fft(int M, int NX, int NY,
		     fftw_complex *yf, double *mask,
		     double lambda_l1, double lambda_tv, double lambda_tsv,
		     double *xvec,
		     struct RESULT *mfista_result)
{
  int i, NY_h = ((int)floor(((double)NY)/2)+1), NN = NX*NY;
  double *mask_h, *x4f, tmp;
  fftw_complex *yf_h, *yAx_fh, *yAx_f;
  fftw_plan fftwplan;

  /* allocate variables */

  x4f    = alloc_vector(NN);
  mask_h = alloc_vector(NX*NY_h);

  /* fftw malloc */

  yAx_f  = (fftw_complex*) fftw_malloc(NN*sizeof(fftw_complex));
  yAx_fh = (fftw_complex*) fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yf_h   = (fftw_complex*) fftw_malloc(NX*NY_h*sizeof(fftw_complex));

  /* preparation for fftw */

  full2half(NX, NY, mask, mask_h);
  fft_full2half(NX, NY, yf, yf_h);

  fftwplan = fftw_plan_dft_r2c_2d( NX, NY, x4f, yAx_fh, FFTW_ESTIMATE);

  /* computing results */

  tmp = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h, &fftwplan, xvec, yAx_fh, x4f, yAx_f);

  /* saving results */

  mfista_result->sq_error = 2*tmp;

  mfista_result->M  = (int)(M/2);
  mfista_result->N  = NN;
  mfista_result->NX = NX;
  mfista_result->NY = NY;

  mfista_result->lambda_l1  = lambda_l1;
  mfista_result->lambda_tv  = lambda_tv;
  mfista_result->lambda_tsv = lambda_tsv;

  mfista_result->mean_sq_error = mfista_result->sq_error/((double)M);

  mfista_result->l1cost   = 0;
  mfista_result->N_active = 0;

  for(i = 0;i < NN;++i){
    tmp = fabs(xvec[i]);
    if(tmp > 0){
      mfista_result->l1cost += tmp;
      ++ mfista_result->N_active;
    }
  }

  mfista_result->finalcost = (mfista_result->sq_error)/2;

  if(lambda_l1 > 0)
    mfista_result->finalcost += lambda_l1*(mfista_result->l1cost);

  if(lambda_tsv > 0){
    mfista_result->tsvcost = TSV(NX, NY, xvec);
    mfista_result->finalcost += lambda_tsv*(mfista_result->tsvcost);
  }
  else if (lambda_tv > 0){
    mfista_result->tvcost = TV(NX, NY, xvec);
    mfista_result->finalcost += lambda_tv*(mfista_result->tvcost);
  }

  /* free */

  free(x4f);
  free(mask_h);

  fftw_free(yAx_f);
  fftw_free(yAx_fh);
  fftw_free(yf_h);

  fftw_destroy_plan(fftwplan);
  fftw_cleanup();
}

/* main subroutine */

void mfista_imaging_core_fft(int *u_idx, int *v_idx,
			     double *y_r, double *y_i, double *noise_stdev,
			     int M, int NX, int NY, int maxiter, double eps,
			     double lambda_l1, double lambda_tv, double lambda_tsv,
			     double cinit, double *xinit, double *xout,
			     int nonneg_flag, int fftw_measure,
			     int box_flag, float *cl_box,
			     struct RESULT *mfista_result)
{
  int i, iter = 0;
  double epsilon, *mask, s_t, e_t, c = cinit;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;
  struct timespec time_spec1, time_spec2;
  fftw_complex *yf;

  for(epsilon=0, i=0;i<M;++i) epsilon += y_r[i]*y_r[i] + y_i[i]*y_i[i];

  epsilon *= eps/((double)M);

  yf   = (fftw_complex*) fftw_malloc(NX*NY*sizeof(fftw_complex));
  mask = alloc_vector(NX*NY);

  idx2mat(M, NX, NY, u_idx, v_idx, y_r, y_i, noise_stdev, yf, mask);

  get_current_time(&time_spec1);

 if ( fftw_measure == 0){
   fftw_plan_flag = FFTW_MEASURE;
 }

  if( lambda_tv == 0 ){
    iter = mfista_L1_TSV_core_fft(NX, NY, maxiter, epsilon,
				  yf, mask, lambda_l1, lambda_tsv, &c, xinit, xout,
				  nonneg_flag, fftw_plan_flag, box_flag, cl_box);
  }
  else if( lambda_tv != 0  && lambda_tsv == 0 ){
    iter = mfista_L1_TV_core_fft(NX, NY, maxiter, epsilon,
				 yf, mask, lambda_l1, lambda_tv, &c, xinit, xout,
				 nonneg_flag, fftw_plan_flag, box_flag, cl_box);
  }
  else{
    printf("You cannot set both of lambda_TV and lambda_TSV positive.\n");
    return;
  }

  get_current_time(&time_spec2);

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result->comp_time = e_t-s_t;
  mfista_result->ITER      = iter;
  mfista_result->nonneg    = nonneg_flag;
  mfista_result->Lip_const = c;
  mfista_result->maxiter   = maxiter;

  calc_result_fft(M, NX, NY, yf, mask, lambda_l1, lambda_tv, lambda_tsv, xout, mfista_result);

  fftw_free(yf);
  free(mask);
}
