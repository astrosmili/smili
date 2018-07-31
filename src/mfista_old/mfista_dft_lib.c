/* 
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista.c'. An optimization algorithm for imaging of
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

#include "mfista.h"

/* subroutines for mfista*/

void calc_yAz(int *M, int *N,
	      double *yvec, double *Amat, double *zvec, double *yAz)
{
  int inc = 1;
  double alpha = -1, beta = 1;

  /* y - A x2 */
  dcopy_(M, yvec, &inc, yAz, &inc);
  dgemv_("N", M, N, &alpha, Amat, M, zvec, &inc, &beta, yAz, &inc);
}

void dL_dx(int *M, int *N, double *yAx, double *Amatrix, double *dfdx)
{
  int inc = 1;
  double beta = 1, gamma = 0;

  /* A' (y - A x) */
  dgemv_("T", M, N, &beta, Amatrix, M, yAx, &inc, &gamma, dfdx, &inc);

}

double calc_F_part(int *M, int *N,
		   double *yvec, double *Amatrix,
		   double *xvec, double *buffvec)
{
  int inc = 1;
  double term1, alpha = -1, beta = 1;

  dcopy_(M, yvec, &inc, buffvec, &inc);

  dgemv_("N", M, N, &alpha, Amatrix, M, 
	 xvec, &inc, &beta, buffvec, &inc);

  term1 = ddot_(M, buffvec, &inc, buffvec, &inc);

  return(term1/2);
}

/* mfista for TV */

int mfista_L1_TV_core(double *yvec, double *Amat, 
		      int *M, int *N, int NX, int NY, int maxiter, double eps,
		      double lambda_l1, double lambda_tv, double *cinit,
		      double *xvec, int nonneg_flag, int box_flag, float *cl_box)
{
  double *ytmp, *zvec, *xtmp, *xnew, *yAz, *dfdx,
    *rmat, *smat, *npmat, *nqmat,
    Qcore, Fval, Qval, c, costtmp, *cost, *pmat, *qmat, *ones,
    mu=1, munew, alpha = 1, gamma=0, tmpa, l1cost, tvcost;
  int i, iter, inc = 1;

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  zvec  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  dfdx  = alloc_vector(*N);
  yAz   = alloc_vector(*M);
  ytmp  = alloc_vector(*M);
  xtmp  = alloc_vector(*N);
  ones  = alloc_vector(*N);

  if( nonneg_flag == 1){
    for(i = 0; i < (*N); ++i) ones[i]=1;
  }

  pmat = alloc_matrix(NX-1,NY);
  qmat = alloc_matrix(NX,NY-1);

  npmat = alloc_matrix(NX-1,NY);
  nqmat = alloc_matrix(NX,NY-1);

  rmat = alloc_matrix(NX-1,NY);
  smat = alloc_matrix(NX,NY-1);

  cost  = alloc_vector(maxiter);

  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = *cinit;

  costtmp = calc_F_part(M, N, yvec, Amat, xvec, ytmp);

  l1cost = dasum_(N, xvec, &inc);
  costtmp += lambda_l1*l1cost;
    
  tvcost = TV(NX, NY, xvec);
  costtmp += lambda_tv*tvcost;

  for(iter = 0; iter < maxiter; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    calc_yAz(M, N,  yvec, Amat, zvec, yAz);
    dgemv_("T", M, N, &alpha, Amat, M, yAz, &inc, &gamma, dfdx, &inc);

    Qcore =  ddot_(M, yAz, &inc, yAz, &inc)/2;

    for( i = 0; i < maxiter; i++){

      if( nonneg_flag == 1){
	dcopy_(N, ones, &inc, xtmp, &inc);
	tmpa = -lambda_l1/c;
	dscal_(N, &tmpa, xtmp, &inc);
	tmpa = 1/c;
	daxpy_(N, &tmpa, dfdx, &inc, xtmp, &inc);
	daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);

	FGP_nonneg_box(N, NX, NY, xtmp, lambda_tv/c, FGPITER,
		       pmat, qmat, rmat, smat, npmat, nqmat, xnew,
		       box_flag, cl_box);
      }
      else{
	dcopy_(N, dfdx, &inc, xtmp, &inc);
	tmpa = 1/c;
	dscal_(N, &tmpa, xtmp, &inc);
	daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);

	FGP_L1_box(N, NX, NY, xtmp, lambda_l1/c, lambda_tv/c, FGPITER,
		   pmat, qmat, rmat, smat, npmat, nqmat, xnew,
		   box_flag, cl_box);
      }

      Fval = calc_F_part(M, N, yvec, Amat, xnew, ytmp);
      Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;
      
      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(N, xnew, &inc);
    Fval += lambda_l1*l1cost;

    tvcost = TV(NX, NY, xnew);
    Fval += lambda_tv*tvcost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(N, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);
	
      dcopy_(N, xnew, &inc, xvec, &inc);
    }	
    else{
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(N, &tmpa, zvec, &inc);
      
      tmpa = mu/munew;
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);

      /* another stopping rule */
      if((iter>1) && (dasum_(N, xvec, &inc) == 0)){
	printf("x becomes a 0 vector.\n");
	break;
      }
    }

    /* stopping rule start */
     
    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<eps)) break;

    /* stopping rule end */

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

  free(ytmp);

  free(xtmp);

  free(zvec);
  free(xnew);

  free(dfdx);
  free(yAz);

  free(ones);
  free(pmat);
  free(qmat);
  free(npmat);
  free(nqmat);

  free(rmat);
  free(smat);
  free(cost);

  return(iter+1);
}

/* subroutines for TSV */

double calc_F_TSV_part(int *M, int NX, int NY, double *yAx, double *xvec, double lambda_tsv)
{
  int inc = 1;
  double term1, term2;

  term1 = ddot_(M, yAx, &inc, yAx, &inc);
  term2 = TSV(NX, NY, xvec);

  return(term1/2+lambda_tsv*term2);
}

void dF_dx(int *M, int *N, int NX, int NY,
	   double *yAx, double *Amatrix, 
	   double *xvec, double lambda_tsv,
	   double *dFdx)
{
  int inc = 1;
  double alpha = 1, beta = -lambda_tsv;

  d_TSV(NX, NY, xvec, dFdx);

  /* A' (y - A x) -lambda_tsv*d_TSV*/
  
  dgemv_("T", M, N, &alpha, Amatrix, M, yAx, &inc, &beta, dFdx, &inc);

}

/* TSV */

int mfista_L1_TSV_core(double *yvec, double *Amat, 
		       int *M, int *N, int NX, int NY, int maxiter, double eps,
		       double lambda_l1, double lambda_tsv, double *cinit,
		       double *xvec, int nonneg_flag, int box_flag, float *cl_box)
{
  void (*soft_th_box)(double *vector, int length, double eta, double *newvec,
		      int box_flag, float *cl_box);
  int i, iter, inc = 1;
  double *xtmp, *xnew, *ytmp, *zvec, *dfdx,
    Qcore, Fval, Qval, c, cinv, tmpa, l1cost, costtmp, *cost,
    mu=1, munew, alpha = 1;

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  cost  = alloc_vector(maxiter);
  dfdx  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  xtmp  = alloc_vector(*N);
  ytmp  = alloc_vector(*M);
  zvec  = alloc_vector(*N);

  /* defining soft_thresholding */
  
  if(nonneg_flag == 0)
    soft_th_box=soft_threshold_box;
  else if(nonneg_flag == 1)
    soft_th_box=soft_threshold_nonneg_box;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return(0);
  }

  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = *cinit;

  calc_yAz(M, N, yvec, Amat, xvec, ytmp);

  if( lambda_tsv > 0)
    costtmp = calc_F_TSV_part(M, NX, NY, ytmp, xvec, lambda_tsv);
  else
    costtmp = ddot_(M, ytmp, &inc, ytmp, &inc)/2;
  
  l1cost = dasum_(N, xvec, &inc);
  costtmp += lambda_l1*l1cost;

  for(iter = 0; iter < maxiter; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f, c = %f \n",(iter+1), cost[iter], c);

    calc_yAz(M, N,  yvec, Amat, zvec, ytmp);

    if(lambda_tsv > 0.0 ){
      dF_dx(M, N, NX, NY, ytmp, Amat, zvec, lambda_tsv, dfdx);
      Qcore = calc_F_TSV_part(M, NX, NY, ytmp, zvec, lambda_tsv);
    }
    else{
      dL_dx(M, N, ytmp, Amat, dfdx);
      Qcore = ddot_(M, ytmp, &inc, ytmp, &inc)/2;
    }

    for( i = 0; i < maxiter; i++){
      dcopy_(N, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(N, &cinv, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);
      soft_th_box(xtmp, *N, lambda_l1/c, xnew, box_flag, cl_box);

      calc_yAz(M, N, yvec, Amat, xnew, ytmp);

      if(lambda_tsv > 0.0 ){
	Fval = calc_F_TSV_part(M, NX, NY, ytmp, xnew, lambda_tsv);
	Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      }
      else{
	Fval = ddot_(M, ytmp, &inc, ytmp, &inc)/2;
	Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      }

      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(N, xnew, &inc);
    Fval += lambda_l1*l1cost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(N, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);
	
      dcopy_(N, xnew, &inc, xvec, &inc);
	    
    }	
    else{
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(N, &tmpa, zvec, &inc);
      
      tmpa = mu/munew;
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);

      /* another stopping rule */
      if((iter>1) && (dasum_(N, xvec, &inc) == 0)) break;
    }

    /* stopping rule start */
     
    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<eps)) break;

    /* stopping rule end */

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

  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(ytmp);
  free(zvec);
  return(iter+1);
}

/* results */

void calc_result(double *yvec, double *Amat,
		 int *M, int *N, int NX, int NY,
		 double lambda_l1, double lambda_tv, double lambda_tsv,
		 double *xvec, int nonneg_flag, int looe_flag,
		 struct RESULT *mfista_result)
{
  int i;
  double *yAx, tmpa;

  printf("summarizing result.\n");

  /* allocate memory space start */ 

  yAx  = alloc_vector(*M);
  mfista_result->residual = alloc_vector(*M);
  
  /* summary of results */

  mfista_result->M = (*M);
  mfista_result->N = (*N);
  mfista_result->NX = NX;
  mfista_result->NY = NY;
	    
  mfista_result->lambda_l1 = lambda_l1;
  mfista_result->lambda_tv = lambda_tv;
  mfista_result->lambda_tsv = lambda_tsv;

  calc_yAz(M, N, yvec, Amat, xvec, yAx);

  /* mean square error */
  mfista_result->sq_error = 0;

  for(i = 0;i< (*M);i++){
    mfista_result->sq_error += yAx[i]*yAx[i];
    mfista_result->residual[i] = yAx[i];
  }

  /* average of mean square error */

  mfista_result->mean_sq_error = mfista_result->sq_error/((double)(*M));

  mfista_result->l1cost   = 0;
  mfista_result->N_active = 0;

  for(i = 0;i < (*N);i++){
    tmpa = fabs(xvec[i]);
    if(tmpa > 0){
      mfista_result->l1cost += tmpa;
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

  /* computing LOOE */

  if(looe_flag == 1 && lambda_tv ==0 ){
    if(lambda_tsv == 0){
      mfista_result->looe_m = compute_LOOE_L1(M, N, lambda_l1, yvec, Amat, xvec, yAx,
					    &(mfista_result->looe_m), &(mfista_result->looe_std));
      printf("%le\n",mfista_result->looe_m);
    }
    else
      mfista_result->looe_m = compute_LOOE_L1_TSV(M, N, NX, NY, lambda_l1, lambda_tsv,
						yvec, Amat, xvec, yAx,
						&(mfista_result->looe_m), &(mfista_result->looe_std));
    if(mfista_result->looe_m == -1){
      mfista_result->Hessian_positive = 0;
      mfista_result->looe_m = 0;
    }
    else{
      mfista_result->Hessian_positive = 1;
    }
  }
  else{
    mfista_result->looe_m = 0;
    mfista_result->Hessian_positive = -1;
  }

  /* clear memory */
  
  free(yAx);
}

/* main subroutine */

void mfista_imaging_core(double *y, double *A, 
			 int *M, int *N, int NX, int NY, int maxiter, double eps,
			 double lambda_l1, double lambda_tv, double lambda_tsv,
			 double cinit, double *xinit, double *xout,
			 int nonneg_flag, int looe_flag,
			 int box_flag, float *cl_box,
			 struct RESULT *mfista_result)
{
  double s_t, e_t, c = cinit;
  int    iter = 0, inc = 1;
  struct timespec time_spec1, time_spec2;

  dcopy_(N, xinit, &inc, xout, &inc);
  
  get_current_time(&time_spec1);

  /* main loop */

  if( lambda_tv == 0){
    iter = mfista_L1_TSV_core(y, A, M, N, NX, NY, maxiter, eps,
			      lambda_l1, lambda_tsv, &c, xout, nonneg_flag, box_flag, cl_box);
  }
  else if( lambda_tv != 0  && lambda_tsv == 0 ){
    iter = mfista_L1_TV_core(y, A, M, N, NX, NY, maxiter, eps,
			     lambda_l1, lambda_tv, &c, xout, nonneg_flag, box_flag, cl_box);
  }
  else{
    printf("You cannot set both of lambda_TV and lambda_TSV positive.\n");
    return;
  }
    
  get_current_time(&time_spec2);

  /* main loop end */

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result->comp_time = e_t-s_t;
  mfista_result->ITER      = iter;
  mfista_result->nonneg    = nonneg_flag;
  mfista_result->Lip_const = c;
  mfista_result->maxiter   = maxiter;

  calc_result(y, A, M, N, NX, NY,
	      lambda_l1, lambda_tv, lambda_tsv, xout, nonneg_flag, looe_flag,
	      mfista_result);

  return;
}
