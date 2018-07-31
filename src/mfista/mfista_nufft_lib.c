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
#include "finufft_c.h"

#define NU_SGN 1
#define NU_EPS 1.0e-12

/* start nufft codes */

double complex *alloc_complex_vector(int length)
{
  return malloc(sizeof(double complex)*length);
}

void calc_yAx_nufft(double complex *yAx, int M, double complex *wvis, double complex *weight)
{
  int i;

  /* main */

  for(i=0; i<M; ++i) yAx[i] = wvis[i] - yAx[i]*weight[i];
}

double calc_F_part_nufft(double complex *yAx,
			 int M, int NX, int NY, double *u_dx, double *v_dy,
			 double complex *wvis, double complex *weight,
			 double *xvec, double complex *tmp_c_vec)
{
  int i;
  double complex chisq;

  for(i=0; i<NX*NY; ++i) tmp_c_vec[i] = xvec[i] + 0.0*I;

  finufft2d2_c(M, u_dx, v_dy, yAx, NU_SGN, NU_EPS, NX, NY, tmp_c_vec);

  calc_yAx_nufft(yAx, M, wvis, weight);

  for(chisq = 0, i = 0 ; i<M; ++i)
    chisq += yAx[i]*conj(yAx[i]);

  return(creal(chisq)/2);
}

void dF_dx_nufft(double *dFdx, int M, int NX, int NY, double *u_dx, double *v_dy,
		 double complex *yAx, double complex *weight, double complex *tmp_c_vec)
{
  int i;

  for(i=0; i<M; ++i) yAx[i] *= weight[i];

  finufft2d1_c(M, u_dx, v_dy, yAx, (-1)*NU_SGN, NU_EPS, NX, NY, tmp_c_vec);

  for(i=0; i<NX*NY; ++i) dFdx[i] = creal(tmp_c_vec[i]);
}

/* TV */

int mfista_L1_TV_core_nufft(double *xout,
			    int M, int NX, int NY, double *u_dx, double *v_dy,
			    int maxiter, double eps,
			    double complex *vis, double *vis_std,
			    double lambda_l1, double lambda_tv,
			    double *cinit, double *xinit, 
			    int nonneg_flag, int box_flag, float *cl_box)
{
  int i, iter, inc = 1, NN;
  double *zvec, *xtmp, *xnew, *dfdx, *ones,
    *rmat, *smat, *npmat, *nqmat, *pmat, *qmat,
    *cost, 
    Qcore, Fval, Qval, c, costtmp, 
    mu=1, munew, alpha = 1, tmpa, l1cost, tvcost;
  double complex *yAx, *tmp_c_vec, *wvis, *weight;

  /* set parameters */
  
  NN = NX*NY;

  printf("computing image with MFISTA.\n");

  printf("stop if iter = %d, or Delta_cost < %e\n", maxiter, eps);

  /* allocate variables */ 

  cost  = alloc_vector(maxiter);
  dfdx  = alloc_vector(NN);
  xnew  = alloc_vector(NN);
  xtmp  = alloc_vector(NN);
  zvec  = alloc_vector(NN);

  /* preparation for TV */

  ones  = alloc_vector(NN);
  for(i = 0; i < NN; ++i) ones[i]=1;

  pmat = alloc_matrix(NX-1,NY);
  qmat = alloc_matrix(NX,NY-1);

  npmat = alloc_matrix(NX-1,NY);
  nqmat = alloc_matrix(NX,NY-1);

  rmat = alloc_matrix(NX-1,NY);
  smat = alloc_matrix(NX,NY-1);

  /* complex malloc */

  yAx       = alloc_complex_vector(M);
  wvis      = alloc_complex_vector(M);
  weight    = alloc_complex_vector(M);
  tmp_c_vec = alloc_complex_vector(NN);

  /* initialize xvec */
  
  dcopy_(&NN, xinit, &inc, xout, &inc);
  dcopy_(&NN, xinit, &inc, zvec, &inc);

  c = *cinit;

  for(i=0;i<M;++i){
    wvis[i] = (double complex) vis[i]/vis_std[i];
    weight[i] = (double complex) 1/vis_std[i];
  }

  /* main */

  costtmp = calc_F_part_nufft(yAx, M, NX, NY, u_dx, v_dy, wvis, weight, xout, tmp_c_vec);

  l1cost = dasum_(&NN, xout, &inc);
  costtmp += lambda_l1*l1cost;

  tvcost = TV(NX, NY, xout);
  costtmp += lambda_tv*tvcost;

  for(iter=0; iter<maxiter; iter++){

    cost[iter] = costtmp;

    /*if((iter % 100) == 0)*/ printf("%d cost = %f, c = %f \n",(iter+1), cost[iter], c);

    Qcore = calc_F_part_nufft(yAx, M, NX, NY, u_dx, v_dy, wvis, weight, zvec, tmp_c_vec);

    dF_dx_nufft(dfdx, M, NX, NY, u_dx, v_dy, yAx, weight, tmp_c_vec);
            
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

      Fval = calc_F_part_nufft(yAx, M, NX, NY, u_dx, v_dy, wvis, weight, xnew, tmp_c_vec);

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

  free(yAx);
  free(wvis);
  free(weight);
  free(tmp_c_vec);

  return(iter+1);
}

/* TSV */

int mfista_L1_TSV_core_nufft(double *xout,
			     int M, int NX, int NY, double *u_dx, double *v_dy,
			     int maxiter, double eps,
			     double complex *vis, double *vis_std,
			     double lambda_l1, double lambda_tsv,
			     double *cinit, double *xinit, 
			     int nonneg_flag, int box_flag, float *cl_box)
{
  void (*soft_th_box)(double *vector, int length, double eta, double *newvec,
		      int box_flag, float *cl_box);
  int NN, i, iter, inc = 1;
  double *xtmp, *xnew, *zvec, *dfdx, *dtmp,
    Qcore, Fval, Qval, c, cinv, tmpa, l1cost, tsvcost, costtmp, *cost,
    mu=1, munew, alpha = 1, beta = -lambda_tsv;
  double complex *yAx, *tmp_c_vec, *wvis, *weight;

  /* set parameters */

  NN = NX*NY;

  printf("computing image with MFISTA with NUFFT.\n");

  printf("stop if iter = %d, or Delta_cost < %e\n", maxiter, eps);

  /* allocate variables */

  cost   = alloc_vector(maxiter);
  dfdx   = alloc_vector(NN);
  xnew   = alloc_vector(NN);
  xtmp   = alloc_vector(NN);
  zvec   = alloc_vector(NN);
  dtmp   = alloc_vector(NN);
 
  /* complex malloc */

  yAx       = alloc_complex_vector(M);
  wvis      = alloc_complex_vector(M);
  weight    = alloc_complex_vector(M);
  tmp_c_vec = alloc_complex_vector(NN);

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

  for(i=0;i<M;++i){
    wvis[i] = (double complex) vis[i]/vis_std[i];
    weight[i] = (double complex) 1/vis_std[i];
  }

  /* main */

  costtmp = calc_F_part_nufft(yAx, M, NX, NY, u_dx, v_dy, wvis, weight, xout, tmp_c_vec);

  l1cost = dasum_(&NN, xout, &inc);
  costtmp += lambda_l1*l1cost;

  if( lambda_tsv > 0 ){
    tsvcost = TSV(NX, NY, xout);
    costtmp += lambda_tsv*tsvcost;
  }

  for(iter = 0; iter < maxiter; iter++){

    cost[iter] = costtmp;

    if((iter % 10) == 0) printf("%d cost = %f, c = %f \n",(iter+1), cost[iter], c);

    Qcore = calc_F_part_nufft(yAx, M, NX, NY, u_dx, v_dy, wvis, weight, zvec, tmp_c_vec);

    dF_dx_nufft(dfdx, M, NX, NY, u_dx, v_dy, yAx, weight, tmp_c_vec);

    if( lambda_tsv > 0.0 ){
      tsvcost = TSV(NX, NY, zvec);
      Qcore += lambda_tsv*tsvcost;

      d_TSV(NX, NY, zvec, dtmp);
      dscal_(&NN, &beta, dtmp, &inc);
      daxpy_(&NN, &alpha, dtmp, &inc, dfdx, &inc);
    }

    for( i = 0; i < maxiter; i++){
      dcopy_(&NN, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(&NN, &cinv, xtmp, &inc);
      daxpy_(&NN, &alpha, zvec, &inc, xtmp, &inc);
      soft_th_box(xtmp, NN, lambda_l1/c, xnew, box_flag, cl_box);

      Fval = calc_F_part_nufft(yAx, M, NX, NY, u_dx, v_dy, wvis, weight, xnew, tmp_c_vec);

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
  free(dtmp);

  free(yAx);
  free(tmp_c_vec);
  free(wvis);
  free(weight);

  return(iter+1);
}

/* results */

void calc_result_nufft(int M, int NX, int NY, double *u_dx, double *v_dy,
		       double complex *vis, double *vis_std,
		       double lambda_l1, double lambda_tv, double lambda_tsv, 
		       double *xvec,
		       struct RESULT *mfista_result)
{
  int i, NN = NX*NY;
  double tmp;
  double complex *yAx, *tmp_c_vec, *wvis, *weight;

  /* complex malloc */

  yAx       = alloc_complex_vector(M);
  wvis      = alloc_complex_vector(M);
  weight    = alloc_complex_vector(M);
  tmp_c_vec = alloc_complex_vector(NN);

  for(i=0;i<M;++i){
    wvis[i] = (double complex) vis[i]/vis_std[i];
    weight[i] = (double complex) 1/vis_std[i];
  }

  /* computing results */
  
  tmp = calc_F_part_nufft(yAx, M, NX, NY, u_dx, v_dy, wvis, weight, xvec, tmp_c_vec);

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
  
  free(yAx);
  free(wvis);
  free(weight);
  free(tmp_c_vec);
}

/* main subroutine */

void mfista_imaging_core_nufft(double *u_dx, double *v_dy, 
			       double *vis_r, double *vis_i, double *vis_std,
			       int M, int NX, int NY, int maxiter, double eps,
			       double lambda_l1, double lambda_tv, double lambda_tsv,
			       double cinit, double *xinit, double *xout,
			       int nonneg_flag, int box_flag, float *cl_box,
			       struct RESULT *mfista_result)
{
  int i, iter = 0;
  double epsilon, s_t, e_t, c = cinit;
  struct timespec time_spec1, time_spec2;
  double complex *vis;

  vis = alloc_complex_vector(M);

  for(epsilon=0, i=0;i<M;++i){
    epsilon += vis_r[i]*vis_r[i] + vis_i[i]*vis_i[i];
    vis[i] = vis_r[i] + vis_i[i]*I;
  }
    
  epsilon *= eps/((double)M);

  get_current_time(&time_spec1);

  if( lambda_tv == 0 ){
    iter = mfista_L1_TSV_core_nufft(xout, M, NX, NY, u_dx, v_dy, maxiter, epsilon, vis, vis_std,
				    lambda_l1, lambda_tsv, &c, xinit, nonneg_flag, box_flag, cl_box);
  }
  else if( lambda_tv != 0  && lambda_tsv == 0 ){
    iter = mfista_L1_TV_core_nufft(xout, M, NX, NY, u_dx, v_dy, maxiter, epsilon, vis, vis_std,
				   lambda_l1, lambda_tv, &c, xinit, nonneg_flag, box_flag, cl_box);
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

  calc_result_nufft(M, NX, NY, u_dx, v_dy, vis, vis_std, lambda_l1, lambda_tv, lambda_tsv, xout, mfista_result);

  free(vis);
}

