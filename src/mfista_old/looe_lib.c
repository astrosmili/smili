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

/* index transform */

int i2r(int i, int NX)
{
  return(i%NX);
}

int i2c(int i, int NX)
{
  double tmp1, tmp2;

  tmp1 = (double)i;
  tmp2 = (double)NX;
  
  return((int)(ceil(((tmp1+1)/tmp2))-1));
}

int rc2i(int r, int c, int NX)
{
  return( r + NX*c);
}


int find_active_set(int N, double *xvec, int *indx_list)
{
  int i, N_active;

  N_active = 0;
  
  for(i = 0;i < N;i++){
    if(fabs(xvec[i]) > 0){
	indx_list[N_active] = i;
	N_active++;
      }
  }

  return(N_active);
}

/* Some routines for computing LOOE */

double *shrink_A(int M, int N, int N_active, int *indx_list,
	      double *Amat)
{
  int i, j, k;
  double *Amat_s;

  Amat_s = alloc_matrix(M, N_active);

  for(j = 0; j < N_active; j++){
    k = indx_list[j];
    for(i = 0; i < M; i++) Amat_s[j*M+i] = Amat[k*M+i];
  }

  return(Amat_s);
}

int solve_lin_looe(int *NA, int *NB, double *Hessian, double *B)
/* Solve A*X = B
   A is a symmetric real matrix with *NA x *NA.
   B is a real matrix with *NA x *NB.
   UpLo is "U" or "L." It tells which part of A is used.
   The function is void and the result is stored in B. */
{
  int  info, lda, ldb;
  char UpLo[2] = {'L','\0'};;
    
    lda  = *NA;
    ldb  = lda;
    
    info = 0;

    printf("Solving a linear equation.\n");

    dposv_(UpLo, NA, NB, Hessian, &lda, B, &ldb, &info );

    printf("solved.\n");

    if (info < 0)      printf("DPOSV: The matrix had an illegal value.\n");
    else if (info > 0) printf("DPOSV: The Hessian matrix is not positive definite.\n");

    return(info);
}

double compute_LOOE_core(int *M, int N_active, 
			 double *yvec, double *Amat, double *xvec,
			 double *yAx,  double *Amat_s, double *Hessian,
			 double *looe_m, double *looe_std)
{
  int i, j, m, n_s;
  double LOOE_m, LOOE_std, *At, *dvec, tmp, tmp_s, info;

  m   = *M;          /* size of Hessian */
  n_s = N_active;  /* number of columns of Amat_s */

  At  = alloc_matrix(n_s,m);

  dvec = alloc_vector(m);

  for(i = 0;i < m; i++)
    for(j = 0;j < n_s; j++)
      At[n_s*i + j] = Amat_s[m*j+i];

  info = solve_lin_looe(&n_s, &m, Hessian, At);

  LOOE_m   = 0;
  LOOE_std = 0;
  
  if(info == 0){

    for(i = 0;i < m;i++){
      dvec[i]=0;
      for(j = 0;j < n_s;j++) dvec[i]+= Amat_s[m*j+i]*At[n_s*i+j];
    }
    
    for(i=0;i<m;++i){
      tmp = yAx[i]/(1-dvec[i]);
      tmp_s = tmp*tmp/2;
      LOOE_m += tmp_s;
      LOOE_std += tmp_s*tmp_s;
    }

    LOOE_m   /= (double)m;

    LOOE_std /= ((double)m-1.0);
    LOOE_std -= LOOE_m*LOOE_m*((double)m)/((double)m-1.0);

    *looe_m   = LOOE_m;
    *looe_std = sqrt(LOOE_std);

    free(dvec);
    free(At);

    return(LOOE_m);
  }
  else{

    *looe_m   = LOOE_m;
    *looe_std = LOOE_std;

    return(-1.0);
  }
}

/* For L1 */

double *compute_Hessian_L1(int *M,  int *indx_list,
			   double *Amat_s, int N_active)
{
  int i,j;
  double *Hessian, alpha = 1, beta = 0;

  printf("The size of Hessian is %d x %d. ",N_active,N_active);

  Hessian = alloc_matrix(N_active,N_active);

  for(i=0;i<N_active;i++)
    for(j=0;j<N_active;j++)
      Hessian[i*N_active+j]=0;

  dsyrk_("L", "T", &N_active, M, &alpha, Amat_s, M,
	 &beta, Hessian, &N_active);

  printf("Done.\n");
  return(Hessian);
}

double compute_LOOE_L1(int *M, int *N, double lambda1, 
		       double *yvec, double *Amat, double *xvec, double *yAx,
		       double *looe_m, double *looe_std)
{
  double *Amat_s, *Hessian, LOOE;
  int    N_active, *indx_list;

  /* computing LOOE */

  indx_list = (int *)malloc(sizeof(int)*(*N));

  N_active = find_active_set(*N, xvec, indx_list);

  Amat_s = shrink_A(*M, *N, N_active, indx_list, Amat);

  printf("The number of active components is %d\n",N_active);

  printf("Computing Hessian matrix.\n");
  Hessian = compute_Hessian_L1(M, indx_list, Amat_s, N_active);

  printf("\n");
  LOOE = compute_LOOE_core(M, N_active, yvec, Amat, xvec, yAx, Amat_s, Hessian,
			   looe_m, looe_std);

  printf("LOOE = %lg\n",LOOE);

  free(Amat_s);
  free(Hessian);
  free(indx_list);

  return(LOOE);
}

/* for TSV */

double d2_TSV(int i, int NX, int NY)
/* Return 4 if i is on corner. Return 6 if it is on an edge, 
and return 8 if it belongs to the interior. */
{
  int r, c;

  r = i2r(i, NX);
  c = i2c(i, NX);

  if(r > 0 && r < NX-1){
    if(c > 0 && c < NX-1) return(8.0);
    else                  return(6.0);
  }
  else{
    if(c > 0 && c < NX-1) return(6.0);
    else                  return(4.0);
  }
  
}

double *compute_Hessian_L1_TSV(int *M, int NX, int NY,
				double lambda_tsv, int *indx_list,
				double *Amat_s, int N_active)
{
  int i,j;
  double *Hessian, alpha = 1, beta = 0;

  printf("The size of Hessian is %d x %d. ",N_active,N_active);

  Hessian = alloc_matrix(N_active,N_active);

  for(i=0;i<N_active;i++)
    for(j=0;j<N_active;j++)
      Hessian[i*N_active+j]=0;

  dsyrk_("L", "T", &N_active, M, &alpha, Amat_s, M,
	 &beta, Hessian, &N_active);

  for(i=0;i<N_active;i++){
    j = indx_list[i];
    Hessian[i*N_active+i] += lambda_tsv*d2_TSV(j, NX, NY);
  }
  printf("Done.\n");
  return(Hessian);
}

double compute_LOOE_L1_TSV(int *M, int *N, int NX, int NY,
			   double lambda_l1, double lambda_tsv,
			   double *yvec, double *Amat, double *xvec, double *yAx,
			   double *looe_m, double *looe_std)
{
  double *Amat_s, *Hessian, LOOE;
  int    N_active, *indx_list;

  /* computing LOOE */

  indx_list = (int *)malloc(sizeof(int)*(*N));

  N_active = find_active_set(*N, xvec, indx_list);

  Amat_s = shrink_A(*M, *N, N_active, indx_list, Amat);

  printf("The number of active components is %d\n",N_active);

  printf("Computing Hessian matrix.\n");
  Hessian = compute_Hessian_L1_TSV(M, NX, NY,
				    lambda_tsv, indx_list, Amat_s, N_active);

  printf("\n");
  LOOE = compute_LOOE_core(M, N_active, yvec, Amat, xvec, yAx, Amat_s, Hessian,
			   looe_m, looe_std);

  printf("LOOE = %lg\n",LOOE);

  free(Amat_s);
  free(Hessian);
  free(indx_list);

  return(LOOE);
}

