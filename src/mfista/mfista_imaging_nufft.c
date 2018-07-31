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

void usage(char *s)
{
  printf("%s <nufft_data fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double c> <X outfile> {X initfile} {-nonneg} {-cl_box box_fname} {-maxiter N} {-eps epsilon} {-log log_fname}\n\n",s);
  printf("  <nufft_data fname>: file name of nufft_file.\n");
  printf("  <double lambda_l1>: value of lambda_l1. Positive.\n");
  printf("  <double lambda_tv>: value of lambda_tv. Positive.\n");
  printf("  <double lambda_tsv>: value of lambda_tsv. Positive.\n");
  printf("  <double c>: value of c. Positive.\n");
  printf("  <X outfile>: file name to write X.\n\n");

  printf(" Options.\n\n");
    
  printf("  {X initfile}: file name of X for initialization.\n");
  printf("  {-nonneg}: Use this if x is nonnegative.\n");
  printf("  {-maxiter N}: maximum number of iteration.\n");
  printf("  {-eps epsilon}: epsilon used to check the convergence.\n");
  printf("  {-cl_box box_fname}: file name of CLEAN box data (float).\n");
  printf("  {-log log_fname}: Specify log file.\n\n");

  printf(" This program solves the following problem with FFT\n\n");

  printf(" argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1\n\n");
  printf("    or\n\n");
  printf(" argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tv TV(x)\n\n");
  printf("    or\n\n");
  printf(" argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tsv TSV(x)\n\n");

  printf(" and write x to <X out file>\n\n");

  printf(" If {-nonneg} option is used, x vector is restricted to be nonnegative.\n\n");

  printf(" c is a parameter used for stepsize. Large c makes the algorithm\n");
  printf(" stable but slow. Around 500000 is fine.\n\n");

  exit(1);
}

int main(int argc, char *argv[]){

  int M, NN, NX, NY, dnum, i, 
    init_flag = 0, box_flag = 0, log_flag = 0, nonneg_flag = 0,
    maxiter = MAXITER;

  char init_fname[1024], box_fname[1024], nufftw_fname[1024], log_fname[1024];
  double *u_dx, *v_dy, *vis_r, *vis_i, *vis_std,
    *xinit, *xvec, cinit, lambda_l1, lambda_tv, lambda_tsv, eps = EPS;

  float *cl_box;
  
  struct IO_FNAMES mfista_io;
  struct RESULT    mfista_result;
  FILE *nufftw_fp, *log_fid;

  /* check the number of variables first. */

  if (argc<7) usage(argv[0]);
	
  /* read parameters */

  lambda_l1 = atof(argv[2]);
  printf("lambda_l1 = %g\n",lambda_l1);

  lambda_tv = atof(argv[3]);
  printf("lambda_tv = %g\n",lambda_tv);

  lambda_tsv = atof(argv[4]);
  printf("lambda_tsv = %g\n",lambda_tsv);

  cinit = atof(argv[5]);
  printf("c = %g\n",cinit);

  if (nonneg_flag == 1)
    printf("x is nonnegative.\n");

  if (log_flag ==1)
    printf("Log will be saved to %s.\n",log_fname);

  printf("\n");

  /* options */

  for(i=7; i<argc ; i++){
    if(strcmp(argv[i],"-log") == 0){
      log_flag = 1;

      ++i;
      strcpy(log_fname,argv[i]);
    }
    else if(strcmp(argv[i],"-maxiter") == 0){
      ++i;
      maxiter = atoi(argv[i]);
    }
    else if(strcmp(argv[i],"-eps") == 0){
      ++i;
      eps = atof(argv[i]);
    }
    else if(strcmp(argv[i],"-nonneg") == 0){
      nonneg_flag = 1;
    }
    else if(strcmp(argv[i],"-cl_box") == 0){
      box_flag = 1;

      ++i;
      strcpy(box_fname,argv[i]);
    }
    else{
      init_flag = 1;
      strcpy(init_fname,argv[i]);
    }
  }

  /* read fftw_data */

  strcpy(nufftw_fname,argv[1]);
  
  nufftw_fp = fopenr(nufftw_fname);

  if (fscanf(nufftw_fp, "M  = %d\n", &M)  !=1){exit(0);}
  if (fscanf(nufftw_fp, "NX = %d\n", &NX)  !=1){exit(0);}
  if (fscanf(nufftw_fp, "NY = %d\n", &NY)  !=1){exit(0);}
  if (fscanf(nufftw_fp, "\n") !=0)            {exit(0);}
  if (fscanf(nufftw_fp, "u_rad, v_rad, vis_r, vis_i, noise_std_dev\n") !=0) {exit(0);}
  if (fscanf(nufftw_fp, "\n") !=0)            {exit(0);}

  printf("number of u-v points:  %d\n",M);
  printf("X-dim of image:        %d\n",NX);
  printf("Y-dim of image:        %d\n",NY);

  u_dx = alloc_vector(M);
  v_dy = alloc_vector(M);
  vis_r    = alloc_vector(M);
  vis_i    = alloc_vector(M);
  vis_std  = alloc_vector(M);
  
  for(i = 0;i<M;++i){
    if(fscanf(nufftw_fp, "%le, %le, %le, %le, %le\n",
	      u_dx+i,v_dy+i,vis_r+i,vis_i+i,vis_std+i)!=5){
      printf("cannot read data.\n");
      exit(0);
    }
  }

  fclose(nufftw_fp);

  /* initialize xvec */

  NN = NX*NY;

  xinit = alloc_vector(NN);
  xvec  = alloc_vector(NN);

  if (init_flag ==1){ 

    printf("Initializing x with %s.\n",init_fname);
    dnum = read_V_vector(init_fname, NN, xinit);

    if(dnum != NN)
      printf("Number of read data is shorter than expected.\n");
  }
  else{
    clear_matrix(xinit, NN, 1);
  }

  cl_box = alloc_f_vector(NN);
  
  if (box_flag ==1){ 

    printf("Restrict x with CLEAN box defined in %s.\n", box_fname);
    dnum = read_f_vector(box_fname, NN, cl_box);

    if(dnum != NN)
      printf("Number of read data is shorter than expected.\n");
  }

  mfista_imaging_core_nufft(u_dx, v_dy, vis_r, vis_i, vis_std,
			    M, NX, NY, maxiter, eps, lambda_l1, lambda_tv, lambda_tsv, cinit,
			    xinit, xvec, nonneg_flag, box_flag, cl_box, &mfista_result);

  write_X_vector(argv[6], NN, xvec);

  mfista_io.fft       = 1;
  mfista_io.fft_fname = argv[1];
  mfista_io.v_fname   = NULL;
  mfista_io.A_fname   = NULL;

  if(init_flag == 1)
    mfista_io.in_fname = init_fname;
  else
    mfista_io.in_fname = NULL;

  mfista_io.out_fname = argv[6];
  show_io_fnames(stdout, argv[0], &mfista_io);
  show_result(stdout, argv[0], &mfista_result);

  if(log_flag == 1){
    log_fid = fopenw(log_fname);
    show_io_fnames(log_fid, argv[0], &mfista_io);
    show_result(log_fid, argv[0], &mfista_result);
    fclose(log_fid);
  }

  /* clear memory */

  free(u_dx);
  free(v_dy);
  free(vis_r);
  free(vis_i);

  free(vis_std);
  free(xinit);
  free(xvec);
  free(cl_box);
 
  return 0;
}
