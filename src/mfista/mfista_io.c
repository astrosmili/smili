#include "mfista.h"

/* file i_o*/

FILE* fopenr(char* fn){
  FILE *fp;

  fp = fopen(fn,"r");
  if (fp==NULL){
    fprintf(stderr," --- Can't fopen(r) %s\n",fn);
    exit(1);
  }
  return(fp);
}

FILE* fopenw(char* fn){
  FILE *fp;

  fp = fopen(fn,"w");
  if (fp==NULL){
    fprintf(stderr," --- Can't fopen(r) %s\n",fn);
    exit(1);
  }
  return(fp);
}

int read_int_vector(char *fname, int length, int *vector)
{
  FILE *fp;
  int n;

  fp = fopenr(fname);
  n  = fread(vector, sizeof(int), length, fp);
  fclose(fp);

  return(n);
}

int read_f_vector(char *fname, int length, float *vector)
{
  FILE *fp;
  int n;

  fp = fopenr(fname);
  n  = fread(vector, sizeof(float), length, fp);
  fclose(fp);

  return(n);
}

int read_V_vector(char *fname, int length, double *vector)
{
  FILE *fp;
  int n;

  fp = fopenr(fname);
  n  = fread(vector, sizeof(double), length, fp);
  fclose(fp);

  return(n);
}

unsigned long read_A_matrix(char *fname, int height, int width, double *matrix)
{
  FILE *fp;
  int i;
  double tmplength;
  unsigned long n, tmp;

  tmplength = (double)ULONG_MAX;
  tmplength /= (double)height;
  tmplength /= (double)width;

  if(tmplength > 1){

    n = height;
    n *= width;
    printf("Reading %d x %d = %ld entries of matrix A.\n",height, width, n);

    fp = fopenr(fname);  
    tmp = fread(matrix, sizeof(double), n, fp);
    fclose(fp);
  }
  else{
    n = 0;
    fp = fopenr(fname);  
    for(i=0;i < height; i++){
      tmp = fread(matrix + n, sizeof(double), width, fp);
      n += tmp;
      if(i%1000==0)
  printf("reading %d line of A.\n",i);
    }
    fclose(fp);
  }
      
  return(n);
}

int write_X_vector(char *fname, int length, double *vector)
{
  FILE *fp;
  int n;

  fp = fopenw(fname);
  n  = fwrite(vector, sizeof(double), length, fp);
  fclose(fp);

  return(n);
}

/* matrix operation*/

void transpose_matrix(double *matrix, int origheight, int origwidth)
/* put transpose of matrix to original matrix space */
{
  int i, j;
  double *tmpmat;

  tmpmat = alloc_matrix(origwidth, origheight);

  for(i=0; i< origheight;i++){
    for(j=0; j< origwidth;j++)
      tmpmat[j*origheight+i] = matrix[i*origwidth+j];
  }

  for(i=0; i< origwidth*origheight;i++)
    matrix[i] = tmpmat[i];

  free(tmpmat);
}

/* display results */

void show_io_fnames(FILE *fid, char *fname, struct IO_FNAMES *mfista_io)
{
  fprintf(fid,"\n\n");

  fprintf(fid,"IO files of %s.\n",fname);

  fprintf(fid,"\n\n");
  
  if ( mfista_io->fft == 0){
    fprintf(fid," input vector file:      %s\n", mfista_io->v_fname);
    fprintf(fid," input matrix file:      %s\n", mfista_io->A_fname);
  }
  else 
    fprintf(fid," FFTW file:              %s\n", mfista_io->fft_fname);  

  if(mfista_io->in_fname != NULL)
    fprintf(fid," x was initialized with: %s\n", mfista_io->in_fname);

  if(mfista_io->out_fname != NULL)
    fprintf(fid," x is saved to:          %s\n", mfista_io->out_fname);

  fprintf(fid,"\n");
}

void show_result(FILE *fid, char *fname, struct RESULT *mfista_result)
{
  fprintf(fid,"\n\n");

  fprintf(fid,"Output of %s.\n",fname);

  fprintf(fid,"\n\n");
  
  fprintf(fid," Size of the problem:\n\n");
  fprintf(fid," size of input vector:   %d\n",mfista_result->M);
  fprintf(fid," size of output vector:  %d\n",mfista_result->N);
  if(mfista_result->NX!=0)
    fprintf(fid," size of image:          %d x %d\n",
	    mfista_result->NX,mfista_result->NY);
  
  fprintf(fid,"\n\n");
  fprintf(fid," Problem Setting:\n\n");

  if(mfista_result->nonneg == 1)
    fprintf(fid," x is a nonnegative vector.\n\n");
  else if (mfista_result->nonneg == 0)
    fprintf(fid," x is a real vector (takes 0, positive, and negative value).\n\n");

  if(mfista_result->lambda_l1 != 0)
    fprintf(fid," Lambda_1:               %e\n", mfista_result->lambda_l1);

  if(mfista_result->lambda_tsv != 0)
    fprintf(fid," Lambda_TSV:             %e\n", mfista_result->lambda_tsv);

  if(mfista_result->lambda_tv != 0)
    fprintf(fid," Lambda_TV:              %e\n", mfista_result->lambda_tv);

  fprintf(fid," MAXITER:                %d\n", mfista_result->maxiter);

  fprintf(fid,"\n\n");
  fprintf(fid," Results:\n\n");

  fprintf(fid," # of iterations:        %d\n", mfista_result->ITER);
  fprintf(fid," cost:                   %e\n", mfista_result->finalcost);
  fprintf(fid," computaion time[sec]:   %e\n", mfista_result->comp_time);
  fprintf(fid," Est. Lipschitzs const:  %e\n\n", mfista_result->Lip_const);

  fprintf(fid," # of nonzero pixels:    %d\n", mfista_result->N_active);
  fprintf(fid," Squared Error (SE):     %e\n", mfista_result->sq_error);
  fprintf(fid," Mean SE:                %e\n", mfista_result->mean_sq_error);

  if(mfista_result->lambda_l1 != 0)
    fprintf(fid," L1 cost:                %e\n", mfista_result->l1cost);

  if(mfista_result->lambda_tsv != 0)
    fprintf(fid," TSV cost:               %e\n", mfista_result->tsvcost);

  if(mfista_result->lambda_tv != 0)
    fprintf(fid," TV cost:                %e\n", mfista_result->tvcost);

  fprintf(fid,"\n");
  
  if(mfista_result->Hessian_positive ==1){
    fprintf(fid," LOOE:(mean)             %e\n", mfista_result->looe_m);
    fprintf(fid," LOOE:(std)              %e\n", mfista_result->looe_std);
  }
  else if (mfista_result->Hessian_positive ==0)
    fprintf(fid," LOOE:    Could not be computed because Hessian was not positive definite.\n");
  else if (mfista_result->Hessian_positive == -1)
    fprintf(fid," LOOE:    Did not compute LOOE.\n");

  fprintf(fid,"\n");

}
