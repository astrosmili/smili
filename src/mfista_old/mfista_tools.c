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

#ifdef __APPLE__
#include <sys/time.h>
#endif

/* memory allocation of matrix and vectors */

int *alloc_int_vector(int length)
{
    return malloc(sizeof(int)*length);
}

float *alloc_f_vector(int length)
{
    return malloc(sizeof(float)*length);
}

double *alloc_vector(int length)
{
    return malloc(sizeof(double)*length);
}

double *alloc_matrix(int height, int width)
{
    return malloc(sizeof(double) * height * width);
}

void clear_matrix(double *matrix, int height, int width)
{
    memset(matrix, 0, sizeof(double) * height * width);
}

/* subroutines for mfista*/

double calc_Q_part(int *N, double *xvec1, double *xvec2,
		   double c, double *AyAz, double *buffxvec1)
{
  int inc = 1;
  double term2, term3, alpha = -1;

  /* x1 - x2 */
  dcopy_(N, xvec1, &inc, buffxvec1, &inc);
  daxpy_(N, &alpha, xvec2, &inc, buffxvec1, &inc);

  /* (x1 - x2)'A'(y - A x2) */
  term2 = ddot_(N, AyAz, &inc, buffxvec1, &inc);
  /* (x1 - x2)'(x1 - x2) */
  term3 = ddot_(N, buffxvec1, &inc, buffxvec1, &inc);

  return(-term2+c*term3/2);
}

/* soft thresholding */

void soft_threshold(double *vector, int length, double eta, 
		    double *newvec)
{
    int i;
    for (i = 0; i < length; i++){
      if(vector[i] >= eta)
	newvec[i] = vector[i] - eta;
      else if(vector[i] <=- eta)
	newvec[i] = vector[i] + eta;
      else
	newvec[i] = 0;
    }
}

void soft_threshold_box(double *vector, int length, double eta, 
			double *newvec, int box_flag, float *cl_box)
{
    int i;

    if(box_flag == 0)
      soft_threshold(vector, length, eta, newvec);
    else{
      for(i = 0; i < length; i++){
	if( (cl_box[i] == 0) || (vector[i] < eta && vector[i] > -eta))
	  newvec[i] = 0;
	else if(vector[i] >= eta)
	  newvec[i] = vector[i] - eta;
	else
	  newvec[i] = vector[i] + eta;
      }
    }
}

void soft_threshold_nonneg(double *vector, int length, double eta, 
			   double *newvec)
{
    int i;
    for (i = 0; i < length; i++){
      if(vector[i] < eta)
	newvec[i] = 0;
      else
	newvec[i] = vector[i] - eta;
    }
}

void soft_threshold_nonneg_box(double *vector, int length, double eta, 
			       double *newvec, int box_flag, float *cl_box)
{
    int i;

    if(box_flag == 0)
      soft_threshold_nonneg(vector, length, eta, newvec);
    else{
      for(i = 0; i < length; i++){
	if((cl_box[i] == 0) || (vector[i] < eta))
	  newvec[i] = 0;
	else
	  newvec[i] = vector[i] - eta;
      }
    }
}

/* TSV */

double TSV(int NX, int NY, double *xvec)
{
  int i, j;
  double tsv = 0;

  for(i = 0; i < NX-1; ++i)
    for(j = 0; j < NY-1; ++j){
      tsv += pow((xvec[NX*j+i]-xvec[NX*j+i+1]),2.0);
      tsv += pow((xvec[NX*j+i]-xvec[NX*(j+1)+i]),2.0);
    }

  for(i = 0; i < NX-1; ++i)
    tsv += pow((xvec[NX*(NY-1)+i]-xvec[NX*(NY-1)+i+1]),2.0);

  for(j = 0; j < NY-1; ++j)
    tsv += pow((xvec[NX*j+(NX-1)]-xvec[NX*(j+1)+(NX-1)]),2.0);

  return(tsv);
}

void d_TSV(int NX, int NY, double *xvec, double *dvec)
{
  int i, j;

  for(j = 0; j < NY; j++) dvec[NX*j+NX-1] = 0;

  for(i = 0; i < NX-1; i++)
    for(j = 0; j < NY; j++)
      dvec[NX*j+i] = 2*(xvec[NX*j+i]-xvec[NX*j+i+1]);

  for(i = 1; i < NX; i++)
    for(j = 0; j < NY; j++)
      dvec[NX*j+i] += 2*(xvec[NX*j+i]-xvec[NX*j+i-1]);

  for(i = 0; i < NX; i++)
    for(j = 0; j < NY-1; j++)
      dvec[NX*j+i] += 2*(xvec[NX*j+i]-xvec[NX*(j+1)+i]);

  for(i = 0; i < NX; i++)
    for(j = 1; j < NY; j++)
      dvec[NX*j+i] += 2*(xvec[NX*j+i]-xvec[NX*(j-1)+i]);
}

/* utility for time measurement */

void get_current_time(struct timespec *t) {
#ifdef __APPLE__
  struct timeval tv;
  struct timezone tz;
  int status = gettimeofday(&tv, &tz);
  if (status == 0) {
    t->tv_sec = tv.tv_sec;
    t->tv_nsec = tv.tv_usec * 1000; /* microsec -> nanosec */
  } else {
    t->tv_sec = 0.0;
    t->tv_nsec = 0.0;
  }
#else
  clock_gettime(CLOCK_MONOTONIC, t);
#endif
}
