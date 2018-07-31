Usage:

when A is a real matrix (M x N), and Y is a vector (M), we provide the
following program.

./mfista_imaging_dft <int M> <int N> <Y_fname> <A_fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-rec Nx} {-nonneg} {-maxiter ITER} {-eps EPS} {-looe} {-cl_box box_filename} {-log logfilename}
 
if the data points are on grids, fft is fater and require less
memory. the following program is implemented with fft.

./mfista_imaging_fft <fft_fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-maxiter ITER} {-eps EPS} {-cl_box box_filename} {-log logfilename}

for <fft_fname> find fft_data.txt and see the data.

Options:

1. {x_init_filename}: This option specifies a initial values for x. If
   not specified, x is set to 0.

2. {-maxiter ITER}: Maximum number of iteration. Default is 50000.

3. {-eps EPS}: EPS is used to determine the convergence. 

4. {-nonneg}: If x is nonnegative, use this option.

5. {-rec Nx}: (only for dft) For mfista_TV, mfista_L1_TV*, and
   mfista_L1_sqTV*. This option should be used when image is not
   square. In the above case, the image size is

   Nx * Ny = 10000

   where Ny is automatically computed from 10000/Nx.

   If you do not set Nx, then Nx = sqrt(N).

6. {-cl_box box_fname}: Use "box_fname file" (float) for clean
   box. This indicates the active pixels.

7. {-looe}: For mfista_imaging_dft, if this is set, Obuchi &
   Kabashima's approximation of Leave One Out Error will be computed.

8. {-log logfilename}: Summary of the run is shown on the screen by
   default. If this option is specified, it will be also saved in the
   file.

Note: For "mfista_imaging_dft," you cannot make both of lambda_tv
and lambda_tsv positive. And when you make lambda_tv positive, you
cannot use {-looe}. For further instruction type "mfista_* --help".

