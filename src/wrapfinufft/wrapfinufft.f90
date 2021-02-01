module wrapfinufft
  !$use omp_lib
  use finufft_fh, only: nufft_opts
  implicit none

  ! Parameters related to NuFFT
  !   FINUFFT's numerical accracy is around 1d-13
  real(kind(1.0d0)), parameter :: ffteps=1d-6
  complex(kind((1.0d0,1.0d0))), parameter :: i_dpc=dcmplx(0d0,1d0)

  interface
    subroutine finufft2d1(nj,xj,yj,cj,iflag,eps,ms,mt,fk,opts,ier)
      use finufft_fh, only: nufft_opts
      implicit none
      type(nufft_opts) opts
      integer :: nj, iflag, ms, mt, ier
      real(kind(1.0d0)) :: xj(nj), yj(nj), eps
      complex(kind((1.0d0,1.0d0))) :: cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
    end subroutine
  end interface

  interface
    subroutine finufft2d2(nj,xj,yj,cj,iflag,eps,ms,mt,fk,opts,ier)
      use finufft_fh, only: nufft_opts
      implicit none
      type(nufft_opts) opts
      integer :: nj, iflag, ms, mt, ier
      real(kind(1.0d0)) :: xj(nj), yj(nj), eps
      complex(kind((1.0d0,1.0d0))) :: cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
    end subroutine
  end interface

  interface
    subroutine finufft_default_opts(opts)
      use finufft_fh, only: nufft_opts
      implicit none
      type(nufft_opts) opts
    end subroutine
  end interface
contains
!------------------------------------------------------------------------------
! NuFFT related functions
!------------------------------------------------------------------------------
subroutine init_opts(opts)
  type(nufft_opts), intent(inout):: opts

  call finufft_default_opts(opts)

  ! init FINUFT options
  opts%modeord = 0
  opts%chkbnds = 1

  opts%debug = 2
  opts%spread_debug = 2
  opts%showwarn = 1

  opts%nthreads = 1
  !opts%fftw = FFTW_ESTIMATE;
  opts%spread_sort = 2
  opts%spread_kerevalmeth = 1
  opts%spread_kerpad = 1
  opts%upsampfac = 1.25d0
  opts%spread_thread = 0
  opts%maxbatchsize = 0
  opts%spread_nthr_atomic = -1
  opts%spread_max_sp_size = 0
end subroutine

subroutine print_opts(opts)
  type(nufft_opts), intent(in):: opts

  ! init FINUFT options
  write (*,*) "modeord:", opts%modeord
  write (*,*) "chkbnds:", opts%chkbnds

  write (*,*) "debug:", opts%debug
  write (*,*) "spread_debug:", opts%spread_debug
  write (*,*) "showwarn:", opts%showwarn

  write (*,*) "nthreads:", opts%nthreads
  write (*,*) "fftw:", opts%fftw
  write (*,*) "spread_sort:", opts%spread_sort
  write (*,*) "spread_kerevalmeth:", opts%spread_kerevalmeth
  write (*,*) "spread_kerpad:", opts%spread_kerpad
  write (*,*) "upsampfac:", opts%upsampfac
  write (*,*) "spread_thread:", opts%spread_thread
  write (*,*) "maxbatchsize:", opts%maxbatchsize
  write (*,*) "spread_nthr_atomic:", opts%spread_nthr_atomic
  write (*,*) "spread_max_sp_size:", opts%spread_max_sp_size
end subroutine

subroutine FINUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  !
  !  Forward Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in)  :: Nx, Ny, Nuv
  real(kind(1.0d0)), intent(in)  :: u(Nuv),v(Nuv)  ! uv coordinates
                                          ! multiplied by 2*pi*dx, 2*pi*dy
  real(kind(1.0d0)), intent(in)  :: I2d(Nx,Ny)     ! Two Dimensional Image
  complex(kind((1.0d0,1.0d0))), intent(out) :: Vcmp(Nuv)  ! Complex Visibility

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the forward Fourier Transformation
  !     0: positive (the standard in Radio Astronomy)
  !     1: negative (the textbook standard; e.g. TMS)
  integer,  parameter :: iflag=0
  !   numerical Accuracy required for FINUFFT
  real(kind(1.0d0)),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  !include 'finufft.fh'
  type(nufft_opts) opts

  ! initialize finufft options
  call init_opts(opts)
  call print_opts(opts)

  ! Call FINUFFT subroutine
  call finufft2d2(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,dcmplx(I2d),opts,ier)

  ! debug
  !print *, ' ier = ',ier
end subroutine

subroutine FINUFFT_fwd_real(u,v,I2d,Vreal,Vimag,Nx,Ny,Nuv)
  !
  !  Forward Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in)  :: Nx, Ny, Nuv
  real(kind(1.0d0)), intent(in)  :: u(Nuv),v(Nuv)  ! uv coordinates
                                          ! multiplied by 2*pi*dx, 2*pi*dy
  real(kind(1.0d0)), intent(in)  :: I2d(Nx,Ny)     ! Two Dimensional Image
  real(kind(1.0d0)), intent(out) :: Vreal(Nuv), Vimag(Nuv) ! Complex Visibility

  complex(kind((1.0d0,1.0d0))) :: Vcmp(Nuv)

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the forward Fourier Transformation
  !     0: positive (the standard in Radio Astronomy)
  !     1: negative (the textbook standard; e.g. TMS)
  integer,  parameter  :: iflag=0
  !   numerical Accuracy required for FINUFFT
  real(kind(1.0d0)),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  !include 'finufft.fh'
  type(nufft_opts) opts

  ! initialize finufft options
  call init_opts(opts)

  ! Call FINUFFT subroutine
  call finufft2d2(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,dcmplx(I2d),opts,ier)

  ! Take real & imaginary parts
  Vreal = dreal(Vcmp)
  Vimag = dimag(Vcmp)

  ! debug
  !print *, ' ier = ',ier
end subroutine

subroutine FINUFFT_adj(u,v,Vcmp,I2d,Nx,Ny,Nuv)
  !
  !  Adjoint Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in) :: Nx, Ny, Nuv
  real(kind(1.0d0)), intent(in) :: u(Nuv),v(Nuv)  ! uv coordinates
                                        ! multiplied by 2*pi*dx, 2*pi*dy
  complex(kind((1.0d0,1.0d0))), intent(in) :: Vcmp(Nuv)  ! Complex Visibility
  complex(kind((1.0d0,1.0d0))), intent(out):: I2d(Nx,Ny) ! Two Dimensional Image

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the adjoint Fourier Transformation
  !     0: positive (the textbook standard TMS)
  !     1: negative (the standard in Radio Astronomy)
  integer, parameter:: iflag=1
  !   numerical Accuracy required for FINUFFT
  real(kind(1.0d0)),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  !include 'finufft.fh'
  type(nufft_opts) opts

  ! initialize finufft options
  call init_opts(opts)
  
  ! Call FINUFFT subroutine
  call finufft2d1(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,I2d,opts,ier)

  ! debug
  !print *, ' ier = ',ier
end subroutine

subroutine FINUFFT_adj_real1D(u,v,Vreal,Vimag,I2d,Nx,Ny,Nuv)
  !
  !  Adjoint Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in) :: Nx, Ny, Nuv
  real(kind(1.0d0)), intent(in) :: u(Nuv),v(Nuv)  ! uv coordinates
                                        ! multiplied by 2*pi*dx, 2*pi*dy
  real(kind(1.0d0)), intent(in) :: Vreal(Nuv),Vimag(Nuv)  ! Complex Visibility
  real(kind(1.0d0)), intent(out):: I2d(Nx*Ny) ! Two Dimensional Image
  complex(kind((1.0d0,1.0d0))):: I2d_cmp(Nx,Ny) ! Two Dimensional Image

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the adjoint Fourier Transformation
  !     0: positive (the textbook standard TMS)
  !     1: negative (the standard in Radio Astronomy)
  integer, parameter:: iflag=1
  !   numerical Accuracy required for FINUFFT
  real(kind(1.0d0)),  parameter :: eps=ffteps
  !   error log
  integer :: ier
  ! Call FINUFFT subroutine

  !include 'finufft.fh'
  type(nufft_opts) opts

  ! initialize finufft options
  call init_opts(opts)
  
  call finufft2d1(Nuv,u,v,dcmplx(Vreal,Vimag),iflag,eps,Nx,Ny,I2d_cmp,opts,ier)
  I2d = reshape(realpart(I2d_cmp), (/Nx*Ny/))

  ! debug
  !print *, ' ier = ',ier
end subroutine

subroutine FINUFFT_adj_real(u,v,Vreal,Vimag,Nx,Ny,Ireal,Iimag,Nuv)
  !
  !  Adjoint Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in) :: Nx, Ny, Nuv
  real(kind(1.0d0)), intent(in) :: u(Nuv),v(Nuv)  ! uv coordinates
                                        ! multiplied by 2*pi*dx, 2*pi*dy
  real(kind(1.0d0)), intent(in) :: Vreal(Nuv)  ! Complex Visibility
  real(kind(1.0d0)), intent(in) :: Vimag(Nuv)  ! Complex Visibility
  real(kind(1.0d0)), intent(out):: Ireal(Nx*Ny) ! Two Dimensional Image
  real(kind(1.0d0)), intent(out):: Iimag(Nx*Ny) ! Two Dimensional Image

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the adjoint Fourier Transformation
  !     0: positive (the textbook standard TMS)
  !     1: negative (the standard in Radio Astronomy)
  integer, parameter:: iflag=1
  !   numerical Accuracy required for FINUFFT
  real(kind(1.0d0)),  parameter :: eps=ffteps
  complex(kind((1.0d0,1.0d0))) :: I2d(Nx,Ny) ! Two Dimensional Image

  !   error log
  integer :: ier

  !include 'finufft.fh'
  type(nufft_opts) opts

  ! initialize finufft options
  call init_opts(opts)
  
  ! Call FINUFFT subroutine
  call finufft2d1(Nuv,u,v,Vreal+i_dpc*Vimag,iflag,eps,Nx,Ny,I2d,opts,ier)
  Ireal = reshape(realpart(I2d), (/Nx*Ny/))
  Iimag = reshape(imagpart(I2d), (/Nx*Ny/))
end subroutine
end module
