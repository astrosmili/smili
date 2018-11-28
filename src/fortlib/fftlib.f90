module fftlib
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  use image, only: I1d_I2d_fwd, I1d_I2d_inv
  implicit none

  ! Parameters related to NuFFT
  !   FINUFFT's numerical accracy is around 1d-13
  real(dp), parameter :: ffteps=1d-12

  interface
    subroutine finufft2d1_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      integer :: nj, iflag, ms, mt, ier
      real(kind(1.0d0)) :: xj(nj), yj(nj), eps
      complex(kind((1.0d0,1.0d0))) :: cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
    end subroutine
  end interface

  interface
    subroutine finufft2d2_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      integer :: nj, iflag, ms, mt, ier
      real(kind(1.0d0)) :: xj(nj), yj(nj), eps
      complex(kind((1.0d0,1.0d0))) :: cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
    end subroutine
  end interface
contains
!-------------------------------------------------------------------------------
! NuFFT related functions
!-------------------------------------------------------------------------------
subroutine NUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  !
  !  Forward Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in)  :: Nx, Ny, Nuv
  real(dp), intent(in)  :: u(Nuv),v(Nuv)  ! uv coordinates
                                          ! multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in)  :: I2d(Nx,Ny)     ! Two Dimensional Image
  complex(dpc), intent(out) :: Vcmp(Nuv)  ! Complex Visibility

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the forward Fourier Transformation
  !     0: positive (the standard in Radio Astronomy)
  !     1: negative (the textbook standard; e.g. TMS)
  integer,  parameter :: iflag=0
  !   numerical Accuracy required for FINUFFT
  real(dp),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  ! Call FINUFFT subroutine
  call finufft2d2_f(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,dcmplx(I2d),ier)

  ! debug
  !print *, ' ier = ',ier
end subroutine


subroutine NUFFT_fwd_real(u,v,I2d,Vreal,Vimag,Nx,Ny,Nuv)
  !
  !  Forward Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in)  :: Nx, Ny, Nuv
  real(dp), intent(in)  :: u(Nuv),v(Nuv)  ! uv coordinates
                                          ! multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in)  :: I2d(Nx,Ny)     ! Two Dimensional Image
  real(dp), intent(out) :: Vreal(Nuv), Vimag(Nuv) ! Complex Visibility

  complex(dpc) :: Vcmp(Nuv)

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the forward Fourier Transformation
  !     0: positive (the standard in Radio Astronomy)
  !     1: negative (the textbook standard; e.g. TMS)
  integer,  parameter  :: iflag=0
  !   numerical Accuracy required for FINUFFT
  real(dp),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  ! Call FINUFFT subroutine
  call finufft2d2_f(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,dcmplx(I2d),ier)

  ! Take real & imaginary parts
  Vreal = dreal(Vcmp)
  Vimag = dimag(Vcmp)

  ! debug
  !print *, ' ier = ',ier
end subroutine


subroutine NUFFT_adj(u,v,Vcmp,I2d,Nx,Ny,Nuv)
  !
  !  Adjoint Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in) :: Nx, Ny, Nuv
  real(dp), intent(in) :: u(Nuv),v(Nuv)  ! uv coordinates
                                         ! multiplied by 2*pi*dx, 2*pi*dy
  complex(dpc), intent(in) :: Vcmp(Nuv)  ! Complex Visibility
  complex(dpc), intent(out):: I2d(Nx,Ny) ! Two Dimensional Image

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the adjoint Fourier Transformation
  !     0: positive (the textbook standard TMS)
  !     1: negative (the standard in Radio Astronomy)
  integer, parameter:: iflag=1
  !   numerical Accuracy required for FINUFFT
  real(dp),  parameter :: eps=ffteps
  !   error log
  integer :: ier

  ! Call FINUFFT subroutine
  call finufft2d1_f(Nuv,u,v,Vcmp,iflag,eps,Nx,Ny,I2d,ier)

  ! debug
  !print *, ' ier = ',ier
end subroutine

subroutine NUFFT_adj_real1D(u,v,Vreal,Vimag,I2d,Nx,Ny,Nuv)
  !
  !  Adjoint Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in) :: Nx, Ny, Nuv
  real(dp), intent(in) :: u(Nuv),v(Nuv)  ! uv coordinates
                                         ! multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in) :: Vreal(Nuv),Vimag(Nuv)  ! Complex Visibility
  real(dp), intent(out):: I2d(Nx*Ny) ! Two Dimensional Image
  complex(dpc):: I2d_cmp(Nx,Ny) ! Two Dimensional Image

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the adjoint Fourier Transformation
  !     0: positive (the textbook standard TMS)
  !     1: negative (the standard in Radio Astronomy)
  integer, parameter:: iflag=1
  !   numerical Accuracy required for FINUFFT
  real(dp),  parameter :: eps=ffteps
  !   error log
  integer :: ier
  ! Call FINUFFT subroutine

  call finufft2d1_f(Nuv,u,v,dcmplx(Vreal,Vimag),iflag,eps,Nx,Ny,I2d_cmp,ier)
  I2d = reshape(realpart(I2d_cmp), (/Nx*Ny/))

  ! debug
  !print *, ' ier = ',ier
end subroutine


subroutine NUFFT_adj_real(u,v,Vreal,Vimag,Nx,Ny,Ireal,Iimag,Nuv)
  !
  !  Adjoint Non-uniform Fast Fourier Transform
  !    This funcion using the FINUFFT library.
  !
  implicit none

  integer,  intent(in) :: Nx, Ny, Nuv
  real(dp), intent(in) :: u(Nuv),v(Nuv)  ! uv coordinates
                                         ! multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in) :: Vreal(Nuv)  ! Complex Visibility
  real(dp), intent(in) :: Vimag(Nuv)  ! Complex Visibility
  real(dp), intent(out):: Ireal(Nx*Ny) ! Two Dimensional Image
  real(dp), intent(out):: Iimag(Nx*Ny) ! Two Dimensional Image

  ! Some Other Parameters for FINUFFT
  !   Sign of the exponent in the adjoint Fourier Transformation
  !     0: positive (the textbook standard TMS)
  !     1: negative (the standard in Radio Astronomy)
  integer, parameter:: iflag=1
  !   numerical Accuracy required for FINUFFT
  real(dp),  parameter :: eps=ffteps
  complex(dpc) :: I2d(Nx,Ny) ! Two Dimensional Image

  !   error log
  integer :: ier

  ! Call FINUFFT subroutine
  call finufft2d1_f(Nuv,u,v,Vreal+i_dpc*Vimag,iflag,eps,Nx,Ny,I2d,ier)
  Ireal = reshape(realpart(I2d), (/Nx*Ny/))
  Iimag = reshape(imagpart(I2d), (/Nx*Ny/))
end subroutine


subroutine NUFFT_adj_resid(u,v,Vre,Vim,I2d,Nx,Ny,Nuv)
  !
  !  This function takes the adjoint non-uniform Fast Fourier Transform
  !  of input visibilities and then sum the real and imag parts of
  !  the transformed image.
  !
  implicit none

  integer,  intent(in):: Nx, Ny, Nuv
  real(dp), intent(in):: u(Nuv),v(Nuv)      ! uv coordinates
                                            ! multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in):: Vre(Nuv),Vim(Nuv)  ! Complex Visibility
  real(dp), intent(out):: I2d(Nx,Ny)        ! Two Dimensional Image

  complex(dpc):: I2dcmp1(Nx,Ny), I2dcmp2(Nx,Ny)

  ! Call adjoint NuFFT
  call NUFFT_adj(u,v,dcmplx(Vre),I2dcmp1,Nx,Ny,Nuv)
  call NUFFT_adj(u,v,dcmplx(Vim),I2dcmp2,Nx,Ny,Nuv)

  ! Take a sum of real part and imaginary part
  I2d = dreal(I2dcmp1)+dimag(I2dcmp2)
end subroutine


subroutine NUFFT_adjrea(u,v,Vcmp,I2d,Nx,Ny,Nuv)
  !
  !  This function takes the adjoint non-uniform Fast Fourier Transform
  !  of input visibilities and then take the real part of the transformed image.
  !
  implicit none

  integer,  intent(in):: Nx, Ny, Nuv
  real(dp), intent(in):: u(Nuv),v(Nuv)  ! uv coordinates
                                        ! multiplied by 2*pi*dx, 2*pi*dy
  complex(dpc), intent(in):: Vcmp(Nuv)  ! Complex Visibility
  real(dp), intent(out):: I2d(Nx,Ny)    ! Two Dimensional Image

  complex(dpc):: I2dcmp(Nx,Ny)

  ! Call adjoint NuFFT
  call NUFFT_adj(u,v,Vcmp,I2dcmp,Nx,Ny,Nuv)

  ! Take a sum of real part and imaginary part
  I2d = dreal(I2dcmp)
end subroutine


subroutine phashift_c2r(u,v,Nxref,Nyref,Nx,Ny,Vcmp_in,Vcmp_out)
  !
  !  This function shift the tracking center of the input full complex visibilities
  !  from the image center to the reference pixel
  !
  implicit none

  integer,  intent(in):: Nx, Ny
  real(dp), intent(in):: u,v            ! uv coordinates multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in):: Nxref, Nyref   ! x,y reference ppixels (1=the leftmost/lowermost pixel)
  complex(dpc), intent(in)  :: Vcmp_in  ! Complex Visibility
  complex(dpc), intent(out) :: Vcmp_out ! Complex Visibility

  real(dp) :: dix, diy

  ! pixels to be shifted
  dix = Nx/2d0 + 1 - Nxref
  diy = Ny/2d0 + 1 - Nyref

  Vcmp_out = Vcmp_in * exp(i_dpc * (u*dix + v*diy))
end subroutine


subroutine phashift_r2c(u,v,Nxref,Nyref,Nx,Ny,Vcmp_in,Vcmp_out)
  !
  !  This function shift the tracking center of the input full complex visibilities
  !  from the reference pixel to the image center
  !
  implicit none

  integer,  intent(in):: Nx, Ny
  real(dp), intent(in):: u,v            ! uv coordinates multiplied by 2*pi*dx, 2*pi*dy
  real(dp), intent(in):: Nxref, Nyref   ! x,y reference pixels
                                        ! (1=the leftmost/lowermost pixel)
  complex(dpc), intent(in)  :: Vcmp_in  ! Complex Visibility
  complex(dpc), intent(out) :: Vcmp_out ! Complex Visibility

  real(dp) :: dix, diy

  ! pixels to be shifted
  dix = Nxref - Nx/2d0 - 1
  diy = Nyref - Ny/2d0 - 1

  Vcmp_out = Vcmp_in * exp(i_dpc * (u*dix + v*diy))
end subroutine


!-------------------------------------------------------------------------------
! Functions to compute chisquares and also residual vectors
!-------------------------------------------------------------------------------
subroutine calc_chisq(&
  Iin,xidx,yidx,Nx,Ny,&
  u,v,&
  isfcv,uvidxfcv,Vfcv,Varfcv,wfcv,&
  isamp,uvidxamp,Vamp,Varamp,wamp,&
  iscp,uvidxcp,CP,Varcp,wcp,&
  isca,uvidxca,CA,Varca,wca,&
  chisq, gradchisq, chisqfcv, chisqamp, chisqcp, chisqca,&
  Npix,Nuv,Nfcv,Namp,Ncp,Nca&
)
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)

  ! Parameters related to full complex visibilities
  logical,      intent(in) :: isfcv           ! is data?
  integer,      intent(in) :: Nfcv            ! number of data
  integer,      intent(in) :: uvidxfcv(Nfcv)  ! uvidx
  complex(dpc), intent(in) :: Vfcv(Nfcv)      ! data
  real(dp),     intent(in) :: Varfcv(Nfcv)    ! variance
  real(dp),     intent(in) :: wfcv            ! data weights

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp           ! is amplitudes?
  integer,  intent(in) :: Namp            ! Number of data
  integer,  intent(in) :: uvidxamp(Namp)  ! uvidx
  real(dp), intent(in) :: Vamp(Namp)      ! data
  real(dp), intent(in) :: Varamp(Namp)    ! variance
  real(dp), intent(in) :: wamp            ! data weights

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp            ! is closure phases?
  integer,  intent(in) :: Ncp             ! Number of data
  integer,  intent(in) :: uvidxcp(3,Ncp)  ! uvidx
  real(dp), intent(in) :: CP(Ncp)         ! data
  real(dp), intent(in) :: Varcp(Ncp)      ! variance
  real(dp), intent(in) :: wcp             ! data weights

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca            ! is closure amplitudes?
  integer,  intent(in) :: Nca             ! Number of data
  integer,  intent(in) :: uvidxca(4,Nca)  ! uvidx
  real(dp), intent(in) :: CA(Nca)         ! data
  real(dp), intent(in) :: Varca(Nca)      ! variance
  real(dp), intent(in) :: wca             ! data weights

  ! Chi-square and its gradient
  real(dp), intent(out) :: chisq           ! weighted sum of chisquares
  real(dp), intent(out) :: chisqfcv        ! chisquare of full complex visibilities
  real(dp), intent(out) :: chisqamp        ! chisquare of amplitudes
  real(dp), intent(out) :: chisqcp         ! chisquare of closure phases
  real(dp), intent(out) :: chisqca         ! chisquare of closure amplitudes
  real(dp), intent(out) :: gradchisq(Npix) ! costfunction and its gradient

  ! allocatable arrays
  real(dp),     allocatable :: I2d(:,:)
  real(dp),     allocatable :: gradchisq2d(:,:)
  real(dp),     allocatable :: Vresre(:),Vresim(:)
  complex(dpc), allocatable :: Vcmp(:)

  ! Initialize the chisquare and its gradient
  chisq        = 0d0
  gradchisq(:) = 0d0

  ! Copy 1d image to 2d image
  allocate(I2d(Nx,Ny))
  I2d(:,:)=0d0
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)

  ! Forward Non-unifrom Fast Fourier Transform
  allocate(Vcmp(Nuv))
  Vcmp(:) = dcmplx(0d0,0d0)
  call NUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  deallocate(I2d)

  ! allocate residual vectors
  allocate(Vresre(Nuv),Vresim(Nuv))
  Vresre(:) = 0d0
  Vresim(:) = 0d0

  ! Full complex visibility
  chisqfcv=0d0
  if (isfcv .eqv. .True.) then
    call calc_chisq_fcv(Vcmp,uvidxfcv,Vfcv,Varfcv,wfcv,chisqfcv,Vresre,Vresim,Nuv,Nfcv)
    chisq = chisq + chisqfcv
    chisqfcv = chisqfcv / wfcv
  end if

  ! Amplitudes
  chisqamp=0d0
  if (isamp .eqv. .True.) then
    call calc_chisq_amp(Vcmp,uvidxamp,Vamp,Varamp,wamp,chisqamp,Vresre,Vresim,Nuv,Namp)
    chisq = chisq + chisqamp
    chisqamp = chisqamp / wamp
  end if

  ! Log closure amplitudes
  chisqca=0d0
  if (isca .eqv. .True.) then
    call calc_chisq_ca(Vcmp,uvidxca,CA,Varca,wca,chisqca,Vresre,Vresim,Nuv,Nca)
    chisq = chisq + chisqca
    chisqca = chisqca / wca
  end if

  ! Closure phases
  chisqcp=0d0
  if (iscp .eqv. .True.) then
    call calc_chisq_cp(Vcmp,uvidxcp,CP,Varcp,wcp,chisqcp,Vresre,Vresim,Nuv,Ncp)
    chisq = chisq + chisqcp
    chisqcp = chisqcp / wcp
  end if

  ! comupute the total chisquare
  deallocate(Vcmp)

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  call NUFFT_adj_resid(u,v,Vresre,Vresim,gradchisq2d(:,:),Nx,Ny,Nuv)
  deallocate(Vresre,Vresim)

  ! copy the gradient of chisquare into that of cost functions
  call I1d_I2d_inv(xidx,yidx,gradchisq,gradchisq2d,Npix,Nx,Ny)
  deallocate(gradchisq2d)
end subroutine

subroutine calc_chisq_fcv(Vcmp,&
                     uvidxfcv,Vfcv,Varfcv,&
                     fnorm,&
                     chisq,Vresre,Vresim,&
                     Nuv,Nfcv)
  !
  !  This subroutine will compute chisquare and its analytic gradient
  !  for full complex visibility data
  !
  implicit none

  ! NuFFT-ed visibilities
  integer,      intent(in) :: Nuv
  complex(dpc), intent(in) :: Vcmp(Nuv)
  ! Data
  integer,  intent(in)     :: Nfcv             ! Number of data
  integer,  intent(in)     :: uvidxfcv(Nfcv)   ! UV Index of FCV data
  complex(dpc), intent(in) :: Vfcv(Nfcv)       ! Full complex visibility (FCV) data
  real(dp), intent(in)     :: Varfcv(Nfcv)     ! variances of FCV data
  ! Normalization Factor of Chisquare
  real(dp), intent(in)     :: fnorm
  ! Outputs
  real(dp), intent(inout)  :: chisq                    ! chisquare
  real(dp), intent(inout)  :: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  complex(dpc):: resid
  real(dp):: factor
  integer:: uvidx, i

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nfcv,fnorm,uvidxfcv,Vcmp,Vfcv,Varfcv) &
  !$OMP   PRIVATE(i,uvidx,resid,factor),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do i=1, Nfcv
    ! pick up uv index
    uvidx = abs(uvidxfcv(i))

    ! take residual
    if (uvidxfcv(i) > 0) then
      resid = Vfcv(i) - Vcmp(uvidx)
    else
      resid = Vfcv(i) - dconjg(Vcmp(uvidx))
    end if

    ! compute chisquare
    chisq = chisq + abs(resid)**2/Varfcv(i)*fnorm

    ! compute residual vector
    factor = -2/Varfcv(i)*fnorm
    Vresre(uvidx) = Vresre(uvidx) + factor*dreal(resid)
    Vresim(uvidx) = Vresim(uvidx) + factor*dimag(resid)*sign(1,uvidxfcv(i))
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine calc_chisq_amp(Vcmp,&
                     uvidxamp,Vamp,Varamp,&
                     fnorm,&
                     chisq,Vresre,Vresim,&
                     Nuv,Namp)
 !
 !  This subroutine will compute chisquare and its analytic gradient
 !  for amplitude data
 !
  implicit none

  ! NuFFT-ed visibilities
  integer,      intent(in):: Nuv
  complex(dpc), intent(in):: Vcmp(Nuv)
  ! Data
  integer,  intent(in):: Namp           ! Number of data
  integer,  intent(in):: uvidxamp(Namp) ! UV Index of Amp data
  real(dp), intent(in):: Vamp(Namp)     ! Amp data
  real(dp), intent(in):: Varamp(Namp)   ! variances of Amp data
  ! Normalization Factor of Chisquare
  real(dp), intent(in):: fnorm
  ! Outputs
  real(dp), intent(inout):: chisq           ! chisquare
  real(dp), intent(inout):: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  real(dp):: resid, factor, model
  integer:: uvidx, i

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Namp,fnorm,uvidxamp,Vcmp,Vamp,Varamp) &
  !$OMP   PRIVATE(i,uvidx,resid,factor,model),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do i=1, Namp
    ! pick up uv index
    uvidx = abs(uvidxamp(i))

    ! take residual
    model = abs(Vcmp(uvidx))
    resid = Vamp(i) - model

    ! compute chisquare
    chisq = chisq + resid**2/Varamp(i)*fnorm

    ! compute residual vector
    factor = -2*resid/Varamp(i)/model*fnorm
    Vresre(uvidx) = Vresre(uvidx) + factor * dreal(Vcmp(uvidx))
    Vresim(uvidx) = Vresim(uvidx) + factor * dimag(Vcmp(uvidx))
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine calc_chisq_ca(Vcmp,&
                    uvidxca,CA,Varca,&
                    fnorm,&
                    chisq,Vresre,Vresim,&
                    Nuv,Nca)
  implicit none
  !
  !  This subroutine will compute chisquare and its analytic gradient
  !  for closure amplitude data
  !
  ! NuFFT-ed visibilities
  integer,      intent(in):: Nuv
  complex(dpc), intent(in):: Vcmp(Nuv)
  ! Data
  integer,  intent(in):: Nca            ! Number of data
  integer,  intent(in):: uvidxca(4,Nca) ! UV Index of Amp data
  real(dp), intent(in):: CA(Nca)        ! Amp data
  real(dp), intent(in):: Varca(Nca)     ! variances of Amp data
  ! Normalization Factor of Chisquare
  real(dp), intent(in):: fnorm
  ! Outputs
  real(dp), intent(inout):: chisq           ! chisquare
  real(dp), intent(inout):: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  real(dp):: resid, factor, model
  real(dp):: Vamp1, Vamp2, Vamp3, Vamp4
  complex(dpc):: Vcmp1, Vcmp2, Vcmp3, Vcmp4
  integer:: uvidx1, uvidx2, uvidx3, uvidx4
  integer:: i

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nca,fnorm,uvidxca,Vcmp,CA,Varca) &
  !$OMP   PRIVATE(i,model,resid,&
  !$OMP           uvidx1,uvidx2,uvidx3,uvidx4,&
  !$OMP           Vcmp1,Vcmp2,Vcmp3,Vcmp4,&
  !$OMP           Vamp1,Vamp2,Vamp3,Vamp4),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do i=1, Nca
    ! pick up uv index
    uvidx1 = abs(uvidxca(1,i))
    uvidx2 = abs(uvidxca(2,i))
    uvidx3 = abs(uvidxca(3,i))
    uvidx4 = abs(uvidxca(4,i))

    ! pick up full complex visibilities
    Vcmp1 = Vcmp(uvidx1)
    Vcmp2 = Vcmp(uvidx2)
    Vcmp3 = Vcmp(uvidx3)
    Vcmp4 = Vcmp(uvidx4)
    Vamp1 = abs(Vcmp1)
    Vamp2 = abs(Vcmp2)
    Vamp3 = abs(Vcmp3)
    Vamp4 = abs(Vcmp4)

    ! calculate model log closure amplitude and residual
    model = log(Vamp1)+log(Vamp2)-log(Vamp3)-log(Vamp4)
    resid = CA(i) - model

    ! compute chisquare
    chisq = chisq + resid**2/Varca(i)*fnorm

    ! compute residual vectors
    factor = -2*resid/Varca(i)*fnorm
    ! re
    Vresre(uvidx1) = Vresre(uvidx1) + factor / Vamp1**2 * dreal(Vcmp1)
    Vresre(uvidx2) = Vresre(uvidx2) + factor / Vamp2**2 * dreal(Vcmp2)
    Vresre(uvidx3) = Vresre(uvidx3) - factor / Vamp3**2 * dreal(Vcmp3)
    Vresre(uvidx4) = Vresre(uvidx4) - factor / Vamp4**2 * dreal(Vcmp4)
    ! im
    Vresim(uvidx1) = Vresim(uvidx1) + factor / Vamp1**2 * dimag(Vcmp1)
    Vresim(uvidx2) = Vresim(uvidx2) + factor / Vamp2**2 * dimag(Vcmp2)
    Vresim(uvidx3) = Vresim(uvidx3) - factor / Vamp3**2 * dimag(Vcmp3)
    Vresim(uvidx4) = Vresim(uvidx4) - factor / Vamp4**2 * dimag(Vcmp4)
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine calc_chisq_cp(Vcmp,&
                    uvidxcp,CP,Varcp,&
                    fnorm,&
                    chisq,Vresre,Vresim,&
                    Nuv,Ncp)
  implicit none
  !
  !  This subroutine will compute chisquare and its analytic gradient
  !  for closure phase data
  !
  ! NuFFT-ed visibilities
  integer,      intent(in):: Nuv
  complex(dpc), intent(in):: Vcmp(Nuv)
  ! Data
  integer,  intent(in):: Ncp            ! Number of data
  integer,  intent(in):: uvidxcp(3,Ncp) ! UV Index of Amp data
  real(dp), intent(in):: CP(Ncp)        ! Amp data
  real(dp), intent(in):: Varcp(Ncp)     ! variances of Amp data
  ! Normalization Factor of Chisquare
  real(dp), intent(in):: fnorm
  ! Outputs
  real(dp), intent(inout):: chisq ! chisquare
  real(dp), intent(inout):: Vresre(Nuv), Vresim(Nuv) ! residual vector
                                            !   its adjoint FT provides
                                            !   the gradient of chisquare)

  real(dp):: resid, factor, model
  real(dp):: Vampsq1, Vampsq2, Vampsq3
  complex(dpc):: Vcmp1, Vcmp2, Vcmp3
  integer:: uvidx1, uvidx2, uvidx3
  integer:: i
  integer:: sign1, sign2, sign3

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Ncp,fnorm,uvidxcp,Vcmp,CP,Varcp) &
  !$OMP   PRIVATE(i,model,resid,&
  !$OMP           uvidx1,uvidx2,uvidx3,&
  !$OMP           Vcmp1,Vcmp2,Vcmp3,&
  !$OMP           Vampsq1,Vampsq2,Vampsq3,&
  !$OMP           sign1,sign2,sign3),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim)
  do i=1, Ncp
    ! pick up uv index
    uvidx1 = abs(uvidxcp(1,i))
    uvidx2 = abs(uvidxcp(2,i))
    uvidx3 = abs(uvidxcp(3,i))

    ! pick up full complex visibilities
    Vcmp1 = Vcmp(uvidx1)
    Vcmp2 = Vcmp(uvidx2)
    Vcmp3 = Vcmp(uvidx3)
    sign1 = sign(1,uvidxcp(1,i))
    sign2 = sign(1,uvidxcp(2,i))
    sign3 = sign(1,uvidxcp(3,i))
    Vampsq1 = abs(Vcmp1)**2
    Vampsq2 = abs(Vcmp2)**2
    Vampsq3 = abs(Vcmp3)**2

    ! calculate model closure phases and residual
    model = atan2(dimag(Vcmp1),dreal(Vcmp1))*sign1
    model = atan2(dimag(Vcmp2),dreal(Vcmp2))*sign2 + model
    model = atan2(dimag(Vcmp3),dreal(Vcmp3))*sign3 + model
    resid = CP(i) - model
    resid = atan2(sin(resid),cos(resid))

    ! compute chisquare
    chisq = chisq + resid**2/Varcp(i)*fnorm

    ! compute residual vectors
    factor = -2*resid/Varcp(i)*fnorm

    Vresre(uvidx1) = Vresre(uvidx1) - factor/Vampsq1*dimag(Vcmp1)*sign1
    Vresre(uvidx2) = Vresre(uvidx2) - factor/Vampsq2*dimag(Vcmp2)*sign2
    Vresre(uvidx3) = Vresre(uvidx3) - factor/Vampsq3*dimag(Vcmp3)*sign3

    Vresim(uvidx1) = Vresim(uvidx1) + factor/Vampsq1*dreal(Vcmp1)*sign1
    Vresim(uvidx2) = Vresim(uvidx2) + factor/Vampsq2*dreal(Vcmp2)*sign2
    Vresim(uvidx3) = Vresim(uvidx3) + factor/Vampsq3*dreal(Vcmp3)*sign3
  end do
  !$OMP END PARALLEL DO
end subroutine


!-------------------------------------------------------------------------------
! Functions to compute chisquares and also residual vectors
!-------------------------------------------------------------------------------
subroutine model_fcv(Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
                     u,v,&
                     uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
                     chisq,gradchisq,modelr,modeli,residr,residi,&
                     Npix,Nuv,Nfcv)
 !
 !  This subroutine will compute model, residual, chisquare and
 !  its analytic gradient of full complex visibility data sets
 !  from input image data
 !
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! NuFFT-ed visibilities
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy
  ! Data
  integer,  intent(in):: Nfcv                     ! Number of data
  integer,  intent(in):: uvidxfcv(Nfcv)           ! UV Index of FCV data
  real(dp), intent(in):: Vfcvr(Nfcv),Vfcvi(Nfcv)  ! Full complex visibility (FCV) data
  real(dp), intent(in):: Varfcv(Nfcv)             ! variances of FCV data
  ! Outputs
  real(dp), intent(out):: chisq                        ! chisquare
  real(dp), intent(out):: modelr(Nfcv), modeli(Nfcv)  ! Model Vector
  real(dp), intent(out):: residr(Nfcv), residi(Nfcv)  ! Residual Vector
  real(dp), intent(out):: gradchisq(Npix)   !   its adjoint FT provides
                                            !   the gradient of chisquare

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:),gradchisq2d(:,:)
  real(dp), allocatable :: Vresre(:),Vresim(:)
  complex(dpc), allocatable :: Vcmp(:)
  complex(dpc), allocatable :: Vfcv(:),resid(:),model(:)

  ! other factors
  real(dp):: factor
  integer:: uvidx

  ! loop variables
  integer :: i

  ! initialize full complex visibilities
  !   allocate arrays
  allocate(Vfcv(Nfcv))
  Vfcv = dcmplx(Vfcvr,Vfcvi)
  !   shift tracking center of full complex visibilities from the reference
  !   pixel to the center of the image
  !write(*,*) 'Shift Tracking Center of Full complex visibilities.'
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(u,v,Nxref,Nyref,Nx,Ny,Nfcv) &
  !$OMP   PRIVATE(i)
  do i=1,Nfcv
    call phashift_r2c(u(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i)),&
                      v(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i)),&
                      Nxref,Nyref,Nx,Ny,&
                      Vfcv(i),Vfcv(i))
  end do
  !$OMP END PARALLEL DO

  ! Copy 1d image to 2d image
  !   allocate array
  allocate(I2d(Nx,Ny))
  I2d(:,:)=0d0
  !   copy image
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)

  ! Forward Non-unifrom Fast Fourier Transform
  !   allocate array
  allocate(Vcmp(Nuv))
  Vcmp(:) = dcmplx(0d0,0d0)
  !   Forward NUFFT
  call NUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  deallocate(I2d)

  ! Compute Chisquare
  !  allocate array
  allocate(model(Nfcv),resid(Nfcv))
  resid(:) = dcmplx(0d0,0d0)
  model(:) = dcmplx(0d0,0d0)
  allocate(Vresre(Nuv),Vresim(Nuv))
  Vresre(:) = 0d0
  Vresim(:) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nfcv,uvidxfcv,Vcmp,Vfcv,Varfcv) &
  !$OMP   PRIVATE(i,uvidx,factor),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim,resid,model)
  do i=1, Nfcv
    ! pick up uv index
    uvidx = abs(uvidxfcv(i))

    ! take residual
    if (uvidxfcv(i) > 0) then
      model(i) = Vcmp(uvidx)
    else
      model(i) = dconjg(Vcmp(uvidx))
    end if
    resid(i) = Vfcv(i) - model(i)

    ! compute chisquare
    chisq = chisq + abs(resid(i))**2/Varfcv(i)

    ! compute residual vector
    factor = -2/Varfcv(i)
    Vresre(uvidx) = Vresre(uvidx) + factor*dreal(resid(i))
    Vresim(uvidx) = Vresim(uvidx) + factor*dimag(resid(i))*sign(1,uvidxfcv(i))
  end do
  !$OMP END PARALLEL DO
  deallocate(Vfcv)
  deallocate(Vcmp)

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  call NUFFT_adj_resid(u,v,Vresre,Vresim,gradchisq2d(:,:),Nx,Ny,Nuv)
  deallocate(Vresre,Vresim)

  ! copy the gradient of chisquare into that of cost functions
  call I1d_I2d_inv(xidx,yidx,gradchisq,gradchisq2d,Npix,Nx,Ny)
  deallocate(gradchisq2d)

  !   shift tracking center of full complex visibilities from the reference
  !   pixel to the center of the image
  !write(*,*) 'Shift Tracking Center of Model and Residual visibilities.'
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(u,v,Nxref,Nyref,Nx,Ny,Nfcv) &
  !$OMP   PRIVATE(i)
  do i=1,Nfcv
    call phashift_c2r(u(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i)),&
                      v(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i)),&
                      Nxref,Nyref,Nx,Ny,&
                      resid(i),resid(i))
    call phashift_c2r(u(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i)),&
                      v(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i)),&
                      Nxref,Nyref,Nx,Ny,&
                      model(i),model(i))
  end do
  !$OMP END PARALLEL DO

  modelr = dreal(model)
  modeli = dimag(model)
  residr = dreal(resid)
  residi = dimag(resid)

  deallocate(resid,model)
end subroutine


subroutine model_amp(Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
                     u,v,&
                     uvidxamp,Vamp,Varamp,&
                     chisq,gradchisq,model,resid,&
                     Npix,Nuv,Namp)
 !
 !  This subroutine will compute model, residual, chisquare and
 !  its analytic gradient of amplitude data sets
 !  from input image data
 !
  implicit none
  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! NuFFT-ed visibilities
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy
  ! Data
  integer,  intent(in):: Namp           ! Number of data
  integer,  intent(in):: uvidxamp(Namp) ! UV Index of amp data
  real(dp), intent(in):: Vamp(Namp)     ! Amplitude data
  real(dp), intent(in):: Varamp(Namp)   ! variances of amp data
  ! Outputs
  real(dp), intent(out):: chisq           ! chisquare
  real(dp), intent(out):: model(Namp)     ! Model Vector
  real(dp), intent(out):: resid(Namp)     ! Residual Vector
  real(dp), intent(out):: gradchisq(Npix) ! its adjoint FT provides
                                          ! the gradient of chisquare

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:),gradchisq2d(:,:)
  real(dp), allocatable :: Vresre(:),Vresim(:)
  complex(dpc), allocatable :: Vcmp(:)

  ! other factors
  real(dp):: factor
  integer:: uvidx

  ! loop variables
  integer :: i

  ! Copy 1d image to 2d image
  !   allocate array
  allocate(I2d(Nx,Ny))
  I2d(:,:)=0d0
  !   copy image
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)

  ! Forward Non-unifrom Fast Fourier Transform
  !   allocate array
  allocate(Vcmp(Nuv))
  Vcmp(:) = dcmplx(0d0,0d0)
  !   Forward NUFFT
  call NUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  deallocate(I2d)

  ! Compute Chisquare
  !  allocate array
  allocate(Vresre(Nuv),Vresim(Nuv))
  Vresre(:) = 0d0
  Vresim(:) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Namp,uvidxamp,Vcmp,Vamp,Varamp) &
  !$OMP   PRIVATE(i,uvidx,factor),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim,resid,model)
  do i=1, Namp
    ! pick up uv index
    uvidx = abs(uvidxamp(i))

    ! take residual
    model(i) = abs(Vcmp(uvidx))
    resid(i) = Vamp(i) - model(i)

    ! compute chisquare
    chisq = chisq + resid(i)**2/Varamp(i)

    ! compute residual vector
    factor = -2*resid(i)/Varamp(i)/model(i)
    Vresre(uvidx) = Vresre(uvidx) + factor * dreal(Vcmp(uvidx))
    Vresim(uvidx) = Vresim(uvidx) + factor * dimag(Vcmp(uvidx))
  end do
  !$OMP END PARALLEL DO
  deallocate(Vcmp)

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  call NUFFT_adj_resid(u,v,Vresre,Vresim,gradchisq2d(:,:),Nx,Ny,Nuv)
  deallocate(Vresre,Vresim)

  ! copy the gradient of chisquare into that of cost functions
  call I1d_I2d_inv(xidx,yidx,gradchisq,gradchisq2d,Npix,Nx,Ny)
  deallocate(gradchisq2d)
end subroutine


subroutine model_ca(Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
                     u,v,&
                     uvidxca,CA,Varca,&
                     chisq,gradchisq,model,resid,&
                     Npix,Nuv,Nca)
  !
  !  This subroutine will compute model, residual, chisquare and
  !  its analytic gradient of closure amplitude data sets
  !  from input image data
  !
  implicit none
  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! NuFFT-ed visibilities
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy
  ! Data
  integer,  intent(in):: Nca              ! Number of data
  integer,  intent(in):: uvidxca(4,Nca)   ! UV Index of ca data
  real(dp), intent(in):: CA(Nca)          ! Closure Amplitude Data
  real(dp), intent(in):: Varca(Nca)       ! variances of ca data
  ! Outputs
  real(dp), intent(out):: chisq           ! chisquare
  real(dp), intent(out):: model(Nca)      ! Model Vector
  real(dp), intent(out):: resid(Nca)      ! Residual Vector
  real(dp), intent(out):: gradchisq(Npix) ! its adjoint FT provides
                                          ! the gradient of chisquare

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:),gradchisq2d(:,:)
  real(dp), allocatable :: Vresre(:),Vresim(:)
  complex(dpc), allocatable :: Vcmp(:)

  ! other factors
  real(dp):: factor
  real(dp):: Vamp1, Vamp2, Vamp3, Vamp4
  complex(dpc):: Vcmp1, Vcmp2, Vcmp3, Vcmp4
  integer:: uvidx1, uvidx2, uvidx3, uvidx4

  ! loop variables
  integer :: i

  ! Copy 1d image to 2d image
  !   allocate array
  allocate(I2d(Nx,Ny))
  I2d(:,:)=0d0
  !   copy image
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)

  ! Forward Non-unifrom Fast Fourier Transform
  !   allocate array
  allocate(Vcmp(Nuv))
  Vcmp(:) = dcmplx(0d0,0d0)
  !   Forward NUFFT
  call NUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  deallocate(I2d)

  ! Compute Chisquare
  !  allocate array
  allocate(Vresre(Nuv),Vresim(Nuv))
  Vresre(:) = 0d0
  Vresim(:) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nca,uvidxca,Vcmp,CA,Varca) &
  !$OMP   PRIVATE(i,&
  !$OMP           uvidx1,uvidx2,uvidx3,uvidx4,&
  !$OMP           Vcmp1,Vcmp2,Vcmp3,Vcmp4,&
  !$OMP           Vamp1,Vamp2,Vamp3,Vamp4),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim,resid,model)
  do i=1, Nca
    ! pick up uv index
    uvidx1 = abs(uvidxca(1,i))
    uvidx2 = abs(uvidxca(2,i))
    uvidx3 = abs(uvidxca(3,i))
    uvidx4 = abs(uvidxca(4,i))

    ! pick up full complex visibilities
    Vcmp1 = Vcmp(uvidx1)
    Vcmp2 = Vcmp(uvidx2)
    Vcmp3 = Vcmp(uvidx3)
    Vcmp4 = Vcmp(uvidx4)
    Vamp1 = abs(Vcmp1)
    Vamp2 = abs(Vcmp2)
    Vamp3 = abs(Vcmp3)
    Vamp4 = abs(Vcmp4)

    ! calculate model log closure amplitude and residual
    model(i) = log(Vamp1)+log(Vamp2)-log(Vamp3)-log(Vamp4)
    resid(i) = CA(i) - model(i)

    ! compute chisquare
    chisq = chisq + resid(i)**2/Varca(i)

    ! compute residual vectors
    factor = -2*resid(i)/Varca(i)
    ! re
    Vresre(uvidx1) = Vresre(uvidx1) + factor / Vamp1**2 * dreal(Vcmp1)
    Vresre(uvidx2) = Vresre(uvidx2) + factor / Vamp2**2 * dreal(Vcmp2)
    Vresre(uvidx3) = Vresre(uvidx3) - factor / Vamp3**2 * dreal(Vcmp3)
    Vresre(uvidx4) = Vresre(uvidx4) - factor / Vamp4**2 * dreal(Vcmp4)
    ! im
    Vresim(uvidx1) = Vresim(uvidx1) + factor / Vamp1**2 * dimag(Vcmp1)
    Vresim(uvidx2) = Vresim(uvidx2) + factor / Vamp2**2 * dimag(Vcmp2)
    Vresim(uvidx3) = Vresim(uvidx3) - factor / Vamp3**2 * dimag(Vcmp3)
    Vresim(uvidx4) = Vresim(uvidx4) - factor / Vamp4**2 * dimag(Vcmp4)
  end do
  !$OMP END PARALLEL DO
  deallocate(Vcmp)

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  call NUFFT_adj_resid(u,v,Vresre,Vresim,gradchisq2d(:,:),Nx,Ny,Nuv)
  deallocate(Vresre,Vresim)

  ! copy the gradient of chisquare into that of cost functions
  call I1d_I2d_inv(xidx,yidx,gradchisq,gradchisq2d,Npix,Nx,Ny)
  deallocate(gradchisq2d)
end subroutine


subroutine model_cp(Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
                     u,v,&
                     uvidxcp,CP,Varcp,&
                     chisq,gradchisq,model,resid,&
                     Npix,Nuv,Ncp)
  !
  !  This subroutine will compute model, residual, chisquare and
  !  its analytic gradient of closure phase data sets
  !  from input image data
  !
  implicit none
  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! NuFFT-ed visibilities
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy
  ! Data
  integer,  intent(in) :: Ncp            ! Number of data
  integer,  intent(in) :: uvidxcp(3,Ncp) ! UV Index of cp data
  real(dp), intent(in) :: CP(Ncp)        ! Closure Phase data
  real(dp), intent(in) :: Varcp(Ncp)     ! variances of ca data
  ! Outputs
  real(dp), intent(out) :: chisq           ! chisquare
  real(dp), intent(out) :: model(Ncp)      ! Model Vector
  real(dp), intent(out) :: resid(Ncp)      ! Residual Vector
  real(dp), intent(out) :: gradchisq(Npix) ! its adjoint FT provides
                                          ! the gradient of chisquare

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:),gradchisq2d(:,:)
  real(dp), allocatable :: Vresre(:),Vresim(:)
  complex(dpc), allocatable :: Vcmp(:)

  ! other factors
  real(dp):: factor
  real(dp):: Vampsq1, Vampsq2, Vampsq3
  complex(dpc):: Vcmp1, Vcmp2, Vcmp3
  integer:: uvidx1, uvidx2, uvidx3
  integer:: sign1, sign2, sign3

  ! loop variables
  integer :: i

  ! Copy 1d image to 2d image
  !   allocate array
  allocate(I2d(Nx,Ny))
  I2d(:,:)=0d0
  !   copy image
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)

  ! Forward Non-unifrom Fast Fourier Transform
  !   allocate array
  allocate(Vcmp(Nuv))
  Vcmp(:) = dcmplx(0d0,0d0)
  !   Forward NUFFT
  call NUFFT_fwd(u,v,I2d,Vcmp,Nx,Ny,Nuv)
  deallocate(I2d)

  ! Compute Chisquare
  !  allocate array
  allocate(Vresre(Nuv),Vresim(Nuv))
  Vresre(:) = 0d0
  Vresim(:) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Ncp,uvidxcp,Vcmp,CP,Varcp) &
  !$OMP   PRIVATE(i,&
  !$OMP           uvidx1,uvidx2,uvidx3,&
  !$OMP           Vcmp1,Vcmp2,Vcmp3,&
  !$OMP           Vampsq1,Vampsq2,Vampsq3,&
  !$OMP           sign1,sign2,sign3),&
  !$OMP   REDUCTION(+:chisq,Vresre,Vresim,model,resid)
  do i=1, Ncp
    ! pick up uv index
    uvidx1 = abs(uvidxcp(1,i))
    uvidx2 = abs(uvidxcp(2,i))
    uvidx3 = abs(uvidxcp(3,i))

    ! pick up full complex visibilities
    Vcmp1 = Vcmp(uvidx1)
    Vcmp2 = Vcmp(uvidx2)
    Vcmp3 = Vcmp(uvidx3)
    sign1 = sign(1,uvidxcp(1,i))
    sign2 = sign(1,uvidxcp(2,i))
    sign3 = sign(1,uvidxcp(3,i))
    Vampsq1 = abs(Vcmp1)**2
    Vampsq2 = abs(Vcmp2)**2
    Vampsq3 = abs(Vcmp3)**2

    ! calculate model closure phases and residual
    model(i) = atan2(dimag(Vcmp1),dreal(Vcmp1))*sign1
    model(i) = atan2(dimag(Vcmp2),dreal(Vcmp2))*sign2 + model(i)
    model(i) = atan2(dimag(Vcmp3),dreal(Vcmp3))*sign3 + model(i)
    resid(i) = CP(i) - model(i)
    resid(i) = atan2(sin(resid(i)),cos(resid(i)))

    ! compute chisquare
    chisq = chisq + resid(i)**2/Varcp(i)

    ! compute residual vectors
    factor = -2*resid(i)/Varcp(i)

    Vresre(uvidx1) = Vresre(uvidx1) - factor/Vampsq1*dimag(Vcmp1)*sign1
    Vresre(uvidx2) = Vresre(uvidx2) - factor/Vampsq2*dimag(Vcmp2)*sign2
    Vresre(uvidx3) = Vresre(uvidx3) - factor/Vampsq3*dimag(Vcmp3)*sign3

    Vresim(uvidx1) = Vresim(uvidx1) + factor/Vampsq1*dreal(Vcmp1)*sign1
    Vresim(uvidx2) = Vresim(uvidx2) + factor/Vampsq2*dreal(Vcmp2)*sign2
    Vresim(uvidx3) = Vresim(uvidx3) + factor/Vampsq3*dreal(Vcmp3)*sign3
  end do
  !$OMP END PARALLEL DO
  deallocate(Vcmp)

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  call NUFFT_adj_resid(u,v,Vresre,Vresim,gradchisq2d(:,:),Nx,Ny,Nuv)
  deallocate(Vresre,Vresim)

  ! copy the gradient of chisquare into that of cost functions
  call I1d_I2d_inv(xidx,yidx,gradchisq,gradchisq2d,Npix,Nx,Ny)
  deallocate(gradchisq2d)
end subroutine
end module
