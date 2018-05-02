module fftim3d
  !$use omp_lib
  use param, only: dp, deps
  use fftlib, only: calc_F, grad_amppha, DFT,&
                    chisq_fcv, chisq_amp, chisq_camp, chisq_cphase,&
                    model_fcv, model_amp, model_camp, model_cphase,&
                    tv, tsv, gradtve, gradtsve, calc_I2d
  use image, only:
  implicit none

  ! BLAS function (external)
  interface
    real(kind(1d0)) function dasum(n, x, incx)
      integer,         intent(in) :: n, incx
      real(kind(1d0)), intent(in) :: x(n)
    end function
  end interface

  interface
    real(kind(1d0)) function ddot(n, x, incx, y, incy)
      integer,         intent(in) :: n, incx, incy
      real(kind(1d0)), intent(in) :: x(n), y(n)
    end function
  end interface
contains
!
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
!
subroutine dynamic_imaging(
  Iin, x, y, xidx, yidx, Nx, Ny, Nz, u, v, &
  lambl1, lambtv, lambtsv, nonneg, logreg, Niter, &
  isfcv, uvidxfcv, Vfcvr, Vfcvi, Varfcv, &
  isamp, uvidxamp, Vamp, Varamp, &
  iscp, uvidxcp, CP, Varcp, &
  isca, uvidxca, CA, Varca, &
  m, factr, pgdeps, &
  Iout, &
  Npix, Nuv, Nfcv, Namp, Ncp, Nca &
)
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny, Nz
  real(dp), intent(in) :: Iin(Npix, Nz)
  real(dp), intent(in) :: x(Npix, Nz),y(Npix, Nz) ! xy coordinates in radian.
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv, Nz), v(Nuv, Nz)  ! uv coordinates in wavelength

  ! Imaging parameters
  real(dp), intent(in) :: lambl1  ! imaging parameter for L1-norm
  real(dp), intent(in) :: lambtv  ! imaging parameter for iso-TV
  real(dp), intent(in) :: lambtsv ! imaging parameter for TSV
  logical,  intent(in) :: nonneg  ! if nonneg > 0, the image will be solved
                                  ! with a non-negative condition
  logical,  intent(in) :: logreg  ! regularize image in log domain
  integer,  intent(in) :: Niter   ! the number of iterations

  ! Parameters related to full complex visibilities
  logical,  intent(in) :: isfcv
  integer,  intent(in) :: Nfcv
  integer,  intent(in) :: uvidxfcv(Nfcv, Nz, Nz) ! uvidx
  real(dp), intent(in) :: Vfcvr(Nfcv, Nz), Vfcvi(Nfcv) ! Real and Imag parts of data
  real(dp), intent(in) :: Varfcv(Nfcv, Nz) ! Variance

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp
  integer,  intent(in) :: Namp
  integer,  intent(in) :: uvidxamp(Namp, Nz)
  real(dp), intent(in) :: Vamp(Namp, Nz)
  real(dp), intent(in) :: Varamp(Namp, Nz)

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp
  integer,  intent(in) :: Ncp
  integer,  intent(in) :: uvidxcp(3,Ncp, Nz)
  real(dp), intent(in) :: CP(Ncp, Nz)
  real(dp), intent(in) :: Varcp(Ncp, Nz)

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca
  integer,  intent(in) :: Nca
  integer,  intent(in) :: uvidxca(4,Nca, Nz)
  real(dp), intent(in) :: CA(Nca, Nz)
  real(dp), intent(in) :: Varca(Nca, Nz)

  ! Paramters related to the L-BFGS-B
  integer,  intent(in) :: m
  real(dp), intent(in) :: factr, pgdeps
  !
  ! Output Image
  real(dp), intent(out) :: Iout(1:Npix, 1:Nz)

  ! chisquare and grad chisq
  real(dp) :: cost
  real(dp) :: gradcost(1:Npix, 1:Nz)

  ! Number of Data
  integer :: Ndata

  ! Fourier Matrix
  real(dp), allocatable :: Freal(:,:,:), Fimag(:,:,:)

  ! variables and parameters tuning L-BFGS-B
  integer,  parameter   :: iprint = 1
  character(len=60)     :: task, csave
  logical               :: lsave(4)
  integer               :: isave(44)
  real(dp)              :: dsave(29)
  integer,  allocatable :: nbd(:),iwa(:)
  real(dp), allocatable :: lower(:),upper(:),wa(:)

  ! loop variables
  integer :: iuv, iz

  ! Check Ndata
  Ndata = 0
  if (isfcv .eqv. .True.) then
    Ndata = Ndata + Nfcv
  end if
  if (isamp .eqv. .True.) then
    Ndata = Ndata + Namp
  end if
  if (iscp .eqv. .True.) then
    Ndata = Ndata + Ncp
  end if
  if (isca .eqv. .True.) then
    Ndata = Ndata + Nca
  end if

  !-------------------------------------
  ! Preperation
  !-------------------------------------
  ! allocate vectors and arrays
  !write(*,*) 'stdftim.imaging: Allocate Freal, Fimag'
  allocate(Freal(Npix, Nuv), Fimag(Npix, Nuv))
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv, Nz) &
  !$OMP   PRIVATE(iuv, iz)
  do iz=1,Nz
    do iuv=1,Nuv
      Freal(1:Npix,i) = 0d0
      Fimag(1:Npix,i) = 0d0
    end do
  end do
  !$OMP END PARALLEL DO

  ! calculate Fourier Matrix
  !write(*,*) 'stdftim.imaging: Calc the Fourier Matrix'
  call calc_F(x,y,u,v,Freal,Fimag,Npix,Nuv,Nz)

  ! copy images
  !write(*,*) 'stdftim.imaging: Initialize the image'
  Iout(1:Npix, 1:Nz) = Iin(1:Npix, 1:Nz)
  !-------------------------------------
  ! L-BFGS-B
  !-------------------------------------
  !write(*,*) 'stdftim.imaging: Initialize the L-BFGS-B'
  ! initialise L-BFGS-B
  !   Allocate some arrays
  allocate(iwa(3*Npix))
  allocate(wa(2*m*Npix + 5*Npix + 11*m*m + 8*m))

  !   set boundary conditions
  allocate(lower(Npix),upper(Npix),nbd(Npix))
  if (nonneg .eqv. .True.) then
    nbd(:) = 1  ! put lower limit
    lower(:) = 0d0  ! put lower limit
  else
    nbd(:) = 0 ! no boundary conditions
  end if

  ! start L-BFGS-B
  !write(*,*) 'stdftim.imaging: start L-BFGS-B calculations'
  task = 'START'
  do while(task(1:2) == 'FG' &
          .or. task == 'NEW_X' &
          .or. task == 'START')
    ! This is the call to the L-BFGS-B code.
    call setulb ( Npix, m, Iout, lower, upper, nbd, cost, gradcost, &
                  factr, pgdeps, wa, iwa, task, iprint,&
                  csave, lsave, isave, dsave )

    if (task(1:2) == 'FG') then
      ! Calculate cost function and gradcostent of cost function
      call cost_3d(&
          Iout, xidx, yidx, Npix, Nx, Ny, Nz, Nuv, Ndata, &
          lambl1, lambtv, lambtsv, logreg, &
          Freal, Fimag,&
          isfcv, Nfcv, uvidxfcv, Vfcvr, Vfcvi, Varfcv,&
          isamp, Namp, uvidxamp, Vamp, Varamp,&
          iscp,  Ncp, uvidxcp, CP, Varcp,&
          isca,  Nca, uvidxca, CA, Varca,&
          cost, gradcost &
      )
    else
      ! If iteration number exceeds the total iteration number, make a flag
      ! to STOP L-BFGS-B iterations
      if (isave(30) > Niter) then
        task='STOP: TOTAL ITERATION NUMBER EXCEEDS LIMIT'
      end if

      ! If we have a flag to STOP the L-BFGS-B algorithm, print it out.
      if (task(1:4) .eq. 'STOP') then
        write (6,*) task
      end if
    end if
  end do

  !-------------------------------------
  ! deallocate vectors
  !-------------------------------------
  deallocate(Freal,Fimag)
end subroutine
!
!
!-------------------------------------------------------------------------------
! calc cost functions
!-------------------------------------------------------------------------------
!
subroutine cost_3d(&
    Iin, xidx, yidx, Npix, Nx, Ny, Nz, Nuv, Ndata, &
    lambl1, lambtv, lambtsv, lambmem, logreg,&
    u, v,&
    isfcv, Nfcv, uvidxfcv, Vfcvr, Vfcvi, Varfcv,&
    isamp, Namp, uvidxamp, Vamp, Varamp,&
    iscp,  Ncp, uvidxcp, CP, Varcp,&
    isca,  Nca, uvidxca, CA, Varca,&
    cost, gradcost)
  !
  ! Calculate Cost Functions
  !
  implicit none

  ! Number of uv points and data
  integer,  intent(in) :: Nuv
  integer,  intent(in) :: Ndata

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny, Nz
  real(dp), intent(in) :: Nxref, Nyref ! reference pixel: 1 will be the lower/leftmost end
  real(dp), intent(in) :: Iin(Npix, Nz)
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! Imaging parameters
  real(dp), intent(in) :: lambl1,lambtv,lambtsv,lambmem
  logical,  intent(in) :: logreg  ! regularize image in log domain

  ! uv coordinates
  real(dp), intent(in) :: u(Nuv, Nz), v(Nuv, Nz) !uv coordinates multiplied by 2*pi*dx, 2*pi*dy

  ! Parameters related to full complex visibilities
  logical,  intent(in) :: isfcv
  integer,  intent(in) :: Nfcv
  integer,  intent(in) :: uvidxfcv(Nfcv, Nz)
  real(dp), intent(in) :: Vfcvr(Nfcv, Nz), Vfcvi(Nfcv, Nz)
  real(dp), intent(in) :: Varfcv(Nfcv, Nz)

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp
  integer,  intent(in) :: Namp
  integer,  intent(in) :: uvidxamp(Namp, Nz)
  real(dp), intent(in) :: Vamp(Namp, Nz)
  real(dp), intent(in) :: Varamp(Namp, Nz)

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp
  integer,  intent(in) :: Ncp
  integer,  intent(in) :: uvidxcp(3,Ncp, Nz)
  real(dp), intent(in) :: CP(Ncp, Nz)
  real(dp), intent(in) :: Varcp(Ncp, Nz)

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca
  integer,  intent(in) :: Nca
  integer,  intent(in) :: uvidxca(4,Nca, Nz)
  real(dp), intent(in) :: CA(Nca, Nz)
  real(dp), intent(in) :: Varca(Nca, Nz)

  ! Outputs
  real(dp), intent(out) :: cost(1:Nz)
  real(dp), intent(out) :: gradcost(1:Npix, 1:Nz)

  ! integer
  integer :: ipix, iuv, iz

  ! chisquares, gradients of each term of equations
  real(dp) :: chisq(Nz), reg(Nz)
  real(dp) :: fnorm(Nz)   ! normalization factor for chisquares

  ! allocatable arrays
  real(dp), allocatable :: I3d(:,:,:), Iin_reg(:,:), gradchisq(:,:), gradreg(:,:)
  complex(dpc), allocatable :: Vresid(:,:), Vcmp(:,:)

  !-----------------------------------------------------------------------------
  ! Initialize outputs, and some parameters
  !-----------------------------------------------------------------------------
  ! Initialize the chisquare and its gradient
  !write(*,*) 'stdftim.calc_cost: initialize cost and gradcost'
  cost(:) = 0d0
  gradcost(:,:) = 0d0

  ! Normalization factor of chisquare
  fnorm(:) = Ndata

  !-----------------------------------------------------------------------------
  ! Compute chisquares, its gradient
  !-----------------------------------------------------------------------------
  ! Allocating arrays
  allocate(I2d(Nx,Ny))
  allocate(Iin_reg(Npix, Nz), gradchisq(Npix, Nz))
  allocate(Vresid(Nuv, Nz), Vcmp(Nuv, Nz))

  ! Initialize
  I2d(:,:)=0d0
  chisq(:) = 0d0
  gradchisq(:,:) = 0d0
  Vresid(1:Nuv, 1:Nz) = dcmplx(0d0,0d0)
  Vcmp(1:Nuv, 1:Nz) = dcmplx(0d0,0d0)

  ! Copy 1d image to 2d image
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)
  call I2d_I3d_fwd(xidx,yidx,Iin,I3d,Npix,Nx,Ny,Nz)

  !-----------------------------------------------------------------------------
  ! NuFFT all images
  !-----------------------------------------------------------------------------
  ! Forward Non-unifrom Fast Fourier Transform
  do iz=1, Nz
    call fNUFFT(u,v,I3d(:,:,iz),Vcmp,Nx,Ny,Nuv)

    ! Full complex visibility
    if (isfcv .eqv. .True.) then
      call chisq_fcv(
        Vcmp(:,iz),uvidxfcv(:,iz),Vfcv(:,iz),Varfcv(:,iz),fnorm,
        chisq(iz),Vresid(:,iz),Nuv(:,iz),Nfcv(:,iz)
        )
    end if

    ! Amplitudes
    if (isamp .eqv. .True.) then
      call chisq_amp(Vcmp(:,iz),uvidxamp(:,iz),Vamp(:,iz),Varamp(:,iz),fnorm,
        chisq(iz),Vresid(:,iz),Nuv(:,iz),Namp(:,iz)
        )
    end if

    ! Log closure amplitudes
    if (isca .eqv. .True.) then
      call chisq_ca(Vcmp(:,iz)(:,iz),uvidxca(:,iz),CA(:,iz),Varca(:,iz),fnorm,
        chisq(iz),Vresid(:,iz),Nuv(:,iz),Nca(:,iz)
        )
    end if

    ! Closure phases
    if (iscp .eqv. .True.) then
      call chisq_cp(Vcmp(:,iz),uvidxcp(:,iz),CP(:,iz),Varcp(:,iz),fnorm,
        chisq(iz),Vresid(:,iz),Nuv(:,iz),Ncp(:,iz)
        )
    end if
  end do

  !-----------------------------------------------------------------------------
  ! calculate gradient of chi-squares
  !-----------------------------------------------------------------------------
  ! Adjoint Non-unifrom Fast Fourier Transform
  call aNUFFTsum(u(:,:),v(:,:),Vresid(:,:),gradcost(:,:),Nx,Ny,Nuv,Nz)
  deallocate(Vresid)

  !-----------------------------------------------------------------------------
  ! Transform Image
  !-----------------------------------------------------------------------------
  if (logreg .eqv. .True.) then
    allocate(Iin_reg(Npix,Nz))
    call logconv_fwd(Iin,Iin_reg,Npix)
  else
    call dcopy(Npix,Iin,1,Iin_reg,1)
  end if

  ! Single pixel based regularizations
  if (lambl1 > 0 .or. lambmem > 0) then
    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(Npix, Iin_reg, lambl1, lambmem) &
    !$OMP   PRIVATE(ipix) &
    !$OMP   REDUCTION(+: reg, gradreg)
    do iz=1, Nz
      do ipix=1, Npix
        if (lambl1 > 0) then
          reg(iz) = reg(iz) + lambl1 * l1_e(Iin_reg(ipix,iz))
          gradreg(ipix,iz) = gradreg(ipix,iz) + lambl1 * l1_grade(Iin_reg(ipix,iz))
        end if
        if (lambmem > 0) then
          reg(iz) = reg(iz) + lambmem * mem_e(Iin_reg(ipix,iz))
          gradreg(ipix,iz) = gradreg(ipix,iz) + lambmem * mem_grade(Iin_reg(ipix,iz))
        end if
      end do
    end do
    !$OMP END PARALLEL DO
  end if
end subroutine
end module
