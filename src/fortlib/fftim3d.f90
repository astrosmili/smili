module fftim3d
  !$use omp_lib
  use param, only: dp, dpc, deps
  use fftlib, only: NUFFT_fwd, NUFFT_adj, NUFFT_adj_resid, phashift_r2c,&
                    chisq_fcv, chisq_amp, chisq_ca, chisq_cp
  use image, only: I1d_I2d_fwd, I1d_I2d_inv,&
                   log_fwd, log_grad,&
                   gamma_fwd, gamma_grad,&
                   l1_e, l1_grade,&
                   tv_e, tv_grade,&
                   tsv_e, tsv_grade,&
                   mem_e, mem_grade,&
                   comreg, zeroeps
  use image3d, only: d2, dkl, rt_d2grad, rt_dklgrad,&
                     ri_d2grad, ri_dklgrad
  implicit none
contains
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
subroutine imaging(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
  u,v,Nuvs,&
  lambl1,lambtv,lambtsv,lambmem,lambcom,&
  lambrt,lambri,lambrs,&
  Niter,nonneg,transtype,transprm,pcom,&
  isfcv,uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
  isamp,uvidxamp,Vamp,Varamp,&
  iscp,uvidxcp,CP,Varcp,&
  isca,uvidxca,CA,Varca,&
  m,factr,pgtol,&
  Iout,&
  Npix,Nuv,Nfcv,Namp,Ncp,Nca&
)
  !
  ! Core function of three-dimensional imaging
  !
  implicit none
  ! Image
  integer,  intent(in) :: Npix, Nx, Ny, Nz
  real(dp), intent(in) :: Iin(Npix*Nz)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! uv coordinates
  integer,  intent(in) :: Nuv
  integer,  intent(in) :: Nuvs(Nz)        ! number of uv data for each frame
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy

  ! Regularization Parameters
  real(dp), intent(in) :: lambl1    ! Regularization Parameter for L1-norm
  real(dp), intent(in) :: lambtv    ! Regularization Parameter for iso-TV
  real(dp), intent(in) :: lambtsv   ! Regularization Parameter for TSV
  real(dp), intent(in) :: lambmem   ! Regularization Parameter for MEM
  real(dp), intent(in) :: lambcom   ! Regularization Parameter for Center of Mass
  real(dp), intent(in) :: lambrt    ! Regularization Parameter for Dynamical Imaging (delta-t)
  real(dp), intent(in) :: lambri    ! Regularization Parameter for Dynamical Imaging (delta-I)
  real(dp), intent(in) :: lambrs    ! Regularization Parameter for Dynamical Imaging (entropy continuity)

  ! Imaging Parameter
  integer,  intent(in) :: Niter     ! iteration number
  logical,  intent(in) :: nonneg    ! if nonneg > 0, the image will be solved
                                    ! with a non-negative condition
  integer,  intent(in) :: transtype ! 0: No transform
                                    ! 1: log correction
                                    ! 2: gamma correction
  real(dp), intent(in) :: transprm  ! transtype=1: theshold for log
                                    ! transtype=2: power of gamma correction
  real(dp), intent(in) :: pcom      ! power weight of C.O.M regularization

  ! Parameters related to full complex visibilities
  logical,  intent(in) :: isfcv           ! is data?
  integer,  intent(in) :: Nfcv            ! number of data
  integer,  intent(in) :: uvidxfcv(Nfcv)  ! uvidx
  real(dp), intent(in) :: Vfcvr(Nfcv)     ! data
  real(dp), intent(in) :: Vfcvi(Nfcv)     ! data
  real(dp), intent(in) :: Varfcv(Nfcv)    ! variance

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp           ! is amplitudes?
  integer,  intent(in) :: Namp            ! Number of data
  integer,  intent(in) :: uvidxamp(Namp)  ! uvidx
  real(dp), intent(in) :: Vamp(Namp)      ! data
  real(dp), intent(in) :: Varamp(Namp)    ! variance

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp            ! is closure phases?
  integer,  intent(in) :: Ncp             ! Number of data
  integer,  intent(in) :: uvidxcp(3,Ncp)  ! uvidx
  real(dp), intent(in) :: CP(Ncp)         ! data
  real(dp), intent(in) :: Varcp(Ncp)      ! variance

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca            ! is closure amplitudes?
  integer,  intent(in) :: Nca             ! Number of data
  integer,  intent(in) :: uvidxca(4,Nca)  ! uvidx
  real(dp), intent(in) :: CA(Nca)         ! data
  real(dp), intent(in) :: Varca(Nca)      ! variance

  ! Paramters related to the L-BFGS-B
  integer,  intent(in) :: m
  real(dp), intent(in) :: factr, pgtol

  ! Output Image
  real(dp), intent(out) :: Iout(Npix*Nz)

  ! full complex visibilities to be used for calculations
  complex(dpc), allocatable :: Vfcv(:)

  ! chisquare and grad chisq
  real(dp) :: cost              ! cost function
  real(dp) :: gradcost(1:Npix*Nz)  ! its gradient

  ! Number of Data
  integer :: Ndata, Nparm   ! number of data, parameters
  real(dp) :: fnorm         ! normalization factor for chisquares
  integer :: Nuvs_sum(Nz)

  ! variables and parameters tuning L-BFGS-B
  integer,  parameter   :: iprint = 1
  character(len=60)     :: task, csave
  logical               :: lsave(4)
  integer               :: isave(44)
  real(dp)              :: dsave(29)
  integer,  allocatable :: nbd(:),iwa(:)
  real(dp), allocatable :: lower(:),upper(:),wa(:)

  ! loop variables
  integer :: i
  real(dp) :: u_tmp, v_tmp

  !-------------------------------------
  ! Initialize Data
  !-------------------------------------
  ! Number of Parameters
  Nparm = Npix * Nz

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
  fnorm = real(Ndata)
  write(*,*) 'Number of Data          ', Ndata
  write(*,*) 'Number of Paramter/frame', Npix
  write(*,*) 'Number of Frames        ', Nz
  write(*,*) 'Number of uv coordinates', Nuv

  ! copy images (Iin -> Iout)
  write(*,*) 'Initialize the parameter vector'
  call dcopy(Nparm,Iin,1,Iout,1)

  ! shift tracking center of full complex visibilities from the reference pixel
  ! to the center of the image
  allocate(Vfcv(Nfcv))
  Vfcv = dcmplx(Vfcvr,Vfcvi)
  if (isfcv .eqv. .True.) then
    write(*,*) 'Shift Tracking Center of Full complex visibilities.'
    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(u,v,Nxref,Nyref,Nx,Ny,Nfcv) &
    !$OMP   PRIVATE(i,u_tmp,v_tmp)
    do i=1,Nfcv
      u_tmp = u(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i))
      v_tmp = v(abs(uvidxfcv(i))) * sign(1,uvidxfcv(i))
      call phashift_r2c(u_tmp,v_tmp,&
                        Nxref,Nyref,Nx,Ny,&
                        Vfcv(i),Vfcv(i))
    end do
    !$OMP END PARALLEL DO
  end if

  ! Compute accumulated number of uvdata before each frame
  !   Nuvs_sum(i) + 1 will be the start index number for i-th frame
  !   Nuvs_sum(i) + Nuvs(i) will be the end index number for i-th frame
  Nuvs_sum(1)=0
  do i=2, Nz
    Nuvs_sum(i) = Nuvs_sum(i-1) + Nuvs(i-1)
  end do
  !-------------------------------------
  ! L-BFGS-B
  !-------------------------------------
  write(*,*) 'Initialize the L-BFGS-B'
  ! initialise L-BFGS-B
  !   Allocate some arrays
  allocate(iwa(3*Nparm))
  allocate(wa(2*m*Nparm + 5*Nparm + 11*m*m + 8*m))

  !   set boundary conditions
  allocate(lower(Nparm),upper(Nparm),nbd(Nparm))
  if (nonneg .eqv. .True.) then
    nbd(:) = 1  ! put lower limit
    lower(:) = 0d0  ! put lower limit
  else
    nbd(:) = 0 ! no boundary conditions
  end if

  ! start L-BFGS-B
  write(*,*) 'Start L-BFGS-B calculations'
  task = 'START'
  do while(task(1:2) == 'FG' &
          .or. task == 'NEW_X' &
          .or. task == 'START')
    ! This is the call to the L-BFGS-B code.
    call setulb ( Nparm, m, Iout, lower, upper, nbd, cost, gradcost, &
                  factr, pgtol, wa, iwa, task, iprint,&
                  csave, lsave, isave, dsave )

    if (task(1:2) == 'FG') then
      ! Calculate cost function and gradcostent of cost function
      call calc_cost(&
        Iout,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
        u,v,Nuvs,Nuvs_sum,&
        lambl1,lambtv,lambtsv,lambmem,lambcom,&
        lambrt,lambri,lambrs,&
        fnorm,transtype,transprm,pcom,&
        isfcv,uvidxfcv,Vfcv,Varfcv,&
        isamp,uvidxamp,Vamp,Varamp,&
        iscp,uvidxcp,CP,Varcp,&
        isca,uvidxca,CA,Varca,&
        cost,gradcost,&
        Nparm,Npix,Nuv,Nfcv,Namp,Ncp,Nca&
      )
    else
      ! If iteration number exceeds the total iteration number, make a flag
      ! to STOP L-BFGS-B iterations
      if (isave(30) > Niter) then
        task='STOP: TOTAL ITERATION NUMBER EXCEEDS LIMIT'
      else if (mod(isave(30),100) == 0) then
        print '("Iteration :",I5,"/",I5,"  Cost :",D13.6)',isave(30),Niter,cost
      end if
    end if
  end do
  print '("Iteration :",I5,"/",I5,"  Cost :",D13.6)',isave(30),Niter,cost
  write (6,*) task
  !
  ! deallocate arrays
  deallocate(Vfcv)
  deallocate(iwa,wa,lower,upper,nbd)
  where(abs(Iout)<zeroeps) Iout=0d0
end subroutine
!
!-------------------------------------------------------------------------------
! calc cost functions
!-------------------------------------------------------------------------------
!
subroutine calc_cost(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
  u,v,Nuvs,Nuvs_sum,&
  lambl1,lambtv,lambtsv,lambmem,lambcom,&
  lambrt,lambri,lambrs,&
  fnorm,transtype,transprm,pcom,&
  isfcv,uvidxfcv,Vfcv,Varfcv,&
  isamp,uvidxamp,Vamp,Varamp,&
  iscp,uvidxcp,CP,Varcp,&
  isca,uvidxca,CA,Varca,&
  cost,gradcost,&
  Nparm,Npix,Nuv,Nfcv,Namp,Ncp,Nca&
)
  !
  ! Calculate Cost Functions
  !
  implicit none

  ! Image
  integer,  intent(in) :: Nparm, Npix, Nx, Ny, Nz
  real(dp), intent(in) :: Iin(Nparm), Nxref, Nyref
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  integer,  intent(in) :: Nuvs_sum(Nz)    ! accumulated number of uv data *before* each frame
  integer,  intent(in) :: Nuvs(Nz)        ! number of uv data for each frame
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy

  ! Regularization Parameters
  real(dp), intent(in) :: lambl1  ! Regularization Parameter for L1-norm
  real(dp), intent(in) :: lambtv  ! Regularization Parameter for iso-TV
  real(dp), intent(in) :: lambtsv ! Regularization Parameter for TSV
  real(dp), intent(in) :: lambmem ! Regularization Parameter for MEM
  real(dp), intent(in) :: lambcom ! Regularization Parameter for Center of Mass
  real(dp), intent(in) :: lambrt    ! Regularization Parameter for Dynamical Imaging (delta-t)
  real(dp), intent(in) :: lambri    ! Regularization Parameter for Dynamical Imaging (delta-I)
  real(dp), intent(in) :: lambrs    ! Regularization Parameter for Dynamical Imaging (entropy continuity)

  ! Imaging Parameter
  real(dp), intent(in) :: fnorm     ! normalization factor for chisquare
  integer,  intent(in) :: transtype ! 0: No transform
                                    ! 1: log correction
                                    ! 2: gamma correction
  real(dp), intent(in) :: transprm  ! transtype=1: theshold for log
                                    ! transtype=2: power of gamma correction
  real(dp), intent(in) :: pcom      ! power weight of C.O.M regularization

  ! Parameters related to full complex visibilities
  logical,      intent(in) :: isfcv           ! is data?
  integer,      intent(in) :: Nfcv            ! number of data
  integer,      intent(in) :: uvidxfcv(Nfcv)  ! uvidx
  complex(dpc), intent(in) :: Vfcv(Nfcv)      ! data
  real(dp),     intent(in) :: Varfcv(Nfcv)    ! variance

  ! Parameters related to amplitude
  logical,  intent(in) :: isamp           ! is amplitudes?
  integer,  intent(in) :: Namp            ! Number of data
  integer,  intent(in) :: uvidxamp(Namp)  ! uvidx
  real(dp), intent(in) :: Vamp(Namp)      ! data
  real(dp), intent(in) :: Varamp(Namp)    ! variance

  ! Parameters related to the closure phase
  logical,  intent(in) :: iscp            ! is closure phases?
  integer,  intent(in) :: Ncp             ! Number of data
  integer,  intent(in) :: uvidxcp(3,Ncp)  ! uvidx
  real(dp), intent(in) :: CP(Ncp)         ! data
  real(dp), intent(in) :: Varcp(Ncp)      ! variance

  ! Parameters related to the closure amplitude
  logical,  intent(in) :: isca            ! is closure amplitudes?
  integer,  intent(in) :: Nca             ! Number of data
  integer,  intent(in) :: uvidxca(4,Nca)  ! uvidx
  real(dp), intent(in) :: CA(Nca)         ! data
  real(dp), intent(in) :: Varca(Nca)      ! variance

  ! Outputs
  real(dp), intent(out) :: cost
  real(dp), intent(out) :: gradcost(Nparm)

  ! integer
  integer :: ipix, iz, iparm, istart, iend

  ! chisquares, gradients of each term of equations
  real(dp) :: chisq, reg                 ! chisquare and regularization
  real(dp) :: rint_s, reg_frm            ! regularization for interpolation

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:),I2dl(:,:),I2du(:,:),Iin_reg(:)
  real(dp), allocatable :: Iavg(:), Iavg2d(:,:), Ij(:), Itmp(:), Itmp2d(:,:)
  real(dp), allocatable :: gradchisq2d(:,:)
  real(dp), allocatable :: gradreg(:), gradreg_tmp(:)
  real(dp), allocatable :: Vresre(:),Vresim(:)
  complex(dpc), allocatable :: Vcmp(:)
  real(dp), allocatable :: gradreg_frm(:), regset(:), gradregset(:), gradrint_s(:)

  !------------------------------------
  ! Initialize outputs, and some parameters
  !------------------------------------
  ! Initialize the chisquare and its gradient
  !write(*,*) 'stdftim.calc_cost: initialize cost and gradcost'
  cost = 0d0
  gradcost(:) = 0d0

  !------------------------------------
  ! Compute chisquare and its gradient
  !------------------------------------
  ! Initialize
  !   scalars
  chisq = 0d0

  ! Forward Non-unifrom Fast Fourier Transform
  !   allocatable arrays
  allocate(Vcmp(Nuv))
  Vcmp(:) = dcmplx(0d0,0d0)
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nx,Ny,Nz,Npix,Nuvs,Nuvs_sum,u,v) &
  !$OMP   PRIVATE(iz, istart, iend, I2d) &
  !$OMP   REDUCTION(+:Vcmp)
  do iz=1, Nz
    ! If there is a data corresponding to this frame
    if (Nuvs(iz) /= 0) then
      ! allocate 2D image for imaging
      allocate(I2d(Nx,Ny))
      I2d(:,:) = 0d0
      call I1d_I2d_fwd(xidx,yidx,Iin((iz-1)*Npix+1:iz*Npix),I2d,Npix,Nx,Ny)

      ! Index of data
      istart = Nuvs_sum(iz) + 1
      iend   = Nuvs_sum(iz) + Nuvs(iz)

      ! run forward NUFFT
      call NUFFT_fwd(u(istart:iend),v(istart:iend),I2d,Vcmp(istart:iend),&
                     Nx,Ny,Nuvs(iz))

      ! deallocate array
      deallocate(I2d)
    end if
  end do
  !$OMP END PARALLEL DO

  ! allocate arrays for residuals
  allocate(Vresre(Nuv), Vresim(Nuv))
  Vresre(:) = 0d0
  Vresim(:) = 0d0

  ! Full complex visibility
  if (isfcv .eqv. .True.) then
    call chisq_fcv(Vcmp,uvidxfcv,Vfcv,Varfcv,fnorm,chisq,Vresre,Vresim,Nuv,Nfcv)
  end if

  ! Amplitudes
  if (isamp .eqv. .True.) then
    call chisq_amp(Vcmp,uvidxamp,Vamp,Varamp,fnorm,chisq,Vresre,Vresim,Nuv,Namp)
  end if

  ! Log closure amplitudes
  if (isca .eqv. .True.) then
    call chisq_ca(Vcmp,uvidxca,CA,Varca,fnorm,chisq,Vresre,Vresim,Nuv,Nca)
  end if

  ! Closure phases
  if (iscp .eqv. .True.) then
    call chisq_cp(Vcmp,uvidxcp,CP,Varcp,fnorm,chisq,Vresre,Vresim,Nuv,Ncp)
  end if

  deallocate(Vcmp)

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nx,Ny,Nz,Npix,Nuvs,Nuvs_sum,u,v,Vresre,Vresim) &
  !$OMP   PRIVATE(iz, istart, iend, gradchisq2d) &
  !$OMP   REDUCTION(+:gradcost)
  do iz=1, Nz
    ! If there is a data corresponding to this frame
    if(Nuvs(iz) /= 0) then
      ! allocate 2D image for imaging
      allocate(gradchisq2d(Nx,Ny))
      gradchisq2d(:,:) = 0d0

      ! Index of data
      istart = Nuvs_sum(iz) + 1
      iend   = Nuvs_sum(iz) + Nuvs(iz)

      ! run adujoint NUFFT
      call NUFFT_adj_resid(u(istart:iend),v(istart:iend),&
                           Vresre(istart:iend),Vresim(istart:iend),&
                           gradchisq2d,Nx,Ny,Nuvs(iz))

      ! copy the gradient of chisquare into that of cost functions
      call I1d_I2d_inv(xidx,yidx,gradcost((iz-1)*Npix+1:iz*Npix),&
                       gradchisq2d,Npix,Nx,Ny)

      ! deallocate array
      deallocate(gradchisq2d)
    end if
  end do
  !$OMP END PARALLEL DO
  deallocate(Vresre,Vresim)

  ! copy the chisquare into that of cost functions
  cost = chisq
  !write(*,*) "chisq: ", chisq

  !------------------------------------
  ! Centoroid Regularizer
  !------------------------------------
  ! if (lambcom > 0) then
  !   ! initialize
  !   !   scalars
  !   reg = 0d0
  !   !   allocatable arrays
  !   allocate(gradreg(Nparm))
  !   gradreg(:) = 0d0
  !   !$OMP PARALLEL DO DEFAULT(SHARED) &
  !   !$OMP   FIRSTPRIVATE(Npix,Nx,Ny,Nz,Nxref,Nyref,lambcom,Iin,xidx,yidx) &
  !   !$OMP   PRIVATE(iz, istart, iend) &
  !   !$OMP   REDUCTION(+: reg, gradreg)
  !   do iz=1, Nz
  !     istart = (iz-1)*Npix+1
  !     iend = iz*Npix
  !     call comreg(xidx,yidx,Nxref,Nyref,pcom,Iin(istart:iend),reg,gradreg(istart:iend),Npix)
  !   end do
  !   !$OMP END PARALLEL DO
  !   cost = cost + lambcom * reg
  !   call daxpy(Nparm, lambcom, gradreg, 1, gradcost, 1) ! gradcost := lambcom * gradreg + gradcost
  !   deallocate(gradreg)
  ! end if
  !
  if (lambcom > 0) then
    ! initialize
    !   scalars
    reg = 0d0
    !   allocatable arrays
    allocate(gradreg_tmp(Npix))
    allocate(gradreg(Nparm))
    allocate(Iavg(Npix))
    Iavg(:) = 0d0
    gradreg_tmp(:) = 0d0
    gradreg(:) = 0d0
    !
    ! Averaged Image
    do iz=1, Nz
      Iavg = Iavg + Iin((iz-1)*Npix+1:iz*Npix)
    end do
    do ipix=1,Npix
      Iavg(ipix) = Iavg(ipix)/Nz
    end do
    !
    ! calc cost and its gradient
    call comreg(xidx,yidx,Nxref,Nyref,pcom,Iavg,reg,gradreg_tmp,Npix)
    cost = cost + lambcom * reg
    !
    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(Npix,Nz,gradreg_tmp) &
    !$OMP   PRIVATE(iz, ipix, istart) &
    !$OMP   REDUCTION(+: gradreg)
    do iz=1, Nz
      istart = (iz-1)*Npix+1
      !iend = iz*Npix
      do ipix=1,Npix
        gradreg(istart+ipix-1) = gradreg_tmp(ipix)/Nz
      end do
    end do
    !$OMP END PARALLEL DO
    call daxpy(Nparm, lambcom, gradreg, 1, gradcost, 1) ! gradcost := lambcom * gradreg + gradcost
    ! deallocate array
    deallocate(gradreg_tmp)
    deallocate(gradreg)
    deallocate(Iavg)
  end if

  !------------------------------------
  ! 2D & 3D Regularization Functions
  !------------------------------------
  ! Initialize
  !   scalars
  reg = 0d0

  !   allocatable arrays
  allocate(gradreg(Nparm),Iin_reg(Nparm))
  gradreg(:) = 0d0
  Iin_reg(:) = 0d0

  ! Transform Image
  if (transtype == 1) then
    ! Log Forward
    call log_fwd(transprm,Iin,Iin_reg,Nparm)
  else if (transtype == 2) then
    ! Gamma contrast
    call gamma_fwd(transprm,Iin,Iin_reg,Nparm)
  else
    call dcopy(Nparm,Iin,1,Iin_reg,1)
  end if

  ! 3D regularizers
  if (lambri > 0) then
    !   get an 3D averaged image
    allocate(Iavg(Npix))
    allocate(Iavg2d(Nx,Ny))
    Iavg(:) = 0d0
    Iavg2d(:,:) = 0d0
    do iz=1, Nz
      Iavg = Iavg + Iin((iz-1)*Npix+1:iz*Npix)
    end do
    do ipix=1,Npix
      Iavg(ipix) = Iavg(ipix)/Nz
    end do
    call I1d_I2d_fwd(xidx,yidx,Iavg,Iavg2d,Npix,Nx,Ny)

    !   temporal I-ratio for the gradient of Ri regularizer
    allocate(Ij(Npix),Itmp(Npix))
    allocate(Itmp2d(Nx,Ny))
    Ij(:) = 0d0
    Itmp(:) = 0d0
    Itmp2d(:,:) = 0d0
    do iz=1, Nz
      Ij = Iin((iz-1)*Npix+1:iz*Npix)
      do ipix=1, Npix
        if (Iavg(ipix) == 0 .or. Ij(ipix) == 0) then
          Itmp(ipix) = Itmp(ipix)
        else if (Iavg(ipix) /= 0 .and. Ij(ipix) /= 0) then
          Itmp(ipix) = Itmp(ipix) + log(Iavg(ipix)/Ij(ipix))
        end if
      end do
    end do
    call I1d_I2d_fwd(xidx,yidx,Itmp,Itmp2d,Npix,Nx,Ny)
    deallocate(Ij)
  end if

  !   scalars
  rint_s = 0d0
  !   arrays
  allocate(regset(Nz),gradregset(Nparm))
  regset(:) = 0d0
  gradregset(:) = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, Nz, lambl1, lambmem, lambtv, lambtsv, lambrt, lambri&
  !$OMP                Iin_reg, xidx, yidx, regset, gradregset) &
  !$OMP   PRIVATE(iz, ipix, iparm, I2d, I2dl, I2du, reg_frm, gradreg_frm) &
  !$OMP   REDUCTION(+: reg, gradreg)
  do iz=1, Nz
    ! allocate 2d image if lambtv/tsv/rt > 0
    if (lambtv > 0 .or. lambtsv > 0 .or. lambrt > 0 .or. lambri > 0) then
      allocate(I2d(Nx,Ny))
      allocate(I2dl(Nx,Ny))
      allocate(I2du(Nx,Ny))
      I2d(:,:)=0d0
      I2dl(:,:)=0d0
      I2du(:,:)=0d0

      call I1d_I2d_fwd(xidx,yidx,Iin_reg((iz-1)*Npix+1:iz*Npix),I2d,Npix,Nx,Ny)
      ! get a former frame
      if (iz > 1) then
        call I1d_I2d_fwd(xidx,yidx,Iin_reg((iz-2)*Npix+1:(iz-1)*Npix),I2dl,Npix,Nx,Ny)
      end if
      ! get a latter frame
      if (iz < Nz) then
        call I1d_I2d_fwd(xidx,yidx,Iin_reg(iz*Npix+1:(iz+1)*Npix),I2du,Npix,Nx,Ny)
      end if
    end if

    reg_frm = 0d0
    allocate(gradreg_frm(Npix))
    gradreg_frm(:) = 0d0

    ! compute regularization function
    do ipix=1, Npix
      iparm = (iz-1)*Npix + ipix
      ! L1
      if (lambl1 > 0) then
        reg_frm = reg_frm + l1_e(Iin_reg(iparm))
        gradreg_frm(ipix) = gradreg_frm(ipix) + l1_grade(Iin_reg(iparm))
        reg = reg + lambl1 * l1_e(Iin_reg(iparm))
        gradreg(iparm) = gradreg(iparm) + lambl1 * l1_grade(Iin_reg(iparm))
      end if

      ! MEM
      if (lambmem > 0) then
        reg_frm = reg_frm + mem_e(Iin_reg(iparm))
        gradreg_frm(ipix) = gradreg_frm(ipix) + mem_grade(Iin_reg(iparm))
        reg = reg + lambmem * mem_e(Iin_reg(iparm))
        gradreg(iparm) = gradreg(iparm) + lambmem * mem_grade(Iin_reg(iparm))
      end if

      ! TV
      if (lambtv > 0) then
        reg_frm = reg_frm + tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg_frm(ipix) = gradreg_frm(ipix) + tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        reg = reg + lambtv * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambtv * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! TSV
      if (lambtsv > 0) then
        reg_frm = reg_frm + tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg_frm(ipix) = gradreg_frm(ipix) + tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        reg = reg + lambtsv * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambtsv * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! Dynamical Imaging
      if (lambrt > 0 .and. iz < Nz) then
        ! Rt from Dp (p=2) distance
        reg = reg + lambrt * d2(xidx(ipix),yidx(ipix),I2d,I2du,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambrt * rt_d2grad(xidx(ipix),yidx(ipix),iz,I2d,I2dl,I2du,Nx,Ny,Nz)
        ! Rt from Kullback-Leibler divergence
        !reg = reg + lambrt * dkl(xidx(ipix),yidx(ipix),I2d,I2du,Nx,Ny)
        !gradreg(iparm) = gradreg(iparm) + lambrt * rt_dklgrad(xidx(ipix),yidx(ipix),iz,I2d,I2dl,I2du,Nx,Ny,Nz)
      end if

      if (lambri > 0) then
        ! Ri from Dp (p=2) distance
        reg = reg + lambri * d2(xidx(ipix),yidx(ipix),I2d,Iavg2d,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambri * ri_d2grad(xidx(ipix),yidx(ipix),I2d,Iavg2d,Nx,Ny)
        ! Ri from Kullback-Leibler divergence
        !reg = reg + lambri * dkl(xidx(ipix),yidx(ipix),I2d,Iavg2d,Nx,Ny)
        !gradreg(iparm) = gradreg(iparm) + lambri * ri_dklgrad(xidx(ipix),yidx(ipix),I2d,Iavg2d,Itmp2d,Nx,Ny,Nz)
      end if
    end do

    regset(iz) = reg_frm
    gradregset((iz-1)*Npix+1:iz*Npix) = gradreg_frm

    ! deallocate I2d
    if (lambtv > 0 .or. lambtsv > 0 .or. lambrt > 0 .or. lambri > 0) then
      deallocate(I2d,I2dl,I2du)
    end if
    !
    deallocate(gradreg_frm)
  end do
  !$OMP END PARALLEL DO
  deallocate(Iin_reg)
  !
  if (lambri > 0) then
    deallocate(Iavg,Iavg2d)
    deallocate(Itmp,Itmp2d)
  end if

  !-------------------------------------------------------------------------------
  ! Constraints for interpolation between frames
  !-------------------------------------------------------------------------------
  !
  if (lambrs > 0) then
    allocate(gradrint_s(Nparm))
    gradrint_s(:) = 0d0

    ! continuity of image entropy
    do iz=1, Nz
      ! regularizer
      if (iz < Nz) then
        rint_s = rint_s + (regset(iz) - regset(iz+1))**2
      end if
      ! gradient of regularizer
      do ipix=1, Npix
        iparm = (iz-1)*Npix + ipix
        if (iz > 1) then
          gradrint_s(iparm) = gradrint_s(iparm) + 2*(regset(iz) - regset(iz-1))
        end if
        if (iz < Nz) then
          gradrint_s(iparm) = gradrint_s(iparm) + (regset(iz) - regset(iz+1))*gradregset(iparm)
        end if
      end do
    end do
    reg = reg + lambrs * rint_s
    gradreg = gradreg + lambrs * gradrint_s
    deallocate(regset,gradregset)
    deallocate(gradrint_s)
    !
    ! continuity of total flux
    !
  end if
  !

  ! multiply variable conversion factor to gradients
  if (transtype == 1) then
    ! Log Forward
    call log_grad(transprm,Iin,gradreg,Nparm)
  else if (transtype == 2) then
    ! Gamma contrast
    call gamma_grad(transprm,Iin,gradreg,Nparm)
  end if

  ! add regularization function and its gradient to cost function and its gradient.
  cost = cost + reg
  call daxpy(Nparm, 1d0, gradreg, 1, gradcost, 1) ! gradcost := gradreg + gradcos

  ! deallocate arrays
  deallocate(gradreg)

end subroutine
end module
