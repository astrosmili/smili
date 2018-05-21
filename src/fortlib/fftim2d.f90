module fftim2d
  !$use omp_lib
  use param, only: dp, dpc, deps
  use fftlib, only: NUFFT_fwd, NUFFT_adj_resid, phashift_r2c,&
                    chisq_fcv, chisq_amp, chisq_ca, chisq_cp
  use image, only: I1d_I2d_fwd, I1d_I2d_inv,&
                   log_fwd, log_grad,&
                   gamma_fwd, gamma_grad,&
                   l1_e, l1_grade,&
                   tv_e, tv_grade,&
                   tsv_e, tsv_grade,&
                   mem_e, mem_grade,&
                   comreg, zeroeps
  implicit none
contains
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
subroutine imaging(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
  u,v,&
  lambl1,lambtv,lambtsv,lambmem,lambcom,doweight,&
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
  ! Core function of two-dimensional imaging
  !
  implicit none
  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! uv coordinates
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy

  ! Regularization Parameters
  real(dp), intent(in) :: lambl1  ! Regularization Parameter for L1-norm
  real(dp), intent(in) :: lambtv  ! Regularization Parameter for iso-TV
  real(dp), intent(in) :: lambtsv ! Regularization Parameter for TSV
  real(dp), intent(in) :: lambmem ! Regularization Parameter for MEM
  real(dp), intent(in) :: lambcom ! Regularization Parameter for Center of Mass
  integer,  intent(in) :: doweight ! if postive, reweight images

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
  logical,      intent(in) :: isfcv           ! is data?
  integer,      intent(in) :: Nfcv            ! number of data
  integer,      intent(in) :: uvidxfcv(Nfcv)  ! uvidx
  real(dp),     intent(in) :: Vfcvr(Nfcv)     ! data
  real(dp),     intent(in) :: Vfcvi(Nfcv)     ! data
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

  ! Paramters related to the L-BFGS-B
  integer,  intent(in) :: m
  real(dp), intent(in) :: factr, pgtol
  !
  ! Output Image
  real(dp), intent(out) :: Iout(Npix)

  ! Reweighting factor for Images
  real(dp), allocatable :: l1_w(:), tv_w(:), tsv_w(:), I2d(:,:)
  real(dp) :: l1_w_norm, tv_w_norm, tsv_w_norm

  ! full complex visibilities to be used for calculations
  complex(dpc), allocatable :: Vfcv(:)

  ! chisquare and grad chisq
  real(dp) :: cost              ! cost function
  real(dp) :: gradcost(1:Npix)  ! its gradient

  ! Number of Data
  integer :: Ndata  ! number of data
  real(dp) :: fnorm ! normalization factor for chisquares

  ! variables and parameters tuning L-BFGS-B
  integer,  parameter   :: iprint = -1
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
  write(*,*) 'Number of uv coordinates', Nuv

  ! copy images (Iin -> Iout)
  write(*,*) 'Initialize the parameter vector'
  call dcopy(Npix,Iin,1,Iout,1)

  ! shift tracking center of full complex visibilities from the reference pixel
  ! to the center of the image
  allocate(Vfcv(Nfcv))
  Vfcv = dcmplx(Vfcvr,Vfcvi)
  if (isfcv .eqv. .True.) then
    write(*,*) 'Shift Tracking Center of Full complex visibilities.'
    !write(*,*) 'Vfcv before',Vfcv(1)
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
    !write(*,*) 'Vfcv after ',Vfcv(1)
  end if


  !-------------------------------------
  ! Reweighting factor for l1, tv, tsv
  !-------------------------------------
  allocate(l1_w(Npix),tv_w(Npix),tsv_w(Npix))
  l1_w = 1
  tv_w = 1
  tsv_w = 1

  if (doweight > 0 ) then
    write(*,*) 'Calculating re-weighting factor for l1, tv, tsv regularizations'
    allocate(I2d(Nx,Ny))
    call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)
    l1_w_norm=0
    tv_w_norm=0
    tsv_w_norm=0
    do i=1, Npix
      l1_w(i) = 1/(l1_e(Iin(i))+zeroeps)
      tv_w(i) = 1/(tv_e(xidx(i),yidx(i),I2d,Nx,Ny)+zeroeps)
      tsv_w(i) = 1/(l1_e(Iin(i)*Iin(i))+zeroeps)
      l1_w_norm = l1_w_norm + l1_e(Iin(i))*l1_w(i)
      tv_w_norm = tv_w_norm + tv_e(xidx(i),yidx(i),I2d,Nx,Ny)*tv_w(i)
      tsv_w_norm = tsv_w_norm + tsv_e(xidx(i),yidx(i),I2d,Nx,Ny)*tsv_w(i)
    end do
    l1_w = l1_w/l1_w_norm
    tv_w = tv_w/tv_w_norm
    tsv_w = tsv_w/tsv_w_norm
    deallocate(I2d)
  end if
  !-------------------------------------
  ! L-BFGS-B
  !-------------------------------------
  write(*,*) 'Initialize the L-BFGS-B'
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
  write(*,*) 'Start L-BFGS-B calculations'
  task = 'START'
  do while(task(1:2) == 'FG' &
          .or. task == 'NEW_X' &
          .or. task == 'START')
    ! This is the call to the L-BFGS-B code.
    call setulb ( Npix, m, Iout, lower, upper, nbd, cost, gradcost, &
                  factr, pgtol, wa, iwa, task, iprint,&
                  csave, lsave, isave, dsave )

    if (task(1:2) == 'FG') then
      ! thresholding
      !where(abs(Iout)<zeroeps) Iout=0d0

      ! Calculate cost function and gradcostent of cost function
      call calc_cost(&
        Iout,xidx,yidx,Nxref,Nyref,Nx,Ny,&
        u,v,&
        lambl1,lambtv,lambtsv,lambmem,lambcom,&
        doweight,l1_w,tv_w,tsv_w,&
        fnorm,transtype,transprm,pcom,&
        isfcv,uvidxfcv,Vfcv,Varfcv,&
        isamp,uvidxamp,Vamp,Varamp,&
        iscp,uvidxcp,CP,Varcp,&
        isca,uvidxca,CA,Varca,&
        cost,gradcost,&
        Npix,Nuv,Nfcv,Namp,Ncp,Nca&
      )
    else
      ! If iteration number exceeds the total iteration number, make a flag
      ! to STOP L-BFGS-B iterations
      if (isave(30) > Niter) then
        task='STOP: TOTAL ITERATION NUMBER EXCEEDS LIMIT'
      else if (mod(isave(30),100) == 0) then
        print '("Iteration :",I5,"/",I5,"  Cost :",D13.6)',isave(30),Niter,cost
        !print '("Iteration :",I5,"/",I5)',isave(30),Niter
        !print '("  Cost Function         ",D13.6)',cost
        !print '("  Chi-Square Total      ",D13.6)',chisq
        !print '("    Complex Visibilities",D13.6)',chisq_fcv
        !print '("    Amplitudes          ",D13.6)',chisq_amp
        !print '("    Closure Phase       ",D13.6)',chisq_cp
        !print '("    Closure Amplitudes  ",D13.6)',chisq_ca
        !print '("  Reguarization Function",D13.6)',reg
        !print '("    Center of Mass:     ",D13.6)',reg_com
        !print '("    l1-norm             ",D13.6)',reg_l1
        !print '("    tv                  ",D13.6)',reg_tv
        !print '("    tsv                 ",D13.6)',reg_tsv
        !print '("    mem                 ",D13.6)',reg_mem
      end if
    end if
  end do
  print '("Iteration :",I5,"/",I5,"  Cost :",D13.6)',isave(30),Niter,cost
  write (6,*) task

  ! deallocate arrays
  deallocate(Vfcv)
  deallocate(iwa,wa,lower,upper,nbd)
  deallocate(l1_w,tv_w,tsv_w)
  !where(abs(Iout)<zeroeps) Iout=0d0
end subroutine
!
!-------------------------------------------------------------------------------
! calc cost functions
!-------------------------------------------------------------------------------
!
subroutine calc_cost(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
  u,v,&
  lambl1,lambtv,lambtsv,lambmem,lambcom,&
  doweight,l1_w,tv_w,tsv_w,&
  fnorm,transtype,transprm,pcom,&
  isfcv,uvidxfcv,Vfcv,Varfcv,&
  isamp,uvidxamp,Vamp,Varamp,&
  iscp,uvidxcp,CP,Varcp,&
  isca,uvidxca,CA,Varca,&
  cost,gradcost,&
  Npix,Nuv,Nfcv,Namp,Ncp,Nca&
)
  !
  ! Calculate Cost Functions (for imaging_2d)
  !
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix), Nxref, Nyref
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy

  ! Regularization Parameters
  real(dp), intent(in) :: lambl1  ! Regularization Parameter for L1-norm
  real(dp), intent(in) :: lambtv  ! Regularization Parameter for iso-TV
  real(dp), intent(in) :: lambtsv ! Regularization Parameter for TSV
  real(dp), intent(in) :: lambmem ! Regularization Parameter for MEM
  real(dp), intent(in) :: lambcom ! Regularization Parameter for Center of Mass

  ! Reweighting
  integer,  intent(in) :: doweight ! if postive, reweight l1,tsv,tv terms
  real(dp), intent(in) :: l1_w(Npix), tv_w(Npix), tsv_w(Npix) ! reweight

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
  real(dp), intent(out) :: gradcost(1:Npix)

  ! integer
  integer :: ipix

  ! chisquares, gradients of each term of equations
  real(dp) :: chisq, reg  ! chisquare and regularization

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:), Iin_reg(:)
  real(dp), allocatable :: gradchisq2d(:,:)
  real(dp), allocatable :: gradreg(:)
  real(dp), allocatable :: Vresre(:),Vresim(:)
  complex(dpc), allocatable :: Vcmp(:)

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
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  call NUFFT_adj_resid(u,v,Vresre,Vresim,gradchisq2d(:,:),Nx,Ny,Nuv)
  deallocate(Vresre,Vresim)

  ! copy the gradient of chisquare into that of cost functions
  cost = chisq
  call I1d_I2d_inv(xidx,yidx,gradcost,gradchisq2d,Npix,Nx,Ny)
  deallocate(gradchisq2d)

  !------------------------------------
  ! Centoroid Regularizer
  !------------------------------------
  if (lambcom > 0) then
    ! initialize
    !   scalars
    reg = 0d0
    !   allocatable arrays
    allocate(gradreg(Npix))
    gradreg(:) = 0d0

    ! calc cost and its gradient
    call comreg(xidx,yidx,Nxref,Nyref,pcom,Iin,reg,gradreg,Npix)
    cost = cost + lambcom * reg
    call daxpy(Npix, lambcom, gradreg, 1, gradcost, 1) ! gradcost := lambcom * gradreg + gradcost

    ! deallocate array
    deallocate(gradreg)
  end if

  !------------------------------------
  ! Regularization Functions
  !------------------------------------
  ! Initialize
  !   scalars
  reg = 0d0
  !   allocatable arrays
  allocate(gradreg(Npix),Iin_reg(Npix))
  gradreg(:) = 0d0
  Iin_reg(:) = 0d0
  if (lambtv > 0 .or. lambtsv > 0) then
    allocate(I2d(Nx,Ny))
    I2d(:,:)=0d0
  end if

  ! Transform Image
  if (transtype == 1) then
    ! Log Forward
    call log_fwd(transprm,Iin,Iin_reg,Npix)
  else if (transtype == 2) then
    ! Gamma contrast
    call gamma_fwd(transprm,Iin,Iin_reg,Npix)
  else
    call dcopy(Npix,Iin,1,Iin_reg,1)
  end if

  ! Copy transformed image to I2d
  if (lambtv > 0 .or. lambtsv > 0) then
    call I1d_I2d_fwd(xidx,yidx,Iin_reg,I2d,Npix,Nx,Ny)
  end if

  ! Compute regularization term
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, Iin_reg, lambl1, lambmem, lambtv, lambtsv,&
  !$OMP                I2d, xidx, yidx, l1_w, tv_w, tsv_w) &
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+: reg, gradreg)
  do ipix=1, Npix
    if (doweight > 0) then
      ! L1
      if (lambl1 > 0) then
        reg = reg + lambl1 * l1_w(ipix) * l1_e(Iin_reg(ipix))
        gradreg(ipix) = gradreg(ipix) + lambl1 * l1_w(ipix) * l1_grade(Iin_reg(ipix))
      end if

      ! TV
      if (lambtv > 0) then
        reg = reg + lambtv * tv_w(ipix) * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(ipix) = gradreg(ipix) + lambtv * tv_w(ipix) * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! TSV
      if (lambtsv > 0) then
        reg = reg + lambtsv * tsv_w(ipix) * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(ipix) = gradreg(ipix) + lambtsv * tsv_w(ipix) * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if
    else
      ! L1
      if (lambl1 > 0) then
        reg = reg + lambl1 * l1_e(Iin_reg(ipix))
        gradreg(ipix) = gradreg(ipix) + lambl1 * l1_grade(Iin_reg(ipix))
      end if

      ! TV
      if (lambtv > 0) then
        reg = reg + lambtv * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(ipix) = gradreg(ipix) + lambtv * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! TSV
      if (lambtsv > 0) then
        reg = reg + lambtsv * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(ipix) = gradreg(ipix) + lambtsv * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if
    end if

    ! MEM
    if (lambmem > 0) then
      reg = reg + lambmem * mem_e(Iin_reg(ipix))
      gradreg(ipix) = gradreg(ipix) + lambmem * mem_grade(Iin_reg(ipix))
    end if
  end do
  !$OMP END PARALLEL DO

  ! multiply variable conversion factor to gradients
  if (transtype == 1) then
    ! Log Forward
    call log_grad(transprm,Iin,gradreg,Npix)
  else if (transtype == 2) then
    ! Gamma contrast
    call gamma_grad(transprm,Iin,gradreg,Npix)
  end if

  ! add regularization function and its gradient to cost function and its gradient.
  cost = cost + reg
  call daxpy(Npix, 1d0, gradreg, 1, gradcost, 1) ! gradcost := gradreg + gradcos

  ! deallocate arrays
  deallocate(gradreg,Iin_reg)
  if (lambtv > 0 .or. lambtsv > 0) then
    deallocate(I2d)
  end if
end subroutine
end module
