module fftim2dqu
  !$use omp_lib
  use param,  only: dp, dpc, deps
  use fftlib, only: calc_chisq, phashift_r2c
  use image,  only: calc_cost_reg
  implicit none
contains
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
subroutine imaging_qu(&
  Iin,Qin,Uin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
  u,v,&
  Niter,nonneg,nprint,&
  pl1_l, pl1_wgt, pl1_Nwgt,&
  ptv_l,  ptv_wgt,  ptv_Nwgt,&
  ptsv_l, ptsv_wgt, ptsv_Nwgt,&
  pkl_l, pkl_wgt, pkl_Nwgt,&
  pgs_l, pgs_wgt, pgs_Nwgt,&
  pkl_l, pkl_wgt, pkl_Nwgt,&
  mtv_l,  mtv_wgt,  mtv_Nwgt,&
  mtsv_l, mtsv_wgt, mtsv_Nwgt,&
  hwe_l, hwe_wgt, hew_mmax, hew_Nwgt&
  isqvis,uvidxqvis,qvisr,qvisi,Varqvis,wqvis,&
  isuvis,uvidxuvis,uvisr,uvisi,Varuvis,wuvis,&
  ismq,uvidxmq,mqr,mqi,Varmq,wmq,&
  ismu,uvidxmu,mur,mui,Varmu,wmu,&
  m,factr,pgtol,&
  Qout,Uout,&
  chisq, chisqqvis, chisquvis, chisqmq, chisqmu,&
  reg, l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, cen_cost, &
  cost, &
  Npix,Nuv,Nfcv,Namp,Ncp,Nca&
)
  !
  ! Core function of two-dimensional imaging
  !
  implicit none
  !
  ! Initial Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! combined uv coordinates
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy

  ! Imaging Parameter
  integer,  intent(in) :: Niter     ! iteration number
  logical,  intent(in) :: nonneg    ! if nonneg > 0, the image will be solved
                                    ! with a non-negative condition
  integer,  intent(in) :: nprint    ! interval

  ! Regularization parameters
  !   parameter for l1
  real(dp), intent(in)  :: l1_l               ! lambda
  integer,  intent(in)  :: l1_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: l1_wgt(l1_Nwgt)    ! weight
  !   parameter for total variation
  real(dp), intent(in)  :: tv_l               ! lambda
  integer,  intent(in)  :: tv_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: tv_wgt(tv_Nwgt)    ! weight
  !   parameter for total squared variation
  real(dp), intent(in)  :: tsv_l              ! lambda
  integer,  intent(in)  :: tsv_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: tsv_wgt(tsv_Nwgt)  ! weight
  !   parameter for kl divergence
  real(dp), intent(in)  :: kl_l               ! lambda
  integer,  intent(in)  :: kl_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: kl_wgt(kl_Nwgt)    ! weight
  !   parameter for Gull & Skilling entropy
  real(dp), intent(in)  :: gs_l               ! lambda
  integer,  intent(in)  :: gs_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: gs_wgt(gs_Nwgt)    ! weight
  !   parameter for the total flux density regularization
  real(dp), intent(in)  :: tfd_l              ! lambda (Normalized)
  real(dp), intent(in)  :: tfd_tgtfd          ! target total flux
  !   parameter for the centoroid regularization
  real(dp), intent(in)  :: cen_l              ! lambda (Normalized)
  real(dp), intent(in)  :: cen_alpha          ! alpha

  ! Parameters related to full complex visibilities
  logical,  intent(in) :: isfcv           ! is data?
  integer,  intent(in) :: Nfcv            ! number of data
  integer,  intent(in) :: uvidxfcv(Nfcv)  ! uvidx
  real(dp), intent(in) :: Vfcvr(Nfcv)     ! data
  real(dp), intent(in) :: Vfcvi(Nfcv)     ! data
  real(dp), intent(in) :: Varfcv(Nfcv)    ! variance
  real(dp), intent(in) :: wfcv            ! data weights

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

  ! Paramters related to the L-BFGS-B
  integer,  intent(in) :: m
  real(dp), intent(in) :: factr, pgtol
  !
  ! Output image
  real(dp), intent(out) :: Qout(Npix) ! Output Stokes Q image
  real(dp), intent(out) :: Uout(Npix) ! Output Stokes U image

  ! Outputs
  !   Chi-squares
  real(dp), intent(out) :: chisq    ! weighted sum of chisquares
  real(dp), intent(out) :: chisqqvis ! chisquare of Stokes Q visibilities
  real(dp), intent(out) :: chisquvis ! chisquare of Stokes U visibilities
  real(dp), intent(out) :: chisqmq   ! chisquare of Stokes Q polarimetric ratio
  real(dp), intent(out) :: chisqmu   ! chisquare of Stokes U polarimetric ratio

  !   Regularization Functions
  real(dp), intent(out) :: reg        ! total cost
  real(dp), intent(out) :: pl1_cost   ! cost of l1  for abs(p)
  real(dp), intent(out) :: ptv_cost   ! cost of tv  for abs(p)
  real(dp), intent(out) :: ptsv_cost  ! cost of tsv for abs(p)
  real(dp), intent(out) :: pkl_cost   ! cost of KL divergence   for abs(p)
  real(dp), intent(out) :: pgs_cost   ! cost of GS entropy      for abs(p)
  real(dp), intent(out) :: pctv_cost  ! cost of complex tv  for linear polarization
  real(dp), intent(out) :: pctsv_cost ! cost of complex tsv for linear polarization
  real(dp), intent(out) :: mtv_cost   ! cost of tv   for abs(m)
  real(dp), intent(out) :: mtsv_cost  ! cost of tsv  for abs(m)
  real(dp), intent(out) :: mctv_cost  ! cost of complex tv  for m
  real(dp), intent(out) :: mctsv_cost ! cost of complex tsv for m
  real(dp), intent(out) :: qtfd_cost  ! cost of total flux regularization for Q image
  real(dp), intent(out) :: utfd_cost  ! cost of total flux regularization for U image

  !   Total Cost function
  real(dp), intent(out) :: cost

  ! full complex visibilities to be used for calculations
  complex(dpc), allocatable :: Vfcv(:)

  ! chisquare and grad chisq
  real(dp) :: gradcost(1:Npix)  ! its gradient

  ! Number of Data
  integer   :: Ndata  ! number of data
  real(dp)  :: wfcv_n, wamp_n, wcp_n, wca_n, wtotal

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
  Ndata  = 0
  wtotal = 0
  if (isqvis .eqv. .True.) then
    Ndata = Ndata + 2 * Nqvis
    wtotal = wtotal + 2 * Nqvis * wqvis
  end if
  if (isuvis .eqv. .True.) then
    Ndata = Ndata + 2 * Nuvis
    wtotal = wtotal + 2 * Nuvis * wuvis
  end if
  if (ismq .eqv. .True.) then
    Ndata = Ndata + 2 * Nmq
    wtotal = wtotal + 2 * Nmq * wmq
  end if
  if (ismu .eqv. .True.) then
    Ndata = Ndata + 2 * Nmu
    wtotal = wtotal + 2 * Nmu * wmu
  end if
  ! compute weight
  wqvis_n = 0
  if (isqvis .eqv. .True.) then
    wqvis_n = wqvis_n / wtotal
  end if
  wuvis_n = 0
  if (isuvis .eqv. .True.) then
    wuvis_n = wuvis_n / wtotal
  end if
  wmq_n = 0
  if (ismq .eqv. .True.) then
    wmq_n = wmq / wtotal
  end if
  wmu_n = 0
  if (ismu .eqv. .True.) then
    wmu_n = wmu / wtotal
  end if

  write(*,*) 'Check Data and Weights'
  write(*,*) '  Number of Data          ', Ndata
  write(*,*) '  Number of uv coordinates', Nuv

  ! copy images (Iin -> Iout)
  write(*,*) 'Initialize the parameter vector'
  allocate(mphi(Npix*2))
  !    Here zeroeps is to avoid numerical error in m at Iin = 0
  call mphi_QU_inv(Iin+zeroeps,mphi(1:Npix),mphi(1+Npix:2*Npix),Qin,Uin,Npix)
  !    make m as nonnegative

  ! allocate complex vector for mq and mu
  allocate(mq(Nmq),mu(Nmu))
  mq = dcmplx(mqr,mqi)
  mu = dcmplx(mur,mui)
  ! precompute Stokes I vsibilities
  if ((isqvis .eqv. .True.) .or. (ismq .eqv. .True.)) then
    write(*,*) 'Precompute Stokes I visbilities for the polarimetric ratio'
    ! allocate array
    Nicmp = Nuv
    allocate(Icmp(Nicmp))

    ! do FFT
    allocate(I2d(Nx,Ny))
    I2d(:,:)=0d0
    call I1d_I2d_fwd(xidx,yidx,Iin(1:Npix),I2d,Npix,Nx,Ny)
    call NUFFT_fwd(u,v,I2d,Icmp,Nx,Ny,Nuv)
    deallocate(I2d)
  else
    Nicmp = 1
    allocate(Icmp(Nicmp))
  end if

  ! shift tracking center of full complex visibilities from the reference pixel
  ! to the center of the image
  allocate(qvis(Nqvis),uvis(Nqvis))
  qvis = dcmplx(qvisr,qvisi)
  uvis = dcmplx(uvisr,uvisi)
  if (isqvis .eqv. .True.) then
    write(*,*) 'Shift Tracking Center of Stokes Q complex visibilities.'
    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(u,v,Nxref,Nyref,uvidxqvis,Nx,Ny,Nqvis) &
    !$OMP   PRIVATE(i,u_tmp,v_tmp)
    do i=1,Nqvis
      u_tmp = u(abs(uvidxqvis(i))) * sign(1,uvidxqvis(i))
      v_tmp = v(abs(uvidxqvis(i))) * sign(1,uvidxqvis(i))
      call phashift_r2c(u_tmp,v_tmp,&
                        Nxref,Nyref,Nx,Ny,&
                        qvis(i),qvis(i))
    end do
    !$OMP END PARALLEL DO
  end if
  if (isuvis .eqv. .True.) then
    write(*,*) 'Shift Tracking Center of Stokes U complex visibilities.'
    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(u,v,Nxref,Nyref,uvidxuvis,Nx,Ny,Nuvis) &
    !$OMP   PRIVATE(i,u_tmp,v_tmp)
    do i=1,Nuvis
      u_tmp = u(abs(uvidxuvis(i))) * sign(1,uvidxuvis(i))
      v_tmp = v(abs(uvidxuvis(i))) * sign(1,uvidxuvis(i))
      call phashift_r2c(u_tmp,v_tmp,&
                        Nxref,Nyref,Nx,Ny,&
                        uvis(i),uvis(i))
    end do
    !$OMP END PARALLEL DO
  end if

  print '("Summary of the initial image")'
  call calc_cost(
  )

  !-------------------------------------
  ! L-BFGS-B
  !-------------------------------------
  write(*,*) 'Initialize the L-BFGS-B'
  ! initialise L-BFGS-B
  !   Allocate some arrays
  allocate(iwa(3*Npix*2))
  allocate(wa(2*m*Npix*2 + 5*Npix*2 + 11*m*m + 8*m))

  !   set boundary conditions
  allocate(lower(Npix*2),upper(Npix*2),nbd(Npix*2))
  nbd(:) = 0      ! no boundary conditions
  lower(:) = 0d0  ! put lower limit
  upper(:) = 1d0  ! put lower limit
  if (constm .eqv. .True.) then
    nbd(1:Npix) = 2       ! put lower & upper limit for m
    lower(1:Npix) = 0d0   ! put lower limit
    upper(1:Npix) = mmax  ! put upper limit
  else
    nbd(1:Npix) = 1       ! just add non-negative constraint on m
    lower(1:Npix) = 0d0   ! put only lower limit
  endif

  ! start L-BFGS-B
  write(*,*) 'Start L-BFGS-B calculations'
  task = 'START'
  do while(task(1:2) == 'FG' &
          .or. task == 'NEW_X' &
          .or. task == 'START')
    ! This is the call to the L-BFGS-B code.
    call setulb ( Npix, m, mphi, lower, upper, nbd, cost, gradcost, &
                  factr, pgtol, wa, iwa, task, iprint,&
                  csave, lsave, isave, dsave )

    if (task(1:2) == 'FG') then
      ! thresholding
      !where(abs(Iout)<zeroeps) Iout=0d0

      ! Calculate cost function and gradcostent of cost function
      call calc_cost(
      )
    else
      ! If iteration number exceeds the total iteration number, make a flag
      ! to STOP L-BFGS-B iterations
      if (isave(30) > Niter) then
        task='STOP: TOTAL ITERATION NUMBER EXCEEDS LIMIT'
      else if (mod(isave(30),nprint) == 0) then
        print '("Iteration :",I5,"/",I5)',isave(30),Niter
        call calc_cost(
        )
      end if
    end if
  end do

  ! Finary print summary again
  print '("Final Summary")'
  print '("  Iteration ent at ",I5,"/",I5)',isave(30),Niter
  call calc_cost(
  )
  write (6,*) task

  ! deallocate arrays
  deallocate(iwa,wa,lower,upper,nbd)
  !where(abs(Iout)<zeroeps) Iout=0d0
end subroutine
!
!-------------------------------------------------------------------------------
! calc cost functions
!-------------------------------------------------------------------------------
subroutine calc_cost(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
  u,v,&
  l1_l, l1_wgt, l1_Nwgt,&
  tv_l, tv_wgt, tv_Nwgt,&
  tsv_l, tsv_wgt, tsv_Nwgt,&
  kl_l, kl_wgt, kl_Nwgt,&
  gs_l, gs_wgt, gs_Nwgt,&
  tfd_l, tfd_tgtfd,&
  cen_l, cen_alpha,&
  isfcv,uvidxfcv,Vfcv,Varfcv,wfcv,&
  isamp,uvidxamp,Vamp,Varamp,wamp,&
  iscp,uvidxcp,CP,Varcp,wcp,&
  isca,uvidxca,CA,Varca,wca,&
  doprint,&
  chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
  reg, l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, cen_cost, &
  cost, gradcost, &
  Npix,Nuv,Nfcv,Namp,Ncp,Nca&
)
  !
  ! Calculate Cost Functions (for imaging_2d)
  !
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: mphi(Npix*2), Nxref, Nyref
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy

  ! Regularization Parameters
  !   parameter for l1
  real(dp), intent(in)  :: l1_l               ! lambda
  integer,  intent(in)  :: l1_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: l1_wgt(l1_Nwgt)    ! weight
  !   parameter for total variation
  real(dp), intent(in)  :: tv_l               ! lambda
  integer,  intent(in)  :: tv_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: tv_wgt(tv_Nwgt)    ! weight
  !   parameter for total squared variation
  real(dp), intent(in)  :: tsv_l              ! lambda
  integer,  intent(in)  :: tsv_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: tsv_wgt(tsv_Nwgt)  ! weight
  !   parameter for kl divergence
  real(dp), intent(in)  :: kl_l               ! lambda
  integer,  intent(in)  :: kl_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: kl_wgt(kl_Nwgt)    ! weight
  !   parameter for Gull & Skilling entropy
  real(dp), intent(in)  :: gs_l               ! lambda
  integer,  intent(in)  :: gs_Nwgt            ! size of the weight vector
  real(dp), intent(in)  :: gs_wgt(gs_Nwgt)    ! weight
  !   parameter for the total flux density regularization
  real(dp), intent(in)  :: tfd_l              ! lambda (Normalized)
  real(dp), intent(in)  :: tfd_tgtfd          ! target total flux
  !   parameter for the centoroid regularization
  real(dp), intent(in)  :: cen_l              ! lambda (Normalized)
  real(dp), intent(in)  :: cen_alpha          ! alpha

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

  ! print option
  integer,  intent(in) :: doprint   ! if doprint > 0 then print summary

  ! Outputs
  !   Chi-squares
  real(dp), intent(out) :: chisq    ! weighted sum of chisquares
  real(dp), intent(out) :: chisqfcv ! chisquare of full complex visibilities
  real(dp), intent(out) :: chisqamp ! chisquare of amplitudes
  real(dp), intent(out) :: chisqcp  ! chisquare of closure phases
  real(dp), intent(out) :: chisqca  ! chisquare of closure amplitudes

  !   Regularization Functions
  real(dp), intent(out) :: reg      ! total sum of the imaging cost
  real(dp), intent(out) :: l1_cost  ! cost of l1
  real(dp), intent(out) :: tv_cost  ! cost of tv
  real(dp), intent(out) :: tsv_cost ! cost of tsv
  real(dp), intent(out) :: kl_cost  ! cost of KL divergence
  real(dp), intent(out) :: gs_cost  ! cost of GS entropy
  real(dp), intent(out) :: tfd_cost ! cost of total flux regularization
  real(dp), intent(out) :: cen_cost ! cost of centoroid regularizaiton

  !   Total Cost function
  real(dp), intent(out) :: cost
  real(dp), intent(out) :: gradcost(1:Npix)

  ! Gradients of each term
  real(dp) :: gradreg(Npix), gradchisq(Npix)
  real(dp), allocatable :: Qin(:), Uin(:), Qg(:), Ug(:)

  !------------------------------------
  ! Convert m, theta to Q, U
  !------------------------------------
  call mphi_QU_fwd(Iin,mphi(1:Npix),mphi(1+Npix:2*Npix),Qin,Uin,Npix)

  !------------------------------------
  ! Initialize outputs, and some parameters
  !------------------------------------
  ! Initialize the chisquare and its gradient
  !write(*,*) 'stdftim.calc_cost: initialize cost and gradcost'
  cost        = 0d0
  gradcost(:) = 0d0

  call calc_chisq_qu(&
    Qin,Uin,xidx,yidx,Nx,Ny,&
    u,v,Icmp,&
    isqvis,uvidxqvis,qvis,Varqvis,wqvis,&
    isuvis,uvidxuvis,uvis,Varuvis,wuvis,&
    ismq,uvidxmq,mq,Varmq,wmq,&
    ismu,uvidxmu,mu,Varmu,wmu,&
    chisq, gradchisq, chisqqvis, chisquvis, chisqmq, chisqmu,&
    Npix,Nuv,Nicmp,Nqvis,Nuvis,Nmq,Nmu&
  )

  call calc_cost_reg(&
      Iin, xidx, yidx, Nxref, Nyref, Nx, Ny,&
      l1_l, l1_wgt, l1_Nwgt,&
      tv_l, tv_wgt, tv_Nwgt,&
      tsv_l, tsv_wgt, tsv_Nwgt,&
      kl_l, kl_wgt, kl_Nwgt,&
      gs_l, gs_wgt, gs_Nwgt,&
      tfd_l, tfd_tgtfd,&
      cen_l, cen_alpha,&
      l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost,&
      tfd_cost, cen_cost,&
      reg, gradreg, Npix)

  ! Finally take the sum of the cost and gradient functions
  cost     = chisq     + reg
  gradcost = gradchisq + gradreg

  ! Print summary if requested
  if (doprint > 0) then
    print '("  Cost Function          : ",D13.6)',cost

    print '("  Data term              : ",D13.6)',chisq
    if (isqvis .eqv. .True.) then
      print '("    Stokes Q Vis.       : ",D13.6)',chisqqvis
    end if
    if (isuvis .eqv. .True.) then
      print '("    Stokes U Vis.       : ",D13.6)',chisquvis
    end if
    if (ismq .eqv. .True.) then
      print '("    Stokes Q Frac. Vis. : ",D13.6)',chisqmq
    end if
    if (ismu .eqv. .True.) then
      print '("    Stokes U Frac. Vis. : ",D13.6)',chisqmu
    end if

    print '("  Reguarization term     : ",D13.6)',reg
    if (pl1_l > 0) then
      print '("    l1-norm of |P|         : ",D13.6)',pl1_cost
    end if
    if (ptv_l > 0) then
      print '("    TV of |P|              : ",D13.6)',ptv_cost
    end if
    if (ptsv_l > 0) then
      print '("    TSV of |P|             : ",D13.6)',ptsv_cost
    end if
    if (pkl_l > 0) then
      print '("    KL divergence of |P|   : ",D13.6)',pkl_cost
    end if
    if (pgs_l > 0) then
      print '("    GS entropy of |P|      : ",D13.6)',pgs_cost
    end if
    if (pctv_l > 0) then
      print '("    2D TV of P             : ",D13.6)',pctv_cost
    end if
    if (pctsv_l > 0) then
      print '("    2D TSV of P            : ",D13.6)',pctsv_cost
    end if
    if (mtv_l > 0) then
      print '("    TV of |m|              : ",D13.6)',mtv_cost
    end if
    if (mtsv_l > 0) then
      print '("    TSV of |m|             : ",D13.6)',mtsv_cost
    end if
    if (mctv_l > 0) then
      print '("    2D TV of m             : ",D13.6)',mctv_cost
    end if
    if (mctsv_l > 0) then
      print '("    2D TSV of m            : ",D13.6)',mctsv_cost
    end if
    if (qtfd_l > 0) then
      print '("    Total Flux Density of Q: ",D13.6)',qtfd_cost
    end if
    if (utfd_l > 0) then
      print '("    Total Flux Density of U: ",D13.6)',utfd_cost
    end if

    print '("  Total flux at Q : ",D13.6)',sum(Qin)
    print '("  Total flux at U : ",D13.6)',sum(Uin)
  end if
end subroutine
end module
