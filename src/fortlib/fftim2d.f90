module fftim2d
  !$use omp_lib
  use param,  only: dp, dpc, deps
  use fftlib, only: calc_chisq, phashift_r2c
  use image,  only: calc_cost_reg
  implicit none
contains
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
subroutine imaging(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
  u,v,&
  Niter,nonneg,nprint,&
  l1_l, l1_wgt, l1_Nwgt,&
  tv_l, tv_wgt, tv_Nwgt,&
  tsv_l, tsv_wgt, tsv_Nwgt,&
  kl_l, kl_wgt, kl_Nwgt,&
  gs_l, gs_wgt, gs_Nwgt,&
  tfd_l, tfd_tgtfd,&
  cen_l, cen_alpha,&
  isfcv,uvidxfcv,Vfcvr,Vfcvi,Varfcv,wfcv,&
  isamp,uvidxamp,Vamp,Varamp,wamp,&
  iscp,uvidxcp,CP,Varcp,wcp,&
  isca,uvidxca,CA,Varca,wca,&
  m,factr,pgtol,&
  Iout,&
  chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
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
  real(dp), intent(out) :: Iout(Npix)
  ! Outputs
  !   Chi-squares
  real(dp), intent(out) :: chisq    ! weighted sum of chisquares
  real(dp), intent(out) :: chisqfcv ! chisquare of full complex visibilities
  real(dp), intent(out) :: chisqamp ! chisquare of amplitudes
  real(dp), intent(out) :: chisqcp  ! chisquare of closure phases
  real(dp), intent(out) :: chisqca  ! chisquare of closure amplitudes

  !   Regularization Functions
  real(dp), intent(out) :: reg      ! total cost
  real(dp), intent(out) :: l1_cost  ! cost of l1
  real(dp), intent(out) :: tv_cost  ! cost of tv
  real(dp), intent(out) :: tsv_cost ! cost of tsv
  real(dp), intent(out) :: kl_cost  ! cost of KL divergence
  real(dp), intent(out) :: gs_cost  ! cost of GS entropy
  real(dp), intent(out) :: tfd_cost ! cost of total flux regularization
  real(dp), intent(out) :: cen_cost ! cost of centoroid regularizaiton

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
  if (isfcv .eqv. .True.) then
    Ndata = Ndata + 2 * Nfcv
    wtotal = wtotal + 2 * Nfcv * wfcv
  end if
  if (isamp .eqv. .True.) then
    Ndata = Ndata + Namp
    wtotal = wtotal + Namp * wamp
  end if
  if (iscp .eqv. .True.) then
    Ndata = Ndata + Ncp
    wtotal = wtotal + Ncp * wcp
  end if
  if (isca .eqv. .True.) then
    Ndata = Ndata + Nca
    wtotal = wtotal + Nca * wca
  end if
  ! compute weight
  wfcv_n = 0
  if (isfcv .eqv. .True.) then
    wfcv_n = wfcv / wtotal
  end if
  wamp_n = 0
  if (isamp .eqv. .True.) then
    wamp_n = wamp / wtotal
  end if
  wcp_n = 0
  if (iscp .eqv. .True.) then
    wcp_n = wcp / wtotal
  end if
  wca_n = 0
  if (isca .eqv. .True.) then
    wca_n = wca / wtotal
  end if
  write(*,*) 'Check Data and Weights'
  write(*,*) '  Number of Data          ', Ndata
  write(*,*) '  Number of uv coordinates', Nuv

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

  print '("Summary of the initial image")'
  call calc_cost(&
    Iout,xidx,yidx,Nxref,Nyref,Nx,Ny,&
    u,v,&
    l1_l, l1_wgt, l1_Nwgt,&
    tv_l, tv_wgt, tv_Nwgt,&
    tsv_l, tsv_wgt, tsv_Nwgt,&
    kl_l, kl_wgt, kl_Nwgt,&
    gs_l, gs_wgt, gs_Nwgt,&
    tfd_l, tfd_tgtfd,&
    cen_l, cen_alpha,&
    isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
    isamp,uvidxamp,Vamp,Varamp,wamp_n,&
    iscp,uvidxcp,CP,Varcp,wcp_n,&
    isca,uvidxca,CA,Varca,wca_n,&
    1,&
    chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
    reg, l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, cen_cost, &
    cost, gradcost, &
    Npix,Nuv,Nfcv,Namp,Ncp,Nca&
  )

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
        l1_l, l1_wgt, l1_Nwgt,&
        tv_l, tv_wgt, tv_Nwgt,&
        tsv_l, tsv_wgt, tsv_Nwgt,&
        kl_l, kl_wgt, kl_Nwgt,&
        gs_l, gs_wgt, gs_Nwgt,&
        tfd_l, tfd_tgtfd,&
        cen_l, cen_alpha,&
        isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
        isamp,uvidxamp,Vamp,Varamp,wamp_n,&
        iscp,uvidxcp,CP,Varcp,wcp_n,&
        isca,uvidxca,CA,Varca,wca_n,&
        -1,&
        chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
        reg, l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, cen_cost, &
        cost, gradcost, &
        Npix,Nuv,Nfcv,Namp,Ncp,Nca&
      )
    else
      ! If iteration number exceeds the total iteration number, make a flag
      ! to STOP L-BFGS-B iterations
      if (isave(30) > Niter) then
        task='STOP: TOTAL ITERATION NUMBER EXCEEDS LIMIT'
      else if (mod(isave(30),nprint) == 0) then
        print '("Iteration :",I5,"/",I5)',isave(30),Niter
        call calc_cost(&
          Iout,xidx,yidx,Nxref,Nyref,Nx,Ny,&
          u,v,&
          l1_l, l1_wgt, l1_Nwgt,&
          tv_l, tv_wgt, tv_Nwgt,&
          tsv_l, tsv_wgt, tsv_Nwgt,&
          kl_l, kl_wgt, kl_Nwgt,&
          gs_l, gs_wgt, gs_Nwgt,&
          tfd_l, tfd_tgtfd,&
          cen_l, cen_alpha,&
          isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
          isamp,uvidxamp,Vamp,Varamp,wamp_n,&
          iscp,uvidxcp,CP,Varcp,wcp_n,&
          isca,uvidxca,CA,Varca,wca_n,&
          1,&
          chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
          reg, l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, cen_cost, &
          cost, gradcost, &
          Npix,Nuv,Nfcv,Namp,Ncp,Nca&
        )
      end if
    end if
  end do

  ! Finary print summary again
  print '("Final Summary")'
  print '("  Iteration ent at ",I5,"/",I5)',isave(30),Niter
  call calc_cost(&
    Iout,xidx,yidx,Nxref,Nyref,Nx,Ny,&
    u,v,&
    l1_l, l1_wgt, l1_Nwgt,&
    tv_l, tv_wgt, tv_Nwgt,&
    tsv_l, tsv_wgt, tsv_Nwgt,&
    kl_l, kl_wgt, kl_Nwgt,&
    gs_l, gs_wgt, gs_Nwgt,&
    tfd_l, tfd_tgtfd,&
    cen_l, cen_alpha,&
    isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
    isamp,uvidxamp,Vamp,Varamp,wamp_n,&
    iscp,uvidxcp,CP,Varcp,wcp_n,&
    isca,uvidxca,CA,Varca,wca_n,&
    1,&
    chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
    reg, l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, cen_cost, &
    cost, gradcost, &
    Npix,Nuv,Nfcv,Namp,Ncp,Nca&
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
  real(dp), intent(in) :: Iin(Npix), Nxref, Nyref
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

  !------------------------------------
  ! Initialize outputs, and some parameters
  !------------------------------------
  ! Initialize the chisquare and its gradient
  !write(*,*) 'stdftim.calc_cost: initialize cost and gradcost'
  cost        = 0d0
  gradcost(:) = 0d0

  call calc_chisq(&
    Iin,xidx,yidx,Nx,Ny,&
    u,v,&
    isfcv,uvidxfcv,Vfcv,Varfcv,wfcv,&
    isamp,uvidxamp,Vamp,Varamp,wamp,&
    iscp,uvidxcp,CP,Varcp,wcp,&
    isca,uvidxca,CA,Varca,wca,&
    chisq, gradchisq, chisqfcv, chisqamp, chisqcp, chisqca,&
    Npix,Nuv,Nfcv,Namp,Ncp,Nca &
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
    if (isfcv .eqv. .True.) then
      print '("    Complex Visibilities : ",D13.6)',chisqfcv
    end if
    if (isamp .eqv. .True.) then
      print '("    Amplitudes           : ",D13.6)',chisqamp
    end if
    if (iscp .eqv. .True.) then
      print '("    Closure Phase        : ",D13.6)',chisqcp
    end if
    if (isca .eqv. .True.) then
      print '("    Closure Amplitudes   : ",D13.6)',chisqca
    end if

    print '("  Reguarization term     : ",D13.6)',reg
    if (l1_l > 0) then
      print '("    l1-norm              : ",D13.6)',l1_cost
    end if
    if (tv_l > 0) then
      print '("    TV                   : ",D13.6)',tv_cost
    end if
    if (tsv_l > 0) then
      print '("    TSV                  : ",D13.6)',tsv_cost
    end if
    if (kl_l > 0) then
      print '("    KL divergence        : ",D13.6)',kl_cost
    end if
    if (gs_l > 0) then
      print '("    GS entropy           : ",D13.6)',gs_cost
    end if
    if (tfd_l > 0) then
      print '("    Total Flux Density   : ",D13.6)',tfd_cost
    end if
    if (cen_l > 0) then
      print '("    Centoroid Reg.       : ",D13.6)',cen_cost
    endif

    print '("  Total flux             : ",D13.6)',sum(Iin)
  end if
end subroutine
end module
