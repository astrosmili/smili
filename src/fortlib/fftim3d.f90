module fftim3d
  !$use omp_lib
  use param, only: dp, dpc, deps
  use fftlib, only: calc_chisq, phashift_r2c
  use fftlib3d, only: calc_chisq3d
  use image, only: calc_cost_reg, zeroeps
  use image3d, only: calc_cost_reg3d
  implicit none
contains
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
subroutine imaging(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
  u,v,Nuvs,&
  Niter,nonneg,nprint,&
  l1_l, l1_wgt, l1_Nwgt,&
  sm_l, sm_maj, sm_min, sm_phi,&
  tv_l, tv_wgt, tv_Nwgt,&
  tsv_l, tsv_wgt, tsv_Nwgt,&
  kl_l, kl_wgt, kl_Nwgt,&
  gs_l, gs_wgt, gs_Nwgt,&
  tfd_l, tfd_tgtfd,&
  lc_l, lc_tgtfd,lc_nidx,&
  cen_l, cen_alpha,&
  rt_l, ri_l, rs_l, rf_l, &
  isfcv,uvidxfcv,Vfcvr,Vfcvi,Varfcv,wfcv,&
  isamp,uvidxamp,Vamp,Varamp,wamp,&
  iscp,uvidxcp,CP,Varcp,wcp,&
  isca,uvidxca,CA,Varca,wca,&
  inormfactr,&
  m,factr,pgtol,&
  Iout,&
  chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
  reg, l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, lc_cost, cen_cost, &
  rt_cost, ri_cost, rs_cost, rf_cost, &
  cost, &
  Npix,Nuv,Nfcv,Namp,Ncp,Nca&
)
  !
  ! Core function of three-dimensional imaging
  !
  implicit none
  !
  ! Initial Image
  integer,  intent(in) :: Npix, Nx, Ny, Nz
  real(dp), intent(in) :: Iin(Npix*Nz)
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! combined uv coordinates
  integer,  intent(in) :: Nuv
  integer,  intent(in) :: Nuvs(Nz)        ! number of uv data for each frame
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
  !   parameter for second moment
  real(dp), intent(in)  :: sm_l               ! lambda
  real(dp), intent(in)  :: sm_maj,sm_min,sm_phi ! major and minor size and position angle
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
  !   parameter for the light curve regularization
  real(dp), intent(in)  :: lc_l(Nz)           ! lambda array (Normalized)
  real(dp), intent(in)  :: lc_tgtfd(Nz)       ! target light curve
  integer,  intent(in)  :: lc_nidx            ! regularizer normalization with lc_tgtfd
  !   parameter for the centoroid regularization
  real(dp), intent(in)  :: cen_l              ! lambda (Normalized)
  real(dp), intent(in)  :: cen_alpha          ! alpha

  ! Regularization parameters of dynamical imaging
  real(dp), intent(in) :: rt_l    ! Regularization Parameter for Dynamical Imaging (delta-t)
  real(dp), intent(in) :: ri_l    ! Regularization Parameter for Dynamical Imaging (delta-I)
  real(dp), intent(in) :: rs_l    ! Regularization Parameter for Dynamical Imaging (entropy continuity)
  real(dp), intent(in) :: rf_l    ! Regularization Parameter for Dynamical Imaging (total flux continuity)

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

  ! Intensity Scale Parameter
  real(dp), intent(in) :: inormfactr

  ! Paramters related to the L-BFGS-B
  integer,  intent(in) :: m
  real(dp), intent(in) :: factr, pgtol
  !
  ! Output Image
  real(dp), intent(out) :: Iout(Npix*Nz)

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
  real(dp), intent(out) :: sm_cost  ! cost of second moment
  real(dp), intent(out) :: tv_cost  ! cost of tv
  real(dp), intent(out) :: tsv_cost ! cost of tsv
  real(dp), intent(out) :: kl_cost  ! cost of KL divergence
  real(dp), intent(out) :: gs_cost  ! cost of GS entropy
  real(dp), intent(out) :: tfd_cost ! cost of total flux regularization
  real(dp), intent(out) :: lc_cost  ! cost of light curve regularization
  real(dp), intent(out) :: cen_cost ! cost of centoroid regularizaiton

  ! Regularization Functions for dynamical imaging
  real(dp), intent(out) :: rt_cost  ! cost of Rt reularization
  real(dp), intent(out) :: ri_cost  ! cost of Ri regularization
  real(dp), intent(out) :: rs_cost  ! cost of Rs regularization
  real(dp), intent(out) :: rf_cost  ! cost of Rf regularization

  !   Total Cost function
  real(dp), intent(out) :: cost

  ! full complex visibilities to be used for calculations
  complex(dpc), allocatable :: Vfcv(:)

  ! chisquare and grad chisq
  real(dp) :: gradcost(1:Npix*Nz)  ! its gradient

  ! Number of Data
  integer :: Ndata, Npixz   ! number of data and (pixel * time bin)
  real(dp)  :: wfcv_n, wamp_n, wcp_n, wca_n, wtotal

  ! Index of (u,v) data for all time frames
  integer :: Nuvs_sum(Nz)

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
  ! Number of Parameters
  Npixz = Npix * Nz

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
  write(*,*) '  Number of Data           ', Ndata
  write(*,*) 'Number of pixel/frame      ', Npix
  write(*,*) 'Number of Frames           ', Nz
  write(*,*) '  Number of uv coordinates ', Nuv

  ! copy images (Iin -> Iout)
  write(*,*) 'Initialize the parameter vector'
  call dcopy(Npixz,Iin,1,Iout,1)

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

  ! Compute accumulated number of uvdata before each frame
  !   Nuvs_sum(i) + 1 will be the start index number for i-th frame
  !   Nuvs_sum(i) + Nuvs(i) will be the end index number for i-th frame
  Nuvs_sum(1)=0
  do i=2, Nz
    Nuvs_sum(i) = Nuvs_sum(i-1) + Nuvs(i-1)
  end do

  print '("Summary of the initial image")'
  call calc_cost(&
    Iout,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
    u, v, Nuvs,Nuvs_sum,&
    l1_l, l1_wgt, l1_Nwgt,&
    sm_l, sm_maj, sm_min, sm_phi,&
    tv_l, tv_wgt, tv_Nwgt,&
    tsv_l, tsv_wgt, tsv_Nwgt,&
    kl_l, kl_wgt, kl_Nwgt,&
    gs_l, gs_wgt, gs_Nwgt,&
    tfd_l, tfd_tgtfd,&
    lc_l, lc_tgtfd,lc_nidx,&
    cen_l, cen_alpha,&
    rt_l,ri_l,rs_l, rf_l, &
    isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
    isamp,uvidxamp,Vamp,Varamp,wamp_n,&
    iscp,uvidxcp,CP,Varcp,wcp_n,&
    isca,uvidxca,CA,Varca,wca_n,&
    1,inormfactr,&
    chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
    reg, l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, lc_cost, cen_cost, &
    rt_cost, ri_cost, rs_cost, rf_cost, &
    cost, gradcost, &
    Npix, Nuv, Nfcv, Namp, Ncp, Nca &
  )


  !-------------------------------------
  ! L-BFGS-B
  !-------------------------------------
  write(*,*) 'Initialize the L-BFGS-B'
  ! initialise L-BFGS-B
  !   Allocate some arrays
  allocate(iwa(3*Npixz))
  allocate(wa(2*m*Npixz + 5*Npixz + 11*m*m + 8*m))

  !   set boundary conditions
  allocate(lower(Npixz),upper(Npixz),nbd(Npixz))
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
    call setulb ( Npixz, m, Iout, lower, upper, nbd, cost, gradcost, &
                  factr, pgtol, wa, iwa, task, iprint,&
                  csave, lsave, isave, dsave )

    if (task(1:2) == 'FG') then
      ! thresholding
      !where(abs(Iout)<zeroeps) Iout=0d0

      ! Calculate cost function and gradcostent of cost function
      call calc_cost(&
        Iout,xidx,yidx,Nxref,Nyref,Nx,Ny, Nz,&
        u,v,Nuvs,Nuvs_sum,&
        l1_l, l1_wgt, l1_Nwgt,&
        sm_l, sm_maj, sm_min, sm_phi,&
        tv_l, tv_wgt, tv_Nwgt,&
        tsv_l, tsv_wgt, tsv_Nwgt,&
        kl_l, kl_wgt, kl_Nwgt,&
        gs_l, gs_wgt, gs_Nwgt,&
        tfd_l, tfd_tgtfd,&
        lc_l, lc_tgtfd,lc_nidx,&
        cen_l, cen_alpha,&
        rt_l, ri_l, rs_l, rf_l, &
        isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
        isamp,uvidxamp,Vamp,Varamp,wamp_n,&
        iscp,uvidxcp,CP,Varcp,wcp_n,&
        isca,uvidxca,CA,Varca,wca_n,&
        -1,inormfactr,&
        chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
        reg, l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, lc_cost, cen_cost, &
        rt_cost, ri_cost, rs_cost, rf_cost, &
        cost, gradcost, &
        Npix, Nuv, Nfcv, Namp, Ncp, Nca&
      )

    else
      ! If iteration number exceeds the total iteration number, make a flag
      ! to STOP L-BFGS-B iterations
      if (isave(30) > Niter) then
        task='STOP: TOTAL ITERATION NUMBER EXCEEDS LIMIT'
      else if (mod(isave(30),nprint) == 0) then
        print '("Iteration :",I5,"/",I5)',isave(30),Niter
        call calc_cost(&
          Iout,xidx,yidx,Nxref,Nyref,Nx,Ny, Nz,&
          u,v,Nuvs,Nuvs_sum,&
          l1_l, l1_wgt, l1_Nwgt,&
          sm_l, sm_maj, sm_min, sm_phi,&
          tv_l, tv_wgt, tv_Nwgt,&
          tsv_l, tsv_wgt, tsv_Nwgt,&
          kl_l, kl_wgt, kl_Nwgt,&
          gs_l, gs_wgt, gs_Nwgt,&
          tfd_l, tfd_tgtfd,&
          lc_l, lc_tgtfd,lc_nidx,&
          cen_l, cen_alpha,&
          rt_l, ri_l, rs_l, rf_l, &
          isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
          isamp,uvidxamp,Vamp,Varamp,wamp_n,&
          iscp,uvidxcp,CP,Varcp,wcp_n,&
          isca,uvidxca,CA,Varca,wca_n,&
          1,inormfactr,&
          chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
          reg, l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, lc_cost, cen_cost, &
          rt_cost, ri_cost, rs_cost, rf_cost, &
          cost, gradcost, &
          Npix, Nuv, Nfcv, Namp, Ncp, Nca&
        )
      end if
    end if
  end do


  ! Finary print summary again
  print '("Final Summary")'
  print '("  Iteration ent at ",I5,"/",I5)',isave(30),Niter
  call calc_cost(&
    Iout,xidx,yidx,Nxref,Nyref,Nx,Ny, Nz,&
    u,v,Nuvs,Nuvs_sum,&
    l1_l, l1_wgt, l1_Nwgt,&
    sm_l, sm_maj, sm_min, sm_phi,&
    tv_l, tv_wgt, tv_Nwgt,&
    tsv_l, tsv_wgt, tsv_Nwgt,&
    kl_l, kl_wgt, kl_Nwgt,&
    gs_l, gs_wgt, gs_Nwgt,&
    tfd_l, tfd_tgtfd,&
    lc_l, lc_tgtfd,lc_nidx,&
    cen_l, cen_alpha,&
    rt_l, ri_l, rs_l, rf_l, &
    isfcv,uvidxfcv,Vfcv,Varfcv,wfcv_n,&
    isamp,uvidxamp,Vamp,Varamp,wamp_n,&
    iscp,uvidxcp,CP,Varcp,wcp_n,&
    isca,uvidxca,CA,Varca,wca_n,&
    1,inormfactr,&
    chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
    reg, l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, lc_cost, cen_cost, &
    rt_cost, ri_cost, rs_cost, rf_cost, &
    cost, gradcost, &
    Npix, Nuv, Nfcv, Namp, Ncp, Nca&
  )
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
subroutine calc_cost(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
  u,v, Nuvs, Nuvs_sum,&
  l1_l, l1_wgt, l1_Nwgt,&
  sm_l, sm_maj, sm_min, sm_phi,&
  tv_l, tv_wgt, tv_Nwgt,&
  tsv_l, tsv_wgt, tsv_Nwgt,&
  kl_l, kl_wgt, kl_Nwgt,&
  gs_l, gs_wgt, gs_Nwgt,&
  tfd_l, tfd_tgtfd,&
  lc_l, lc_tgtfd,lc_nidx,&
  cen_l, cen_alpha,&
  rt_l, ri_l, rs_l, rf_l,&
  isfcv,uvidxfcv,Vfcv,Varfcv,wfcv,&
  isamp,uvidxamp,Vamp,Varamp,wamp,&
  iscp,uvidxcp,CP,Varcp,wcp,&
  isca,uvidxca,CA,Varca,wca,&
  doprint,inormfactr,&
  chisq, chisqfcv, chisqamp, chisqcp, chisqca,&
  reg, l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost, tfd_cost, lc_cost, cen_cost, &
  rt_cost, ri_cost, rs_cost, rf_cost, &
  cost, gradcost, &
  Npix, Nuv, Nfcv, Namp, Ncp, Nca &
)
  !
  ! Calculate Cost Functions (for imaging_2d)
  !
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny, Nz
  real(dp), intent(in) :: Iin(Npix*Nz), Nxref, Nyref
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,  intent(in) :: Nuv
  integer,  intent(in) :: Nuvs(Nz)        ! number of uv data for each frame
  integer,  intent(in) :: Nuvs_sum(Nz)    ! accumulated number of uv data *before* each frame

  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy

  ! Regularization Parameters
  !   parameter for l1
  real(dp), intent(in)  :: l1_l               ! lambda
  real(dp), intent(in)  :: l1_wgt(l1_Nwgt)    ! weight
  integer,  intent(in)  :: l1_Nwgt            ! size of the weight vector
  !   parameter for second moment
  real(dp), intent(in)  :: sm_l               ! lambda
  real(dp), intent(in)  :: sm_maj,sm_min,sm_phi ! major and minor size and position angle
  !   parameter for total variation
  real(dp), intent(in)  :: tv_l               ! lambda
  real(dp), intent(in)  :: tv_wgt(tv_Nwgt)    ! weight
  integer,  intent(in)  :: tv_Nwgt            ! size of the weight vector
  !   parameter for total squared variation
  real(dp), intent(in)  :: tsv_l              ! lambda
  real(dp), intent(in)  :: tsv_wgt(tsv_Nwgt)  ! weight
  integer,  intent(in)  :: tsv_Nwgt           ! size of the weight vector
  !   parameter for kl divergence
  real(dp), intent(in)  :: kl_l               ! lambda
  real(dp), intent(in)  :: kl_wgt(kl_Nwgt)    ! weight
  integer,  intent(in)  :: kl_Nwgt            ! size of the weight vector
  !   parameter for Gull & Skilling entropy
  real(dp), intent(in)  :: gs_l               ! lambda
  real(dp), intent(in)  :: gs_wgt(gs_Nwgt)    ! weight
  integer,  intent(in)  :: gs_Nwgt            ! size of the weight vector
  !   parameter for the total flux density regularization
  real(dp), intent(in)  :: tfd_l              ! lambda (Normalized)
  real(dp), intent(in)  :: tfd_tgtfd          ! target total flux
  !   parameter for the light curve regularization
  real(dp), intent(in)  :: lc_l(Nz)           ! lambda array (Normalized)
  real(dp), intent(in)  :: lc_tgtfd(Nz)       ! target light curve
  integer,  intent(in)  :: lc_nidx            ! regularizer normalization with lc_tgtfd
  !   parameter for the centoroid regularization
  real(dp), intent(in)  :: cen_l              ! lambda (Normalized)
  real(dp), intent(in)  :: cen_alpha          ! alpha

  ! Regularization parameters for dynamical imagings
  real(dp), intent(in)  :: rt_l, ri_l, rs_l
  real(dp), intent(in)  ::  rf_l

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
  real(dp), intent(in) :: inormfactr  ! intensity normalization factor

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
  real(dp), intent(out) :: sm_cost  ! cost of second moment
  real(dp), intent(out) :: tv_cost  ! cost of tv
  real(dp), intent(out) :: tsv_cost ! cost of tsv
  real(dp), intent(out) :: kl_cost  ! cost of KL divergence
  real(dp), intent(out) :: gs_cost  ! cost of GS entropy
  real(dp), intent(out) :: tfd_cost ! cost of total flux regularization
  real(dp), intent(out) :: lc_cost  ! cost of light curve regularization
  real(dp), intent(out) :: cen_cost ! cost of centoroid regularizaiton

  ! Regularization functions for dynamical imagings
  real(dp), intent(out) :: rt_cost
  real(dp), intent(out) :: ri_cost
  real(dp), intent(out) :: rs_cost
  real(dp), intent(out) :: rf_cost

  !   Total Cost function
  real(dp), intent(out) :: cost
  real(dp), intent(out) :: gradcost(1:Npix*Nz)

  ! Gradients of each term
  real(dp) :: gradreg(Npix*Nz), gradchisq(Npix*Nz)

  ! second moment variables
  real(dp) :: out_maj, out_min, out_phi, pi

  ! dynamical imaging
  integer :: iz
  !------------------------------------
  ! Initialize outputs, and some parameters
  !------------------------------------
  ! Initialize the chisquare and its gradient
  !write(*,*) 'stdftim.calc_cost: initialize cost and gradcost'
  chisq       = 0d0
  gradchisq   = 0d0
  cost        = 0d0
  gradcost(:) = 0d0

  call calc_chisq3d(&
    Iin,&
    xidx,yidx,Nx,Ny,&
    u, v, Nuvs, Nuvs_sum,&
    isfcv,uvidxfcv,Vfcv,Varfcv,wfcv,&
    isamp,uvidxamp,Vamp,Varamp,wamp,&
    iscp,uvidxcp,CP,Varcp,wcp,&
    isca,uvidxca,CA,Varca,wca,&
    chisq, gradchisq, chisqfcv, chisqamp, chisqcp, chisqca,&
    Npix, Nz, Nuv, Nfcv, Namp, Ncp, Nca &
  )

  call calc_cost_reg3d(&
      Iin, xidx, yidx, Nxref, Nyref, Nx, Ny,&
      l1_l, l1_wgt, l1_Nwgt,&
      sm_l, sm_maj, sm_min, sm_phi,&
      tv_l, tv_wgt, tv_Nwgt,&
      tsv_l, tsv_wgt, tsv_Nwgt,&
      kl_l, kl_wgt, kl_Nwgt,&
      gs_l, gs_wgt, gs_Nwgt,&
      tfd_l, tfd_tgtfd,&
      lc_l, lc_tgtfd,lc_nidx,&
      cen_l, cen_alpha,&
      rt_l, ri_l, rs_l, rf_l, &
      l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost,&
      tfd_cost, lc_cost, cen_cost,&
      rt_cost, ri_cost, rs_cost, rf_cost,&
      out_maj, out_min, out_phi,&
      reg, gradreg, Npix, Nz &
  )



  ! Finally take the sum of the cost and gradient functions

  cost     = chisq     + reg
  gradcost = gradchisq + gradreg

  ! print '(" Debug:  Print summary if requested")'
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

    if (sm_l > 0) then
      print '("    second momentum      : ",D13.6)',sm_cost
      out_maj = sqrt(8.*log(2.)*out_maj)
      out_min = sqrt(8.*log(2.)*out_min)
      pi = 3.1415926535
      print '("    input  maj, min, phi : ",D13.6, D13.6, D13.6)', sqrt(8.*log(2.)*sm_maj), sqrt(8.*log(2.)*sm_min), &
                                                                   sm_phi * 180./pi
      print '("    output maj, min, phi : ",D13.6, D13.6, D13.6)', out_maj, out_min, out_phi * 180./pi
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
    if (sum(lc_l) > 0) then
      print '("    Light curve          : ",D13.6)',lc_cost
    end if
    if (cen_l > 0) then
      print '("    Centoroid Reg.       : ",D13.6)',cen_cost
    endif

    print '("  Total flux (avg)       : ",D13.6)',sum(Iin)/Nz/inormfactr
    ! regularization parameters for dynamical imaging
    if (rt_l > 0) then
      print '("  Dynamical  Rt Reg.     : ",D13.6)', rt_cost
    end if
    if (ri_l > 0) then
      print '("  Dynamical  Ri Reg.     : ",D13.6)', ri_cost
    end if
    if (rs_l > 0) then
      print '("  Dynamical  Rs Reg.     : ",D13.6)', rs_cost
    end if
    if (rf_l > 0) then
      print '("  Dynamical  Rf Reg.     : ",D13.6)', rf_cost
    end if

  end if
end subroutine

end module
