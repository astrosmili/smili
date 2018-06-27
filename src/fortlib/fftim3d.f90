module fftim3d
  !$use omp_lib
  use param, only: dp, dpc, deps
  use fftlib, only: NUFFT_fwd, NUFFT_adj, NUFFT_adj_resid, phashift_r2c,&
                    chisq_fcv, chisq_amp, chisq_ca, chisq_cp
  use image, only: I1d_I2d_fwd, I1d_I2d_inv,&
                   l1_e, l1_grade,&
                   tv_e, tv_grade,&
                   tsv_e, tsv_grade,&
                   she_e, she_grade,&
                   gse_e, gse_grade,&
                   zeroeps, ixy2ixiy, ixiy2ixy
  use image3d, only: di, comreg3d,&
                     dt_e, dt_grade, &
                     dtf_e, dtf_grade, &
                     calc_l1_w_3d, calc_tv_w_3d, calc_tsv_w_3d, &
                     calc_dt_w, calc_di_w, calc_dtf_w
  implicit none
contains
!-------------------------------------------------------------------------------
! Imaging Routine
!-------------------------------------------------------------------------------
subroutine imaging(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
  u,v,Nuvs,&
  lambl1,lambtv,lambtsv,lambshe,lambgse,&
  lambdt,lambdi,lambdtf,lambcom,&
  doweight,tgtdyrange,ent_p,&
  Niter,nonneg,pcom,&
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
  real(dp), intent(in) :: lambl1  ! Regularization Parameter for L1-norm
  real(dp), intent(in) :: lambtv  ! Regularization Parameter for iso-TV
  real(dp), intent(in) :: lambtsv ! Regularization Parameter for TSV
  real(dp), intent(in) :: lambshe ! Regularization Parameter for Shannon Entropy
  real(dp), intent(in) :: lambgse ! Regularization Parameter for Gull & Skilling Entropy
  real(dp), intent(in) :: lambcom ! Regularization Parameter for Center of Mass
  real(dp), intent(in) :: lambdt  ! Regularization Parameter for Dynamical Imaging (delta-t)
  real(dp), intent(in) :: lambdi  ! Regularization Parameter for Dynamical Imaging (delta-I)
  real(dp), intent(in) :: lambdtf  ! Regularization Parameter for Dynamical Imaging (entropy continuity)

  ! reweighting
  integer,  intent(in) :: doweight   ! if postive, reweight images for L1, isoTV, TSV
  real(dp), intent(in) :: tgtdyrange ! target dynamic range for reweighting
  real(dp), intent(in) :: ent_p(Npix)! prior image for the maximum entropy methods (gse, she)

  ! Imaging Parameter
  integer,  intent(in) :: Niter     ! iteration number
  logical,  intent(in) :: nonneg    ! if nonneg > 0, the image will be solved
                                    ! with a non-negative condition
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

  ! Reweighting factor for Images
  real(dp), allocatable :: l1_w(:), tv_w(:), tsv_w(:)
  real(dp), allocatable :: dt_w(:), di_w(:), dtf_w(:)

  ! full complex visibilities to be used for calculations
  complex(dpc), allocatable :: Vfcv(:)

  ! chisquare and grad chisq
  real(dp) :: cost                      ! cost function
  real(dp) :: gradcost(1:Npix*Nz)       ! its gradient

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
  write(*,*) 'Number of Data                        ', Ndata
  write(*,*) 'Number of Paramter/frame              ', Npix
  write(*,*) 'Number of Image pixels                ', Nx, Ny
  write(*,*) 'Number of Frames                      ', Nz
  write(*,*) 'Number of non redundant uv coordinates', Nuv

  ! copy images (Iin -> Iout)
  call dcopy(Nparm,Iin,1,Iout,1)

  ! shift tracking center of full complex visibilities from the reference pixel
  ! to the center of the image
  allocate(Vfcv(Nfcv))
  Vfcv = dcmplx(Vfcvr,Vfcvi)
  if (isfcv .eqv. .True.) then
    write(*,*) 'Shift tracking center of full complex visibilities.'
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

  !-----------------------------------------------------------------------------
  ! Reweighting factor for l1, tv, tsv
  !-----------------------------------------------------------------------------
  write(*,*) 'Regularization Parameters'
  write(*,*) ' Reweighting:', doweight
  write(*,*) ' lambl1     :', lambl1
  write(*,*) ' lambtv     :', lambtv
  write(*,*) ' lambtsv    :', lambtsv
  write(*,*) ' lambshe    :', lambshe
  write(*,*) ' lambgse    :', lambgse
  write(*,*) ' lambdt     :', lambdt
  write(*,*) ' lambdi     :', lambdi
  write(*,*) ' lambdtf    :', lambdtf
  ! allocate(l1_w(Nparm),tv_w(Nparm),tsv_w(Nparm),dt_w(Nparm),di_w(Npix),dtf_w(Nz))
  ! if (doweight > 0 ) then
  !   write(*,*) 'Calculating re-weighting factor for l1, tv, tsv, di, dt, dtf regularizations'
  !   call calc_l1_w_3d(Iin,tgtdyrange,l1_w,Nparm)
  !   call calc_tv_w_3d(Iin,xidx,yidx,tgtdyrange,tv_w,Nparm,Npix,Nx,Ny,Nz)
  !   call calc_tsv_w_3d(Iin,xidx,yidx,tsv_w,Nparm,Npix,Nx,Ny,Nz)
  !   call calc_dt_w(Iin,dt_w,Nparm,Npix,Nz)
  !   call calc_di_w(Iin,di_w,Nparm,Npix,Nz)
  !   call calc_dtf_w(Iin,dtf_w,Nparm,Npix,Nz)
  ! else
  !   l1_w = 1
  !   tv_w = 1
  !   tsv_w = 1
  !   dt_w = 1
  !   di_w = 1
  !   dtf_w = 1
  ! end if

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
!        doweight,l1_w,tv_w,tsv_w,dt_w,di_w,dtf_w,ent_p,&
      call calc_cost(&
        Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
        u,v,Nuvs,Nuvs_sum,&
        lambl1,lambtv,lambtsv,lambshe,lambgse,lambcom,&
        lambdt,lambdi,lambdtf,&
        ent_p,&
        fnorm,pcom,&
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
      else if (mod(isave(30),10) == 0) then
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
  ! deallocate(l1_w,tv_w,tsv_w,dt_w,di_w,dtf_w)
  where(abs(Iout)<zeroeps) Iout=0d0
end subroutine
!
!-------------------------------------------------------------------------------
! calc cost functions
!-------------------------------------------------------------------------------
!
!  doweight,l1_w,tv_w,tsv_w,dt_w,di_w,dtf_w,&
subroutine calc_cost(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
  u,v,Nuvs,Nuvs_sum,&
  lambl1,lambtv,lambtsv,lambshe,lambgse,lambcom,&
  lambdt,lambdi,lambdtf,&
  ent_p,&
  fnorm,pcom,&
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
  real(dp), intent(in) :: lambshe ! Regularization Parameter for Shannon Information Entropy
  real(dp), intent(in) :: lambgse ! Regularization Parameter for Gull & Skilling Entropy
  real(dp), intent(in) :: lambcom ! Regularization Parameter for Center of Mass
  real(dp), intent(in) :: lambdt  ! Regularization Parameter for Delta-T (D2)
  real(dp), intent(in) :: lambdi  ! Regularization Parameter for Delta-I (D2)
  real(dp), intent(in) :: lambdtf ! Regularization Parameter for Delta-TF (D2)

  ! Reweighting
!  integer,  intent(in) :: doweight ! if postive, reweight l1,tsv,tv terms
!  real(dp), intent(in) :: l1_w(Nparm), tv_w(Nparm), tsv_w(Nparm) ! reweight
!  real(dp), intent(in) :: dt_w(Nparm), di_w(Npix), dtf_w(Nz) ! reweight
  real(dp), intent(in) :: ent_p(Npix) ! prior image for MEM

  ! Imaging Parameter
  real(dp), intent(in) :: fnorm     ! normalization factor for chisquare
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

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:)
  real(dp), allocatable :: Isum(:)
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

  ! Forward Non-unifrom Fast Fourier Transform
  !   allocatable arrays
  allocate(Vcmp(Nuv))
  allocate(I2d(Nx,Ny))
  Vcmp(:) = dcmplx(0d0,0d0)
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nx,Ny,Nz,Npix,Nuvs,Nuvs_sum,u,v,xidx,yidx) &
  !$OMP   PRIVATE(iz, istart, iend, I2d) &
  !$OMP   REDUCTION(+:Vcmp)
  do iz=1, Nz
    ! If there is a data corresponding to this frame
    if (Nuvs(iz) /= 0) then
      ! allocate 2D image for imaging
      I2d(:,:) = 0d0
      call I1d_I2d_fwd(xidx,yidx,Iin((iz-1)*Npix+1:iz*Npix),I2d,Npix,Nx,Ny)

      ! Index of data
      istart = Nuvs_sum(iz) + 1
      iend   = Nuvs_sum(iz) + Nuvs(iz)

      ! run forward NUFFT
      call NUFFT_fwd(u(istart:iend),v(istart:iend),I2d,Vcmp(istart:iend),&
                     Nx,Ny,Nuvs(iz))
    end if
  end do
  !$OMP END PARALLEL DO

  ! deallocate array
  deallocate(I2d)

  ! allocate arrays for residuals
  allocate(Vresre(Nuv), Vresim(Nuv))
  Vresre(:) = 0d0
  Vresim(:) = 0d0

  ! Full complex visibility
  if (isfcv .eqv. .True.) then
    call chisq_fcv(Vcmp,uvidxfcv,Vfcv,Varfcv,fnorm,chisq,Vresre,Vresim,Nuv,Nfcv)
    !print '("chisq fcv :",D13.6)',chisq
  end if

  ! Amplitudes
  if (isamp .eqv. .True.) then
    call chisq_amp(Vcmp,uvidxamp,Vamp,Varamp,fnorm,chisq,Vresre,Vresim,Nuv,Namp)
    !print '("chisq amp :",D13.6)',chisq
  end if

  ! Log closure amplitudes
  if (isca .eqv. .True.) then
    call chisq_ca(Vcmp,uvidxca,CA,Varca,fnorm,chisq,Vresre,Vresim,Nuv,Nca)
    !print '("chisq ca :",D13.6)',chisq
  end if

  ! Closure phases
  if (iscp .eqv. .True.) then
    call chisq_cp(Vcmp,uvidxcp,CP,Varcp,fnorm,chisq,Vresre,Vresim,Nuv,Ncp)
    !print '("chisq cp :",D13.6)',chisq
  end if
  deallocate(Vcmp)
  !print '("chisq total :",D13.6)',chisq

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  ! allocate 2D image for imaging
  allocate(gradchisq2d(Nx,Ny))
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nx,Ny,Nz,Npix,Nuvs,Nuvs_sum,u,v,xidx,yidx,Vresre,Vresim) &
  !$OMP   PRIVATE(iz, istart, iend, gradchisq2d) &
  !$OMP   REDUCTION(+:gradcost)
  do iz=1, Nz
    ! If there is a data corresponding to this frame
    if(Nuvs(iz) /= 0) then

      ! Index of data
      istart = Nuvs_sum(iz) + 1
      iend   = Nuvs_sum(iz) + Nuvs(iz)

      ! run adujoint NUFFT
      gradchisq2d(:,:) = 0d0
      call NUFFT_adj_resid(u(istart:iend),v(istart:iend),&
                           Vresre(istart:iend),Vresim(istart:iend),&
                           gradchisq2d,Nx,Ny,Nuvs(iz))

      ! copy the gradient of chisquare into that of cost functions
      call I1d_I2d_inv(xidx,yidx,gradcost((iz-1)*Npix+1:iz*Npix),&
                       gradchisq2d,Npix,Nx,Ny)
    end if
  end do
  !$OMP END PARALLEL DO
  ! deallocate array
  deallocate(gradchisq2d)
  deallocate(Vresre,Vresim)

  ! copy the chisquare into that of cost functions
  cost = chisq

  !-----------------------------------------------------------------------------
  ! Centoroid Regularizer
  !-----------------------------------------------------------------------------
  if (lambcom > 0) then
    ! initialize
    !   scalars
    reg = 0d0
    !   allocatable arrays
    allocate(gradreg(Nparm))
    gradreg(:) = 0d0

    ! calc cost and its gradient (modify)
    call comreg3d(xidx,yidx,Nxref,Nyref,pcom,Iin,reg,gradreg,Npix,Nz,Nparm)

    ! update cost
    cost = cost + lambcom * reg
    !   gradcost := lambcom * gradreg + gradcost
    call daxpy(Nparm, lambcom, gradreg, 1, gradcost, 1)

    !print '("cost com :",D13.6,"*",D13.6,"=",D13.6)',lambcom,reg,lambcom*reg
    ! deallocate array
    deallocate(gradreg)
  end if

  !-----------------------------------------------------------------------------
  ! Delta I dynamical regularization
  !-----------------------------------------------------------------------------
  if (lambdi > 0) then
    ! initialize
    !   scalars
    reg = 0d0
    !   allocatable arrays
    allocate(gradreg(Nparm))
    gradreg(:) = 0d0

    ! calc cost and its gradient (modify)
    !call di(Iin,di_w,doweight,reg,gradreg,Nparm,Npix,Nz)
    allocate(Isum(Npix)) ! this is dammy arrays
    call di(Iin,Isum,-1,reg,gradreg,Nparm,Npix,Nz)
    deallocate(Isum)

    ! Update cost
    cost = cost + lambdi * reg
    !    gradcost := lambdi * gradreg + gradcost
    call daxpy(Nparm, lambdi, gradreg, 1, gradcost, 1)
    !print '("cost DI :",D13.6,"*",D13.6,"=",D13.6)',lambdi,reg,lambdi * reg

    ! deallocate array
    deallocate(gradreg)
  end if

  !-----------------------------------------------------------------------------
  ! Other 2D & 3D Regularization Functions
  !-----------------------------------------------------------------------------
  if (lambdtf > 0) then
    !   get the total flux of each image
    allocate(Isum(Nz))
    Isum = sum(reshape(Iin,(/Npix,Nz/)),1)
  end if

  if (lambtv > 0 .or. lambtsv > 0) then
    allocate(I2d(Nx,Ny))
  end if

  !!$OMP                xidx, yidx, l1_w, tv_w, tsv_w, dt_w, dtf_w, ent_p) &
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, Nx, Ny, Nz, Nparm, Iin, Isum, &
  !$OMP                lambl1, lambtv, lambtsv, lambshe, lambgse, &
  !$OMP                lambdt, lambdtf, &
  !$OMP                xidx, yidx) &
  !$OMP   PRIVATE(iz, ipix, iparm, I2d, istart, iend) &
  !$OMP   REDUCTION(+: cost, gradcost)
  do iz=1, Nz
    istart = (iz-1)*Npix+1
    iend = iz*Npix

    ! allocate 2d image if lambtv/tsv/rt > 0
    if (lambtv > 0 .or. lambtsv > 0) then
      I2d(:,:)=0d0
      call I1d_I2d_fwd(xidx,yidx,Iin(istart:iend),I2d,Npix,Nx,Ny)
    end if

    ! total flux regularization
    if (lambdtf > 0) then
      ! if (doweight > 0) then
      !   cost = cost + lambdtf * dtf_w(iz) * dtf_e(iz, Isum, Nz)
      !   gradcost(istart:iend) = gradcost(istart:iend)&
      !                         + lambdtf*dtf_w(iz)*dtf_grade(iz,Isum,Nz)
      ! else
      !   cost = cost + lambdtf * dtf_e(iz, Isum, Nz)
      !   gradcost(istart:iend) = gradcost(istart:iend)&
      !                         + lambdtf*dtf_grade(iz,Isum,Nz)
      ! end if
      cost = cost + lambdtf * dtf_e(iz, Isum, Nz)
      gradcost(istart:iend) = gradcost(istart:iend)&
                            + lambdtf*dtf_grade(iz,Isum,Nz)
    end if

    ! compute regularization function
    do ipix=1, Npix
      call ixy2ixiy(ipix,iz,iparm,Npix)
      ! Compute L1, TV, TSV, DT, DI
      ! if (doweight > 0) then
      !   ! L1
      !   if (lambl1 > 0) then
      !     cost = cost + lambl1 * l1_w(iparm) * l1_e(Iin(iparm))
      !     gradcost(iparm) = gradcost(iparm)&
      !                     + lambl1 * l1_w(iparm) * l1_grade(Iin(iparm))
      !   end if
      !
      !   ! TV
      !   if (lambtv > 0) then
      !     cost = cost + lambtv * tv_w(iparm) * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !     gradcost(iparm) = gradcost(iparm) &
      !                     + lambtv * tv_w(iparm) * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !   end if
      !
      !   ! TSV
      !   if (lambtsv > 0) then
      !     cost = cost + lambtsv * tsv_w(iparm) * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !     gradcost(iparm) = gradcost(iparm) &
      !                     + lambtsv * tsv_w(iparm) * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !   end if
      !
      !   ! Delta T dynamical regularization
      !   if (lambdt > 0) then
      !     cost = cost + lambdt * dt_w(iparm) * dt_e(iparm,iz,Iin,Npix,Nz,Nparm)
      !     gradcost(iparm) = gradcost(iparm) + lambdt * dt_w(iparm) * dt_grade(iparm,iz,Iin,Npix,Nz,Nparm)
      !   end if
      ! else
      !   ! L1
      !   if (lambl1 > 0) then
      !     cost = cost + lambl1 * l1_e(Iin(iparm))
      !     gradcost(iparm) = gradcost(iparm) + lambl1 * l1_grade(Iin(iparm))
      !   end if
      !
      !   ! TV
      !   if (lambtv > 0) then
      !     cost = cost + lambtv * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !     gradcost(iparm) = gradcost(iparm) + lambtv * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !   end if
      !
      !   ! TSV
      !   if (lambtsv > 0) then
      !     cost = cost + lambtsv * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !     gradcost(iparm) = gradcost(iparm) + lambtsv * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      !   end if
      !
      !   ! Delta T dynamical regularization
      !   if (lambdt > 0) then
      !     cost = cost + lambdt * dt_e(iparm,iz,Iin,Npix,Nz,Nparm)
      !     gradcost(iparm) = gradcost(iparm) + lambdt * dt_grade(iparm,iz,Iin,Npix,Nz,Nparm)
      !   end if
      ! end if
      ! L1
      if (lambl1 > 0) then
        cost = cost + lambl1 * l1_e(Iin(iparm))
        gradcost(iparm) = gradcost(iparm) + lambl1 * l1_grade(Iin(iparm))
      end if

      ! TV
      if (lambtv > 0) then
        cost = cost + lambtv * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradcost(iparm) = gradcost(iparm) + &
                          lambtv * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! TSV
      if (lambtsv > 0) then
        cost = cost + lambtsv * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradcost(iparm) = gradcost(iparm) + &
                          lambtsv * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! Delta T dynamical regularization
      if (lambdt > 0) then
        cost = cost + lambdt * dt_e(iparm,iz,Iin,Npix,Nz,Nparm)
        gradcost(iparm) = gradcost(iparm) + &
                          lambdt * dt_grade(iparm,iz,Iin,Npix,Nz,Nparm)
      end if

      ! Shannon Entropy
      if (lambshe > 0) then
        cost = cost + lambshe * she_e(Iin(iparm),ent_p(ipix))
        gradcost(iparm) = gradcost(iparm) + &
                          lambshe * she_grade(Iin(iparm),ent_p(ipix))
      end if

      ! Gull & Skilling Entropy
      if (lambgse > 0) then
        cost = cost + lambgse * gse_e(Iin(iparm),ent_p(ipix))
        gradcost(iparm) = gradcost(iparm) + &
                          lambgse * gse_grade(Iin(iparm),ent_p(ipix))
      end if
    end do
  end do
  !$OMP END PARALLEL DO

  ! deallocate arrays
  if (lambtv > 0 .or. lambtsv > 0) then
    deallocate(I2d)
  end if

  if (lambdtf > 0) then
    deallocate(Isum)
  end if
end subroutine
end module
