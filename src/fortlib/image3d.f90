module image3d
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  use fftlib, only: NUFFT_fwd, NUFFT_adj, NUFFT_adj_resid, phashift_r2c,&
                    chisq_fcv, chisq_amp, chisq_ca, chisq_cp
  use image, only : ixiy2ixy, ixy2ixiy, comreg, zeroeps,&
                    I1d_I2d_fwd,I1d_I2d_inv,&
                    log_fwd, log_grad,&
                    gamma_fwd, gamma_grad,&
                    l1_e, l1_grade,&
                    tv_e, tv_grade,&
                    tsv_e, tsv_grade,&
                    comreg3d
  implicit none
contains
!
!-------------------------------------------------------------------------------
! Regularization Function for Dynamical Imaging
!-------------------------------------------------------------------------------
!
!-------------------------------------------------------------------------------
! Distances
!-------------------------------------------------------------------------------
!
! Dp (p=2) distance
!
real(dp) function d2(xidx,yidx,I2d,I2du,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  i = xidx
  j = yidx
  !
  d2  = (I2du(i,j) - I2d(i,j))**2
end function
!
! Kullback-Leibler divergence
!
real(dp) function dkl(xidx,yidx,I2d,I2du,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  i = xidx
  j = yidx
  !
  if (I2d(i,j) /= 0 .and. I2du(i,j) /= 0) then
    dkl  = I2d(i,j) * log(I2d(i,j)/I2du(i,j))
  else if (I2d(i,j) == 0 .or. I2du(i,j) == 0) then
    dkl  = 0
  end if
  !
end function
!
!-------------------------------------------------------------------------------
! Gradients
!-------------------------------------------------------------------------------
!
! Gradient of Rt from D2 distance
!
real(dp) function rt_d2grad(xidx,yidx,zidx,I2d,I2dl,I2du,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,Nz
  integer, intent(in)  :: xidx, yidx, zidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2dl(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  rt_d2grad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  if (zidx > 1) then
    rt_d2grad = rt_d2grad + 2*(I2d(i,j) - I2dl(i,j))
  end if
  if (zidx < Nz) then
    rt_d2grad = rt_d2grad - 2*(I2du(i,j) - I2d(i,j))
  end if
  !
end function
!
! Gradient of Rt from Kullback-Leibler divergence
!
real(dp) function rt_dklgrad(xidx,yidx,zidx,I2d,I2dl,I2du,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,Nz
  integer, intent(in)  :: xidx, yidx, zidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2dl(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  rt_dklgrad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  if (zidx > 1 .and. I2d(i,j) /= 0 .and. I2dl(i,j) /= 0) then
    rt_dklgrad = rt_dklgrad + (1 + log(I2d(i,j)/I2dl(i,j)))
  end if
  if (zidx < Nz .and. I2d(i,j) /= 0) then
    rt_dklgrad = rt_dklgrad - (I2du(i,j)/I2d(i,j))
  end if
  !
end function
!
! Gradient of Ri from D2 distance
!
real(dp) function ri_d2grad(xidx,yidx,I2d,Iavg2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx, yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),Iavg2d(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  ri_d2grad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  ri_d2grad = 2 * (I2d(i,j) - Iavg2d(i,j))
  !
end function
!
! Gradient of Ri from Kullback-Leibler divergence
!
real(dp) function ri_dklgrad(xidx,yidx,I2d,Iavg2d,Itmp2d,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,Nz,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),Iavg2d(Nx,Ny),Itmp2d(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  ri_dklgrad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  ri_dklgrad = 1 - (Iavg2d(i,j)/I2d(i,j)) + Itmp2d(i,j)/Nz
  !
end function
!
!-------------------------------------------------------------------------------
! calc cost functions
!-------------------------------------------------------------------------------
!
subroutine calc_costs(&
  Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,Nz,&
  u,v,Nuvs,Nuvs_sum,&
  lambl1,lambtv,lambtsv,lambmem,lambcom,&
  lambrt,lambri,lambrs,&
  fnorm,transtype,transprm,pcom,&
  isfcv,uvidxfcv,Vfcv,Varfcv,&
  isamp,uvidxamp,Vamp,Varamp,&
  iscp,uvidxcp,CP,Varcp,&
  isca,uvidxca,CA,Varca,&
  costs,gradcosts,&
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
  real(dp), intent(out) :: costs(10)
  real(dp), intent(out) :: gradcosts(Nparm,10)

  ! integer
  integer :: ipix, iz, iparm, istart, iend

  ! chisquares, gradients of each term of equations
  real(dp) :: chisq, reg                 ! chisquare and regularization
  real(dp) :: rint_s, reg_frm            ! regularization for interpolation
  real(dp) :: cost,gradcost(Nparm)       ! for costs, and gradcosts
  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:),I2dl(:,:),I2du(:,:),Iin_reg(:)
  real(dp), allocatable :: Iavg(:), Iavg2d(:,:), Ij(:), Itmp(:), Itmp2d(:,:)

  ! modifiy
  real(dp), allocatable :: Iavgin(:,:)
  real(dp), allocatable :: gradreg_tmpin(:,:)

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
  cost     = 0d0
  costs(:) = 0d0
  gradcosts(:,:) = 0d0
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

  !------------------------------------
  ! Initialize outputs, and some parameters
  !------------------------------------
  ! Initialize the chisquare and its gradient
  !write(*,*) 'stdftim.calc_cost: initialize cost and gradcost'
  costs(:) = 0d0
  gradcosts(:,:) = 0d0
  gradcost(:) = 0d0


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
  costs(1) = chisq
  !write(*,*) "chisq: ", chisq

  if (lambcom > 0) then
    ! initialize
    !   scalars
    reg = 0d0

    !  allocatable arrays (modify)
    allocate(Iavgin(Npix,Nz))
    allocate(gradreg_tmpin(Npix,Nz))
    allocate(gradreg(Nparm))
    Iavgin(:,:)        = 0d0
    gradreg_tmpin(:,:) = 0d0
    gradreg(:) = 0d0

    ! Averaged Image (modify)
    do iz=1, Nz
      do ipix=1,Npix
        Iavgin(ipix,iz)=Iin((iz-1)*Npix+ipix)
       end do
    end do

    ! calc cost and its gradient (modify)
    call comreg3d(xidx,yidx,Nxref,Nyref,pcom,Iavgin,reg,gradreg_tmpin,Npix,Nz)

    costs(2) = costs(2) + lambcom * reg
    !
    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(Npix,Nz,gradreg_tmp) &
    !$OMP   PRIVATE(iz, ipix, istart) &
    !$OMP   REDUCTION(+: gradreg)

    ! modify
    do iz=1, Nz
      istart = (iz-1)*Npix+1
      !iend = iz*Npix
      do ipix=1,Npix
        gradreg(istart+ipix-1) = gradreg_tmpin(ipix,iz)
      end do
    end do

    !$OMP END PARALLEL DO
    call daxpy(Nparm, lambcom, gradreg, 1, gradcost, 1) ! gradcost := lambcom * gradreg + gradcost
    ! deallocate array
    !deallocate(gradreg_tmp)
    !deallocate(gradreg)
    !deallocate(Iavg)

    ! deallocate array (modify)
    deallocate(gradreg_tmpin)
    deallocate(gradreg)
    deallocate(Iavgin)
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
        !reg_frm = reg_frm + l1_e(Iin_reg(iparm))
        gradreg_frm(ipix) = gradreg_frm(ipix) + l1_grade(Iin_reg(iparm))
        costs(3) = costs(3) + lambl1 * l1_e(Iin_reg(iparm))
        gradreg(iparm) = gradreg(iparm) + lambl1 * l1_grade(Iin_reg(iparm))
      end if

      ! TV
      if (lambtv > 0) then
        !reg_frm = reg_frm + tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg_frm(ipix) = gradreg_frm(ipix) + tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        costs(4) = costs(4) + lambtv * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambtv * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! TSV
      if (lambtsv > 0) then
        !reg_frm = reg_frm + tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg_frm(ipix) = gradreg_frm(ipix) + tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        costs(5) = costs(5) + lambtsv * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambtsv * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      end if

      ! Dynamical Imaging
      if (lambrt > 0 .and. iz < Nz) then
        ! Rt from Dp (p=2) distance
        costs(6) = costs(6) + lambrt * d2(xidx(ipix),yidx(ipix),I2d,I2du,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambrt * rt_d2grad(xidx(ipix),yidx(ipix),iz,I2d,I2dl,I2du,Nx,Ny,Nz)
        ! Rt from Kullback-Leibler divergence
        !costs(6) = costs(6) + lambrt * dkl(xidx(ipix),yidx(ipix),I2d,I2du,Nx,Ny)
        !gradreg(iparm) = gradreg(iparm) + lambrt * rt_dklgrad(xidx(ipix),yidx(ipix),iz,I2d,I2dl,I2du,Nx,Ny,Nz)
      end if

      if (lambri > 0) then
        ! Ri from Dp (p=2) distance
        costs(7) = costs(7) + lambri * d2(xidx(ipix),yidx(ipix),I2d,Iavg2d,Nx,Ny)
        gradreg(iparm) = gradreg(iparm) + lambri * ri_d2grad(xidx(ipix),yidx(ipix),I2d,Iavg2d,Nx,Ny)
        ! Ri from Kullback-Leibler divergence
        !costs(7) = costs(7) + lambri * dkl(xidx(ipix),yidx(ipix),I2d,Iavg2d,Nx,Ny)
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
    costs(8) = costs(8) + lambrs * rint_s
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
  !costs(9) = costs(9)
  call daxpy(Nparm, 1d0, gradreg, 1, gradcost, 1) ! gradcost := gradreg + gradcos

  ! deallocate arrays
  deallocate(gradreg)


end subroutine


end module
!
!
!-------------------------------------------------------------------------------
! A convinient function to compute regularization functions
! for python interfaces
!-------------------------------------------------------------------------------
!
! subroutine I2d_rtd2(I2d,I2dl,I2du,cost,costmap,gradmap,Nx,Ny,Nz)
!   implicit none
!
!   integer, intent(in)  :: Nx,Ny,Nz
!   real(dp), intent(in) :: I2d(Nx,Ny),I2dl(Nx,Ny),I2du(Nx,Ny)
!   real(dp), intent(out):: cost,costmap(Nx,Ny),gradmap(Nx,Ny)
!
!   integer :: ixy,ix,iy,iz
!
!   ! initialize output
!   cost = 0d0
!   costmap(:,:) = 0d0
!   gradmap(:,:) = 0d0
!
!   !$OMP PARALLEL DO DEFAULT(SHARED) &
!   !$OMP   FIRSTPRIVATE(Nx,Ny,Nz,I2d,I2dl,I2du) &
!   !$OMP   PRIVATE(ixy,ix,iy,iz) &
!   !$OMP   REDUCTION(+:cost,costmap,gradmap)
!   do iz=1, Nz
!     do ixy=1, Nx*Ny
!       iparm = (iz-1)*Nx*Ny + ixy
!       call ixy2ixiy(ixy,ix,iy,Nx)
!       costmap(ix,iy) = d2(ix,iy,I2d,I2du,Nx,Ny)
!       gradmap(ix,iy) = rt_d2grad(ix,iy,iz,I2d,I2dl,I2du,Nx,Ny,Nz)
!       cost = cost+costmap(ix,iy)
!     end do
!   end do
!   !$OMP END PARALLEL DO
! end subroutine
!
