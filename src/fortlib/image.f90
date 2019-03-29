module image
  !$use omp_lib
  use param, only : dp, dpc, pi, e, i_dpc, deps
  implicit none
  ! Epsiron for Zero judgement
  real(dp), parameter :: zeroeps = 1d-10
contains
!-------------------------------------------------------------------------------
! Calc cost function
!-------------------------------------------------------------------------------
subroutine calc_cost_reg(&
    I1d, xidx, yidx, Nxref, Nyref, Nx, Ny, &
    l1_l, l1_wgt, l1_Nwgt,&
    tv_l, tv_wgt, tv_Nwgt,&
    tsv_l, tsv_wgt, tsv_Nwgt,&
    kl_l, kl_wgt, kl_Nwgt,&
    gs_l, gs_wgt, gs_Nwgt,&
    tfd_l, tfd_tgtfd,&
    cen_l, cen_alpha,&
    l1_cost, tv_cost, tsv_cost, kl_cost, gs_cost,&
    tfd_cost, cen_cost,&
    cost, gradcost, &
    N1d)
  implicit none

  ! Image
  integer,  intent(in)  :: N1d, Nx, Ny
  integer,  intent(in)  :: xidx(N1d), yidx(N1d)
  real(dp), intent(in)  :: I1d(N1d)
  real(dp), intent(in)  :: Nxref, Nyref

  ! parameter for l1
  real(dp), intent(in)  :: l1_l              ! lambda
  integer,  intent(in)  :: l1_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: l1_wgt(l1_Nwgt)   ! weight

  ! parameter for total variation
  real(dp), intent(in)  :: tv_l              ! lambda
  integer,  intent(in)  :: tv_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: tv_wgt(tv_Nwgt)   ! weight

  ! parameter for total squared variation
  real(dp), intent(in)  :: tsv_l             ! lambda
  integer,  intent(in)  :: tsv_Nwgt          ! size of the weight vector
  real(dp), intent(in)  :: tsv_wgt(tsv_Nwgt) ! weight

  ! parameter for kl divergence
  real(dp), intent(in)  :: kl_l              ! lambda
  integer,  intent(in)  :: kl_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: kl_wgt(kl_Nwgt)   ! weight

  ! parameter for Gull & Skilling entropy
  real(dp), intent(in)  :: gs_l              ! lambda
  integer,  intent(in)  :: gs_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: gs_wgt(gs_Nwgt)   ! weight

  ! parameter for the total flux density regularization
  real(dp), intent(in)  :: tfd_l             ! lambda (Normalized)
  real(dp), intent(in)  :: tfd_tgtfd         ! target total flux

  ! parameter for the centoroid regularization
  real(dp), intent(in)  :: cen_l             ! lambda (Normalized)
  real(dp), intent(in)  :: cen_alpha         ! alpha

  ! regularization function
  real(dp), intent(out) :: l1_cost    ! cost of l1
  real(dp), intent(out) :: tv_cost    ! cost of tv
  real(dp), intent(out) :: tsv_cost   ! cost of tsv
  real(dp), intent(out) :: kl_cost    ! cost of KL divergence
  real(dp), intent(out) :: gs_cost    ! cost of GS entropy
  real(dp), intent(out) :: tfd_cost   ! cost of total flux regularization
  real(dp), intent(out) :: cen_cost   ! cost of centoroid regularizaiton

  ! Total Cost function
  real(dp), intent(out) :: cost             ! cost function
  real(dp), intent(out) :: gradcost(1:N1d)  ! gradient of the cost function

  ! integer and real
  integer   :: ipix
  real(dp)  :: tmp

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:)
  real(dp), allocatable :: tmp1d(:)

  ! Initialize
  l1_cost     = 0d0
  tv_cost     = 0d0
  tsv_cost    = 0d0
  gs_cost     = 0d0
  kl_cost     = 0d0
  tfd_cost    = 0d0
  cen_cost    = 0d0
  cost        = 0d0
  gradcost(:) = 0d0

  ! Totalflux regularization
  if (tfd_l > 0) then
    call calc_tfdreg(I1d,tfd_tgtfd,tfd_cost,tmp,N1d)
    tfd_cost = tfd_l * tfd_cost
    gradcost = gradcost + tfd_l * tmp
  end if

  ! Center of mass regularization
  if (cen_l > 0) then
    allocate(tmp1d(N1d))
    call calc_cenreg(I1d,xidx,yidx,Nxref,Nyref,cen_alpha,tmp,tmp1d,N1d)
    cen_cost = cen_l * tmp
    gradcost = gradcost + cen_l * tmp1d
    deallocate(tmp1d)
  end if

  ! Compute pixel-based regularizations (MEM, l1, tv, tsv)
  !   Allocate two dimensional array if needed
  if (tv_l > 0 .or. tsv_l > 0) then
    allocate(I2d(Nx,Ny))
    I2d(:,:)=0d0
    call I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  end if
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d, I1d, I2d, xidx, yidx,&
  !$OMP                l1_l,  l1_wgt,&
  !$OMP                tv_l,  tv_wgt,&
  !$OMP                tsv_l, tsv_wgt,&
  !$OMP                gs_l,  gs_wgt,&
  !$OMP                kl_l,  kl_wgt) &
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+: l1_cost, tv_cost, tsv_cost, gs_cost, kl_cost, gradcost)
  do ipix=1, N1d
    ! weighted L1
    if (l1_l > 0) then
      l1_cost = l1_cost + l1_l * l1_wgt(ipix) * l1_e(I1d(ipix))
      gradcost(ipix) = gradcost(ipix) + l1_l * l1_wgt(ipix) * l1_grade(I1d(ipix))
    end if

    ! KL divergence
    if (kl_l > 0) then
      kl_cost = kl_cost + kl_l * kl_e(I1d(ipix),kl_wgt(ipix))
      gradcost(ipix) = gradcost(ipix) + kl_l * kl_grade(I1d(ipix),kl_wgt(ipix))
    end if

    ! GS Entropy
    if (gs_l > 0) then
      gs_cost = gs_cost + gs_l * gs_e(I1d(ipix),gs_wgt(ipix))
      gradcost(ipix) = gradcost(ipix) + gs_l * gs_grade(I1d(ipix),gs_wgt(ipix))
    end if

    ! TV
    if (tv_l > 0) then
      tv_cost = tv_cost + tv_l * tv_wgt(ipix) * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      gradcost(ipix) = gradcost(ipix) + tv_l * tv_wgt(ipix) * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if

    ! TSV
    if (tsv_l > 0) then
      tsv_cost = tsv_cost + tsv_l * tsv_wgt(ipix) * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      gradcost(ipix) = gradcost(ipix) + tsv_l * tsv_wgt(ipix) * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if
  end do
  !$OMP END PARALLEL DO

  ! de allocate array if needed
  if (tv_l > 0 .or. tsv_l > 0) then
    deallocate(I2d)
  end if

  ! take summation of all the cost function
  cost = l1_cost + tv_cost + tsv_cost + kl_cost + gs_cost + tfd_cost + cen_cost
end subroutine


!-------------------------------------------------------------------------------
! Regularization Function: weighted l1-norm (smoothed version)
!-------------------------------------------------------------------------------
! compute normalized weight for the l1-norm
subroutine init_l1reg(l1_prior,l1_wgt,N1d)
  implicit none

  integer,  intent(in)  :: N1d            ! Number of pixels
  real(dp), intent(in)  :: l1_prior(N1d)  ! Input Prior Image
  real(dp), intent(out) :: l1_wgt(N1d)    ! l1 weight

  integer   :: ipix

  ! compute weights
  do ipix=1, N1d
    l1_wgt(ipix) = 1/l1_e(l1_prior(ipix))
  end do

  ! normalize weights
  l1_wgt = l1_wgt / l1(l1_prior,l1_wgt,N1d)
end subroutine


! compute weighted l1-norm
real(dp) function l1(I1d,l1_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d           ! Number of Pixels
  real(dp), intent(in) :: I1d(N1d)      ! Input Image
  real(dp), intent(in) :: l1_wgt(N1d)   ! weight

  integer :: ipix

  ! compute weights
  l1 = 0d0
  do ipix=1, N1d
    l1 = l1 + l1_e(I1d(ipix)) * l1_wgt(ipix)
  end do
end function


! l1-norm of each pixel
real(dp) function l1_e(I)
  implicit none
  real(dp), intent(in) :: I
  l1_e = smabs(I)
end function


! gradient of l1-norm at each pixel
real(dp) function l1_grade(I)
  implicit none
  real(dp), intent(in) :: I
  l1_grade = smabs_diff(I)
end function


!-------------------------------------------------------------------------------
! Regularization Function: Kullback-Leibler Divergence (relative entropy)
!   Approximate that it can be differentiable everywhere
!-------------------------------------------------------------------------------
! Initialize Prior
subroutine init_klreg(kl_l_in,kl_prior,kl_l,kl_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d           ! Number of Pixels
  real(dp), intent(in) :: kl_l_in       ! input lambda
  real(dp), intent(in) :: kl_prior(N1d) ! prior image
  real(dp), intent(out):: kl_l          ! normalized lambda
  real(dp), intent(out):: kl_wgt(N1d)   ! weight

  ! compute prior
  call calc_kl_wgt(kl_prior,kl_wgt,N1d)
  ! normalize Lambda
  kl_l = kl_l_in / kl(kl_wgt,kl_wgt,N1d)
end subroutine


subroutine calc_kl_wgt(I1d,kl_wgt,N1d)
  implicit none

  integer,  intent(in)  :: N1d       ! Number of Pixels
  real(dp), intent(in)  :: I1d(N1d)  ! input image
  real(dp), intent(out) :: kl_wgt(N1d) ! prior image

  integer   :: ipix

  do ipix=1, N1d
    kl_wgt(ipix) = smabs(I1d(ipix))
  end do
end subroutine

! compute the Kullback-Leibler divergence
real(dp) function kl(I1d,kl_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d        ! Number of Pixels
  real(dp), intent(in) :: I1d(N1d)   ! input Image
  real(dp), intent(in) :: kl_wgt(N1d)  ! prior Image

  integer   :: ipix

  ! compute weights
  kl = 0d0
  do ipix=1, N1d
    kl = kl + kl_e(I1d(ipix),kl_wgt(ipix))
  end do
end function

real(dp) function kl_e(I, P)
  ! when I = P, the total sum will be |P|_1/e
  implicit none
  real(dp),intent(in) :: I, P  ! P must be positive
  real(dp) :: absI

  absI = smabs(I)
  kl_e = absI * log(absI/P) + P/e
end function


real(dp) function kl_grade(I, P)
  implicit none
  real(dp),intent(in) :: I, P       ! P must be positive
  real(dp) :: absI, gradabsI

  absI = smabs(I)
  gradabsI = smabs_diff(I)
  kl_grade = gradabsI * (log(absI/P)+1)
end function


!-------------------------------------------------------------------------------
! Gull & Skilling Entropy
!   Approximate that it can be differentiable everywhere
!-------------------------------------------------------------------------------
! Initialize Prior
subroutine init_gsreg(gs_prior,gs_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d           ! Number of Pixels
  real(dp), intent(in) :: gs_prior(N1d) ! prior image
  real(dp), intent(out):: gs_wgt(N1d)   ! weight

  integer   :: ipix

  do ipix=1, N1d
    gs_wgt(ipix) = smabs(gs_prior(ipix))
  end do
end subroutine

! compute the Kullback-Leibler divergence
real(dp) function gs(I1d,gs_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d         ! Number of Pixels
  real(dp), intent(in) :: I1d(N1d)    ! Input Image
  real(dp), intent(in) :: gs_wgt(N1d) ! weight

  integer   :: ipix

  ! compute weights
  gs = 0d0
  do ipix=1, N1d
    gs = gs + gs_e(I1d(ipix),gs_wgt(ipix))
  end do
end function

real(dp) function gs_e(I, P)
  implicit none
  real(dp),intent(in) :: I, P  ! P must be nonnegative
  real(dp) :: absI

  absI = smabs(I)
  gs_e = absI * (log(absI/P) - 1) + P
end function
!
!
real(dp) function gs_grade(I, P)
  implicit none
  real(dp),intent(in) :: I, P  ! P must be nonnegative
  real(dp) :: absI, gradabsI

  absI = smabs(I)
  gradabsI = smabs_diff(I)
  gs_grade = gradabsI * log(absI/P)
end function
!
!
!-------------------------------------------------------------------------------
! Isotropic Total Variation
!   Approximate that it can be differentiable everywhere
!-------------------------------------------------------------------------------
! Initialize the total variation
subroutine init_tvreg(xidx,yidx,Nx,Ny,tv_isflat,tv_prior,tv_wgt,N1d)

  integer,  intent(in)  :: N1d,Nx,Ny ! Number of Pixels
  integer,  intent(in)  :: xidx(N1d), yidx(N1d)
  logical,  intent(in)  :: tv_isflat
  real(dp), intent(in)  :: tv_prior(N1d)
  real(dp), intent(out) :: tv_wgt(N1d)

  if (tv_isflat .eqv. .True.) then
    ! Flat prior
    tv_wgt(:) = 1/(zeroeps + sum(tv_prior))/N1d
  else
    ! (re-)weighted TV
    call calc_tv_wgt(tv_prior,xidx,yidx,Nx,Ny,tv_wgt,N1d)
  end if
end subroutine


! derive the weights
subroutine calc_tv_wgt(I1d,xidx,yidx,Nx,Ny,tv_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d,Nx,Ny ! Number of Pixels
  real(dp), intent(in) :: I1d(N1d)  ! Input Image
  integer,  intent(in) :: xidx(N1d), yidx(N1d)
  real(dp), intent(out):: tv_wgt(N1d) ! weight

  integer   :: ipix
  real(dp)  :: I2d(Nx,Ny)

  ! copy the array to the 2d image
  I2d = 0d0
  call I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)

  ! compute weights
  do ipix=1, N1d
    tv_wgt(ipix) = 1/tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
  end do

  ! Normalize weights
  tv_wgt = tv_wgt/tv(I1d,xidx,yidx,Nx,Ny,tv_wgt,N1d)
end subroutine


! compute the Total Variation
real(dp) function tv(I1d,xidx,yidx,Nx,Ny,tv_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d,Nx,Ny ! Number of Pixels
  real(dp), intent(in) :: I1d(N1d)  ! Input Image
  integer, intent(in) :: xidx(1:N1d), yidx(1:N1d)
  real(dp), intent(in) :: tv_wgt(N1d) ! weight

  integer   :: ipix
  real(dp)  :: I2d(Nx,Ny)

  ! copy the array to the 2d image
  I2d = 0d0
  call I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)

  ! compute weights
  tv = 0d0
  do ipix=1, N1d
    tv = tv + tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny) * tv_wgt(ipix)
  end do
end function
!
!
real(dp) function tv_e(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer,  intent(in)  ::Nx,Ny,xidx,yidx
  real(dp), intent(in)  ::I2d(Nx,Ny)
  !
  integer   ::i1,j1,i2,j2
  real(dp)  ::dIx,dIy
  !
  i1 = xidx
  j1 = yidx
  i2 = i1 + 1
  j2 = j1 + 1

  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx = I2d(i2,j1) - I2d(i1,j1)
  end if

  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  ! smooth TV
  tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
end function
!
!
! Gradient of Isotropic Total Variation
real(dp) function tv_grade(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: Nx,Ny
  integer, intent(in) :: xidx, yidx
  real(dp),intent(in) :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  real(dp):: dIx,dIy,tv_e
  !
  ! initialize tsv term
  tv_grade = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  !-------------------------------------
  ! (i2,j1)-(i1,j1), (i1,j2)-(i1,j1)
  !-------------------------------------
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx = I2d(i2,j1) - I2d(i1,j1)
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  !tv_e = sqrt(dIx*dIx+dIy*dIy)
  !if (tv_e > zeroeps) then
  !  tv_grade = tv_grade - (dIx + dIy)/tv_e
  !end if
  !
  ! Smooth TV
  tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
  tv_grade = tv_grade - (dIx + dIy)/tv_e
  !
  !-------------------------------------
  ! (i1,j1)-(i0,j1), (i0,j2)-(i0,j1)
  !-------------------------------------
  if (i0 > 0) then
    ! dIx = I(i,j) - I(i-1,j)
    dIx = I2d(i1,j1) - I2d(i0,j1)

    ! dIy = I(i-1,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy = I2d(i0,j2) - I2d(i0,j1)
    end if

    !tv_e = sqrt(dIx*dIx+dIy*dIy)
    !if (tv_e > zeroeps) then
    !  tv_grade = tv_grade + dIx/tv_e
    !end if
    !
    ! Smooth TV
    tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
    tv_grade = tv_grade + dIx/tv_e
  end if
  !
  !-------------------------------------
  ! (i2,j0)-(i1,j0), (i1,j1)-(i1,j0)
  !-------------------------------------
  if (j0 > 0) then
    ! dIy = I(i,j) - I(i,j-1)
    dIy = I2d(i1,j1) - I2d(i1,j0)

    ! dIx = I(i+1,j-1) - I(i,j-1)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx = I2d(i2,j0) - I2d(i1,j0)
    end if

    !tv_e = sqrt(dIx*dIx+dIy*dIy)
    !if (tv_e > zeroeps) then
    !  tv_grade = tv_grade + dIy/tv_e
    !end if
    !
    ! Smooth TV
    tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
    tv_grade = tv_grade + dIx/tv_e
  end if
  !
end function
!
!
!-------------------------------------------------------------------------------
! Total Squared Variation
!   Approximate that it can be differentiable everywhere
!-------------------------------------------------------------------------------
! Initialize the total squared variation
subroutine init_tsvreg(xidx,yidx,Nx,Ny,tsv_isflat,tsv_prior,tsv_wgt,N1d)
  integer,  intent(in)  :: N1d, Nx, Ny
  integer,  intent(in)  :: xidx(1:N1d), yidx(1:N1d)  ! pixel coordinate
  logical,  intent(in)  :: tsv_isflat
  real(dp), intent(in)  :: tsv_prior(N1d)
  real(dp), intent(out) :: tsv_wgt(N1d)

  if (tsv_isflat .eqv. .True.) then
    ! flat prior
    tsv_wgt(:) = 1/(zeroeps + sum(tsv_prior)**2/N1d)/N1d
  else
    ! (re)weighted TSV
    call calc_tsv_wgt(tsv_prior,xidx,yidx,Nx,Ny,tsv_wgt,N1d)
  end if
end subroutine


! derive the weights
subroutine calc_tsv_wgt(I1d,xidx,yidx,Nx,Ny,tsv_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d,Nx, Ny    ! Number of Pixels,
  integer,  intent(in) :: xidx(1:N1d), yidx(1:N1d)  ! pixel coordinate
  real(dp), intent(in) :: I1d(N1d)      ! Input Image
  real(dp), intent(out):: tsv_wgt(N1d)  ! weight

  ! compute weights
  tsv_wgt = 1/(I1d**2+zeroeps)

  ! Normalize weights
  tsv_wgt = tsv_wgt/tsv(I1d,xidx,yidx,Nx,Ny,tsv_wgt,N1d)
end subroutine


! compute the Total Variation
real(dp) function tsv(I1d,xidx,yidx,Nx,Ny,tsv_wgt,N1d)
  implicit none

  integer,  intent(in) :: N1d,Nx,Ny                 ! Number of Pixels
  real(dp), intent(in) :: I1d(N1d)                  ! Input Image
  integer,  intent(in) :: xidx(1:N1d), yidx(1:N1d)  ! pixel coordinate
  real(dp), intent(in) :: tsv_wgt(N1d)              ! weight

  integer   :: ipix
  real(dp)  :: I2d(Nx,Ny)

  ! copy the array to the 2d image
  I2d = 0d0
  call I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)

  ! compute weights
  tsv = 0d0
  do ipix=1, N1d
    tsv = tsv + tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny) * tsv_wgt(ipix)
  end do
end function


real(dp) function tsv_e(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i1,j1,i2,j2
  real(dp):: dIx,dIy
  !
  ! initialize tsv term
  i1 = xidx
  j1 = yidx
  i2 = i1 + 1
  j2 = j1 + 1
  !
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx  = I2d(i2,j1) - I2d(i1,j1)
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy  = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  tsv_e = dIx*dIx+dIy*dIy
end function


! Gradient of Total Squared Variation
real(dp) function tsv_grade(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  integer, intent(in)  :: xidx, yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  !
  ! initialize tsv term
  tsv_grade = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 <= Nx) then
    tsv_grade = tsv_grade - 2*(I2d(i2,j1) - I2d(i1,j1))
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 <= Ny) then
    tsv_grade = tsv_grade - 2*(I2d(i1,j2) - I2d(i1,j1))
  end if
  !
  ! dIx = I(i,j) - I(i-1,j)
  if (i0 > 0) then
    tsv_grade = tsv_grade + 2*(I2d(i1,j1) - I2d(i0,j1))
  end if
  !
  ! dIy = I(i,j) - I(i,j-1)
  if (j0 > 0) then
    tsv_grade = tsv_grade + 2*(I2d(i1,j1) - I2d(i1,j0))
  end if
  !
end function


!-------------------------------------------------------------------------------
! Total Flux Regularization
!-------------------------------------------------------------------------------
subroutine init_tfdreg(tfd_l_in, tfd_tgtfd, tfd_tgter, tfd_l)
  implicit none
  !
  real(dp),intent(in)  :: tfd_l_in   ! input Lambda value for this regularization
  real(dp),intent(in)  :: tfd_tgtfd  ! Target total flux density
  real(dp),intent(in)  :: tfd_tgter  ! Target fractional error
  real(dp),intent(out) :: tfd_l      ! Normalized Lambda

  ! normalize lambda with the target total flux density and fractional error
  tfd_l = tfd_l_in / (tfd_tgtfd * tfd_tgter)**2
end subroutine


subroutine calc_tfdreg(I1d,tgtfd,cost,gradcost,N1d)
  implicit none
  !
  integer, intent(in) :: N1d      ! number of data sets
  real(dp),intent(in) :: I1d(N1d) ! input image
  real(dp),intent(in) :: tgtfd    ! target total flux
  real(dp),intent(out):: cost     ! cost function
  real(dp),intent(out):: gradcost ! gradient is constant,
                                  ! so this is just 1 real value
  real(dp) :: resid

  resid = sum(I1d)-tgtfd   ! take the total flux and residual
  cost = resid**2
  gradcost = 2*resid
end subroutine


!-------------------------------------------------------------------------------
! Centoroid regularization
!-------------------------------------------------------------------------------
subroutine init_cenreg(cen_l_in, cen_prior, cen_alpha, cen_l, N1d)
  implicit none
  !
  integer,intent(in)   :: N1d            ! the number of the pixel in the image
  real(dp),intent(in)  :: cen_l_in        ! input Lambda value for this regularization
  real(dp),intent(in)  :: cen_prior(N1d) ! Prior Image
  real(dp),intent(in)  :: cen_alpha       ! power
  real(dp),intent(out) :: cen_l           ! Normalized Lambda

  integer :: ipix
  real(dp) :: Ip, sumI

  ! taking the sum of the intensity
  sumI = 0
  do ipix=1,N1d
    if (abs(cen_alpha-1)<zeroeps) then
      Ip = smabs(cen_prior(ipix))
    else
      Ip = smabs(cen_prior(ipix))**cen_alpha
    end if
    sumI = sumI + Ip
  end do

  ! normalize lambda with the total intensity
  cen_l = cen_l_in / (zeroeps + sumI)
end subroutine


subroutine calc_cenreg(I1d,xidx,yidx,Nxref,Nyref,alpha,cost,gradcost,N1d)
  implicit none
  !
  integer, intent(in)  :: N1d
  real(dp),intent(in)  :: I1d(1:N1d)
  integer, intent(in)  :: xidx(1:N1d), yidx(1:N1d)
  real(dp),intent(in)  :: alpha
  real(dp),intent(in)  :: Nxref, Nyref
  real(dp),intent(out) :: cost
  real(dp),intent(out) :: gradcost(1:N1d)
  !
  real(dp) :: dix, diy, Ip, gradIp
  real(dp) :: sumx, sumy

  integer :: ipix

  ! initialize
  sumx = 0d0
  sumy = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,I1d,N1d) &
  !$OMP   PRIVATE(ipix, dix, diy, Ip) &
  !$OMP   REDUCTION(+: sumx, sumy)
  do ipix=1, N1d
    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! take a alpha
    if (abs(alpha-1)<zeroeps) then
      Ip = smabs(I1d(ipix))
    else
      Ip = smabs(I1d(ipix))**alpha
    end if

    ! calculate sum
    sumx = sumx + Ip * dix
    sumy = sumy + Ip * diy
  end do
  !$OMP END PARALLEL DO

  ! cost function
  cost = sqrt(sumx**2+sumy**2+zeroeps)
  ! calculate gradient of cost function
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,I1d,N1d,sumx,sumy,cost) &
  !$OMP   PRIVATE(ipix,dix,diy,gradIp) &
  !$OMP   REDUCTION(+:gradcost)
  do ipix=1, N1d
    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! gradient of sum
    if (abs(alpha-1)<zeroeps) then
      gradIp = smabs_diff(I1d(ipix))
    else
      gradIp = smabs_diff(I1d(ipix)) * alpha*smabs(I1d(ipix))**(alpha-1)
    end if

    ! calculate gradint of cost function
    gradcost(ipix) = (dix*sumx+diy*sumy)/cost*gradIp
  end do
  !$OMP END PARALLEL DO
end subroutine


!-------------------------------------------------------------------------------
! Absolute apporoximater
!-------------------------------------------------------------------------------
! Smoothed absolute opperator
!   |x| ~ sqrt(x^2 + e)
real(dp) function smabs(x)
  implicit none
  real(dp),intent(in) :: x
  smabs = sqrt(x**2+zeroeps)
end function


!   d|x|/dx ~ x/sqrt(x^2 + e)
real(dp) function smabs_diff(x)
  implicit none
  real(dp),intent(in) :: x
  smabs_diff = x/smabs(x)
end function


!-------------------------------------------------------------------------------
! Copy 1D image vector from/to 2D image vector
!-------------------------------------------------------------------------------
!
! I1d --> I2d
!
subroutine I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  !
  !
  implicit none
  !
  integer, intent(in) :: N1d,Nx,Ny
  integer, intent(in) :: xidx(N1d), yidx(N1d)
  real(dp),intent(in) :: I1d(N1d)
  real(dp),intent(inout) :: I2d(Nx,Ny)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d,I1d,xidx,yidx) &
  !$OMP   PRIVATE(i)
  do i=1,N1d
    I2d(xidx(i),yidx(i))=I1d(i)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
! I1d <-- I2d
!
subroutine I1d_I2d_inv(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: N1d,Nx,Ny
  integer, intent(in) :: xidx(N1d), yidx(N1d)
  real(dp),intent(inout) :: I1d(N1d)
  real(dp),intent(in) :: I2d(Nx,Ny)
  !
  integer :: i
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d,I2d,xidx,yidx) &
  !$OMP   PRIVATE(i)
  do i=1,N1d
    I1d(i)=I2d(xidx(i),yidx(i))
  end do
  !$OMP END PARALLEL DO
end subroutine


!-------------------------------------------------------------------------------
! A convinient function to compute regularization functions
! for python interfaces
!-------------------------------------------------------------------------------
subroutine ixy2ixiy(ixy,ix,iy,Nx)
  implicit none

  ! arguments
  integer, intent(in):: ixy,Nx
  integer, intent(out):: ix,iy
  !
  ix = mod(ixy-1,Nx)+1
  iy = (ixy-1)/Nx+1
end subroutine
!
!
subroutine ixiy2ixy(ix,iy,ixy,Nx)
  implicit none

  ! arguments
  integer, intent(in):: ix,iy,Nx
  integer, intent(out):: ixy
  !
  ixy = ix + (iy-1) * Nx
end subroutine
end module
