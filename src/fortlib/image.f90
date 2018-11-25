module image
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  implicit none
  ! Epsiron for Zero judgement
  real(dp), parameter :: zeroeps=1d-10
contains
!-------------------------------------------------------------------------------
! Calc cost function
!-------------------------------------------------------------------------------
subroutine calc_cost_reg()
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  integer,  intent(in) :: xidx(Npix), yidx(Npix)
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: Nxref, Nyref

  ! parameter for l1
  real(dp), intent(in) :: l1_l          ! lambda
  real(dp), intent(in) :: l1_w(Npix)    ! weight

  ! parameter for total variation
  real(dp), intent(in) :: tv_l          ! lambda
  real(dp), intent(in) :: tv_w(Npix)    ! weight

  ! parameter for total squared variation
  real(dp), intent(in) :: tsv_l         ! lambda
  real(dp), intent(in) :: tsv_w(Npix)   ! weight

  ! parameter for simple shannon entropy
  real(dp), intent(in) :: shent_l       ! lambda
  real(dp), intent(in) :: shent_p(Npix) ! weight

  ! parameter for Gull & Skilling entropy
  real(dp), intent(in) :: gsent_l       ! lambda
  real(dp), intent(in) :: gsent_p(Npix) ! weight

  ! parameter for the center of mass regularization
  real(dp), intent(in) :: com_l         ! lambda
  real(dp), intent(in) :: com_a         ! weight

  ! parameter for the total flux regularization
  real(dp), intent(in) :: tf_l  ! lambda
  real(dp), intent(in) :: tf_t  ! the target total flux

  ! regularization function
  real(dp), intent(out) :: l1_cost    ! cost of l1
  real(dp), intent(out) :: tv_cost    ! cost of tv
  real(dp), intent(out) :: tsv_cost   ! cost of tsv
  real(dp), intent(out) :: gsent_cost ! cost of Gull & Skilling entropy
  real(dp), intent(out) :: shent_cost ! cost of shannon entropy
  real(dp), intent(out) :: tf_cost    ! cost of total flux

  ! Total Cost function
  real(dp), intent(out) :: cost
  real(dp), intent(out) :: gradcost(1:Npix)

  ! integer
  integer :: ipix

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:)

  !------------------------------------
  ! Regularization Functions
  !------------------------------------
  ! Initialize
  l1_cost = 0d0
  tv_cost = 0d0
  tsv_cost = 0d0
  gsent_cost = 0d0
  shent_cost = 0d0
  tf_cost = 0d0
  cost = 0d0
  gradcost(:) = 0d0

  ! Allocate two dimensional array if needed
  if (tv_l > 0 .or. tsv_l > 0) then
    allocate(I2d(Nx,Ny))
    I2d(:,:)=0d0
    call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)
  end if

  ! Compute regularization term
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, Iin, I2d, xidx, yidx,&
  !$OMP                l1_l, l1_w,&
  !$OMP                tv_l, tv_w,&
  !$OMP                tsv_l, tsv_w,&
  !$OMP                gsent_l, gsent_p,&
  !$OMP                shent_l, shent_p) &
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+: l1_cost, tv_cost, tsv_cost, gsent_cost, shent_cost, cost, gradcost)
  do ipix=1, Npix
    ! L1
    if (l1_l > 0) then
      l1_cost = l1_cost + l1_l * l1_w(ipix) * l1_e(Iin(ipix))
      gradcost(ipix) = gradcost(ipix) + l1_l * l1_w(ipix) * l1_grade(Iin(ipix))
    end if

    ! Shannon Entropy
    if (shent_l > 0) then
      shent_cost = shent_cost + shent_l * shent_e(Iin(ipix),shent_p)
      gradcost(ipix) = gradcost(ipix) + shent_l * shent_grade(Iin(ipix),shent_p)
    end if

    ! Gull & Skilling Entropy
    if (gsent_l > 0) then
      gsent_cost = gsent_cost + gsent_l * gsent_e(Iin(ipix),gsent_p)
      gradcost(ipix) = gradcost(ipix) + gsent_l * gsent_grade(Iin(ipix),gsent_p)
    end if

    ! TV
    if (tv_l > 0) then
      tv_cost = tv_cost + tv_l * tv_w(ipix) * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      gradcost(ipix) = gradcost(ipix) + tv_l * tv_w(ipix) * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if

    ! TSV
    if (tsv_l > 0) then
      tsv_cost = tsv_cost + tsv_l * tsv_w(ipix) * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      gradcost(ipix) = gradcost(ipix) + tsv_l * tsv_w(ipix) * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if
  end do
  !$OMP END PARALLEL DO

  ! Total Flux regularization

  ! Center of Mass regularization

  ! de allocate array if needed
  if (tv_l > 0 .or. tsv_l > 0) then
    deallocate(I2d)
  end if
end subroutine
!
!
!-------------------------------------------------------------------------------
! Regularization Function
!-------------------------------------------------------------------------------
!
! l1-norm (Smoothed version)
!
real(dp) function l1_e(I)
  implicit none
  real(dp), intent(in) :: I
  l1_e = smabs(I)
end function
!
real(dp) function l1_grade(I)
  implicit none
  real(dp), intent(in) :: I
  l1_grade = smabs_diff(I)
end function
!
!
! A simple version of the Shannon Information Entropy
!   Approximate that it can be differentiable everywhere
!
real(dp) function shent_e(I, P)
  implicit none

  real(dp),intent(in) :: I, P
  real(dp) :: absI, absP

  absI = smabs(I)
  absP = smabs(P)
  shent_e = absI * log(absI/absP)
end function
!
real(dp) function shent_grade(I, P)
  implicit none

  real(dp),intent(in) :: I, P
  real(dp) :: absI, absP, gradabsI

  absI = smabs(I)
  absP = smabs(P)
  gradabsI = smabs_diff(I)
  shent_grade = gradabsI * (1+log(absI))
end function
!
! Gull & Skilling Entropy
!   Approximate that it can be differentiable everywhere
!
real(dp) function gsent_e(I, P)
  implicit none

  real(dp),intent(in) :: I, P
  real(dp) :: absI, absP

  absI = smabs(I)
  absP = smabs(P)
  gsent_e = absI * (log(absI/absP)-1)
end function
!
real(dp) function gsent_grade(I, P)
  implicit none

  real(dp),intent(in) :: I, P
  real(dp) :: absI, absP

  absI = smabs(I)
  absP = smabs(P)
  gradabsI = smabs_diff(I)
  ent_grade = gradabsI * (1+log(absI))
end function
!
! Isotropic Total Variation
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
  ! Literal TV
  !tv_e = sqrt(dIx*dIx+dIy*dIy)
  !
  ! smooth TV
  tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
end function
!
! Gradient of Isotropic Total Variation
!
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
! Total Squared Variation
!
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
!
! Gradient of Total Squared Variation
!
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
!
! Centoroid Regularization
!
subroutine comreg(xidx,yidx,Nxref,Nyref,alpha,I1d,cost,gradcost,Npix)
  implicit none
  !
  integer, intent(in) :: Npix
  integer, intent(in) :: xidx(1:Npix), yidx(1:Npix)
  real(dp),intent(in) :: alpha
  real(dp),intent(in) :: Nxref, Nyref
  real(dp),intent(in) :: I1d(1:Npix)
  real(dp),intent(inout) :: cost
  real(dp),intent(inout) :: gradcost(1:Npix)
  !
  real(dp) :: dix, diy, Ip
  real(dp) :: sumx, sumy, sumI
  real(dp) :: gradsumx, gradsumy, gradsumI
  real(dp) :: reg
  !
  integer :: ipix

  sumx = 0d0
  sumy = 0d0
  sumI = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,I1d,Npix) &
  !$OMP   PRIVATE(ipix, dix, diy, Ip) &
  !$OMP   REDUCTION(+: sumx, sumy, sumI)
  do ipix=1, Npix
    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! take a alpha
    if (abs(alpha-1)<zeroeps) then
      Ip = l1_e(I1d(ipix))
    else
      Ip = l1_e(I1d(ipix))**alpha
    end if

    ! calculate sum
    sumx = sumx + Ip * dix
    sumy = sumy + Ip * diy
    sumI = sumI + Ip
  end do
  !$OMP END PARALLEL DO

  ! Smooth Version
  !
  ! calculate cost function
  !   need zeroeps for smoothing sqrt,
  sumI = sumI + zeroeps
  reg = sqrt((sumx/(sumI))**2+(sumy/(sumI))**2+zeroeps)
  cost = cost + reg

  ! calculate gradient of cost function
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,I1d,Npix,sumx,sumy,sumI,reg) &
  !$OMP   PRIVATE(ipix,dix,diy,gradsumI,gradsumx,gradsumy) &
  !$OMP   REDUCTION(+:gradcost)
  do ipix=1, Npix
    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! gradient of sum
    if (abs(alpha-1)<zeroeps) then
      gradsumI = l1_grade(I1d(ipix))
    else
      gradsumI = alpha*l1_e(I1d(ipix))**(alpha-1)*l1_grade(I1d(ipix))
    end if

    gradsumx = gradsumI*dix
    gradsumy = gradsumI*diy

    ! gradient of sumx/sumI or sumy/sumI
    gradsumx = (sumI*gradsumx - gradsumI*sumx)/sumI**2
    gradsumy = (sumI*gradsumy - gradsumI*sumy)/sumI**2

    ! calculate gradint of cost function
    gradcost(ipix) = gradcost(ipix) + (sumx/sumI*gradsumx+sumy/sumI*gradsumy)/reg
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
!-------------------------------------------------------------------------------!
subroutine comreg3d(xidx,yidx,Nxref,Nyref,alpha,Iin,cost,gradcost,Npix,Nz)
  implicit none
  !
  integer, intent(in) :: Npix,Nz
  integer, intent(in) :: xidx(1:Npix), yidx(1:Npix)
  real(dp),intent(in) :: alpha
  real(dp),intent(in) :: Nxref, Nyref
  real(dp),intent(in) :: Iin(Npix,Nz)
  real(dp),intent(inout) :: cost
  real(dp),intent(inout) :: gradcost(Npix,Nz)
  !
  real(dp) :: dix, diy, Ip, Isum(Npix)
  real(dp) :: sumx, sumy, sumI
  real(dp) :: gradsumx, gradsumy, gradsumI
  real(dp) :: reg
  !
  integer  :: ipix,iz

  sumx = 0d0
  sumy = 0d0
  sumI = 0d0

  ! Take summation
  Isum = sum(Iin,2)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,Isum,Npix,Nz) &
  !$OMP   PRIVATE(ipix, dix, diy, Ip) &
  !$OMP   REDUCTION(+: sumx, sumy, sumI)
  do ipix=1, Npix
    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! take a alpha
    if (abs(alpha-1)<zeroeps) then
      Ip = l1_e(Isum(ipix))
    else
      Ip = l1_e(Isum(ipix))**alpha
    end if

    ! calculate sum
    sumx = sumx + Ip * dix
    sumy = sumy + Ip * diy
    sumI = sumI + Ip
  end do
  !$OMP END PARALLEL DO

  ! Smooth Version
  !
  ! calculate cost function
  !   need zeroeps for smoothing sqrt,
  sumI = sumI + zeroeps
  reg = sqrt((sumx/(sumI))**2+(sumy/(sumI))**2+zeroeps)
  cost = cost + reg


  ! calculate gradient of cost function
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,Isum,Npix,sumx,sumy,sumI,reg) &
  !$OMP   PRIVATE(ipix,dix,diy,gradsumI,gradsumx,gradsumy) &
  !$OMP   REDUCTION(+:gradcost)
  do ipix=1, Npix
    do iz=1,Nz
      ! pixel from the reference pixel
      dix = xidx(ipix) - Nxref
      diy = yidx(ipix) - Nyref

      ! gradient of sum
      if (abs(alpha-1)<zeroeps) then
        gradsumI = l1_grade(Iin(ipix,iz))
      else
        gradsumI = alpha*l1_e(Isum(ipix))**(alpha-1)*l1_grade(Iin(ipix,iz))   ! <----
      end if

      gradsumx = gradsumI*dix
      gradsumy = gradsumI*diy

      ! gradient of sumx/sumI or sumy/sumI
      gradsumx = (sumI*gradsumx - gradsumI*sumx)/sumI**2
      gradsumy = (sumI*gradsumy - gradsumI*sumy)/sumI**2

      ! calculate gradint of cost function
      gradcost(ipix,iz) = gradcost(ipix,iz) + (sumx/sumI*gradsumx+sumy/sumI*gradsumy)/reg
    end do
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
!
!   d|x|/dx ~ x/sqrt(x^2 + e)
real(dp) function smabs_diff(x)
  implicit none
  real(dp),intent(in) :: x
  smabs_diff = x/smabs(x)
end function
!
!
!-------------------------------------------------------------------------------
! Copy 1D image vector from/to 2D/3D image vector
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
!
!
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


subroutine ixiy2ixy(ix,iy,ixy,Nx)
  implicit none

  ! arguments
  integer, intent(in):: ix,iy,Nx
  integer, intent(out):: ixy
  !
  ixy = ix + (iy-1) * Nx
end subroutine
end module
