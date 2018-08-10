module image
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc, e
  implicit none

  ! Epsiron for Zero judgement
  real(dp), parameter :: zeroeps=1d-12

  ! regularization class
  !   l1-norm
  type l1_parm
    private
      real(dp) :: lambda
      real(dp), allocatable :: prior(:)
      real(dp), allocatable :: weight(:)
  contains
    private
      procedure ::  => circle_area
  end type



contains
!-------------------------------------------------------------------------------
! Compute regularization functions (and print it)
!-------------------------------------------------------------------------------
!
!  This is the fast version for imaging
!
subroutine calc_regfuncs_fast()
  implicit none

  ! image
  integer,  intent(in) :: N1d,Nx,Ny
  integer,  intent(in) :: xidx(N1d),yidx(N1d)
  real(dp), intent(in) :: I1d(Npix)

  ! regularization parameters
  !   l1
  real(dp), intent(in) :: l1_lamb
  real(dp), intent(in) :: l1_w(N1d)
  !   tv
  real(dp), intent(in) :: tv_lamb
  real(dp), intent(in) :: tv_w(N1d)
  !   tsv
  real(dp), intent(in) :: tsv_lamb
  real(dp), intent(in) :: tsv_w(N1d)
  !   gse1 version 1
  real(dp), intent(in) :: gse1_lamb
  real(dp), intent(in) :: gse1_p(N1d)
  !   gse1 version 2
  real(dp), intent(in) :: gse2_lamb
  real(dp), intent(in) :: gse2_p(N1d)
  !   c.o.m. regularization version 1 (non-convex version)
  real(dp), intent(in) :: com1_lamb
  real(dp), intent(in) :: com1_alpha
  !   c.o.m. regularization version 2 (convex version)
  !real(dp), intent(in) :: com2_lamb
  !real(dp), intent(in) :: com2_alpha

  real(dp), intent(out) :: cost
  real(dp), intent(out) :: gradcost

  !------------------------------------
  ! Regularization Functions
  !------------------------------------
  ! Allocate 2D array
  if (lambtv > 0 .or. lambtsv > 0) then
    allocate(I2d(Nx,Ny))
    I2d(:,:)=0d0
    call I1d_I2d_fwd(xidx,yidx,I1d,I2d,Npix,Nx,Ny)
  end if

  ! Compute  term
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(I1d, I2d, xidx, yidx, Nx, Ny, Npix,&
  !$OMP                l1_lamb, l1_w,&
  !$OMP                tv_lamb, tv_w,&
  !$OMP                tsv_lamb, tsv_w,&
  !$OMP                gse
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+: cost, gradcost)
  do ipix=1, Npix
    ! l1
    if (l1_lamb > 0) then
      cost = cost + l1_lamb * l1_w(ipix) * l1_e(I1d(ipix))
      gradcost(ipix) = gradcost(ipix) + l1_lamb * l1_w(ipix) * l1_grade(I1d(ipix))
    end if

    ! total variation
    if (tv_lamb > 0) then
      cost = cost + tv_lamb * tv_w(ipix) * tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      gradcost(ipix) = gradcost(ipix) + tv_lamb * tv_w(ipix) * tv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if

    ! total squared variation
    if (tsv_w > 0) then
      cost = cost + tsv_lamb * tsv_w(ipix) * tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
      gradcost(ipix) = gradcost(ipix) + tsv_lamb * tsv_w(ipix) * tsv_grade(xidx(ipix),yidx(ipix),I2d,Nx,Ny)
    end if

    ! gull and skilling entropy (type 1)
    if (gse1_lamb > 0) then
      cost = cost + gse1_lamb * gse1_e(I1d(ipix),gse1_p(ipix))
      gradcost(ipix) = gradcost(ipix) + gse1_lamb * gse1_grade(I1d(ipix),gse1_p(ipix))
    end if

    ! gull and skilling entropy (type 2)
    if (gse2_lamb > 0) then
      cost = cost + gse2_lamb * gse2_e(I1d(ipix),gse2_p(ipix))
      gradcost(ipix) = gradcost(ipix) + gse2_lamb * gse2_grade(I1d(ipix),gse2_p(ipix))
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine

!-------------------------------------------------------------------------------
! l1-norm (smooth l1)
!-------------------------------------------------------------------------------
real(dp) function l1_e(I)
  implicit none
  real(dp),intent(in) :: I

  l1_e = sqrt(I**2+zeroeps)
end function

real(dp) function l1_grade(I)
  implicit none

  real(dp),intent(in) :: I

  l1_grade = I/l1_e(I)
end function

subroutine l1_P1d_w(P1d, l1_w, N1d)
  implicit none

  integer,  intent(in)  :: N1d
  real(dp), intent(in)  :: P1d(N1d)  ! prior image
  real(dp), intent(out) :: l1_w(N1d) ! gradients

  ! initialize sum
  wsum = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d, P1d) &
  !$OMP   PRIVATE(i) &
  !$OMP   REDUCTION(+:wsum, l1_w)
  do i=1, N1d
    l1_w(i) = 1/(l1_e(P1d(i))+zeroeps)
    wsum = wsum + l1_w(i)
  end do
  !$OMP END PARALLEL DO

  ! normalize
  l1_w = l1_w / wsum
end subroutine

subroutine l1_I1d_reg(I1d, P1d, reg, gradreg, N1d)
  implicit none

  integer,  intent(in)  :: N1d
  real(dp), intent(in)  :: I1d(N1d)     ! image
  real(dp), intent(in)  :: P1d(N1d)     ! prior image
  real(dp), intent(out) :: reg          ! regularization function
  real(dp), intent(out) :: gradreg(N1d) ! gradients

  real(dp) :: l1_w(N1d) ! gradients

  ! initialize
  reg = 0d0
  gradreg = 0d0

  ! calc weights
  call l1_P1d_w(P1d, l1_w, N1d)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d, I1d, l1_w) &
  !$OMP   PRIVATE(i) &
  !$OMP   REDUCTION(+:reg,gradreg)
  do i=1, N1d
    reg = reg + l1_e(I1d(i)) * l1_w(i)
    gradreg(i) = l1_grade(I1d(i)) * l1_w(i)
  end do
  !$OMP END PARALLEL DO
end subroutine

!-------------------------------------------------------------------------------
! Gull & Skilling Entropy type 1 (Simple in EHT-IMAGING)
!-------------------------------------------------------------------------------
real(dp) function gse1_e(I,P)
  implicit none

  real(dp),intent(in) :: I,P
  real(dp) :: absI

  absI = l1_e(I)
  absP = l1_e(P)
  gse1_e = absI * log(absI/absP) + absP/e
end function

real(dp) function gse1_grade(I,P)
  implicit none

  real(dp),intent(in) :: I,P
  real(dp) :: absI, gradabsI

  absI = l1_e(I)
  absP = l1_e(P)
  gradabsI = l1_grade(I)
  gse1_grade = gradabsI * (1+log(absI/absP))
end function

real function gse1_P1d_w(P1d, N1d)
  implicit none

  integer,  intent(in)  :: N1d
  real(dp), intent(in)  :: P1d(N1d)  ! prior image

  ! initialize
  gse1_P1d_w = 0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d, P1d) &
  !$OMP   PRIVATE(i) &
  !$OMP   REDUCTION(+:gse1_P1d_w)
  do i=1, N1d
    gse1_P1d_w = gse1_w + l1_e(P1d(i))
  end do
  !$OMP END PARALLEL DO

  ! normalize
  gse1_P1d_w = e / gse1_P1d_w
end function

subroutine gse1_I1d_reg(I1d, P1d, reg, gradreg, N1d)
  implicit none

  integer,  intent(in)  :: N1d
  real(dp), intent(in)  :: I1d(N1d)     ! image
  real(dp), intent(in)  :: P1d(N1d)     ! prior image
  real(dp), intent(out) :: reg          ! regularization function
  real(dp), intent(out) :: gradreg(N1d) ! gradients

  real(dp) :: gse1_w ! normalization factor

  ! initialize
  reg = 0d0
  gradreg = 0d0

  ! calc weights
  gse1_w = gse1_P1d_w(P1d, N1d)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d, I1d, P1d, gse1_w) &
  !$OMP   PRIVATE(i) &
  !$OMP   REDUCTION(+:reg,gradreg)
  do i=1, N1d
    reg = reg + gse1_e(I1d(i),P1d(i)) * gse1_w
    gradreg(i) = gse1_grade(I1d(i),P1d(i)) * gse1_w
  end do
  !$OMP END PARALLEL DO
end subroutine


!-------------------------------------------------------------------------------
! Gull & Skilling Entropy type 2 (GS in EHT-IMAGING)
!-------------------------------------------------------------------------------
real(dp) function gse2_e(I,P)
  implicit none

  real(dp),intent(in) :: I,P
  real(dp) :: absI

  absI = l1_e(I)
  absP = l1_e(P)
  gse2_e = absI * log(absI/absP) - absI + absP
end function

real(dp) function gse2_grade(I,P)
  implicit none

  real(dp),intent(in) :: I,P
  real(dp) :: absI, gradabsI

  absI = l1_e(I)
  absP = l1_e(P)
  gradabsI = l1_grade(I)
  gse2_grade = gradabsI * log(absI/absP)
end function

subroutine gse2_I1d_reg(I1d, P1d, reg, gradreg, N1d)
  implicit none

  integer,  intent(in)  :: N1d
  real(dp), intent(in)  :: I1d(N1d)     ! image
  real(dp), intent(in)  :: P1d(N1d)     ! prior image
  real(dp), intent(out) :: reg          ! regularization function
  real(dp), intent(out) :: gradreg(N1d) ! gradients

  ! initialize
  reg = 0d0
  gradreg = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d, I1d, P1d) &
  !$OMP   PRIVATE(i) &
  !$OMP   REDUCTION(+:reg,gradreg)
  do i=1, N1d
    reg = reg + gse2_e(I1d(i),P1d(i))
    gradreg(i) = gse2_grade(I1d(i),P1d(i))
  end do
  !$OMP END PARALLEL DO
end subroutine

!-------------------------------------------------------------------------------
! Isotropic TV
!-------------------------------------------------------------------------------
real(dp) function tv_e(xidx,yidx,I2d,Nx,Ny)
  implicit none

  integer,  intent(in)  ::Nx,Ny,xidx,yidx
  real(dp), intent(in)  ::I2d(Nx,Ny)

  integer   ::i1,j1,i2,j2
  real(dp)  ::dIx,dIy

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

  ! smooth TV
  tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
end function

real(dp) function tv_grade(xidx,yidx,I2d,Nx,Ny)
  implicit none

  integer, intent(in) :: Nx,Ny
  integer, intent(in) :: xidx, yidx
  real(dp),intent(in) :: I2d(Nx,Ny)

  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  real(dp):: dIx,dIy,tv_e

  ! initialize tsv term
  tv_grade = 0d0

  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1

  !-------------------------------------
  ! (i2,j1)-(i1,j1), (i1,j2)-(i1,j1)
  !-------------------------------------
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

  ! Smooth TV
  tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
  tv_grade = tv_grade - (dIx + dIy)/tv_e

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

    ! Smooth TV
    tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
    tv_grade = tv_grade + dIx/tv_e
  end if

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

    ! Smooth TV
    tv_e = sqrt(dIx*dIx+dIy*dIy+zeroeps)
    tv_grade = tv_grade + dIx/tv_e
  end if

end function

!-------------------------------------------------------------------------------
! Total Squared Variation (TSV)
!-------------------------------------------------------------------------------
real(dp) function tsv_e(xidx,yidx,I2d,Nx,Ny)
  implicit none

  integer, intent(in)  :: Nx,Ny,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)

  ! variables
  integer :: i1,j1,i2,j2
  real(dp):: dIx,dIy

  i1 = xidx
  j1 = yidx
  i2 = i1 + 1
  j2 = j1 + 1

  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx  = I2d(i2,j1) - I2d(i1,j1)
  end if

  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy  = I2d(i1,j2) - I2d(i1,j1)
  end if

  tsv_e = dIx*dIx+dIy*dIy
end function

real(dp) function tsv_grade(xidx,yidx,I2d,Nx,Ny)
  implicit none

  integer, intent(in)  :: Nx,Ny
  integer, intent(in)  :: xidx, yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)

  ! variables
  integer :: i0,j0,i1,j1,i2,j2

  ! initialize tsv term
  tsv_grade = 0d0

  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1

  ! dIx = I(i+1,j) - I(i,j)
  if (i2 <= Nx) then
    tsv_grade = tsv_grade - 2*(I2d(i2,j1) - I2d(i1,j1))
  end if

  ! dIy = I(i,j+1) - I(i,j)
  if (j2 <= Ny) then
    tsv_grade = tsv_grade - 2*(I2d(i1,j2) - I2d(i1,j1))
  end if

  ! dIx = I(i,j) - I(i-1,j)
  if (i0 > 0) then
    tsv_grade = tsv_grade + 2*(I2d(i1,j1) - I2d(i0,j1))
  end if

  ! dIy = I(i,j) - I(i,j-1)
  if (j0 > 0) then
    tsv_grade = tsv_grade + 2*(I2d(i1,j1) - I2d(i1,j0))
  end if
end function

!-------------------------------------------------------------------------------
! COM Regularization Version 1 (non-convex version)
!-------------------------------------------------------------------------------
subroutine I1d_com1_reg(xidx,yidx,Nxref,Nyref,alpha,I1d,reg,gradreg,Npix)
  implicit none

  integer, intent(in) :: Npix
  integer, intent(in) :: xidx(1:Npix), yidx(1:Npix)
  real(dp),intent(in) :: I1d(1:Npix)
  real(dp),intent(in) :: Nxref, Nyref
  real(dp),intent(in) :: alpha
  real(dp),intent(out) :: reg
  real(dp),intent(out) :: gradreg(1:Npix)

  real(dp) :: dix, diy, Ip
  real(dp) :: sumx, sumy, sumI
  real(dp) :: gradsumx, gradsumy, gradsumI

  integer :: ipix

  ! initialize sum
  sumx = 0d0
  sumy = 0d0
  sumI = 0d0
  reg = 0
  gradreg = 0

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

  ! calculate reg function
  !   need zeroeps for smoothing sqrt,
  sumI = sumI + zeroeps
  reg = sqrt((sumx/(sumI))**2+(sumy/(sumI))**2+zeroeps)

  ! calculate gradient of reg function
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,I1d,Npix,sumx,sumy,sumI,reg) &
  !$OMP   PRIVATE(ipix,dix,diy,gradsumI,gradsumx,gradsumy) &
  !$OMP   REDUCTION(+:gradreg)
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

    ! calculate gradint of reg function
    gradreg(ipix) = gradreg(ipix) + (sumx/sumI*gradsumx+sumy/sumI*gradsumy)/reg
  end do
  !$OMP END PARALLEL DO
end subroutine

!-------------------------------------------------------------------------------
! Copy 1D image vector from/to 2D/3D image vector
!-------------------------------------------------------------------------------
! I2d <-- I1d
subroutine I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  implicit none

  integer, intent(in)    :: N1d,Nx,Ny
  integer, intent(in)    :: xidx(N1d), yidx(N1d)
  real(dp),intent(in)    :: I1d(N1d)
  real(dp),intent(inout) :: I2d(Nx,Ny)

  integer :: i

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(N1d,I1d,xidx,yidx) &
  !$OMP   PRIVATE(i)
  do i=1,N1d
    I2d(xidx(i),yidx(i))=I1d(i)
  end do
  !$OMP END PARALLEL DO
end subroutine

! I1d <-- I2d
subroutine I1d_I2d_inv(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  implicit none

  integer, intent(in)    :: N1d,Nx,Ny
  integer, intent(in)    :: xidx(N1d), yidx(N1d)
  real(dp),intent(inout) :: I1d(N1d)
  real(dp),intent(in)    :: I2d(Nx,Ny)

  integer :: i

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

  ix = mod(ixy-1,Nx)+1
  iy = (ixy-1)/Nx+1
end subroutine


subroutine ixiy2ixy(ix,iy,ixy,Nx)
  implicit none

  ! arguments
  integer, intent(in):: ix,iy,Nx
  integer, intent(out):: ixy

  ixy = ix + (iy-1) * Nx
end subroutine
end module
