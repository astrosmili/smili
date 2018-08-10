module funcs
  use param
  implicit none
contains
!-------------------------------------------------------------------------------
! Smooth absolute operator
!-------------------------------------------------------------------------------
real(dp) function smabs(x)
  implicit none
  real(dp), intent(in) :: x
  smabs = sqrt(x**2+zeroeps)
end function

real(dp) function gradsmabs(x)
  implicit none
  real(dp), intent(in) :: x
  gradsmabs = x/smabs(x)
end function
!-------------------------------------------------------------------------------
! Copy 1D image vector from/to 2D/3D image vector
!-------------------------------------------------------------------------------
! I2d <-- I1d
subroutine I1d_I2d_fwd(xidx,yidx,I1d,I2d,N1d,Nx,Ny)
  implicit none

  integer(i4b), intent(in)  :: N1d,Nx,Ny
  integer(i4b), intent(in)  :: xidx(N1d), yidx(N1d)
  real(dp),     intent(in)  :: I1d(N1d)
  real(dp),     intent(out) :: I2d(Nx,Ny)

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

  integer(i4b), intent(in)  :: N1d,Nx,Ny
  integer(i4b), intent(in)  :: xidx(N1d), yidx(N1d)
  real(dp),     intent(out) :: I1d(N1d)
  real(dp),     intent(in)  :: I2d(Nx,Ny)

  integer :: i

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
  integer(i4b), intent(in):: ixy,Nx
  integer(i4b), intent(out):: ix,iy

  ix = mod(ixy-1,Nx)+1
  iy = (ixy-1)/Nx+1
end subroutine

subroutine ixiy2ixy(ix,iy,ixy,Nx)
  implicit none

  ! arguments
  integer(i4b), intent(in):: ix,iy,Nx
  integer(i4b), intent(out):: ixy

  ixy = ix + (iy-1) * Nx
end subroutine
end module
