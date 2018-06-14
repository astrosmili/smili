module image3d
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  use image, only : ixiy2ixy, ixy2ixiy, comreg, zeroeps
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
