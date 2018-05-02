module houghlib
  !$use omp_lib
  use param, only: dp, pi
  use interp, only: splie2, splin2grid
  implicit none
contains
!
!  subroutines and functions
!
subroutine circle_hough(Iin,x,y,R,Nth,H,Nx,Ny,Nr)
  !
  ! This is a subroutine to calculate the circle Hough transform H(Nx, Ny, Nr)
  ! of the input two dimentional function Iin(Nx, Ny).
  !
  ! R(Nr) is the radius used in the circle Hough transformation, x(Nx) and y(Ny)
  ! are tablulated coordinates of Iin along x and y axises, respectively.
  !
  ! Nth is the number of circular shifts used in the circle Hough transform.
  implicit none

  integer,  intent(in) :: Nx,Ny,Nr,Nth
  real(dp), intent(in) :: Iin(1:Nx,1:Ny), x(1:Nx), y(1:Ny)
  real(dp), intent(in) :: R(1:Nr)
  real(dp), intent(out) :: H(1:Nx,1:Ny,1:Nr)
  integer :: ir, ith
  real(dp) :: x1(1:Nx), y1(1:Ny), z1(1:Nx,1:Ny)
  real(dp) :: Iin2(1:Nx,1:Ny)
  real(dp) :: theta

  ! Derive Spline Co-efficient
  call splie2(x, y, Iin, Iin2, Nx, Ny)

  ! Initialize H
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP PRIVATE(ir)
  do ir=1,Nr
    H(1:Nx,1:Ny,ir)=0d0
  end do
  !$OMP END PARALLEL DO

  ! Calculate Classic Hough Transform
  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(Nx,Ny,Nr,Nth,Iin,Iin2,x,y,R) &
  !$OMP   PRIVATE(ir,ith,x1,y1,z1,theta)
  do ir=1,Nr
    do ith=1,Nth
      theta = 2*pi*real(ith-1)/Nth
      x1 = x + R(ir) * cos(theta)
      y1 = y + R(ir) * sin(theta)
      call splin2grid(x(1:Nx), y(1:Ny), Iin(1:Nx,1:Ny), Iin2(1:Nx,1:Ny),&
                      x1(1:Nx), y1(1:Ny), z1(1:Nx,1:Ny), Nx, Ny, Nx, Ny)
      H(1:Nx,1:Ny,ir) = H(1:Nx,1:Ny,ir) + z1(1:Nx,1:Ny)
    end do
    H(1:Nx,1:Ny,ir) = H(1:Nx,1:Ny,ir) * 2 * pi * R(ir)
  end do
  !$OMP END PARALLEL DO
end subroutine
end module
