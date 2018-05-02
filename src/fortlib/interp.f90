module interp
  use param, only: dp
contains
!
!  subroutines and functions
!
subroutine solve_tridiag(a,b,c,d,x,n)
  !
  ! This subroutine solves a linear equation (Ax=d), where A is a tridiagonal
  ! matrix consisting of vectors a, b, c.
  !
  ! a - sub-diagonal (means it is the diagonal below the main diagonal)
  ! b - the main diagonal
  ! c - sup-diagonal (means it is the diagonal above the main diagonal)
  ! d - right part
  ! x - the answer
  ! n - number of equations
  !
  ! This function is originally from https://en.wikibooks.org, and slightly
  ! modified by Kazu Akiyama.
  !
  implicit none
  integer,intent(in) :: n
  real(dp), intent(in) :: a(2:n),b(1:n),c(1:n-1),d(1:n)
  real(dp), intent(out) :: x(1:n)
  real(dp) :: dc(1:n),dd(1:n)
  real(dp) :: m
  integer i

  ! initialize c-prime and d-prime
  dc(1) = c(1)/b(1)
  dd(1) = d(1)/b(1)

  ! solve for vectors c-prime and d-prime
  do i = 2,n
    m = b(i)-dc(i-1)*a(i)
    dc(i) = c(i)/m
    dd(i) = (d(i)-dd(i-1)*a(i))/m
  end do

  ! initialize x
  x(n) = dd(n)

  ! solve for x from the vectors c-prime and d-prime
  do i = n-1, 1, -1
    x(i) = dd(i)-dc(i)*x(i+1)
  end do
end subroutine solve_tridiag


function locate(xx,x,n)
  !
  ! This subroutine is copied from Numerical Recipe F90, and arranged
  ! by Kazu Akiyama
  !
  ! given an array xx(1:n), and given a value x, returns a value j such that x
  ! is between xx(j) and xx(j+1). xx must be monotonic, either increasing or
  ! decreasing. j=0 or j=n is returned to indicate that x is out of range.
  !
  implicit none
  integer,  intent(in) :: n
  real(dp), intent(in) :: xx(1:n)
  real(dp), intent(in) :: x
  integer :: locate
  integer :: jl,jm,ju
  logical :: ascnd

  ascnd = (xx(n) >= xx(1)) ! true if ascending order of table, false otherwise.
  jl=0                     ! initialize lower
  ju=n+1                   ! and upper limits.

  do
    if (ju-jl <= 1) exit   ! repeat until this condition is satisfied.
    jm=(ju+jl)/2           ! compute a midpoint,
    if (ascnd .eqv. (x >= xx(jm))) then
      jl=jm                ! and replace either the lower limit
    else
      ju=jm                ! or the upper limit, as appropriate.
    end if
  end do

  if (x == xx(1)) then  ! then set the output, being careful with the endpoints.
    locate=1
  else if (x == xx(n)) then
    locate=n-1
  else
    locate=jl
  end if

end function


subroutine spline(x, y, yp1, ypn, y2, n)
  !
  ! This subroutine is copied from Numerical Recipe F90, and arranged
  ! by Kazu Akiyama
  !
  ! Given arrays x and y of length N containing a tabulated function i.e.
  ! yi = f(xi), with x1<x2<...<xN, and given values yp1 and ypN for the first
  ! derivative of the interpolating function at point 1 and N, respectively,
  ! this routine returns an array y2 of length N that contains the second
  ! derivertives of the interpolating function at the tabulated points xi.
  ! If yp1 and/or ypn are equal to 1x10e30 or larger (default values),
  ! the routine is signaled to set the corresponding boundary condition for
  ! a natural spline, with zero second derivative on that boundary.
  !
  implicit none
  integer,  intent(in)  :: n
  real(dp), intent(in)  :: x(1:n), y(1:n)
  real(dp), intent(in)  :: yp1, ypn
  real(dp), intent(out) :: y2(1:n)
  real(dp) :: a(1:n), b(1:n), c(1:n), r(1:n)

  c(1:n-1) = x(2:n) - x(1:n-1)
  r(1:n-1) = 6.0d0*((y(2:n)-y(1:n-1))/c(1:n-1))
  r(2:n-1) = r(2:n-1)-r(1:n-2)
  a(2:n-1) = c(1:n-2)
  b(2:n-1) = 2.0d0*(c(2:n-1)+a(2:n-1))
  b(1) = 1.0d0
  b(n) = 1.0d0

  ! set the boundary condition
  if (yp1 > 0.99d30) then
    r(1)=0.0d0
    c(1)=0.0d0
  else
    r(1)=(3.0d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
    c(1)=0.5d0
  end if
  if (ypn > 0.99d30) then
    r(n)=0.0d0
    a(n)=0.0d0
  else
    r(n)=(-3.0d0/(x(n)-x(n-1)))*((y(n)-y(n-1))/(x(n)-x(n-1))-ypn)
    a(n)=0.5d0
  end if

  ! derive the spline coefficients
  call solve_tridiag(a(2:n),b(1:n),c(1:n-1),r(1:n),y2(1:n),n)
end subroutine


subroutine splint(xa,ya,y2a,x,y,na)
  !
  ! This subroutine is copied from Numerical Recipe F90, and arranged
  ! by Kazu Akiyama
  !
  ! Given the arrays xa and ya, which tabulate a function (with the xa(i)’s in
  ! increasing or decreasing order), and given the array y2a, which is the
  ! output from spline above, and given a value of x, this routine returns
  ! a cubic-spline interpolated value. The arrays xa, ya and y2a are all of the
  ! same size.
  !
  implicit none
  integer :: na
  real(dp), intent(in) :: xa(1:na),ya(1:na),y2a(1:na) ! input tablated data
  real(dp), intent(in) :: x ! at each value in x, interpolation is exermined.
  real(dp), intent(out):: y ! cubic-spline interpolated value on x

  integer :: khi,klo
  real(dp) :: a,b,h

  !klo and khi now bracket the input value of x.
  klo=max(min(locate(xa,x,na),na-1),1)
  khi=klo+1
  h=xa(khi)-xa(klo)
  a=(xa(khi)-x)/h
  b=(x-xa(klo))/h
  y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**2)/6.0d0
end subroutine


subroutine splintvec(xa,ya,y2a,x,y,na,n)
  !
  ! This is mostly same to the above subroutine 'splint' but this function
  ! can evalute cubic-spline-interpolated values on a vector x.
  !
  implicit none
  integer :: na, n
  real(dp), intent(in) :: xa(1:na),ya(1:na),y2a(1:na) ! input tablated data
  real(dp), intent(in) :: x(1:n) ! at each value in x, interpolation is exermined.
  real(dp), intent(out):: y(1:n) ! cubic-spline interpolated value on x
  integer :: i

  do i=1, n
    if ((x(i) > maxval(xa)) .or. (x(i) < minval(xa))) then
      y(i) = 0d0
    else
      call splint(xa,ya,y2a,x(i),y(i),na)
    end if
  end do
end subroutine


subroutine splie2(xa, ya, za, z2a, nx, ny)
  !
  ! This subroutine is copied from Numerical Recipe F90, and arranged
  ! by Kazu Akiyama
  !
  implicit none
  integer,  intent(in) :: nx,ny
  real(dp), intent(in) :: xa(1:nx), ya(1:ny), za(1:nx,1:ny)
  real(dp), intent(out):: z2a(1:nx,1:ny)
  integer :: iy
  !
  do iy=1, ny
    call spline(xa(1:nx),za(1:nx,iy), 1d30, 1d30, z2a(1:nx,iy), nx)
  end do
end subroutine


subroutine splin2(xa, ya, za, z2a, x, y, z, nxa, nya)
  !
  ! This subroutine is copied from Numerical Recipe F90, and arranged
  ! by Kazu Akiyama
  !
  implicit none
  integer,  intent(in) :: nxa, nya
  real(dp), intent(in) :: xa(1:nxa),ya(1:nya),za(1:nxa,1:nya),z2a(1:nxa,1:nya)
  real(dp), intent(in) :: x,y
  real(dp), intent(out):: z
  real(dp) :: ztmp(1:nya), zztmp(1:nya)
  integer :: iy
  !
  do iy=1, nya
    call splint(xa(1:nxa),za(1:nxa,iy),z2a(1:nxa,iy),x,zztmp(iy),nxa)
  end do
  call spline(ya(1:nya), zztmp(1:nya), 1d30, 1d30, ztmp(1:nya), nya)
  call splint(ya(1:nya), zztmp(1:nya), ztmp(1:nya), y, z, nya)
end subroutine


subroutine splin2vec(xa, ya, za, z2a, x, y, z, nxa, nya, n)
  !
  ! This is mostly same to the above subroutine 'splin2' but this function
  ! can evalute cubic-spline-interpolated values on vectors (x, y).
  !
  ! Note that, for grid-to-grid interpolation, the next subroutine 'splin2grid'
  ! is much more efficient and faster, because it omits redundant calculations
  ! for sets of data with the same x-axis coordinates.
  !
  implicit none
  integer,  intent(in) :: nxa, nya, n
  real(dp), intent(in) :: xa(1:nxa),ya(1:nya),za(1:nxa,1:nya),z2a(1:nxa,1:nya)
  real(dp), intent(in) :: x(1:n),y(1:n)
  real(dp), intent(out):: z(1:n)
  integer :: i
  !
  do i=1,n
    if ((x(i) > maxval(xa)) .or. (x(i) < minval(xa)) .or. &
        (y(i) > maxval(ya)) .or. (y(i) < minval(ya))) then
      z(i) = 0d0
    else
      call splin2(xa, ya, za, z2a, x(i), y(i), z(i), nxa, nya)
    end if
  end do
end subroutine


subroutine splin2grid(xa, ya, za, z2a, x, y, z, nxa, nya, nx, ny)
  !
  ! This subroutine is a variant of the subroutine splin2 in Numerical Recipe
  ! F90, arranged by Kazu Akiyama. This fucntion is optimized for the
  ! two-dimensional bi-cubic interpolation on the grid-to-grid basis.
  !
  ! This subroutine omits reduandant calculations of spline coefficients
  ! for sets of data with the same x-axis coordinates.
  !
  implicit none
  integer,  intent(in) :: nx, ny, nxa, nya
  real(dp), intent(in) :: xa(1:nxa),ya(1:nya),za(1:nxa,1:nya),z2a(1:nxa,1:nya)
  real(dp), intent(in) :: x(1:nx), y(1:ny)
  real(dp), intent(out):: z(1:nx,1:ny)
  real(dp) :: ztmp(1:nya,1:nx), zztmp(1:nya,1:nx)
  integer :: ix, iy, iya
  real(dp) :: xamin, xamax, yamin, yamax

  ! get minimum and maximam values of reference grids
  xamin = minval(xa)
  xamax = maxval(xa)
  yamin = minval(ya)
  yamax = maxval(ya)

  ! calculate coeffcients using x-axis information
  do ix=1, nx
    if ((x(ix) < xamin) .or. (x(ix) > xamax)) then
      zztmp(:,ix)=0d0
      ztmp(:,ix)=0d0
    else
      do iya=1, nya
        call splint(xa(1:nxa),za(1:nxa,iya),z2a(1:nxa,iya),x(ix),zztmp(iya,ix),nxa)
      end do
      call spline(ya(1:nya),zztmp(1:nya,ix),1d30,1d30,ztmp(1:nya,ix),nya)
    end if
  end do

  ! spline interplate along with the y-axis
  do iy=1, ny
    if ((y(iy) < yamin) .or. (y(iy) > yamax)) then
      z(:,iy) = 0d0
      continue
    else
      do ix=1, nx
        if ((x(ix) < xamin) .or. (x(ix) > xamax)) then
          z(ix,iy) = 0d0
        else
          call splint(ya(1:nya),zztmp(1:nya,ix),ztmp(1:nya,ix),&
                      y(iy),z(ix,iy),nya)
        end if
      end do
    end if
  end do
end subroutine

end module
