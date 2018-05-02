module coord
  !$use omp_lib
  use param, only: dp, pi
  implicit none
contains
!
! Calcurate UV Coverage
!
subroutine calc_uvw(gst,alpha,delta,dx,dy,dz,u,v,w,Ndata)
    implicit none

    ! Arguments
    integer, intent(in)   :: Ndata          ! number of semented time
    real(dp), intent(in)  :: gst(Ndata)    ! Greenwich Sidereal Time (hour)
    real(dp), intent(in)  :: alpha(Ndata)  ! RA (rad)
    real(dp), intent(in)  :: delta(Ndata)  ! Dec (rad)
    real(dp), intent(in)  :: dx(Ndata)     ! Baseline Vecor (XYZ coordinate)
    real(dp), intent(in)  :: dy(Ndata)     ! Baseline Vecor (XYZ coordinate)
    real(dp), intent(in)  :: dz(Ndata)     ! Baseline Vecor (XYZ coordinate)
    real(dp), intent(out) :: u(Ndata)      ! output UVW coordinates (U)
    real(dp), intent(out) :: v(Ndata)      ! output UVW coordinates (V)
    real(dp), intent(out) :: w(Ndata)      ! output UVW coordinates (W)

    ! local variables
    integer  :: i
    real(dp) :: cosgh        ! cos of greenwich hour angle
    real(dp) :: singh        ! sin of greenwich hour angle
    real(dp) :: cosd         ! cos of declination
    real(dp) :: sind         ! sin of declination
    real(dp) :: bl_x         ! baseline vector (x axis)
    real(dp) :: bl_y         ! baseline vector (y axis)
    real(dp) :: bl_z         ! baseline vector (z axis)

    !$OMP PARALLEL DO DEFAULT(SHARED) &
    !$OMP   FIRSTPRIVATE(Ndata,gst,alpha,delta,dx,dy,dz) &
    !$OMP   PRIVATE(i,cosgh,singh,cosd,sind,bl_x,bl_y,bl_z)
    do i=1, Ndata
      ! calcurate cos, sin
      !   ra
      cosgh = cos(gst(i) * pi / 12 - alpha(i))
      singh = sin(gst(i) * pi / 12 - alpha(i))

      !   dec
      cosd = cos(delta(i))
      sind = sin(delta(i))

      ! baseline vector
      bl_x = cosgh*dx(i) - singh*dy(i)
      bl_y = singh*dx(i) + cosgh*dy(i)
      bl_z =                           + dz(i)

      ! calculate uv
      !    u = dot product of bl_vec_vector and eu
      u(i) =   bl_y
      !    v = dot product of bl_vec_vector and ev
      v(i) = - bl_x * sind + bl_z * cosd
      !    w = dot product of bl_vec_vector and ew
      w(i) = + bl_x * cosd + bl_z * sind
    end do
    !$OMP END PARALLEL DO
end subroutine
end module
