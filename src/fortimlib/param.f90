module param
  use iso_c_binding
  implicit none

  !symbolic names for kind types of 2-, 4-, and 8-byte integers:
  integer, parameter :: i2b = c_short
  integer, parameter :: i4b = c_int
  integer, parameter :: i8b = c_long

  !symbolic names for kind types of single- and double-precision reals:
  integer, parameter :: sp = c_float
  integer, parameter :: dp = c_double

  !symbolic names for kind types of single- and double-precision complex:
  integer, parameter :: spc = c_float_complex
  integer, parameter :: dpc = c_double_complex

  !symbolic name for kind type of default logical:
  integer, parameter :: lgt = c_bool

  !frequently used mathematical constants (with precision to spare):
  real(sp), parameter :: hug_sp = huge(1.0)
  real(sp), parameter :: eps_sp = epsilon(1.0)
  real(sp), parameter :: tin_sp = tiny(1.0)
  real(dp), parameter :: hug_dp = huge(1d0)
  real(dp), parameter :: eps_dp = epsilon(1d0)
  real(dp), parameter :: tin_dp = tiny(1d0)
  real(dp), parameter :: pi_dp  = 4*atan(1d0)
  real(dp), parameter :: e_dp   = exp(1d0)
  complex(dpc), parameter :: i_dpc=dcmplx(0d0,1d0)

  !other parameters
  real(dp), parameter :: zeroeps = 1d-12
  real(dp), parameter :: finuffteps = 1d-12
end module
