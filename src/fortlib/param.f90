module param
  implicit none

  !symbolic names for kind types of 4-, 2-, and 1-byte integers:
  integer, parameter :: i4b = selected_int_kind(9)
  integer, parameter :: i2b = selected_int_kind(4)
  integer, parameter :: i1b = selected_int_kind(2)

  !symbolic names for kind types of single- and double-precision reals:
  integer, parameter :: sp = kind(1.0)
  integer, parameter :: dp = kind(1.0d0)

  !symbolic names for kind types of single- and double-precision complex:
  integer, parameter :: spc = kind((1.0,1.0))
  integer, parameter :: dpc = kind((1.0d0,1.0d0))

  !symbolic name for kind type of default logical:
  integer, parameter :: lgt = kind(.true.)

  !frequently used mathematical constants (with precision to spare):
  real(sp), parameter :: shug = huge(1.0)
  real(sp), parameter :: seps = epsilon(1.0)
  real(sp), parameter :: stin = tiny(1.0)
  real(dp), parameter :: dhug = huge(1d0)
  real(dp), parameter :: deps = epsilon(1d0)
  real(dp), parameter :: dtin = tiny(1d0)
  real(dp), parameter :: pi=4*atan(1d0)
  real(dp), parameter :: e=exp(1d0)
  complex(dpc), parameter :: i_dpc=dcmplx(0d0,1d0)
end module
