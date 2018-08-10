module reg_she_class
  !$use omp_lib
  use param, only: dp, i4b, zeroeps, e_dp
  use funcs, only: smabs, gradsmabs
  implicit none

  ! declaration of the class
  type reg_she
    private
      ! lambda
      real(dp) :: lambda       ! original lambda
      real(dp) :: lambda_norm  ! normalized lambda (used when prior is specified)
      ! Option for prior
      !   0: flat prior
      !   1: use prior (same dimension to input image(s))
      integer(i4b) :: isprior
      ! prior
      real(dp), allocatable :: prior(:)
      ! size of the image
      integer(i4b) :: Npix
  contains
    private
      procedure :: set_prior => reg_she_set_prior
      procedure :: cost_e => reg_she_cost_e
      procedure :: grad_e => reg_she_grad_e
      procedure :: cost_1d => reg_she_cost_I1d
  end type

  ! declaration of the constructor
  interface reg_she
    module procedure reg_she_init
  end interface reg_she
contains
!-------------------------------------------------------------------------------
! Constructor
!-------------------------------------------------------------------------------
type(reg_she) function reg_she_init(lambda, isprior, Npix, prior)
  implicit none

  real(dp),     intent(in) :: lambda
  integer(i4b), intent(in) :: isprior

  integer(i4b), intent(in) :: Npix
  real(dp),     intent(in), optional :: prior(Npix)

  ! set parameters
  reg_she_init%lambda = lambda
  reg_she_init%lambda_norm = lambda
  reg_she_init%isprior = isprior

  ! if prior is specified, set prior
  if (isprior > 0 .and. lambda > 0) then
    ! set prior
    call reg_she_init%set_prior(prior, Npix)
  end if
end function
!-------------------------------------------------------------------------------
! Set Prior
!-------------------------------------------------------------------------------
subroutine reg_she_set_prior(self, prior, Npix)
  implicit none
  class(reg_she) :: self

  integer(i4b), intent(in) :: Npix
  real(dp),     intent(in) :: prior(Npix)

  integer(i4b) :: ipix
  real(dp)     :: wsum

  ! set prior
  !   check if prior is allocated
  if (allocated(self%prior) .eqv. .True.) then
    deallocate(self%prior)
  end if
  !   allocate prior
  allocate(self%prior(Npix))
  self%prior = prior

  ! set prior size
  self%Npix = Npix

  ! initialize sum
  wsum = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, prior) &
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+:wsum)
  do ipix=1, Npix
    wsum = wsum + smabs(prior(ipix))
  end do
  !$OMP END PARALLEL DO

  ! normalize
  self%lambda_norm = self%lambda_norm * e_dp / wsum
end subroutine
!-------------------------------------------------------------------------------
! element wise cost function calculation
!-------------------------------------------------------------------------------
real(dp) function reg_she_cost_e(self, I, ipix)
  implicit none
  class(reg_she) :: self

  integer(i4b), intent(in) :: ipix
  real(dp),     intent(in) :: I
  real(dp) :: absI, absP

  if (self%isprior .eq. 0) then
    absI = smabs(I)
    reg_she_cost_e = self%lambda * (absI * log(absI/1) + 1/e_dp)
  else
    absI = smabs(I)
    absP = smabs(self%prior(ipix))
    reg_she_cost_e = self%lambda_norm * (absI * log(absI/absP) + absP/e_dp)
  end if
end function

real(dp) function reg_she_grad_e(self, I, ipix)
  implicit none
  class(reg_she) :: self

  integer(i4b), intent(in) :: ipix
  real(dp),     intent(in) :: I
  real(dp) :: absI, absP, gradabsI

  if (self%isprior .eq. 0) then
    absI = smabs(I)
    gradabsI = gradsmabs(I)
    reg_she_grad_e = self%lambda * gradabsI * (1+log(absI))
  else
    absI = smabs(I)
    absP = smabs(self%prior(ipix))
    gradabsI = gradsmabs(I)
    reg_she_grad_e = self%lambda_norm * gradabsI * (1+log(absI/absP))
  end if
end function
!-------------------------------------------------------------------------------
! compute cost functions and gradients from 1D image
!-------------------------------------------------------------------------------
subroutine reg_she_cost_I1d(self, I1d, cost, gradcost, Npix)
  implicit none
  class(reg_she) :: self

  integer(i4b), intent(in)  :: Npix
  real(dp),     intent(in)  :: I1d(Npix)
  real(dp),     intent(out) :: cost
  real(dp),     intent(out) :: gradcost(Npix)

  integer(i4b) :: ipix

  ! initialize
  cost = 0d0
  gradcost = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, I1d) &
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+:cost, gradcost)
  do ipix=1, Npix
    cost = cost + self%cost_e(I1d(ipix), ipix)
    gradcost(ipix) = self%grad_e(I1d(ipix), ipix)
  end do
  !$OMP END PARALLEL DO
end subroutine
end module
