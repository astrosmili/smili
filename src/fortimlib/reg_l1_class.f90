module reg_l1_class
  !$use omp_lib
  use param, only: dp, i4b, zeroeps
  use funcs, only: smabs, gradsmabs
  implicit none

  ! declaration of the class
  type reg_l1
    private
      ! lambda
      real(dp) :: lambda
      real(dp) :: lambda_norm

      ! Option for re-weighting
      !   0: no-normalization
      !   1: scaled by the total flux
      !   2: reweight with prior image (same dimension to input image(s))
      integer(i4b) :: doweight

      ! weight
      real(dp), allocatable :: weight(:)

      ! size of the image
      integer(i4b) :: Npix
  contains
    private
      procedure :: calc_weight => reg_l1_calc_weight
      procedure :: cost_e => reg_l1_cost_e
      procedure :: grad_e => reg_l1_grad_e
      procedure :: cost_1d => reg_l1_cost_I1d
  end type

  ! declaration of the constructor
  interface reg_l1
    module procedure reg_l1_init
  end interface reg_l1
contains
!-------------------------------------------------------------------------------
! Constructor
!-------------------------------------------------------------------------------
type(reg_l1) function reg_l1_init(lambda, doweight, totalflux, Npix, prior)
  implicit none

  real(dp),     intent(in) :: lambda
  integer(i4b), intent(in) :: doweight
  integer(i4b), intent(in) :: totalflux

  integer(i4b), intent(in) :: Npix
  real(dp),     intent(in), optional :: prior(Npix)

  ! set parameters
  reg_l1_init%lambda = lambda
  reg_l1_init%lambda_norm = lambda/totalflux
  reg_l1_init%doweight = doweight
  
  ! if weighting is enabled, compute weights
  if (doweight > 0 .and. lambda > 0) then
    ! compute weights
    call reg_l1_init%calc_weight(prior, Npix)
  end if
end function
!-------------------------------------------------------------------------------
! Compute weights
!-------------------------------------------------------------------------------
subroutine reg_l1_calc_weight(self, prior, Npix)
  implicit none
  class(reg_l1) :: self

  integer(i4b), intent(in) :: Npix
  real(dp),     intent(in) :: prior(Npix)

  integer(i4b) :: ipix
  real(dp)     :: wsum

  ! allocate weight
  !   check if weight is allocated
  if (allocated(self%weight) .eqv. .True.) then
    deallocate(self%weight)
  end if
  !   allocate weight
  allocate(self%weight(Npix))

  ! set weight size
  self%Npix = Npix

  ! initialize sum
  wsum = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, prior) &
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+:wsum, self%weight)
  do ipix=1, Npix
    self%weight(ipix) = 1/(smabs(prior(ipix))+zeroeps)
    wsum = wsum + self%weight(ipix)
  end do
  !$OMP END PARALLEL DO

  ! normalize
  self%weight = self%weight / wsum
end subroutine
!-------------------------------------------------------------------------------
! element wise cost function calculation
!-------------------------------------------------------------------------------
real(dp) function reg_l1_cost_e(self, I, ipix)
  implicit none
  class(reg_l1) :: self

  integer(i4b), intent(in) :: ipix
  real(dp),     intent(in) :: I

  if (self%doweight .eq. 0) then
    reg_l1_cost_e = self%lambda * smabs(I)
  else
    reg_l1_cost_e = self%lambda * self%weight(ipix) * smabs(I)
  end if
end function

real(dp) function reg_l1_grad_e(self, I, ipix)
  implicit none
  class(reg_l1) :: self

  integer(i4b), intent(in) :: ipix
  real(dp),     intent(in) :: I

  if (self%doweight .eq. 0) then
    reg_l1_grad_e = self%lambda * gradsmabs(I)
  else
    reg_l1_grad_e = self%lambda * self%weight(ipix) * gradsmabs(I)
  end if
end function
!-------------------------------------------------------------------------------
! compute cost functions and gradients from 1D image
!-------------------------------------------------------------------------------
subroutine reg_l1_cost_I1d(self, I1d, cost, gradcost, Npix)
  implicit none
  class(reg_l1) :: self

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
