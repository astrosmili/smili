module fftlibqu
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  use image, only: I1d_I2d_fwd, I1d_I2d_inv
  use fftlib, only: NUFFT_fwd, NUFFT_adj_resid, &
                    phashift_c2r, phashift_r2c, &
                    calc_chisq_fcv
  implicit none
contains
!-------------------------------------------------------------------------------
! Functions to compute chisquares and also residual vectors
!-------------------------------------------------------------------------------
subroutine calc_chisq_qu(&
  Iin,xidx,yidx,Nx,Ny,&
  u,v,Icmp,&
  isqvis,uvidxqvis,qvis,Varqvis,wqvis,&
  isuvis,uvidxuvis,uvis,Varuvis,wuvis,&
  ismq,uvidxmq,mq,Varmq,wmq,&
  ismu,uvidxmu,mu,Varmu,wmu,&
  chisq, gradchisq, chisqqvis, chisquvis, chisqmq, chisqmu,&
  Npix,Nuv,Nicmp,Nqvis,Nuvis,Nmq,Nmu&
)
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix*2) ! for qu
  integer,  intent(in) :: xidx(Npix), yidx(Npix)

  ! uv coordinate
  integer,      intent(in) :: Nuv
  real(dp),     intent(in) :: u(Nuv), v(Nuv)

  ! Stokes I visibilities
  integer,      intent(in) :: Nicmp            ! 1 or Nuv depending on using mvis
  complex(dp),  intent(in) :: Icmp(Nicmp)      ! Model Stokes I visibility

  ! Parameters related to Stokes Q visibilities
  logical,      intent(in) :: isqvis           ! is data?
  integer,      intent(in) :: Nqvis            ! number of data
  integer,      intent(in) :: uvidxqvis(Nqvis) ! uvidx
  complex(dpc), intent(in) :: qvis(Nqvis)      ! data
  real(dp),     intent(in) :: Varqvis(Nqvis)   ! variance
  real(dp),     intent(in) :: wqvis            ! data weights

  ! Parameters related to Stokes U visibilities
  logical,      intent(in) :: isuvis           ! is data?
  integer,      intent(in) :: Nuvis            ! number of data
  integer,      intent(in) :: uvidxuvis(Nuvis) ! uvidx
  complex(dpc), intent(in) :: uvis(Nuvis)      ! data
  real(dp),     intent(in) :: Varuvis(Nuvis)   ! variance
  real(dp),     intent(in) :: wuvis            ! data weights

  ! Parameters related to fractional Stokes Q visibilities
  logical,      intent(in) :: ismq           ! is data?
  integer,      intent(in) :: Nmq            ! number of data
  integer,      intent(in) :: uvidxmq(Nmq)   ! uvidx
  complex(dpc), intent(in) :: mq(Nmq)        ! data
  real(dp),     intent(in) :: Varmq(Nmq)     ! variance
  real(dp),     intent(in) :: wmq            ! data weights

  ! Parameters related to fractional Stokes U visibilities
  logical,      intent(in) :: ismu           ! is data?
  integer,      intent(in) :: Nmu            ! number of data
  integer,      intent(in) :: uvidxmu(Nmu)   ! uvidx
  complex(dpc), intent(in) :: mu(Nmu)        ! data
  real(dp),     intent(in) :: Varmu(Nmu)     ! variance
  real(dp),     intent(in) :: wmu            ! data weights

  ! Chi-square and its gradient
  real(dp), intent(out) :: chisq           ! weighted sum of chisquares
  real(dp), intent(out) :: chisqqvis       ! chisquare of Stokes Q vis
  real(dp), intent(out) :: chisquvis       ! chisquare of Stokes U vis
  real(dp), intent(out) :: chisqmq         ! chisquare of mQ
  real(dp), intent(out) :: chisqmu         ! chisquare of mU
  real(dp), intent(out) :: gradchisq(Npix) ! costfunction and its gradient

  ! allocatable arrays
  real(dp),     allocatable :: I2d(:,:)
  real(dp),     allocatable :: gradchisq2d(:,:)
  complex(dpc), allocatable :: Qresre(:),Qresim(:),Uresre(:),Uresim(:)
  complex(dpc), allocatable :: Qcmp(:), Ucmp(:)

  ! Initialize the chisquare and its gradient
  chisq        = 0d0
  gradchisq(:) = 0d0

  ! Copy 1d image to 2d image
  allocate(I2d(Nx,Ny))
  I2d(:,:)=0d0

  ! Compute Stokes Q visibilities
  if ((isqvis .eqv. .True.) .or. (ismq .eqv. .True.)) then
    call I1d_I2d_fwd(xidx,yidx,Iin(1:Npix),I2d,Npix,Nx,Ny)
    !     Forward Non-unifrom Fast Fourier Transform
    allocate(Qcmp(Nuv))
    Qcmp(:) = dcmplx(0d0,0d0)
    call NUFFT_fwd(u,v,I2d,Qcmp,Nx,Ny,Nuv)
  end if

  ! Compute Stokes U visibilities
  if ((isuvis .eqv. .True.) .or. (ismu .eqv. .True.)) then
    call I1d_I2d_fwd(xidx,yidx,Iin(Npix+1:2*Npix),I2d,Npix,Nx,Ny)
    !     Forward Non-unifrom Fast Fourier Transform
    allocate(Ucmp(Nuv))
    Ucmp(:) = dcmplx(0d0,0d0)
    call NUFFT_fwd(u,v,I2d,Ucmp,Nx,Ny,Nuv)
  end if

  ! deallocate qvis and uuvis
  deallocate(I2d)

  ! allocate residual vectors
  if ((isqvis .eqv. .True.) .or. (ismq .eqv. .True.)) then
    allocate(Qresre(Nuv),Qresim(Nuv))
    Qresre(:) = dcmplx(0d0,0d0)
    Qresim(:) = dcmplx(0d0,0d0)
  end if

  if ((isuvis .eqv. .True.) .or. (ismu .eqv. .True.)) then
    allocate(Uresre(Nuv),Uresim(Nuv))
    Uresre(:) = dcmplx(0d0,0d0)
    Uresim(:) = dcmplx(0d0,0d0)
  end if

  ! Stokes Q visibility
  chisqqvis=0d0
  if (isqvis .eqv. .True.) then
    call calc_chisq_fcv(Qcmp,uvidxqvis,qvis,Varqvis,wqvis,chisqqvis,Qresre,Qresim,Nuv,Nqvis)
    chisq = chisq + chisqqvis
    chisqqvis = chisqqvis / wqvis / (2*Nqvis)
  end if

  ! Stokes U visibility
  chisquvis=0d0
  if (isuvis .eqv. .True.) then
    call calc_chisq_fcv(Ucmp,uvidxuvis,uvis,Varuvis,wuvis,chisquvis,Uresre,Uresim,Nuv,Nuvis)
    chisq = chisq + chisquvis
    chisquvis = chisquvis / wuvis / (2*Nuvis)
  end if

  ! Fractional Stokes Q visibility
  chisqmq=0d0
  if (ismq .eqv. .True.) then
    call calc_chisq_m(Qcmp,Icmp,uvidxmq,mq,Varmq,wmq,chisqmq,Qresre,Qresim,Nuv,Nmq)
    chisq = chisq + chisqmq
    chisqmq = chisqmq / wmq / (2*Nmq)
  end if

  ! Fractional Stokes U visibility
  chisqmu=0d0
  if (ismu .eqv. .True.) then
    call calc_chisq_m(Ucmp,Icmp,uvidxmu,mu,Varmu,wmu,chisqmu,Uresre,Uresim,Nuv,Nmu)
    chisq = chisq + chisqmu
    chisqmu = chisqmu / wmu / (2*Nmu)
  end if

  if ((isqvis .eqv. .True.) .or. (ismq .eqv. .True.)) then
    deallocate(Qcmp)
  end if

  if ((isuvis .eqv. .True.) .or. (ismu .eqv. .True.)) then
    deallocate(Ucmp)
  end if

  ! Adjoint Non-unifrom Fast Fourier Transform
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  !  for Stokes Q
  if ((isqvis .eqv. .True.) .or. (ismq .eqv. .True.)) then
    !  this will provide gradient of chisquare functions
    call NUFFT_adj_resid(u,v,Qresre,Qresim,gradchisq2d(:,:),Nx,Ny,Nuv)
    deallocate(Qresre,Qresim)
    ! copy the gradient of chisquare into that of cost functions
    call I1d_I2d_inv(xidx,yidx,gradchisq(1:Npix),gradchisq2d,Npix,Nx,Ny)
  endif
  !  for Stokes U
  if ((isqvis .eqv. .True.) .or. (ismq .eqv. .True.)) then
    !  this will provide gradient of chisquare functions
    call NUFFT_adj_resid(u,v,Uresre,Uresim,gradchisq2d(:,:),Nx,Ny,Nuv)
    deallocate(Uresre,Uresim)
    ! copy the gradient of chisquare into that of cost functions
    call I1d_I2d_inv(xidx,yidx,gradchisq(Npix+1:2*Npix),gradchisq2d,Npix,Nx,Ny)
  endif
  ! deallocate array
  deallocate(gradchisq2d)
end subroutine

! Chisquare function for a simple single stoke parameter
subroutine calc_chisq_m(Scmp,Icmp,&
                     uvidxm,m,Varm,&
                     fnorm,&
                     chisq,mresre,mresim,&
                     Nuv,Nm)
  !
  !  This subroutine will compute chisquare and its analytic gradient
  !  for full complex visibility data
  !
  implicit none

  ! NuFFT-ed visibilities
  integer,      intent(in) :: Nuv
  complex(dpc), intent(in) :: Scmp(Nuv) ! Stokes S visibility
  complex(dpc), intent(in) :: Icmp(Nuv) ! Stokes I visibility
  ! Data
  integer,  intent(in)     :: Nm          ! Number of data
  integer,  intent(in)     :: uvidxm(Nm)  ! UV Index of m data
  complex(dpc), intent(in) :: m(Nm)       ! Full complex visibility (m) data
  real(dp), intent(in)     :: Varm(Nm)    ! variances of m data
  ! Normalization Factor of Chisquare
  real(dp), intent(in)     :: fnorm
  ! Outputs
  real(dp), intent(inout)  :: chisq       ! chisquare
  complex(dpc), intent(inout) :: mresre(Nuv), mresim(Nuv) ! residual vector
                                                          !   its adjoint FT provides
                                                          !   the gradient of chisquare)

  complex(dpc):: resid
  complex(dpc):: factor
  integer:: uvidx, i

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nm,fnorm,uvidxm,Scmp,Icmp,m,Varm) &
  !$OMP   PRIVATE(i,uvidx,resid,factor),&
  !$OMP   REDUCTION(+:chisq,mresre,mresim)
  do i=1, Nm
    ! pick up uv index
    uvidx = abs(uvidxm(i))

    ! take residual
    if (uvidxm(i) > 0) then
      resid = m(i) - Scmp(uvidx)/Icmp(uvidx)
    else
      resid = m(i) - dconjg(Scmp(uvidx)/Icmp(uvidx))
    end if

    ! compute chisquare
    chisq = chisq + abs(resid)**2/Varm(i)*fnorm

    ! compute residual vector
    if (uvidxm(i) > 0) then
      factor = -2/Varm(i)*fnorm/Icmp(uvidx)
      mresre(uvidx) = mresre(uvidx) + factor*dreal(resid)
      mresim(uvidx) = mresim(uvidx) + factor*dimag(resid)
    else
      factor = -2/Varm(i)*fnorm/dconjg(Icmp(uvidx))
      mresre(uvidx) = mresre(uvidx) + factor*dreal(resid)
      mresim(uvidx) = mresim(uvidx) - factor*dimag(resid)
    end if
  end do
  !$OMP END PARALLEL DO
end subroutine


!-------------------------------------------------------------------------------
! Functions to compute chisquares and also residual vectors
!-------------------------------------------------------------------------------
subroutine model_m(&
  Pin,Iin,xidx,yidx,Nxref,Nyref,Nx,Ny,&
  u,v,&
  uvidxm,mr,mi,Varm,&
  chisq,gradchisq,&
  modelr,modeli,&
  residr,residi,&
  Npix,Nuv,Nm)
 !
 !  This subroutine will compute model, residual, chisquare and
 !  its analytic gradient of full complex visibility data sets
 !  from input image data
 !
  implicit none

  ! Image
  integer,  intent(in) :: Npix, Nx, Ny
  real(dp), intent(in) :: Iin(Npix)     ! Total Intensity Image
  real(dp), intent(in) :: Pin(Npix)     ! Stokes Image
  real(dp), intent(in) :: Nxref, Nyref  ! x,y reference ppixels
                                        ! 1 = the leftmost/lowermost pixel
  integer,  intent(in) :: xidx(Npix), yidx(Npix)  ! x,y pixel number

  ! NuFFT-ed visibilities
  integer,  intent(in) :: Nuv
  real(dp), intent(in) :: u(Nuv), v(Nuv)  ! uv coordinates mutiplied by 2*pi*dx, 2*pi*dy
  ! Data
  integer,  intent(in):: Nm                     ! Number of data
  integer,  intent(in):: uvidxm(Nm)           ! UV Index of m data
  real(dp), intent(in):: mr(Nm),mi(Nm)  ! Full complex visibility (m) data
  real(dp), intent(in):: Varm(Nm)             ! variances of m data
  ! Outputs
  real(dp), intent(out):: chisq                        ! chisquare
  real(dp), intent(out):: modelr(Nm), modeli(Nm)  ! Model Vector
  real(dp), intent(out):: residr(Nm), residi(Nm)  ! Residual Vector
  real(dp), intent(out):: gradchisq(Npix)   !   its adjoint FT provides
                                            !   the gradient of chisquare

  ! allocatable arrays
  real(dp), allocatable :: I2d(:,:),gradchisq2d(:,:)
  complex(dpc), allocatable :: Icmp(:), Pcmp(:),mresre(:),mresim(:)
  complex(dpc), allocatable :: m(:),resid(:),model(:)

  ! other factors
  complex(dpc):: factor
  integer:: uvidx

  ! loop variables
  integer :: i

  ! initialize full complex fractional polarization
  !   allocate arrays
  allocate(m(Nm))
  m = dcmplx(mr,mi)

  ! Copy 1d image to 2d image
  !   allocate array
  allocate(I2d(Nx,Ny), Icmp(Nuv), Pcmp(Nuv))
  Icmp(:) = dcmplx(0d0,0d0)
  Pcmp(:) = dcmplx(0d0,0d0)
  !   Copy image and run FFT for Stokes I
  I2d(:,:)=0d0
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)
  call NUFFT_fwd(u,v,I2d,Icmp,Nx,Ny,Nuv)
  !   Copy image and run FFT for Stokes P
  I2d(:,:)=0d0
  call I1d_I2d_fwd(xidx,yidx,Pin,I2d,Npix,Nx,Ny)
  call NUFFT_fwd(u,v,I2d,Pcmp,Nx,Ny,Nuv)
  !   deallocate 2d image
  deallocate(I2d)

  ! Compute Chisquare
  !  allocate array
  allocate(model(Nm),resid(Nm))
  resid(:) = dcmplx(0d0,0d0)
  model(:) = dcmplx(0d0,0d0)
  allocate(mresre(Nuv),mresim(Nuv))
  mresre(:) = dcmplx(0d0,0d0)
  mresim(:) = dcmplx(0d0,0d0)
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nm,uvidxm,Icmp,Pcmp,m,Varm) &
  !$OMP   PRIVATE(i,uvidx,factor),&
  !$OMP   REDUCTION(+:chisq,mresre,mresim,resid,model)
  do i=1, Nm
    ! pick up uv index
    uvidx = abs(uvidxm(i))

    ! take residual
    if (uvidxm(i) > 0) then
      model(i) = Pcmp(uvidx)/Icmp(uvidx)
    else
      model(i) = dconjg(Pcmp(uvidx)/Icmp(uvidx))
    end if
    resid(i) = m(i) - model(i)

    ! compute chisquare
    chisq = chisq + abs(resid(i))**2/Varm(i)

    ! compute residual vector
    if (uvidxm(i) > 0) then
      factor = -2/Varm(i)/Icmp(uvidx)
      mresre(uvidx) = mresre(uvidx) + factor*dreal(resid(i))
      mresim(uvidx) = mresim(uvidx) + factor*dimag(resid(i))
    else
      factor = -2/Varm(i)/dconjg(Icmp(uvidx))
      mresre(uvidx) = mresre(uvidx) + factor*dreal(resid(i))
      mresim(uvidx) = mresim(uvidx) - factor*dimag(resid(i))
    end if
  end do
  !$OMP END PARALLEL DO
  deallocate(m, Icmp, Pcmp)

  ! Adjoint Non-unifrom Fast Fourier Transform
  !  this will provide gradient of chisquare functions
  allocate(gradchisq2d(Nx,Ny))
  gradchisq2d(:,:) = 0d0
  call NUFFT_adj_resid(u,v,mresre,mresim,gradchisq2d(:,:),Nx,Ny,Nuv)
  deallocate(mresre,mresim)

  ! copy the gradient of chisquare into that of cost functions
  call I1d_I2d_inv(xidx,yidx,gradchisq,gradchisq2d,Npix,Nx,Ny)
  deallocate(gradchisq2d)

  modelr = dreal(model)
  modeli = dimag(model)
  residr = dreal(resid)
  residi = dimag(resid)

  deallocate(resid,model)
end subroutine
end module
