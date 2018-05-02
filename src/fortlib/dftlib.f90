module dftlib
  !$use omp_lib
  use param, only : dp, deps, pi
  implicit none  ! BLAS function (external)
  interface
    real(kind(1d0)) function dasum(n, x, incx)
      integer,         intent(in) :: n, incx
      real(kind(1d0)), intent(in) :: x(n)
    end function
  end interface

  interface
    real(kind(1d0)) function ddot(n, x, incx, y, incy)
      integer,         intent(in) :: n, incx, incy
      real(kind(1d0)), intent(in) :: x(n), y(n)
    end function
  end interface
contains
!
!-------------------------------------------------------------------------------
! calc the Fourier Matrix
!-------------------------------------------------------------------------------
!
subroutine calc_F(x,y,u,v,Freal,Fimag,Npix,Nuv)
  !
  ! This subroutine calculates the DFT matrix from the input data
  !
  implicit none

  integer,  intent(in)  :: Npix,Nuv
  real(dp), intent(in)  :: x(Npix),y(Npix)
  real(dp), intent(in)  :: u(Nuv),v(Nuv)
  real(dp), intent(out) :: Freal(Npix,Nuv), Fimag(Npix,Nuv)

  integer :: iuv
  real(dp):: phase(1:Npix)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv,Npix,x,y,u,v) &
  !$OMP   PRIVATE(iuv, phase)
  do iuv = 1, Nuv
    ! Calculate phases first
    phase(1:Npix) = 0d0
    call daxpy(Npix,2*pi*u(iuv),x(1:Npix),1,phase(1:Npix),1)
    call daxpy(Npix,2*pi*v(iuv),y(1:Npix),1,phase(1:Npix),1)
    !write(*,*) maxval(v(iuv)*y), minval(v(iuv)*y)

    ! Calculate the matrix
    Freal(1:Npix, iuv) = cos(phase(1:Npix))
    Fimag(1:Npix, iuv) = sin(phase(1:Npix))
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine grad_amppha(Freal,Fimag,Vreal,Vimag,gradVamp,gradVpha,Npix,Nuv)
  implicit none

  integer,  intent(in)  :: Npix,Nuv
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv),Vimag(Nuv)
  real(dp), intent(out) :: gradVamp(Npix,Nuv),gradVpha(Npix,Nuv)

  real(dp) :: Vamp, Vampsq
  integer  :: iuv

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv,Npix,Vreal,Vimag) &
  !$OMP   PRIVATE(iuv, Vamp, Vampsq)
  do iuv=1, Nuv
    Vamp = sqrt(Vreal(iuv)*Vreal(iuv)+Vimag(iuv)*Vimag(iuv))
    Vampsq = Vamp*Vamp

    ! using BLAS functions
    gradVamp(1:Npix,iuv) = 0d0
    call daxpy(Npix,Vreal(iuv)/Vamp,Freal(1:Npix,iuv),1,gradVamp(1:Npix,iuv),1)
    call daxpy(Npix,Vimag(iuv)/Vamp,Fimag(1:Npix,iuv),1,gradVamp(1:Npix,iuv),1)
    gradVpha(1:Npix,iuv) = 0d0
    call daxpy(Npix, Vreal(iuv)/Vampsq,Fimag(1:Npix,iuv),1,gradVpha(1:Npix,iuv),1)
    call daxpy(Npix,-Vimag(iuv)/Vampsq,Freal(1:Npix,iuv),1,gradVpha(1:Npix,iuv),1)

    ! using Fortran functions
    !gradVamp(1:Npix,iuv)=Vreal(iuv)/Vamp*Freal(1:Npix, iuv)&
    !                    +Vimag(iuv)/Vamp*Fimag(1:Npix, iuv)
    !gradVpha(1:Npix,iuv)=Vreal(iuv)/Vampsq*Fimag(1:Npix, iuv)&
    !                    -Vimag(iuv)/Vampsq*Freal(1:Npix, iuv)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
!-------------------------------------------------------------------------------
! DFT
!-------------------------------------------------------------------------------
!
subroutine DFT(I,Freal,Fimag,Vreal,Vimag,Npix,Nuv)
  ! This subroutine do DFT
  implicit none
  integer,  intent(in)  :: Npix,Nuv
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: I(Npix)
  real(dp), intent(out) :: Vreal(Nuv),Vimag(Nuv)

  integer  :: iuv

  !call dgemv('T',Nuv,Npix,1d0,Freal(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vreal(1:Nuv),1)
  !call dgemv('T',Nuv,Npix,1d0,Fimag(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vimag(1:Nuv),1)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nuv,Npix,I) &
  !$OMP   PRIVATE(iuv)
  do iuv=1, Nuv
    Vreal(iuv) = ddot(Npix, I(1:Npix), 1, Freal(1:Npix,iuv), 1)
    Vimag(iuv) = ddot(Npix, I(1:Npix), 1, Fimag(1:Npix,iuv), 1)
  end do
  !$OMP END PARALLEL DO
  !write(*,*) maxval(I),maxval(Freal),maxval(Fimag),minval(Freal),minval(Fimag)
end subroutine
!
!-------------------------------------------------------------------------------
! Bessel Fourier Transformation
!-------------------------------------------------------------------------------
!
subroutine iBFT(beta,B,I,Nbeta,Npix)
  ! This subroutine do inverse Bessel Fourier Transformation
  implicit none
  integer,  intent(in)  :: Nbeta, Npix
  real(dp), intent(in)  :: B(Nbeta,Npix)
  real(dp), intent(in)  :: beta(Nbeta)
  real(dp), intent(out) :: I(Npix)

  integer  :: ipix

  !call dgemv('T',Nuv,Npix,1d0,Freal(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vreal(1:Nuv),1)
  !call dgemv('T',Nuv,Npix,1d0,Fimag(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vimag(1:Nuv),1)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nbeta,beta) &
  !$OMP   PRIVATE(ipix)
  do ipix=1, Npix
    I(ipix) = ddot(Nbeta, beta(1:Nbeta), 1, B(1:Nbeta,ipix), 1)
  end do
  !$OMP END PARALLEL DO
  !write(*,*) maxval(I),maxval(Freal),maxval(Fimag),minval(Freal),minval(Fimag)
end subroutine
!
!
subroutine BFT(I,C,beta,Npix,Nbeta)
  ! This subroutine do Bessel Fourier Transformation
  implicit none
  integer,  intent(in)  :: Nbeta, Npix
  real(dp), intent(in)  :: C(Npix,Nbeta)
  real(dp), intent(in)  :: I(Npix)
  real(dp), intent(out) :: beta(Nbeta)

  integer  :: ibeta

  !call dgemv('T',Nuv,Npix,1d0,Freal(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vreal(1:Nuv),1)
  !call dgemv('T',Nuv,Npix,1d0,Fimag(1:Npix,1:Nuv),&
  !           Npix,I(1:Npix),1,0d0,Vimag(1:Nuv),1)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nbeta,I) &
  !$OMP   PRIVATE(ibeta)
  do ibeta=1, Nbeta
    beta(ibeta) = ddot(Npix, I(1:Npix), 1, C(1:Npix,ibeta), 1)
  end do
  !$OMP END PARALLEL DO
  !write(*,*) maxval(I),maxval(Freal),maxval(Fimag),minval(Freal),minval(Fimag)
end subroutine
!
!
!-------------------------------------------------------------------------------
! calc chisquares
!-------------------------------------------------------------------------------
!
!
subroutine chisq_fcv(Freal,Fimag,Vreal,Vimag,&
                     uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
                     chisq,gradchisq,Npix,Nuv,Nfcv)
  integer,  intent(in)  :: Npix,Nuv,Nfcv
  integer,  intent(in)  :: uvidxfcv(Nfcv)
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vfcvr(Nfcv),Vfcvi(Nfcv)
  real(dp), intent(in)  :: Varfcv(Nfcv)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid1,resid2,factor1,factor2
  integer   :: uvidx,ifcv

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nfcv,Vreal,Vimag,uvidxfcv,Vfcvr,Vfcvi,Varfcv) &
  !$OMP   PRIVATE(ifcv,uvidx,resid1,resid2,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ifcv=1, Nfcv
    ! calc model fcvlitude
    uvidx = abs(uvidxfcv(ifcv))
    resid1 = Vreal(uvidx)-Vfcvr(ifcv)
    resid2 = sign(1,uvidxfcv(ifcv))*Vimag(uvidx)-Vfcvi(ifcv)

    ! calc chi-square
    chisq = chisq + (resid1*resid1+resid2*resid2)/Varfcv(ifcv)

    ! calc gradient of chi-square
    factor1 = 2/Varfcv(ifcv)*resid1
    factor2 = 2/Varfcv(ifcv)*resid2*sign(1,uvidxfcv(ifcv))
    call daxpy(Npix,factor1,Freal(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    call daxpy(Npix,factor2,Fimag(1:Npix,uvidx),1,gradchisq(1:Npix),1)
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine chisq_amp(gradVamp,Vreal,Vimag,&
                     uvidxamp,Vamp,Varamp,&
                     chisq,gradchisq,Npix,Nuv,Namp)
  integer,  intent(in)  :: Npix, Nuv, Namp
  integer,  intent(in)  :: uvidxamp(Namp)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vamp(Namp),Varamp(Namp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid
  integer   :: uvidx,iamp

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Namp,Vreal,Vimag,uvidxamp,Vamp,Varamp) &
  !$OMP   PRIVATE(iamp,uvidx,resid)  &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do iamp=1, Namp
    ! calc model amplitude
    uvidx = abs(uvidxamp(iamp))
    resid = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))-Vamp(iamp)

    ! calc chi-square
    chisq = chisq + resid*resid/Varamp(iamp)

    ! calc gradient of chisquare
    !  using BLAS
    call daxpy(Npix,2/Varamp(iamp)*resid,&
               gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    !  not using BLAS
    !gradchisq(1:Npix) = gradchisq(1:Npix) &
    !                  + 2/Varamp(iamp)*resid*gradVamp(1:Npix,uvidx)
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine chisq_cphase(gradVpha,Vreal,Vimag,&
                        uvidxcp,CP,Varcp,&
                        chisq,gradchisq,Npix,Nuv,Ncp)
  integer,  intent(in)  :: Npix, Nuv, Ncp
  integer,  intent(in)  :: uvidxcp(3,Ncp)
  real(dp), intent(in)  :: gradVpha(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CP(Ncp),Varcp(Ncp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,factor
  integer   :: uvidx,icp,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Ncp,Vreal,Vimag,uvidxcp,CP,Varcp) &
  !$OMP   PRIVATE(icp,ibl,resid,uvidx,factor) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do icp=1, Ncp
    ! calc model closure phase
    resid = 0d0
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      resid = resid + atan2(sign(1,uvidxcp(ibl,icp))*Vimag(uvidx),Vreal(uvidx))
    end do

    ! take a residual betweem model and data
    resid = resid - CP(icp)

    ! adjust a residual phase
    !do while (resid > pi)
    !  resid = resid - 2*pi
    !end do
    !do while (resid < -pi)
    !  resid = resid + 2*pi
    !end do
    resid = atan2(sin(resid),cos(resid))

    ! calc chi-square
    chisq = chisq + resid*resid/Varcp(icp)

    ! calc gradient of chi-square
    factor=2/Varcp(icp)*resid
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      call daxpy(Npix,sign(1,uvidxcp(ibl,icp))*factor,&
                 gradVpha(1:Npix,uvidx),1,gradchisq(1:Npix),1)
      !gradchisq(1:Npix) = gradchisq(1:Npix)&
      !                  + sign(1,uvidxcp(ibl,icp))*factor*gradVpha(1:Npix,uvidx)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine chisq_camp(gradVamp,Vreal,Vimag,&
                      uvidxca,CA,Varca,&
                      chisq,gradchisq,Npix,Nuv,Nca)
  integer,  intent(in)  :: Npix, Nuv, Nca
  integer,  intent(in)  :: uvidxca(4,Nca)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CA(Nca),Varca(Nca)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,amp,factor1,factor2(4)
  integer   :: uvidx,ica,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nca,Vreal,Vimag,uvidxca,CA,Varca) &
  !$OMP   PRIVATE(ica,ibl,resid,uvidx,amp,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ica=1, Nca
    ! calc model closure phase
    resid = 0d0
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      amp = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))
      if (ibl < 3) then
        factor2(ibl) = 1/amp
        resid = resid + log(amp)
      else
        factor2(ibl) = -1/amp
        resid = resid - log(amp)
      end if
    end do

    ! take a residual betweem model and data
    resid = resid - CA(ica)

    ! calc chi-square
    chisq = chisq + resid*resid/Varca(ica)

    ! calc gradient of chisquare
    factor1=2/Varca(ica)*resid
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      call daxpy(Npix,factor1*factor2(ibl),&
                 gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine model_fcv(Freal,Fimag,Vreal,Vimag,&
                     uvidxfcv,Vfcvr,Vfcvi,Varfcv,&
                     Vfcvrmod,Vfcvimod,Vfcvres,&
                     chisq,gradchisq,Npix,Nuv,Nfcv)
  integer,  intent(in)  :: Npix,Nuv,Nfcv
  integer,  intent(in)  :: uvidxfcv(Nfcv)
  real(dp), intent(in)  :: Freal(Npix,Nuv), Fimag(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vfcvr(Nfcv),Vfcvi(Nfcv)
  real(dp), intent(in)  :: Varfcv(Nfcv)
  real(dp), intent(out) :: Vfcvrmod(Nfcv),Vfcvimod(Nfcv),Vfcvres(Nfcv)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid1,resid2,factor1,factor2
  integer   :: uvidx,ifcv

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  Vfcvrmod(1:Nfcv) = 0d0
  Vfcvimod(1:Nfcv) = 0d0
  Vfcvres(1:Nfcv) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nfcv,Vreal,Vimag,uvidxfcv,Vfcvr,Vfcvi,Varfcv) &
  !$OMP   PRIVATE(ifcv,uvidx,resid1,resid2,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ifcv=1, Nfcv
    ! calc model fcvlitude
    uvidx = abs(uvidxfcv(ifcv))
    Vfcvrmod(ifcv) = Vreal(uvidx)
    Vfcvimod(ifcv) = sign(1,uvidxfcv(ifcv))*Vimag(uvidx)
    resid1 = Vfcvrmod(ifcv)-Vfcvr(ifcv)
    resid2 = Vfcvimod(ifcv)-Vfcvi(ifcv)

    ! calc chi-square
    Vfcvres(ifcv)=sqrt(resid1*resid1+resid2*resid2)
    chisq = chisq + (resid1*resid1+resid2*resid2)/Varfcv(ifcv)

    ! calc gradient of chi-square
    factor1 = 2/Varfcv(ifcv)*resid1
    factor2 = 2/Varfcv(ifcv)*resid2*sign(1,uvidxfcv(ifcv))
    call daxpy(Npix,factor1,Freal(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    call daxpy(Npix,factor2,Fimag(1:Npix,uvidx),1,gradchisq(1:Npix),1)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine model_amp(gradVamp,Vreal,Vimag,&
                     uvidxamp,Vamp,Varamp,&
                     Vampmod,Vampres,&
                     chisq,gradchisq,Npix,Nuv,Namp)
  integer,  intent(in)  :: Npix, Nuv, Namp
  integer,  intent(in)  :: uvidxamp(Namp)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: Vamp(Namp), Varamp(Namp)
  real(dp), intent(out) :: Vampmod(Namp),Vampres(Namp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  !real(dp)  :: resid
  integer   :: uvidx,iamp

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  Vampmod(1:Namp) = 0d0
  Vampres(1:Namp) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Namp,Vreal,Vimag,uvidxamp,Vamp,Varamp) &
  !$OMP   PRIVATE(iamp,uvidx)  &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do iamp=1, Namp
    ! calc model amplitude
    uvidx = abs(uvidxamp(iamp))
    Vampmod(iamp) = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))
    Vampres(iamp) = Vampmod(iamp)-Vamp(iamp)

    ! calc chi-square
    chisq = chisq + Vampres(iamp)*Vampres(iamp)/Varamp(iamp)

    ! calc gradient of chisquare
    !  using BLAS
    call daxpy(Npix,2/Varamp(iamp)*Vampres(iamp),&
               gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    !  not using BLAS
    !gradchisq(1:Npix) = gradchisq(1:Npix) &
    !                  + 2/Varamp(iamp)*resid*gradVamp(1:Npix,uvidx)
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine model_cphase(gradVpha,Vreal,Vimag,&
                        uvidxcp,CP,Varcp,&
                        CPmod,CPres,&
                        chisq,gradchisq,Npix,Nuv,Ncp)
  integer,  intent(in)  :: Npix, Nuv, Ncp
  integer,  intent(in)  :: uvidxcp(3,Ncp)
  real(dp), intent(in)  :: gradVpha(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CP(Ncp),Varcp(Ncp)
  real(dp), intent(out) :: CPmod(Ncp),CPres(Ncp)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,factor
  integer   :: uvidx,icp,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  CPmod(1:Ncp) = 0d0
  CPres(1:Ncp) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Ncp,Vreal,Vimag,uvidxcp,CP,Varcp) &
  !$OMP   PRIVATE(icp,ibl,resid,uvidx,factor) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do icp=1, Ncp
    ! calc model closure phase
    resid = 0d0
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      resid = resid + atan2(sign(1,uvidxcp(ibl,icp))*Vimag(uvidx),Vreal(uvidx))
    end do
    CPmod(icp)=atan2(sin(resid),cos(resid))

    ! take a residual betweem model and data
    resid = resid - CP(icp)

    ! adjust a residual phase
    resid = atan2(sin(resid),cos(resid))
    CPres(icp) = resid

    ! calc chi-square
    chisq = chisq + resid*resid/Varcp(icp)

    ! calc gradient of chi-square
    factor=2/Varcp(icp)*resid
    do ibl=1,3
      uvidx = abs(uvidxcp(ibl,icp))
      call daxpy(Npix,sign(1,uvidxcp(ibl,icp))*factor,&
                 gradVpha(1:Npix,uvidx),1,gradchisq(1:Npix),1)
      !gradchisq(1:Npix) = gradchisq(1:Npix)&
      !                  + sign(1,uvidxcp(ibl,icp))*factor*gradVpha(1:Npix,uvidx)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine model_camp(gradVamp,Vreal,Vimag,&
                      uvidxca,CA,Varca,&
                      CAmod,CAres,&
                      chisq,gradchisq,Npix,Nuv,Nca)
  integer,  intent(in)  :: Npix, Nuv, Nca
  integer,  intent(in)  :: uvidxca(4,Nca)
  real(dp), intent(in)  :: gradVamp(Npix,Nuv)
  real(dp), intent(in)  :: Vreal(Nuv), Vimag(Nuv)
  real(dp), intent(in)  :: CA(Nca),Varca(Nca)
  real(dp), intent(out) :: CAmod(Nca),CAres(Nca)
  real(dp), intent(out) :: chisq
  real(dp), intent(out) :: gradchisq(Npix)

  real(dp)  :: resid,amp,factor1,factor2(4)
  integer   :: uvidx,ica,ibl

  chisq = 0d0
  gradchisq(1:Npix) = 0d0
  CAmod(1:Nca) = 0d0
  CAres(1:Nca) = 0d0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Nuv,Nca,Vreal,Vimag,uvidxca,CA,Varca) &
  !$OMP   PRIVATE(ica,ibl,resid,uvidx,amp,factor1,factor2) &
  !$OMP   REDUCTION(+:chisq,gradchisq)
  do ica=1, Nca
    ! calc model closure phase
    resid = 0d0
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      amp = sqrt(Vreal(uvidx)*Vreal(uvidx)+Vimag(uvidx)*Vimag(uvidx))
      if (ibl < 3) then
        factor2(ibl) = 1/amp
        resid = resid + log(amp)
      else
        factor2(ibl) = -1/amp
        resid = resid - log(amp)
      end if
    end do
    CAmod(ica) = resid

    ! take a residual betweem model and data
    resid = resid - CA(ica)
    CAres(ica) = resid

    ! calc chi-square
    chisq = chisq + resid*resid/Varca(ica)

    ! calc gradient of chisquare
    factor1=2/Varca(ica)*resid
    do ibl=1,4
      uvidx = abs(uvidxca(ibl,ica))
      call daxpy(Npix,factor1*factor2(ibl),&
                 gradVamp(1:Npix,uvidx),1,gradchisq(1:Npix),1)
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine


subroutine calc_I2d(Iin,xidx,yidx,I2d,Npix,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: Npix,Nx,Ny
  integer, intent(in) :: xidx(Npix), yidx(Npix)
  real(dp),intent(in) :: Iin(Npix)
  real(dp),intent(inout) :: I2d(Nx,Ny)
  !
  integer :: ipix
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Iin,xidx,yidx) &
  !$OMP   PRIVATE(ipix)
  do ipix=1,Npix
    I2d(xidx(ipix),yidx(ipix))=Iin(ipix)
  end do
  !$OMP END PARALLEL DO
end subroutine


real(dp) function tv(I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: Nxy
  integer :: i1,j1,i2,j2,ixy
  real(dp):: dIx,dIy
  !
  ! initialize tv term
  tv = 0d0
  Nxy= Nx*Ny
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nxy,Nx,Ny,I2d) &
  !$OMP   PRIVATE(i1,j1,i2,j2,dIx,dIy) &
  !$OMP   REDUCTION(+:tv)
  do ixy=1,Nxy
    call ixy2ixiy(ixy,i1,j1,Nx)
    i2 = i1 + 1             ! i+1
    j2 = j1 + 1             ! j+1
    !
    ! dIx = I(i+1,j) - I(i,j)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx = I2d(i2,j1) - I2d(i1,j1)
    end if
    !
    ! dIy = I(i,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy = I2d(i1,j2) - I2d(i1,j1)
    end if
    !
    tv = tv + sqrt(dIx*dIx+dIy*dIy)
  end do
  !$OMP END PARALLEL DO
end function


real(dp) function tsv(I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: Nxy
  integer :: i1,j1,i2,j2,ixy
  real(dp):: dIx,dIy
  !
  ! initialize tsv term
  tsv = 0d0
  Nxy= Nx*Ny
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nxy,Nx,Ny,I2d) &
  !$OMP   PRIVATE(i1,j1,i2,j2,dIx,dIy,ixy) &
  !$OMP   REDUCTION(+:tsv)
  do ixy=1,Nxy
    call ixy2ixiy(ixy,i1,j1,Nx)
    i2 = i1 + 1             ! i+1
    j2 = j1 + 1             ! j+1
    !
    ! dIx = I(i+1,j) - I(i,j)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx  = I2d(i2,j1) - I2d(i1,j1)
    end if
    !
    ! dIy = I(i,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy  = I2d(i1,j2) - I2d(i1,j1)
    end if
    !
    tsv = tsv + dIx*dIx+dIy*dIy
  end do
  !$OMP END PARALLEL DO
end function


real(dp) function gradtve(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: Nx,Ny
  integer, intent(in) :: xidx, yidx
  real(dp),intent(in) :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  real(dp):: dIx,dIy,tve
  !
  ! initialize tsv term
  gradtve = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  !-------------------------------------
  ! (i2,j1)-(i1,j1), (i1,j2)-(i1,j1)
  !-------------------------------------
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx = I2d(i2,j1) - I2d(i1,j1)
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  tve = sqrt(dIx*dIx+dIy*dIy)
  if (tve > deps) then
    gradtve = gradtve - (dIx + dIy)/tve
  end if
  !
  !-------------------------------------
  ! (i1,j1)-(i0,j1), (i0,j2)-(i0,j1)
  !-------------------------------------
  if (i0 > 0) then
    ! dIx = I(i,j) - I(i-1,j)
    dIx = I2d(i1,j1) - I2d(i0,j1)

    ! dIy = I(i-1,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy = I2d(i0,j2) - I2d(i0,j1)
    end if

    tve = sqrt(dIx*dIx+dIy*dIy)
    if (tve > deps) then
      gradtve = gradtve + dIx/tve
    end if
  end if
  !
  !-------------------------------------
  ! (i2,j0)-(i1,j0), (i1,j1)-(i1,j0)
  !-------------------------------------
  if (j0 > 0) then
    ! dIy = I(i,j) - I(i,j-1)
    dIy = I2d(i1,j1) - I2d(i1,j0)

    ! dIx = I(i+1,j-1) - I(i,j-1)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx = I2d(i2,j0) - I2d(i1,j0)
    end if

    tve = sqrt(dIx*dIx+dIy*dIy)
    if (tve > deps) then
      gradtve = gradtve + dIy/tve
    end if
  end if
  !
end function


real(dp) function gradtsve(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  integer, intent(in)  :: xidx, yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  !
  ! initialize tsv term
  gradtsve = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 <= Nx) then
    gradtsve = gradtsve - 2*(I2d(i2,j1) - I2d(i1,j1))
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 <= Ny) then
    gradtsve = gradtsve - 2*(I2d(i1,j2) - I2d(i1,j1))
  end if
  !
  if (i0 > 0) then
    gradtsve = gradtsve + 2*(I2d(i1,j1) - I2d(i0,j1))
  end if
  !
  if (j0 > 0) then
    gradtsve = gradtsve + 2*(I2d(i1,j1) - I2d(i1,j0))
  end if
  !
end function


real(dp) function pwall(Iin, scale, Npix)
  implicit none

  ! arguments
  integer, intent(in) :: Npix
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: scale

  integer :: ipix

  pwall=0
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Iin,Npix) &
  !$OMP   PRIVATE(ipix) &
  !$OMP   REDUCTION(+:pwall)
  do ipix=1,Npix
    if (Iin(ipix) < 0) then
      pwall = pwall + scale * Iin(ipix) ** 2
    end if
  end do
  !$OMP END PARALLEL DO
end function


real(dp) function gradpwalle(ipix, Iin, scale, Npix)
  implicit none

  ! arguments
  integer, intent(in) :: Npix, ipix
  real(dp), intent(in) :: Iin(Npix)
  real(dp), intent(in) :: scale

  if (Iin(ipix) < 0) then
    gradpwalle = 2 * scale * Iin(ipix)
  else
    gradpwalle = 0d0
  end if
end function


subroutine ixy2ixiy(ixy,ix,iy,Nx)
  implicit none

  ! arguments
  integer, intent(in):: ixy,Nx
  integer, intent(out):: ix,iy
  !
  ix = mod(ixy-1,Nx)+1
  iy = (ixy-1)/Nx+1
end subroutine


subroutine ixiy2ixy(ix,iy,ixy,Nx)
  implicit none

  ! arguments
  integer, intent(in):: ix,iy,Nx
  integer, intent(out):: ixy
  !
  ixy = ix + (iy-1) * Nx
end subroutine


end module
