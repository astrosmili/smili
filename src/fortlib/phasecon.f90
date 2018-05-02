module phasecon_lib
  !$use omp_lib
  use param, only: dp, pi, deps, seps
  implicit none
contains
!
!  subroutines and functions
!
subroutine pc_car2d(x,y,u,v,Vreal,Vimag,weight,&
                    PC1,PC2,PC3,PC4,dmap,&
                    Nxy,Nuv)
  !
  implicit none
  !
  integer,  intent(in) :: Nuv, Nxy
  real(dp), intent(in) :: x(Nxy), y(Nxy)
  real(dp), intent(in) :: u(Nuv), v(Nuv), Vreal(Nuv), Vimag(Nuv), weight(Nuv)
  real(dp), intent(out) :: PC1(Nxy),PC2(Nxy),PC3(Nxy),PC4(Nxy),dmap(Nxy)

  integer :: ixy,iuv
  real(dp) :: wasum,wacsum1,wacsum2,wassum1,wassum2
  real(dp) :: wsum,wcsum1,wcsum2,wssum1,wssum2
  real(dp) :: A(1:Nuv), phi(1:Nuv)
  real(dp) :: barphi1, barphi2

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(x,y,u,v,Vreal,Vimag,weight,Nxy,Nuv) &
  !$OMP   PRIVATE(ixy,iuv,wasum,wacsum1,wacsum2,wassum1,wassum2, &
  !$OMP           wsum,wcsum1,wcsum2,wssum1,wssum2,A,phi,barphi1,barphi2)
  do ixy=1,Nxy
    ! using full complex visibilities
    wasum=0d0
    wacsum1=0d0
    wacsum2=0d0
    wassum1=0d0
    wassum2=0d0
    dmap(ixy)=0
    ! without amplitude
    wsum=0d0
    wcsum1=0d0
    wssum1=0d0
    wcsum2=0d0
    wssum2=0d0
    do iuv=1,Nuv
      A(iuv)=sqrt(Vreal(iuv)*Vreal(iuv)+Vimag(iuv)*Vimag(iuv))
      phi(iuv)=atan2(Vimag(iuv),Vreal(iuv))-2*pi*(u(iuv)*x(ixy)+v(iuv)*y(ixy))
      ! with amplitude
      wasum   = wasum   + weight(iuv)*A(iuv)
      wacsum1 = wacsum1 + weight(iuv)*A(iuv)*cos(phi(iuv))
      wassum1 = wassum1 + weight(iuv)*A(iuv)*sin(phi(iuv))
      ! without amplitude
      wsum   = wsum   + weight(iuv)
      wcsum1 = wcsum1 + weight(iuv)*cos(phi(iuv))
      wssum1 = wssum1 + weight(iuv)*sin(phi(iuv))
      dmap(ixy) = dmap(ixy) + A(iuv)*cos(phi(iuv))
    end do
    barphi1 = atan2(wassum1,wacsum1) ! with amplitude
    barphi2 = atan2(wssum1, wcsum1)  ! without amplitude
    do iuv=1,Nuv
      wacsum2 = wacsum2 +     weight(iuv)*A(iuv)*cos(phi(iuv)-barphi1)
      wassum2 = wassum2 + abs(weight(iuv)*A(iuv)*sin(phi(iuv)-barphi1))
      ! without amplitude
      wcsum2 = wcsum2 +     weight(iuv)*cos(phi(iuv)-barphi2)
      wssum2 = wssum2 + abs(weight(iuv)*sin(phi(iuv)-barphi2))
    end do

    PC1(ixy)=wacsum2/wasum
    PC2(ixy)=(wacsum2-wassum2)/wasum
    PC3(ixy)=wcsum2/wsum
    PC4(ixy)=(wcsum2-wssum2)/wsum
  end do
  !$OMP END PARALLEL DO
end subroutine
!
!
subroutine pc2_2d(x,y,u,v,L,Vreal,Vimag,weight,&
                  Nscale,Ntheta,&
                  cutoff,gamma,minpix,multi,siguvd,&
                  PCmax, PCmin,&
                  Nxy,Nuv)
  !
  implicit none
  !
  integer,  intent(in) :: Nuv, Nxy, Nscale, Ntheta
  real(dp), intent(in) :: x(Nxy), y(Nxy)
  real(dp), intent(in) :: u(Nuv), v(Nuv)
  real(dp), intent(in) :: L  ! FOV
  real(dp), intent(in) :: Vreal(Nuv), Vimag(Nuv), weight(Nuv)
  real(dp), intent(in) :: cutoff,gamma,minpix, multi, siguvd
  real(dp), intent(out) :: PCmax(Nxy), PCmin(Nxy)

  integer :: ixy,iuv,iscale,itheta
  real(dp) :: uvd(1:Nuv), uvp(1:Nuv), Vamp(1:Nuv), Vpha(1:Nuv)
  real(dp) :: amp, pha, G, siguvp, uvd0, uvp0, dpa
  real(dp) :: An(Nscale), en(Nscale), on(Nscale)
  real(dp) :: sumAn, sumen, sumon, maxAn, meanen, meanon
  real(dp) :: E, s, W, PC, covx, covy, covx2, covy2, covxy, denom

  ! standard deviation of log-gabor filter in angu
  siguvp = pi/Ntheta

  ! Compute uvdistance and uvposition angle
  uvd = sqrt(u**2+v**2) ! uv distance
  uvp = atan(v/u)      ! position angle (in degree)

  ! Amplitudes and Phases
  Vamp = sqrt(Vreal**2+Vimag**2)
  Vpha = atan2(Vimag,Vreal)

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(x,y,u,v,L,minpix,multi,&
  !$OMP                siguvd,siguvp,uvd,uvp,&
  !$OMP                Vamp,Vpha,weight,&
  !$OMP                Nxy,Nuv,Nscale,Ntheta) &
  !$OMP   PRIVATE(ixy,iuv,iscale,itheta,&
  !$OMP           amp, pha, G, uvd0, uvp0, dpa, &
  !$OMP           An, en, on, &
  !$OMP           sumAn, sumen, sumon, maxAn, meanen, meanon, &
  !$OMP           E, s, W, PC, covx, covy, covx2, covy2, covxy, denom) &
  !$OMP   REDUCTION(+:PCmax,PCmin)
  do ixy=1,Nxy
    do itheta=1, Ntheta
      ! Get current position angle of the gabor filter
      uvp0 = -pi/2 + (itheta-1)*pi/Ntheta

      ! Computes sumAn, sumen, sumon, maxAn
      sumAn=0d0
      sumen=0d0
      sumon=0d0
      maxAn=0d0
      covx2=0d0
      covy2=0d0
      covxy=0d0
      do iscale=1, Nscale
        ! select a scale for the log-gabor filter
        uvd0=L/Nscale * minpix * multi**(iscale-1)
        uvd0=1/uvd0

        ! Compute adjoint Fourier Transform of
        ! the log-Gabor filetered visibilities
        en(iscale) = 0d0
        on(iscale) = 0d0
        do iuv=1,Nuv
          ! compute delta pa
          dpa = uvp(iuv)-uvp0
          dpa = atan(sin(dpa)/(cos(dpa)+seps))

          ! compute log-Gabor filter
          G = -log(uvd(iuv)/uvd0)**2/2/log(siguvd)**2 - dpa**2/2/siguvp**2
          G = exp(G)

          ! Visibility Amplitudes (Gabor filtered)
          amp = Vamp(iuv) * G
          pha = Vpha(iuv) - 2*pi*(u(iuv)*x(ixy)+v(iuv)*y(ixy))

          ! Take summations
          en(iscale) = takesum(en(iscale), + amp * cos(pha) * weight(iuv))
          on(iscale) = takesum(on(iscale), + amp * sin(pha) * weight(iuv))
        end do ! Nuv

        ! Compute An: Amplitudes of the filtered image
        An(iscale) = sqrt(en(iscale)**2+on(iscale)**2)

        ! Take Sum
        sumAn = sumAn + An(iscale)
        sumen = sumen + en(iscale)
        sumon = sumon + on(iscale)
      end do ! scale
      maxAn = maxval(An)

      ! Get weighted mean filter response vector, this gives the weighted
      ! mean phase angle.
      E = sqrt(sumen**2+sumon**2)+seps
      meanen = sumen/E
      meanon = sumon/E

      ! Now calculate An(cos(phase_deviation)-| sin(phase_deviation))| by
      ! using dot and cross products between the weighted mean filter
      ! response vector and the individual filter response vectors at each
      ! scale. This quantity is phase congruency multiplied by An, which we
      ! call energy.
      E = sum(en(:) * meanen + on(:) * meanon - abs(en(:)*meanon - on(:)*meanen))

      ! Spread and Weight Function
      s = sumAn/(maxAn + seps)/Nscale
      W = 1/(1+exp(gamma*(cutoff-s)))

      ! Phase Congruency
      PC = W * E / (sumAn + seps)

      ! accumulate covariance data
      covx = PC * cos(uvp0)
      covy = PC * sin(uvp0)
      covx2 = covx2 + covx * covx
      covy2 = covy2 + covy * covy
      covxy = covxy + covx * covy
    end do ! theta

    ! First normalise covariance values by the number of orientations/2
    covx2 = covx2 / Ntheta * 2.
    covy2 = covx2 / Ntheta * 2.
    covxy = covxy / Ntheta * 4.
    denom = sqrt(covxy * covxy + (covx2 - covy2) * (covx2 - covy2)) + seps

    ! Maximum and minimum moments
    PCmax(ixy) = (covx2 + covy2 + denom) / 2.
    PCmin(ixy) = (covx2 + covy2 - denom) / 2.
  end do
  !$OMP END PARALLEL DO
end subroutine

function takesum(a, b)
  real(dp), intent(in) :: a
  real(dp), intent(in) :: b
  real(dp) :: takesum
  if (isnan(b) .eqv. .False.) takesum = a + b
end function
end module
