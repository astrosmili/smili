module image3d
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  use fftlib, only: NUFFT_fwd, NUFFT_adj, NUFFT_adj_resid, phashift_r2c,&
                    chisq_fcv, chisq_amp, chisq_ca, chisq_cp
  use image, only : ixiy2ixy, ixy2ixiy, comreg, zeroeps,&
                    I1d_I2d_fwd,I1d_I2d_inv,&
                    log_fwd, log_grad,&
                    gamma_fwd, gamma_grad,&
                    l1_e, l1_grade,&
                    tv_e, tv_grade,&
                    tsv_e, tsv_grade
  implicit none
contains
!-------------------------------------------------------------------------------
! Dynamical Imaging Delta T regularization using D2 distance
!-------------------------------------------------------------------------------
! Delta t regularization (using D2 distance)
real(dp) function dt_e(iparm,iz,Iin,Npix,Nz,Nparm)
  implicit none
  !
  integer, intent(in) :: iparm, iz, Npix, Nz, Nparm
  real(dp),intent(in) :: Iin(Nparm)

  if (iz < Nz) then
    dt_e = (Iin(iparm)-Iin(iparm+Npix))**2
  else
    dt_e = 0d0
  end if
end function


! Delta t regularization (using D2 distance)
real(dp) function dt_grade(iparm,iz,Iin,Npix,Nz,Nparm)
  implicit none
  !
  integer, intent(in) :: iparm, iz, Npix, Nz, Nparm
  real(dp),intent(in) :: Iin(Nparm)

  ! initialize output
  dt_grade = 0d0

  if (iz > 1) then
    dt_grade = dt_grade + 2 * (Iin(iparm)-Iin(iparm-Npix))
  end if
  if (iz < Nz) then
    dt_grade = dt_grade + 2 * (Iin(iparm)-Iin(iparm+Npix))
  end if
end function


!-------------------------------------------------------------------------------
! Dynamical Imaging Delta I regularization using D2 distance
!-------------------------------------------------------------------------------
subroutine di(Iin,di_w,doweight,reg,gradreg,Nparm,Npix,Nz)
  implicit none
  !
  integer, intent(in) :: Npix, Nz, Nparm, doweight
  real(dp),intent(in) :: Iin(Nparm),di_w(Npix)
  real(dp),intent(inout) :: reg
  real(dp),intent(inout) :: gradreg(Nparm)

  real(dp) :: Iave(Npix)
  real(dp) :: Iressum(Npix)
  real(dp) :: resid

  integer :: iparm, ipix, iz

  Iave(:) = 0d0
  Iressum(:) = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nparm,Npix,Nz) &
  !$OMP   PRIVATE(iparm,ipix,iz) &
  !$OMP   REDUCTION(+: Iave)
  do iparm=1, Nparm
    call ixy2ixiy(iparm,ipix,iz,Npix)
    Iave(ipix) = Iave(ipix) + Iin(iparm)/Nz
  end do
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nparm,Npix,Nz) &
  !$OMP   PRIVATE(iparm,ipix,iz,resid) &
  !$OMP   REDUCTION(+:reg,Iressum,gradreg)
  do iparm=1, Nparm
    call ixy2ixiy(iparm,ipix,iz,Npix)
    resid = Iin(iparm) - Iave(ipix)
    if (doweight>0) then
      reg = reg + di_w(ipix) * resid**2
      gradreg(iparm) = gradreg(iparm) + 2 * di_w(ipix) * resid
      Iressum(ipix) = Iressum(ipix) - 2 * di_w(ipix) * resid / Npix
    else
      reg = reg + resid**2
      gradreg(iparm) = gradreg(iparm) + 2 * resid
      Iressum(ipix) = Iressum(ipix) - 2 * resid / Npix
    end if
  end do
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nparm,Npix,Nz) &
  !$OMP   PRIVATE(iparm,ipix,iz) &
  !$OMP   REDUCTION(+:gradreg)
  do iparm=1, Nparm
    call ixy2ixiy(iparm,ipix,iz,Npix)
    gradreg(iparm) = gradreg(iparm) + Iressum(ipix)
  end do
  !$OMP END PARALLEL DO
end subroutine


!-------------------------------------------------------------------------------
! Dynamical Imaging Total flux continuity using D2 distance
!-------------------------------------------------------------------------------
! Delta t regularization (using D2 distance)
real(dp) function dtf_e(iz, Isum, Nz)
  implicit none
  !
  integer, intent(in) :: iz, Nz
  real(dp),intent(in) :: Isum(Nz)

  if (iz < Nz) then
    dtf_e = (Isum(iz)-Isum(iz+1))**2
  else
    dtf_e = 0d0
  end if
end function


! Delta t regularization (using D2 distance)
real(dp) function dtf_grade(iz, Isum, Nz)
  implicit none
  !
  integer, intent(in) :: iz, Nz
  real(dp),intent(in) :: Isum(Nz)

  dtf_grade = 0d0
  if (iz < Nz) then
    dtf_grade = dtf_grade + 2 * (Isum(iz) - Isum(iz+1))
  end if
  if (iz > 1) then
    dtf_grade = dtf_grade + 2 * (Isum(iz) - Isum(iz-1))
  end if
end function

!-------------------------------------------------------------------------------
! Dynamical Imaging: Center of Mass regularizations
!-------------------------------------------------------------------------------
subroutine comreg3d(xidx,yidx,Nxref,Nyref,alpha,Iin,cost,gradcost,Npix,Nz,Nparm)
  implicit none
  !
  integer, intent(in) :: Npix,Nz,Nparm
  integer, intent(in) :: xidx(1:Npix), yidx(1:Npix)
  real(dp),intent(in) :: alpha
  real(dp),intent(in) :: Nxref, Nyref
  real(dp),intent(in) :: Iin(Nparm)
  real(dp),intent(inout) :: cost
  real(dp),intent(inout) :: gradcost(Nparm)
  !
  real(dp) :: dix, diy, Ip, Isum(Npix)
  real(dp) :: sumx, sumy, sumI
  real(dp) :: gradsumx, gradsumy, gradsumI
  real(dp) :: reg
  !
  integer  :: ipix,iz,iparm

  sumx = 0d0
  sumy = 0d0
  sumI = 0d0

  ! Take summation
  Isum = sum(reshape(Iin,(/Npix,Nz/)),1)

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,Isum,Npix) &
  !$OMP   PRIVATE(ipix, dix, diy, Ip) &
  !$OMP   REDUCTION(+: sumx, sumy, sumI)
  do ipix=1, Npix
    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! take a alpha
    if (abs(alpha-1)<zeroeps) then
      Ip = l1_e(Isum(ipix))
    else
      Ip = l1_e(Isum(ipix))**alpha
    end if

    ! calculate sum
    sumx = sumx + Ip * dix
    sumy = sumy + Ip * diy
    sumI = sumI + Ip
  end do
  !$OMP END PARALLEL DO

  ! Smooth Version
  !
  ! calculate cost function
  !   need zeroeps for smoothing sqrt,
  sumI = sumI + zeroeps
  reg = sqrt((sumx/(sumI))**2+(sumy/(sumI))**2+zeroeps)
  cost = cost + reg

  ! calculate gradient of cost function
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(xidx,yidx,Nxref,Nyref,alpha,Isum,Npix,Nparm,sumx,sumy,sumI,reg) &
  !$OMP   PRIVATE(iparm,ipix,dix,diy,gradsumI,gradsumx,gradsumy) &
  !$OMP   REDUCTION(+:gradcost)
  do iparm=1, Nparm
    call ixy2ixiy(iparm,ipix,iz,Npix)

    ! pixel from the reference pixel
    dix = xidx(ipix) - Nxref
    diy = yidx(ipix) - Nyref

    ! gradient of sum
    if (abs(alpha-1)<zeroeps) then
      gradsumI = l1_grade(Iin(iparm))
    else
      gradsumI = alpha*l1_e(Isum(ipix))**(alpha-1)*l1_grade(Iin(iparm))
    end if

    gradsumx = gradsumI*dix
    gradsumy = gradsumI*diy

    ! gradient of sumx/sumI or sumy/sumI
    gradsumx = (sumI*gradsumx - gradsumI*sumx)/sumI**2
    gradsumy = (sumI*gradsumy - gradsumI*sumy)/sumI**2

    ! calculate gradint of cost function
    gradcost(iparm) = gradcost(iparm) + (sumx/sumI*gradsumx+sumy/sumI*gradsumy)/reg
  end do
  !$OMP END PARALLEL DO
end subroutine

!-------------------------------------------------------------------------------
! Reweighting
!-------------------------------------------------------------------------------
!
! l1-norm
!
subroutine calc_l1_w_3d(Iin,tgtdyrange,l1_w,Nparm)
  implicit none

  integer, intent(in)   :: Nparm
  real(dp), intent(in)  :: Iin(Nparm)
  real(dp), intent(in)  :: tgtdyrange
  real(dp), intent(out) :: l1_w(Nparm)

  integer :: i
  real(dp):: norm, eps

  eps = maxval(Iin)/tgtdyrange
  norm = 0d0
  do i=1, Nparm
    l1_w(i) = 1/(l1_e(Iin(i))+eps)
    norm = norm + l1_e(Iin(i))*l1_w(i)
  end do
  l1_w = l1_w/norm
end subroutine
!
! TV
!
subroutine calc_tv_w_3d(Iin,xidx,yidx,tgtdyrange,tv_w,Nparm,Npix,Nx,Ny,Nz)
  implicit none

  integer, intent(in) :: Nparm, Npix, Nx, Ny, Nz
  real(dp), intent(in):: Iin(Nparm)
  integer, intent(in) :: xidx(Npix), yidx(Npix)
  real(dp), intent(in):: tgtdyrange
  real(dp), intent(out):: tv_w(Nparm)

  integer :: iparm,ipix,iz
  real(dp):: norm, eps
  real(dp), allocatable:: I2d(:,:)

  eps = maxval(Iin)/tgtdyrange
  allocate(I2d(Nx,Ny))

  ! compute weights
  norm = 0d0
  do iz=1, Nz
    call I1d_I2d_fwd(xidx,yidx,Iin(1+(iz-1)*Nz:iz*Nz),I2d,Npix,Nx,Ny)
    do ipix=1, Npix
      call ixiy2ixy(ipix,iz,iparm,Npix)
      tv_w(iparm) = 1/(tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)+eps)
      norm = norm + tv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)*tv_w(iparm)
    end do
  end do
  deallocate(I2d)

  ! normalize weights
  tv_w = tv_w/norm
end subroutine
!
! TSV
!
subroutine calc_tsv_w_3d(Iin,xidx,yidx,tsv_w,Nparm,Npix,Nx,Ny,Nz)
  implicit none

  integer, intent(in) :: Nparm, Npix, Nx, Ny, Nz
  real(dp), intent(in):: Iin(Nparm)
  integer, intent(in) :: xidx(Npix), yidx(Npix)
  real(dp), intent(out):: tsv_w(Nparm)

  integer :: iparm,ipix,iz
  real(dp):: norm
  real(dp), allocatable:: I2d(:,:)

  allocate(I2d(Nx,Ny))
  call I1d_I2d_fwd(xidx,yidx,Iin,I2d,Npix,Nx,Ny)

  norm = 0d0
  do iz=1, Nz
    call I1d_I2d_fwd(xidx,yidx,Iin(1+(iz-1)*Nz:iz*Nz),I2d,Npix,Nx,Ny)
    do ipix=1, Npix
      call ixiy2ixy(ipix,iz,iparm,Npix)
      tsv_w(iparm) = 1/l1_e(Iin(iparm)**2)
      norm = norm + tsv_e(xidx(ipix),yidx(ipix),I2d,Nx,Ny)*tsv_w(iparm)
    end do
  end do
  deallocate(I2d)

  tsv_w = tsv_w/norm
end subroutine

subroutine calc_dt_w(Iin,dt_w,Nparm,Npix,Nz)
  implicit none

  integer, intent(in) :: Nparm, Npix, Nz
  real(dp), intent(in):: Iin(Nparm)
  real(dp), intent(out):: dt_w(Nparm)

  integer :: iparm, ipix, iz
  real(dp):: norm

  norm = 0d0
  do iparm=1, Nparm
    call ixy2ixiy(iparm,ipix,iz,Npix)
    dt_w(iparm) = 1/l1_e(Iin(iparm)**2)
    norm = norm + dt_w(iparm) * dt_e(iparm,iz,Iin,Npix,Nz,Nparm)
  end do
  dt_w = dt_w/norm
end subroutine

subroutine calc_di_w(Iin,di_w,Nparm,Npix,Nz)
  implicit none

  integer, intent(in) :: Nparm, Npix, Nz
  real(dp), intent(in):: Iin(Nparm)
  real(dp), intent(out):: di_w(Npix)

  integer :: iparm, ipix, iz
  real(dp):: norm, Iave(Npix)

  Iave = sum(reshape(Iin,(/Npix,Nz/)),2)/Nz

  do ipix=1, Npix
    di_w(ipix) = 1/l1_e(Iave(ipix)**2)
  end do

  norm = 0d0
  do iparm=1, Nparm
    call ixy2ixiy(iparm,ipix,iz,Npix)
    norm = norm + di_w(ipix) * (Iin(iparm)-Iave(ipix))**2
  end do
  di_w = di_w/norm
end subroutine

subroutine calc_dtf_w(Iin,dtf_w,Nparm,Npix,Nz)
  implicit none

  integer, intent(in) :: Nparm, Npix, Nz
  real(dp), intent(in):: Iin(Nparm)
  real(dp), intent(out):: dtf_w(Nz)

  integer :: iz
  real(dp):: norm, Isum(Nz)

  Isum = sum(reshape(Iin,(/Npix,Nz/)),1)

  norm = 0d0
  do iz=1, Nz
    dtf_w(iz) = 1/l1_e(Isum(iz)**2)
    norm = norm + dtf_w(iz) * dtf_e(iz, Isum, Nz)
  end do
  dtf_w = dtf_w/norm
end subroutine
end module
