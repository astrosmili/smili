module image3d
  !$use omp_lib
  use param, only : dp, dpc, pi, i_dpc
  use fftlib, only: NUFFT_fwd, NUFFT_adj, NUFFT_adj_resid, phashift_r2c
  use image, only : calc_cost_reg, ixiy2ixy, ixy2ixiy, zeroeps,&
                    I1d_I2d_fwd,I1d_I2d_inv,&
                    l1_e, l1_grade,&
                    tv_e, tv_grade,calc_tfdreg,&
                    tsv_e, tsv_grade, smabs, calc_cenreg
  implicit none
contains
!-------------------------------------------------------------------------------
! Calc cost function
!-------------------------------------------------------------------------------
subroutine calc_cost_reg3d(&
    I1d, xidx, yidx, Nxref, Nyref, Nx, Ny, &
    l1_l, l1_wgt, l1_Nwgt,&
    sm_l, sm_maj, sm_min, sm_phi,&
    tv_l, tv_wgt, tv_Nwgt,&
    tsv_l, tsv_wgt, tsv_Nwgt,&
    kl_l, kl_wgt, kl_Nwgt,&
    gs_l, gs_wgt, gs_Nwgt,&
    tfd_l, tfd_tgtfd,&
    lc_l, lc_tgtfd,lc_nidx,&
    cen_l, cen_alpha,&
    rt_l, ri_l, rs_l, rf_l, &
    l1_cost, sm_cost, tv_cost, tsv_cost, kl_cost, gs_cost,&
    tfd_cost, lc_cost, cen_cost, &
    rt_cost, ri_cost, rs_cost, rf_cost, &
    out_maj, out_min, out_phi,&
    cost, gradcost, &
    Npix, Nz)
  implicit none

  ! Image
  integer,  intent(in)  :: Npix, Nx, Ny, Nz
  integer,  intent(in)  :: xidx(Npix), yidx(Npix)
  real(dp), intent(in)  :: I1d(Npix*Nz)
  real(dp), intent(in)  :: Nxref, Nyref

  ! parameter for l1
  real(dp), intent(in)  :: l1_l              ! lambda
  integer,  intent(in)  :: l1_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: l1_wgt(l1_Nwgt)   ! weight

  ! parameter for second momentum
  real(dp), intent(in)   :: sm_l              ! lambda
  real(dp), intent(in)   :: sm_maj,sm_min,sm_phi ! major, minor size and position angle

  ! parameter for total variation
  real(dp), intent(in)  :: tv_l              ! lambda
  integer,  intent(in)  :: tv_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: tv_wgt(tv_Nwgt)   ! weight

  ! parameter for total squared variation
  real(dp), intent(in)  :: tsv_l             ! lambda
  integer,  intent(in)  :: tsv_Nwgt          ! size of the weight vector
  real(dp), intent(in)  :: tsv_wgt(tsv_Nwgt) ! weight

  ! parameter for kl divergence
  real(dp), intent(in)  :: kl_l              ! lambda
  integer,  intent(in)  :: kl_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: kl_wgt(kl_Nwgt)   ! weight

  ! parameter for Gull & Skilling entropy
  real(dp), intent(in)  :: gs_l              ! lambda
  integer,  intent(in)  :: gs_Nwgt           ! size of the weight vector
  real(dp), intent(in)  :: gs_wgt(gs_Nwgt)   ! weight

  ! parameter for the total flux density regularization
  real(dp), intent(in)  :: tfd_l             ! lambda (Normalized)
  real(dp), intent(in)  :: tfd_tgtfd         ! target total flux

  ! parameter for the light curve regularization
  real(dp), intent(in)  :: lc_l(Nz)          ! lambda (Normalized)
  real(dp), intent(in)  :: lc_tgtfd(Nz)      ! target light curve
  integer,  intent(in)  :: lc_nidx           ! regularizer normalization with lc_tgtfd

  ! parameter for the centoroid regularization
  real(dp), intent(in)  :: cen_l             ! lambda (Normalized)
  real(dp), intent(in)  :: cen_alpha         ! alpha

  ! regularization parameters for dynamical imaging
  real(dp), intent(in) :: rt_l
  real(dp), intent(in) :: ri_l
  real(dp), intent(in) :: rs_l
  real(dp), intent(in) :: rf_l

  ! regularization function
  real(dp), intent(out) :: l1_cost    ! cost of l1
  real(dp), intent(out) :: sm_cost    ! cost of second moment
  real(dp), intent(out) :: tv_cost    ! cost of tv
  real(dp), intent(out) :: tsv_cost   ! cost of tsv
  real(dp), intent(out) :: kl_cost    ! cost of KL divergence
  real(dp), intent(out) :: gs_cost    ! cost of GS entropy
  real(dp), intent(out) :: tfd_cost   ! cost of total flux regularization
  real(dp), intent(out) :: lc_cost    ! cost of light curve regularization
  real(dp), intent(out) :: cen_cost   ! cost of centoroid regularizaiton

  ! regularizer for dynamical imaging
  real(dp), intent(out) :: rt_cost, ri_cost, rs_cost, rf_cost

  ! second moment variables
  real(dp), intent(out) :: out_maj, out_min, out_phi


  ! Total Cost function
  real(dp), intent(out) :: cost                 ! cost function
  real(dp), intent(out) :: gradcost(1:Npix*Nz)  ! gradient of the cost function

  ! gradcost for dynamical imaging
  real(dp) :: di_gradcost(Npix*Nz), di_cost
  ! values for each frame
  real(dp) :: l1_cost_frm, sm_cost_frm, tv_cost_frm, &
              tsv_cost_frm, kl_cost_frm, gs_cost_frm, &
              tfd_cost_frm, cen_cost_frm,&
              out_maj_frm, out_min_frm, out_phi_frm,&
              cost_frm
  real(dp) :: gradcost_frm(Npix)
  real(dp) :: lc_cost_frm, grad_lc_cost_frm
  real(dp) :: l1_wgt_lc(l1_Nwgt), tv_wgt_lc(tv_Nwgt), tsv_wgt_lc(tsv_Nwgt),&
              kl_wgt_lc(kl_Nwgt), gs_wgt_lc(gs_Nwgt)
  integer :: iz

  ! Initialize
  l1_cost  = 0d0
  sm_cost  = 0d0
  tv_cost  = 0d0
  tsv_cost = 0d0
  kl_cost  = 0d0
  gs_cost  = 0d0
  tfd_cost = 0d0
  lc_cost  = 0d0
  cen_cost = 0d0

  rt_cost = 0d0
  ri_cost = 0d0
  rs_cost = 0d0
  rf_cost = 0d0
  di_cost = 0d0

  out_maj  = 0d0
  out_min  = 0d0
  out_phi  = 0d0

  cost     = 0d0
  gradcost(:) = 0d0
  di_gradcost(:) = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix, I1d, xidx, yidx, Nxref, Nyref, Nx, Ny, Nz,&
  !$OMP                l1_l,  l1_wgt,  l1_Nwgt,&
  !$OMP                tv_l,  tv_wgt,  tv_Nwgt,&
  !$OMP                tsv_l, tsv_wgt, tsv_Nwgt,&
  !$OMP                kl_l,  kl_wgt,  kl_Nwgt,&
  !$OMP                gs_l,  gs_wgt,  gs_Nwgt,&
  !$OMP                tfd_l, tfd_tgtfd,&
  !$OMP                lc_l, lc_tgtfd,lc_nidx,&
  !$OMP                cen_l, cen_alpha,&
  !$OMP                sm_l, sm_maj, sm_min, sm_phi) &
  !$OMP   PRIVATE(iz, l1_cost_frm, sm_cost_frm, tv_cost_frm, tsv_cost_frm,&
  !$OMP           kl_cost_frm, gs_cost_frm, tfd_cost_frm, cen_cost_frm,&
  !$OMP           out_maj_frm, out_min_frm, out_phi_frm, cost_frm, gradcost_frm,&
  !$OMP           lc_cost_frm, grad_lc_cost_frm,&
  !$OMP            l1_wgt_lc, tv_wgt_lc, tsv_wgt_lc,kl_wgt_lc, gs_wgt_lc) &
  !$OMP   REDUCTION(+: cost, gradcost, l1_cost, sm_cost, tv_cost, tsv_cost,&
  !$OMP                kl_cost, gs_cost, tfd_cost, lc_cost, cen_cost, out_maj, out_min, out_phi)


  do iz=1, Nz
    if (sum(lc_l) > 0.) then
    if (lc_nidx > 0) then
      l1_wgt_lc(:) = l1_wgt(:) / sum(l1_wgt) * lc_l(iz)
      tv_wgt_lc(:) = tv_wgt_lc(:) / sum(tv_wgt_lc) * lc_l(iz)
      tsv_wgt_lc(:) = tsv_wgt(:) / sum(tsv_wgt) * lc_l(iz) * lc_l(iz)
      kl_wgt_lc(:) = kl_wgt(:) / sum(kl_wgt) * lc_l(iz)
      gs_wgt_lc(:) = gs_wgt(:) / sum(gs_wgt) * lc_l(iz)

    else
      l1_wgt_lc(:) = l1_wgt(:)
      tv_wgt_lc(:) = tv_wgt(:)
      tsv_wgt_lc(:) = tsv_wgt(:)
      kl_wgt_lc(:) = kl_wgt(:)
      gs_wgt_lc(:) = gs_wgt(:)
    end if
    call calc_cost_reg(&
          I1d((iz-1)*Npix+1:iz*Npix), &
          xidx, yidx, Nxref, Nyref, Nx, Ny,&
          l1_l, l1_wgt_lc, l1_Nwgt,&
          tv_l, tv_wgt_lc, tv_Nwgt,&
          tsv_l, tsv_wgt_lc, tsv_Nwgt,&
          kl_l, kl_wgt_lc, kl_Nwgt,&
          gs_l, gs_wgt_lc, gs_Nwgt,&
          lc_l(iz), lc_tgtfd(iz),&
          -1d0, 1d0,& ! cen_l=-1, cen_alpha=1d0,
          sm_l, sm_maj, sm_min, sm_phi,&
          l1_cost_frm, &
          tv_cost_frm, tsv_cost_frm, kl_cost_frm, gs_cost_frm,&
          lc_cost_frm, cen_cost_frm,&
          sm_cost_frm,&
          out_maj_frm, out_min_frm, out_phi_frm,&
          cost_frm, gradcost_frm, Npix &
      )
      lc_cost = lc_cost + lc_cost_frm / Nz

    else
      call calc_cost_reg(&
          I1d((iz-1)*Npix+1:iz*Npix), &
          xidx, yidx, Nxref, Nyref, Nx, Ny,&
          l1_l, l1_wgt, l1_Nwgt,&
          tv_l, tv_wgt, tv_Nwgt,&
          tsv_l, tsv_wgt, tsv_Nwgt,&
          kl_l, kl_wgt, kl_Nwgt,&
          gs_l, gs_wgt, gs_Nwgt,&
          tfd_l, tfd_tgtfd,&
          -1d0, 1d0,& ! cen_l=-1, cen_alpha=1d0,
          sm_l, sm_maj, sm_min, sm_phi,&
          l1_cost_frm, &
          tv_cost_frm, tsv_cost_frm, kl_cost_frm, gs_cost_frm,&
          tfd_cost_frm, cen_cost_frm,&
          sm_cost_frm,&
          out_maj_frm, out_min_frm, out_phi_frm,&
          cost_frm, gradcost_frm, Npix &
      )
      tfd_cost = tfd_cost + tfd_cost_frm / Nz
    end if

    l1_cost  = l1_cost + l1_cost_frm / Nz
    sm_cost  = sm_cost + sm_cost_frm / Nz
    tv_cost  = tv_cost + tv_cost_frm / Nz
    tsv_cost = tsv_cost + tsv_cost_frm / Nz
    kl_cost  = kl_cost + kl_cost_frm / Nz
    gs_cost  = gs_cost + gs_cost_frm / Nz

    out_maj  = out_maj + out_maj_frm / Nz
    out_min  = out_min + out_min_frm / Nz
    out_phi  = out_phi + out_phi_frm / Nz
    cost     = cost + cost_frm / Nz
    gradcost((iz-1)*Npix+1:iz*Npix) = gradcost_frm / Nz
  end do

  !$OMP END PARALLEL DO

  ! dynamical imaging regularizers
  if (cen_l > 0. .or. rt_l > 0 .or. ri_l > 0 .or. rs_l > 0) then
    call calc_cost_dynamical(&
      Npix, Nz, xidx, yidx, Nxref, Nyref, I1d, cen_l, cen_alpha, rt_l, ri_l, rs_l, rf_l, &
      cen_cost, rt_cost, ri_cost, rs_cost, rf_cost, lc_nidx, lc_tgtfd, di_cost, di_gradcost&
    )
    cost     = cost + di_cost
    gradcost(:) = gradcost(:) + di_gradcost(:)
  end if
end subroutine
!
!-------------------------------------------------------------------------------
! Regularization Function for Dynamical Imaging
!-------------------------------------------------------------------------------
subroutine calc_cost_dynamical(&
  Npix, Nz, xidx, yidx, Nxref, Nyref, Iin, cen_l, cen_alpha, rt_l, ri_l, rs_l, rf_l, &
  cen_cost, rt_cost, ri_cost, rs_cost, rf_cost, lc_nidx, lc_tgtfd, &
  cost, gradcost &
)
  implicit none

  ! allocatable arrays
  integer, intent(in) :: Nz, Npix
  real(dp), intent(in)  :: Nxref, Nyref
  real(dp), intent(in) :: Iin(Npix*Nz)
  integer,  intent(in)  :: xidx(Npix), yidx(Npix)
  real(dp), intent(in) :: cen_l, cen_alpha, rt_l, ri_l, rs_l, rf_l
  integer, intent(in) :: lc_nidx
  real(dp), intent(in) :: lc_tgtfd(Nz)
  real(dp), intent(out) :: cen_cost, rt_cost, ri_cost, rs_cost, rf_cost
  real(dp), intent(out) :: cost, gradcost(Npix*Nz)
  ! light curve normalization
  real(dp) :: Iavg(Npix), Iin_frm(Npix), s, su, sl, f, fu, fl, Il, Iu, II

  integer :: ipix, iz
  real(dp) :: totalflux, stotal
  real(dp) :: ri_wgt, rt_wgt, rs_wgt, rf_wgt, gradcost_cen(Npix)

  real(dp)  :: tmp
  real(dp), allocatable :: tmp1d(:)

  ! Initialize
  Iavg(:)     = 0d0
  totalflux   = 0d0
  stotal      = 0d0
  cen_cost    = 0d0
  rt_cost     = 0d0
  ri_cost     = 0d0
  rs_cost     = 0d0
  rf_cost     = 0d0
  cost        = 0d0
  gradcost(:) = 0d0

  ! Calculate averaged intensity map, totalflux and total entropy
  do iz=1, Nz
    Iin_frm = Iin((iz-1)*Npix+1:iz*Npix)
    Iavg = Iavg + Iin_frm / Nz
    totalflux  = f_e(Iavg, Npix)
  end do

  if (cen_l > 0) then
    allocate(tmp1d(Npix))
    call calc_cenreg(Iavg,xidx,yidx,Nxref,Nyref,cen_alpha,tmp,tmp1d,Npix)
    cen_cost = cen_l * tmp

    do iz=1, Nz
      gradcost((iz-1)*Npix+1:iz*Npix) = gradcost((iz-1)*Npix+ipix) + &
                                        cen_l * tmp1d / Nz
    end do
    deallocate(tmp1d)
  end if

  ! uniform weight of each dynamical regularizer
  ri_wgt = 1. / (totalflux * Nz)
  rt_wgt = 1. / (totalflux * Nz)
  rs_wgt = 1. / (stotal * Nz)
  rf_wgt = 1. / (totalflux * Nz)

  do iz=1,Nz
    Iin_frm = Iin((iz-1)*Npix+1:iz*Npix)

    ! Regularization normalization with light curve
    if (lc_nidx >0) then
      ri_wgt = 1. / (lc_tgtfd(iz) * Nz)
      rt_wgt = 1. / (lc_tgtfd(iz) * Nz)
      rf_wgt = 1. / (lc_tgtfd(iz) * Nz)
    end if

    ! For Rs or Rf regularizer
    if (rs_l > 0 .or. rf_l > 0) then
      if (iz < Nz-1) then
        s  = s_e(Iin((iz-1)*Npix+1:iz*Npix),Npix)
        su = s_e(Iin(iz*Npix+1:(iz+1)*Npix),Npix)
        f  = f_e(Iin((iz-1)*Npix+1:iz*Npix),Npix)
        fu = f_e(Iin(iz*Npix+1:(iz+1)*Npix),Npix)
        if(iz > 1) then
          sl = s_e(Iin((iz-2)*Npix+1:(iz-1)*Npix),Npix)
          fl = f_e(Iin((iz-2)*Npix+1:(iz-1)*Npix),Npix)
        end if
      end if
    end if

    do ipix=1,Npix
      II = Iin((iz-1)*Npix+ipix)
      if (iz < Nz-1) then
        Iu = Iin(iz*Npix+ipix)
      else
        Iu = II
      end if
      if (iz > 1) then
        Il = Iin((iz-2)*Npix+ipix)
      else
        Il = II
      end if
      ! Rt regularizer
      if (rt_l > 0) then
        rt_cost  = rt_cost + rt_l * rt_wgt * rt_e(II, Iu)
        gradcost((iz-1)*Npix+ipix) = gradcost((iz-1)*Npix+ipix) + &
                  rt_l * rt_wgt * rt_grade(II, Il, Iu)
      end if

      ! Ri regularizer
      if (ri_l > 0) then
        ri_cost     = ri_cost + ri_l * ri_wgt * ri_e(II, Iavg(ipix))
        gradcost((iz-1)*Npix+ipix) = gradcost((iz-1)*Npix+ipix) + &
                          ri_l * ri_wgt * ri_grade(II, Iavg(ipix))
      end if

      ! Rs regularizer
      if (rs_l > 0) then
        if (iz < Nz-1 .and. iz>1) then
          rs_cost  = rs_cost + rs_l * rs_wgt * rs_e(s, su)
          gradcost((iz-1)*Npix+ipix) = gradcost((iz-1)*Npix+ipix) + &
                    rs_l * rs_wgt * rs_grade(s, sl, su, Iin_frm(ipix))
        end if
      end if

      ! Rf regularizer
      if (rf_l > 0) then
        if (iz < Nz-1 .and. iz>1) then
          rf_cost  = rf_cost + rf_l * rf_wgt * rf_e(f, fu)
          gradcost((iz-1)*Npix+ipix) = gradcost((iz-1)*Npix+ipix) + &
                    rf_l * rf_wgt * rf_grade(f, fl, fu, Iin_frm(ipix))
        end if
      end if

    end do
    ! take summation of all the cost function
    cost = cen_cost + ri_cost + rt_cost + rs_cost + rf_cost

  end do

end subroutine


!-------------------------------------------------------------------------------
! Rt regularizer
!-------------------------------------------------------------------------------
! rt norm at each pixel and time frame
real(dp) function rt_e(I, Iu)
  implicit none
  real(dp), intent(in) :: I, Iu      ! pixel intensity of two time frames

  rt_e = (Iu - I)**2
end function


! gradient of rt norm at each pixel and time frame
real(dp) function rt_grade(I, Il, Iu)
  implicit none
  real(dp), intent(in) :: I, Iu, Il  ! pixel intensity of three time frames

  rt_grade = 2. * ((I - Il) +(I - Iu))
end function

!-------------------------------------------------------------------------------
! Ri regularizer
!-------------------------------------------------------------------------------
!ri norm at each pixel and time frame
real(dp) function ri_e(I, Iavg)
  implicit none
  real(dp), intent(in) :: I, Iavg    ! pixel intensity and averaged one

  ri_e = (I - Iavg)**2
end function


! gradient of ri norm at each pixel and time frame
real(dp) function ri_grade(I, Iavg)
  implicit none
  real(dp), intent(in) :: I, Iavg    ! pixel intensity and averaged one

  ri_grade = 2. *  (I - Iavg)
end function

!-------------------------------------------------------------------------------
! Rs regularizer
!-------------------------------------------------------------------------------
! rs norm at each pixel and time frame
real(dp) function rs_e(s, su)
  implicit none
  real(dp), intent(in) :: s, su      ! maximum entropy of two time frames
  rs_e = (s - su)**2
end function


! calculate maximum entropy
real(dp) function s_e(I1d, Npix)
  implicit none
  integer, intent(in)  :: Npix
  real(dp), intent(in) :: I1d(Npix) ! pixel intensity of a time frame
  integer :: ipix
  real(dp) :: se

  se = 0d0

  do ipix=1, Npix
    se = se - smabs(I1d(ipix))*log(smabs(I1d(ipix)))
  end do

  s_e = se

end function


! gradient of rs norm at each pixel and time frame
real(dp) function rs_grade(s, sl, su, I)
  implicit none
  real(dp), intent(in) :: s, sl, su  ! maximum entropy of three time frames
  real(dp), intent(in) :: I          ! pixel intensity of a time frame
  rs_grade = 2. * (2. * s - sl - su) * s_grade(I)
end function


! gradient of maximum entropy
real(dp) function s_grade(I)
  implicit none
  real(dp), intent(in) :: I          ! pixel intensity of a time frame

  s_grade = -(log(smabs(I))+1.)
end function


!-------------------------------------------------------------------------------
! Rf regularizer
!-------------------------------------------------------------------------------
! rf norm at each pixel and time frame
real(dp) function rf_e(f, fu)
  implicit none
  real(dp), intent(in) :: f, fu      ! maximum entropy of two time frames
  rf_e = (f - fu)**2
end function


! calculate maximum entropy
real(dp) function f_e(I1d, Npix)
  implicit none
  integer, intent(in)  :: Npix
  real(dp), intent(in) :: I1d(Npix) ! pixel intensity of a time frame
  integer :: ipix
  real(dp) :: fe

  fe = 0d0

  do ipix=1, Npix
    fe = fe + smabs(I1d(ipix))
  end do

  f_e = fe

end function


! gradient of rf norm at each pixel and time frame
real(dp) function rf_grade(f, fl, fu, I)
  implicit none
  real(dp), intent(in) :: f, fl, fu  ! total flux of three time frames
  real(dp), intent(in) :: I          ! pixel intensity of a time frame

  rf_grade = 2. * (2. * f - fl - fu)
end function


!-------------------------------------------------------------------------------
! Distances
!-------------------------------------------------------------------------------
!
! Dp (p=2) distance
!
real(dp) function d2(xidx,yidx,I2d,I2du,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  i = xidx
  j = yidx
  !
  d2  = (I2du(i,j) - I2d(i,j))**2
end function
!
! Kullback-Leibler divergence
!
real(dp) function dkl(xidx,yidx,I2d,I2du,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  i = xidx
  j = yidx
  !
  if (I2d(i,j) /= 0 .and. I2du(i,j) /= 0) then
    dkl  = I2d(i,j) * log(I2d(i,j)/I2du(i,j))
  else if (I2d(i,j) == 0 .or. I2du(i,j) == 0) then
    dkl  = 0
  end if
  !
end function
!
!-------------------------------------------------------------------------------
! Gradients
!-------------------------------------------------------------------------------
!
! Gradient of Rt from D2 distance
!
real(dp) function rt_d2grad(xidx,yidx,zidx,I2d,I2dl,I2du,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,Nz
  integer, intent(in)  :: xidx, yidx, zidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2dl(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  rt_d2grad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  if (zidx > 1) then
    rt_d2grad = rt_d2grad + 2*(I2d(i,j) - I2dl(i,j))
  end if
  if (zidx < Nz) then
    rt_d2grad = rt_d2grad - 2*(I2du(i,j) - I2d(i,j))
  end if
  !
end function
!
! Gradient of Rt from Kullback-Leibler divergence
!
real(dp) function rt_dklgrad(xidx,yidx,zidx,I2d,I2dl,I2du,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,Nz
  integer, intent(in)  :: xidx, yidx, zidx
  real(dp),intent(in)  :: I2d(Nx,Ny),I2dl(Nx,Ny),I2du(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  rt_dklgrad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  if (zidx > 1 .and. I2d(i,j) /= 0 .and. I2dl(i,j) /= 0) then
    rt_dklgrad = rt_dklgrad + (1 + log(I2d(i,j)/I2dl(i,j)))
  end if
  if (zidx < Nz .and. I2d(i,j) /= 0) then
    rt_dklgrad = rt_dklgrad - (I2du(i,j)/I2d(i,j))
  end if
  !
end function
!
! Gradient of Ri from D2 distance
!
real(dp) function ri_d2grad(xidx,yidx,I2d,Iavg2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,xidx, yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),Iavg2d(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  ri_d2grad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  ri_d2grad = 2 * (I2d(i,j) - Iavg2d(i,j))
  !
end function
!
! Gradient of Ri from Kullback-Leibler divergence
!
real(dp) function ri_dklgrad(xidx,yidx,I2d,Iavg2d,Itmp2d,Nx,Ny,Nz)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny,Nz,xidx,yidx
  real(dp),intent(in)  :: I2d(Nx,Ny),Iavg2d(Nx,Ny),Itmp2d(Nx,Ny)
  !
  ! variables
  integer :: i,j
  !
  ! initialize rt term
  ri_dklgrad = 0d0
  !
  ! take indice
  i = xidx
  j = yidx
  !
  ri_dklgrad = 1 - (Iavg2d(i,j)/I2d(i,j)) + Itmp2d(i,j)/Nz
  !
end function
!


end module
!
!
!-------------------------------------------------------------------------------
! A convinient function to compute regularization functions
! for python interfaces
!-------------------------------------------------------------------------------
!
! subroutine I2d_rtd2(I2d,I2dl,I2du,cost,costmap,gradmap,Nx,Ny,Nz)
!   implicit none
!
!   integer, intent(in)  :: Nx,Ny,Nz
!   real(dp), intent(in) :: I2d(Nx,Ny),I2dl(Nx,Ny),I2du(Nx,Ny)
!   real(dp), intent(out):: cost,costmap(Nx,Ny),gradmap(Nx,Ny)
!
!   integer :: ixy,ix,iy,iz
!
!   ! initialize output
!   cost = 0d0
!   costmap(:,:) = 0d0
!   gradmap(:,:) = 0d0
!
!   !$OMP PARALLEL DO DEFAULT(SHARED) &
!   !$OMP   FIRSTPRIVATE(Nx,Ny,Nz,I2d,I2dl,I2du) &
!   !$OMP   PRIVATE(ixy,ix,iy,iz) &
!   !$OMP   REDUCTION(+:cost,costmap,gradmap)
!   do iz=1, Nz
!     do ixy=1, Nx*Ny
!       iparm = (iz-1)*Nx*Ny + ixy
!       call ixy2ixiy(ixy,ix,iy,Nx)
!       costmap(ix,iy) = d2(ix,iy,I2d,I2du,Nx,Ny)
!       gradmap(ix,iy) = rt_d2grad(ix,iy,iz,I2d,I2dl,I2du,Nx,Ny,Nz)
!       cost = cost+costmap(ix,iy)
!     end do
!   end do
!   !$OMP END PARALLEL DO
! end subroutine
!
