module uvdata
  !$use omp_lib
  use param, only: dp, sp, seps, shug
  use interp, only: spline, splintvec
  implicit none
contains
!
! average
!
subroutine average(uvdata,u,v,w,tin,tout,start,end,solint,minpoint, &
                   uvdataout,uout,vout,wout,isdata,&
                   Nstokes,Nch,Nif,Nra,Ndec,Ndata,Nt,Nidx)
  ! Number of Data
  integer, intent(in) :: Nstokes,Nch,Nif,Nra,Ndec,Ndata
  ! Number of Time
  integer, intent(in) :: Nt,Nidx
  ! Parameter for time averaging
  real(dp), intent(in) :: solint    ! integration time in sec
  integer, intent(in) :: minpoint   ! minimum points that data will be averaged
  ! Input Data
  real(sp), intent(in) :: uvdata(3,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  real(dp), intent(in) :: u(Ndata), v(Ndata), w(Ndata) ! uvw coordinates
  real(dp), intent(in) :: tin(Ndata)! UTC time of the input data
  real(dp), intent(in) :: tout(Nt)  ! UTC time of the output data
  integer, intent(in) :: start(Nidx), end(Nidx) ! start/end index of each baseline & source
  ! Output Data
  real(sp), intent(out) :: uvdataout(3,Nstokes,Nch,Nif,Nra,Ndec,Nidx*Nt)
  real(dp), intent(out) :: uout(Nidx*Nt),vout(Nidx*Nt),wout(Nidx*Nt)
  logical, intent(out) :: isdata(Nidx*Nt)

  integer :: i1,i2,i3,i4,i5,i6,i7,idx,it
  integer :: Ndata_idx
  integer, allocatable:: cnt(:,:,:,:,:)
  real(sp), allocatable:: vrsum(:,:,:,:,:),visum(:,:,:,:,:),wsum(:,:,:,:,:)
  real(sp), allocatable:: uvdatatmp(:,:,:,:,:,:,:)
  real(dp), allocatable:: tintmp(:),uvwtmp1(:),uvwtmp2(:)

  ! initialize arrays
  uvdataout(:,:,:,:,:,:,:) = 0.0
  isdata(:) = .False.
  uout(:) = 0d0
  vout(:) = 0d0
  wout(:) = 0d0

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(tout,solint,minpoint,start,end,u,v,w,&
  !$OMP                Nstokes,Nch,Nif,Nra,Ndec,Ndata,Nt,Nidx) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,i7,idx,it,Ndata_idx,&
  !$OMP           uvdatatmp,uvwtmp1,uvwtmp2,tintmp,cnt,vrsum,visum,wsum) &
  !$OMP   REDUCTION(+:uvdataout,uout,vout,wout)
  do idx=1, Nidx
    Ndata_idx = end(idx) - start(idx) + 1

    ! temporal time data for this particular data index
    allocate(tintmp(Ndata_idx))
    tintmp(1:Ndata_idx) = tin(start(idx):end(idx))

    ! Interpolate uvw coordinates
    !   allocate arrays
    allocate(uvwtmp1(Ndata_idx),uvwtmp2(Ndata_idx))
    !   U coodinates
    uvwtmp1(1:Ndata_idx) = u(start(idx):end(idx))
    call spline(tintmp, uvwtmp1, 1d30, 1d30, uvwtmp2, Ndata_idx)
    call splintvec(tintmp,uvwtmp1,uvwtmp2,tout,uout((idx-1)*Nt+1:idx*Nt),Ndata_idx,Nt)
    !   V coodinates
    uvwtmp1(1:Ndata_idx) = v(start(idx):end(idx))
    call spline(tintmp, uvwtmp1, 1d30, 1d30, uvwtmp2, Ndata_idx)
    call splintvec(tintmp,uvwtmp1,uvwtmp2,tout,vout((idx-1)*Nt+1:idx*Nt),Ndata_idx,Nt)
    !   W coodinates
    uvwtmp1(1:Ndata_idx) = w(start(idx):end(idx))
    call spline(tintmp, uvwtmp1, 1d30, 1d30, uvwtmp2, Ndata_idx)
    call splintvec(tintmp,uvwtmp1,uvwtmp2,tout,wout((idx-1)*Nt+1:idx*Nt),Ndata_idx,Nt)
    !   deallocate arrays
    deallocate(uvwtmp1,uvwtmp2)

    ! allocate and initialize arrays for averaging uvw data
    !   temporal input data for this particular data index
    allocate(uvdatatmp(3,Nstokes,Nch,Nif,Nra,Ndec,Ndata_idx))
    uvdatatmp(:,:,:,:,:,:,1:Ndata_idx) = uvdata(:,:,:,:,:,:,start(idx):end(idx))

    !   This is a counter that how many data points are included in a specific
    !   time segment
    allocate(cnt(Nstokes,Nch,Nif,Nra,Ndec))
    !   sum of real/imag/weight data
    allocate(vrsum(Nstokes,Nch,Nif,Nra,Ndec))
    allocate(visum(Nstokes,Nch,Nif,Nra,Ndec))
    allocate(wsum(Nstokes,Nch,Nif,Nra,Ndec))
    !
    ! Take weighted sum for each time segments
    do it=1, Nt
      ! allocate and initialize arrays
      cnt=0
      vrsum=0d0
      visum=0d0
      wsum=0d0
      !
      ! take weighted sum
      do i1=1,Ndata_idx
        ! if data is not in the time segment covered by tout(it), skip all precedure
        if (abs(tintmp(i1)-tout(it)) > solint/2) then
          cycle
        end if
        do i2=1,Ndec
          do i3=1,Nra
            do i4=1,Nif
              do i5=1,Nch
                do i6=1,Nstokes
                  if (uvdatatmp(3,i6,i5,i4,i3,i2,i1) < seps) then
                    cycle
                  end if
                  if (uvdatatmp(3,i6,i5,i4,i3,i2,i1) > shug) then
                    cycle
                  end if
                  if (uvdatatmp(3,i6,i5,i4,i3,i2,i1) .ne. uvdatatmp(3,i6,i5,i4,i3,i2,i1)) then
                    cycle
                  end if
                  ! take summation of data
                  vrsum(i6,i5,i4,i3,i2)= uvdatatmp(1,i6,i5,i4,i3,i2,i1) &
                                       * uvdatatmp(3,i6,i5,i4,i3,i2,i1) &
                                       + vrsum(i6,i5,i4,i3,i2)
                  visum(i6,i5,i4,i3,i2)= uvdatatmp(2,i6,i5,i4,i3,i2,i1) &
                                       * uvdatatmp(3,i6,i5,i4,i3,i2,i1) &
                                       + visum(i6,i5,i4,i3,i2)
                  wsum(i6,i5,i4,i3,i2) = uvdatatmp(3,i6,i5,i4,i3,i2,i1) &
                                       + wsum(i6,i5,i4,i3,i2)
                  cnt(i6,i5,i4,i3,i2) = cnt(i6,i5,i4,i3,i2) + 1
                end do !Stokes
              end do !ch
            end do !IF
          end do !RA
        end do !DEC
      end do !Ndata_idx
      !
      ! normalize weighted sum of Vreal, Vimag
      do i2=1,Ndec
        do i3=1,Nra
          do i4=1,Nif
            do i5=1,Nch
              do i6=1,Nstokes
                if (cnt(i6,i5,i4,i3,i2)>=minpoint) then
                  vrsum(i6,i5,i4,i3,i2) = vrsum(i6,i5,i4,i3,i2)/wsum(i6,i5,i4,i3,i2)
                  visum(i6,i5,i4,i3,i2) = visum(i6,i5,i4,i3,i2)/wsum(i6,i5,i4,i3,i2)
                  cnt(i6,i5,i4,i3,i2) = 1
                else
                  vrsum(i6,i5,i4,i3,i2) = 0
                  visum(i6,i5,i4,i3,i2) = 0
                  wsum(i6,i5,i4,i3,i2) = 0
                  cnt(i6,i5,i4,i3,i2) = 0
                end if
              end do !Stokes
            end do !ch
          end do !IF
        end do !RA
      end do !DEC
      !
      ! copy results to output array
      uvdataout(1,:,:,:,:,:,(idx-1)*Nt+it)=vrsum(:,:,:,:,:)
      uvdataout(2,:,:,:,:,:,(idx-1)*Nt+it)=visum(:,:,:,:,:)
      uvdataout(3,:,:,:,:,:,(idx-1)*Nt+it)=wsum(:,:,:,:,:)
      !
      ! check if data exists
      if (sum(cnt) > 0) then
        isdata((idx-1)*Nt+it) = .True.
      end if
    end do ! Nt
    deallocate(uvdatatmp,tintmp,vrsum,visum,wsum,cnt)
  end do ! Nidx
  !$OMP END PARALLEL DO
end subroutine
!
! weightcal
!
subroutine weightcal(uvdata,tsec,ant1,ant2,subarray,source,&
                     solint,dofreq,minpoint,uvdataout,&
                     Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  implicit none

  integer,  intent(in) :: Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata
  integer,  intent(in) :: source(Ndata),ant1(Ndata),ant2(Ndata),subarray(Ndata)
  integer,  intent(in) :: dofreq,minpoint
  real(sp), intent(in) :: uvdata(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  real(dp), intent(in) :: solint,tsec(Ndata)
  real(sp), intent(out) :: uvdataout(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)

  integer :: i1,i2,i3,i4,i5,i6,i7,N
  logical :: flag
  real(dp) :: aver, avei, msqr, msqi, var
  real(dp) :: vmr(Nstokes,Nch,Nif,Nra,Ndec),vmi(Nstokes,Nch,Nif,Nra,Ndec)
  real(dp) :: vrr(Nstokes,Nch,Nif,Nra,Ndec),vri(Nstokes,Nch,Nif,Nra,Ndec)
  integer :: cnt(Nstokes,Nch,Nif,Nra,Ndec)

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(source,ant1,ant2,subarray,tsec,solint,dofreq,minpoint, &
  !$OMP                Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,i7,N,&
  !$OMP           aver,avei,msqr,msqi,var,vmr,vmi,vrr,vri,cnt,flag)
  do i1=1,Ndata
    uvdataout(:,:,:,:,:,:,i1) = uvdata(:,:,:,:,:,:,i1)
    vmr(:,:,:,:,:) = 0d0
    vmi(:,:,:,:,:) = 0d0
    vrr(:,:,:,:,:) = 0d0
    vri(:,:,:,:,:) = 0d0
    cnt(:,:,:,:,:) = 0
    do i2=1,Ndata
      ! check source and stokes
      flag = (source(i1) == source(i2))
      flag = flag .and. (ant1(i1) == ant1(i2))
      flag = flag .and. (ant2(i1) == ant2(i2))
      flag = flag .and. (subarray(i1) == subarray(i2))
      flag = flag .and. (abs(tsec(i2)-tsec(i1)) < solint)
      if (flag .eqv. .False.) then
        cycle
      end if
      ! calc sum and squared-sum of vreal, vimag
      do i3=1, Ndec
        do i4=1, Nra
          do i5=1, Nif
            do i6=1, Nch
              do i7=1, Nstokes
                if (uvdata(3,i7,i6,i5,i4,i3,i2) < seps) then
                  cycle
                end if
                if (uvdata(3,i7,i6,i5,i4,i3,i2) > shug) then
                  cycle
                end if
                if (uvdata(3,i7,i6,i5,i4,i3,i2) .ne. uvdata(3,i7,i6,i5,i4,i3,i2)) then
                  cycle
                end if
                vmr(i7,i6,i5,i4,i3) = uvdata(1,i7,i6,i5,i4,i3,i2) &
                                    + vmr(i7,i6,i5,i4,i3)
                vmi(i7,i6,i5,i4,i3) = uvdata(2,i7,i6,i5,i4,i3,i2) &
                                    + vmi(i7,i6,i5,i4,i3)
                vrr(i7,i6,i5,i4,i3) = uvdata(1,i7,i6,i5,i4,i3,i2) &
                                    * uvdata(1,i7,i6,i5,i4,i3,i2) &
                                    + vrr(i7,i6,i5,i4,i3)
                vri(i7,i6,i5,i4,i3) = uvdata(2,i7,i6,i5,i4,i3,i2) &
                                    * uvdata(2,i7,i6,i5,i4,i3,i2) &
                                    + vri(i7,i6,i5,i4,i3)
                cnt(i7,i6,i5,i4,i3) = cnt(i7,i6,i5,i4,i3)+1
              end do
            end do
          end do
        end do
      end do
    end do

    ! calc weight
    do i2=1, Ndec
      do i3=1, Nra
        if (dofreq .eq. 0) then
          do i4=1, Nstokes
            N = sum(cnt(i4,:,:,i3,i2))
            if (N < minpoint) then
              uvdataout(3,i4,:,:,i3,i2,i1) = 0.0
              cycle
            end if
            aver = sum(vmr(i4,:,:,i3,i2))/N
            avei = sum(vmi(i4,:,:,i3,i2))/N
            msqr = sum(vrr(i4,:,:,i3,i2))/N
            msqi = sum(vri(i4,:,:,i3,i2))/N
            var = 0.5 * (msqr - aver**2 + msqi - avei**2)
            uvdataout(3,i4,:,:,i3,i2,i1) = sngl(1d0/var)
          end do
        else if (dofreq .eq. 1) then
          do i4=1, Nif
            do i5=1, Nstokes
              N = sum(cnt(i5,:,i4,i3,i2))
              if (N < minpoint) then
                uvdataout(3,i5,:,i4,i3,i2,i1) = 0.0
                cycle
              end if
              aver = sum(vmr(i5,:,i4,i3,i2))/N
              avei = sum(vmi(i5,:,i4,i3,i2))/N
              msqr = sum(vrr(i5,:,i4,i3,i2))/N
              msqi = sum(vri(i5,:,i4,i3,i2))/N
              var = 0.5 * (msqr - aver**2 + msqi - avei**2)
              uvdataout(3,i5,:,i4,i3,i2,i1) = sngl(1d0/var)
            end do
          end do
        else
          do i4=1, Nif
            do i5=1, Nch
              do i6=1, Nstokes
                N = cnt(i6,i5,i4,i3,i2)
                if (N < minpoint) then
                  uvdataout(3,i6,i5,i4,i3,i2,i1) = 0.0
                  cycle
                end if
                aver = vmr(i6,i5,i4,i3,i2)/N
                avei = vmi(i6,i5,i4,i3,i2)/N
                msqr = vrr(i6,i5,i4,i3,i2)/N
                msqi = vri(i6,i5,i4,i3,i2)/N
                var = 0.5 * (msqr - aver**2 + msqi - avei**2)
                uvdataout(3,i6,i5,i4,i3,i2,i1) = sngl(1d0/var)
              end do
            end do
          end do
        end if

        do i4=1, Nif
          do i5=1, Nch
            do i6=1, Nstokes
              if (uvdata(3,i6,i5,i4,i3,i2,i1) < seps) then
                uvdataout(3,i6,i5,i4,i3,i2,i1) = 0
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) > shug) then
                uvdataout(3,i6,i5,i4,i3,i2,i1) = 0
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) .ne. uvdata(3,i6,i5,i4,i3,i2,i1)) then
                uvdataout(3,i6,i5,i4,i3,i2,i1) = 0
              end if
            end do
          end do
        end do
      end do
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
! average spectrum
!
subroutine avspc_dofreq0(uvdata,uvdataout,&
                         Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  implicit none

  integer,  intent(in) :: Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata
  real(sp), intent(in) :: uvdata(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  real(sp), intent(out) :: uvdataout(Ncomp,Nstokes,1,1,Nra,Ndec,Ndata)

  real(dp) :: weigsum(1:Nstokes), realsum(1:Nstokes), imagsum(1:Nstokes)
  integer :: i1,i2,i3,i4,i5,i6

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,weigsum,realsum,imagsum)
  do i1=1,Ndata
    do i2=1,Ndec
      do i3=1,Nra
        weigsum(:) = 0
        realsum(:) = 0
        imagsum(:) = 0
        do i4=1,Nif
          do i5=1,Nch
            do i6=1,Nstokes
              if (uvdata(3,i6,i5,i4,i3,i2,i1) < seps) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) > shug) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) .ne. uvdata(3,i6,i5,i4,i3,i2,i1)) then
                cycle
              end if
              weigsum(i6) = weigsum(i6) + uvdata(3,i6,i5,i4,i3,i2,i1)
              realsum(i6) = realsum(i6) + uvdata(1,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
              imagsum(i6) = imagsum(i6) + uvdata(2,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
            end do
          end do
        end do
        do i4=1,Nstokes
          uvdataout(1,i4,1,1,i3,i2,i1) = sngl(realsum(i4)/weigsum(i4))
          uvdataout(2,i4,1,1,i3,i2,i1) = sngl(imagsum(i4)/weigsum(i4))
          uvdataout(3,i4,1,1,i3,i2,i1) = sngl(weigsum(i4))
        end do
      end do
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
! average spectrum
!
subroutine avspc_dofreq1(uvdata,uvdataout,&
                         Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  implicit none

  integer,  intent(in) :: Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata
  real(sp), intent(in) :: uvdata(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata)
  real(sp), intent(out) :: uvdataout(Ncomp,Nstokes,1,Nif,Nra,Ndec,Ndata)

  real(dp) :: weigsum(1:Nstokes), realsum(1:Nstokes), imagsum(1:Nstokes)
  integer :: i1,i2,i3,i4,i5,i6

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(Ncomp,Nstokes,Nch,Nif,Nra,Ndec,Ndata) &
  !$OMP   PRIVATE(i1,i2,i3,i4,i5,i6,weigsum,realsum,imagsum)
  do i1=1,Ndata
    do i2=1,Ndec
      do i3=1,Nra
        do i4=1,Nif
          weigsum(:) = 0
          realsum(:) = 0
          imagsum(:) = 0
          do i5=1,Nch
            do i6=1,Nstokes
              if (uvdata(3,i6,i5,i4,i3,i2,i1) < seps) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) > shug) then
                cycle
              end if
              if (uvdata(3,i6,i5,i4,i3,i2,i1) .ne. uvdata(3,i6,i5,i4,i3,i2,i1)) then
                cycle
              end if
              weigsum(i6) = weigsum(i6) + uvdata(3,i6,i5,i4,i3,i2,i1)
              realsum(i6) = realsum(i6) + uvdata(1,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
              imagsum(i6) = imagsum(i6) + uvdata(2,i6,i5,i4,i3,i2,i1) * uvdata(3,i6,i5,i4,i3,i2,i1)
            end do
          end do
          do i5=1,Nstokes
            uvdataout(1,i5,1,i4,i3,i2,i1) = sngl(realsum(i5)/weigsum(i5))
            uvdataout(2,i5,1,i4,i3,i2,i1) = sngl(imagsum(i5)/weigsum(i5))
            uvdataout(3,i5,1,i4,i3,i2,i1) = sngl(weigsum(i5))
          end do
        end do
      end do
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
!
end module
