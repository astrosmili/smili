module finufft_fh
    implicit none
    
    type nufft_opts
        integer debug, spread_debug,spread_sort,spread_kerevalmeth,&
                spread_kerpad,chkbnds,fftw,modeord
        real*8 upsampfac
        integer spread_thread,maxbatchsize,showwarn,nthreads,&
                spread_nthr_atomic,spread_max_sp_size
    end type
end module