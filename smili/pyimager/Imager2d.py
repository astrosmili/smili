from collections import OrderedDict

class LBFGSB_Imager_2d(object):
    do_listuv  = True
    do_initreg = True

    def __init__(self):
        pass

    def init_params(self):
        '''
        Initialize the Parameter Dictionary
        '''
        # Create Parameters
        param = OrderedDict()

        # Append items
        #   data tables
        param["vistable"] = None
        param["amptable"] = None
        param["bstable"] = None
        param["catable"] = None
        #   Expected/Target Total Flux
        param["totalflux"] = None
        #   Non-negativity
        param["nonneg"] = True
        #   l1
        param["l1_lambda"] = -1
        param["l1_prior"]  = None
        #   tv
        param["tv_lambda"] = -1
        param["tv_prior"]  = None
        #   tsv
        param["tsv_lambda"] = -1
        param["tsv_prior"]  = None
        #   kl
        param["kl_lambda"] = -1
        param["kl_prior"]  = None
        #   gs
        param["gs_lambda"] = -1
        param["gs_prior"]  = None
        #   second moment regularizaion
        #param["secmom_lambda"]  = -1
        #param["secmom_majsize"] = 1
        #param["secmom_minsize"] = None
        #param["secmom_pa"]      = None
        #param["secmom_angunit"] = None
        #   total flux regularization
        param["tfd_lambda"] = -1
        param["tfd_tgterror"] = 0.01
        #   centroid regularization
        param["cen_lambda"] = -1
        param["cen_power"] = 3
        #   L-BFGS-B "m": 5, "factr": 1e1, "pgtol": 0.
        param["lbfgsb_m"] = 5
        param["lbfgsb_factr"] = 1e1
        param["lbfgsb_pgtol"] = 0
        self.param = param

    def set_params(self, **kwargs):
        for key in kwargs:
            if key not in self.param.keys():
                msg = "%s does not exist in the parameter keys."%(key)
                raise ValueError(msg)

            # set values
            self.param[key] = kwargs[key]

            # set flags
            if "table" in key:
                self.do_listuv = True

            if "vistable" in key or "amptable" in key:
                self.do_est_totalflux = True

            if "totalflux" in key:
                self.do_est_totalflux = True
                self.do_initreg = True

    def print_params(self, **argv):
        pass

    def precompute(self):
        self.sanity_check()

        if do_listuv:
            pass

        if do_initl1reg:
            pass

    def sanity_check(self):
        # check presence of data sets
        if ((self.param["vistable"] is None) and
            (self.param["amptable"] is None) and
            (self.param["bstable"] is None) and
            (self.param["catable"] is None)):
            msg = "No data are specified."
            raise ValueError(msg)

        # check if we have something that we can guess total flux
        if ((self.param["vistable"] is None) and
            (self.param["amptable"] is None) and
            (self.param["totalflux"] is None)):
            msg = "No information that can provide an estimate of the total flux "
            msg+= "in the input data sets nor parameter sets. "
            msg+= "You need to specify at least one of vistable, amptable or totalflux."
            raise ValueError(msg)

        # check if we have any amplitude information that we can guess total flux
        if ((self.param["vistable"] is None) and
            (self.param["amptable"] is None) and
            (self.param["tfd_lambda"] is None)):
            msg = "No absolute amplitude information is given "
            msg+= "in the input data sets nor parameter sets. "
            msg+= "You need to (A) specify vistable and/or amptable, and/or "
            msg+= "(B) use the total flux regularization by specifying tfd_lambda."
            raise ValueError(msg)

        # check if we have the absolute phase information
        if ((self.param["vistable"] is None) and
            (self.param["l1_prior"] is None) and
            (self.param["tv_prior"] is None) and
            (self.param["tsv_prior"] is None) and
            (self.param["gs_prior"] is None) and
            (self.param["kl_prior"] is None) and
            (self.param["cen_lambda"] <= 0)):
            msg = "[Warning] None of the absolute phase (vistable), prior images "
            msg+= "for regularizations (xx_prior) nor centroid regularization (cen_lambda) "
            msg+= "is specified. This parameter has a very poor constraint on "
            msg+= "the absolute position information."
            print(msg)

    def _update_initimgarr():
        # Get initial images
        self.Iin = np.float64(initimage.data[istokes, ifreq])
        #   size of images
        Nx = initimage.header["nx"]
        Ny = initimage.header["ny"]
        Nyx = Nx * Ny
        #   pixel coordinates
        x, y = initimage.get_xygrid(twodim=True, angunit="rad")
        xidx = np.arange(Nx) + 1
        yidx = np.arange(Ny) + 1
        xidx, yidx = np.meshgrid(xidx, yidx)
        Nxref = initimage.header["nxref"]
        Nyref = initimage.header["nyref"]
        dx_rad = np.deg2rad(initimage.header["dx"])
        dy_rad = np.deg2rad(initimage.header["dy"])
        pass

    def _listuv(self):
        '''
        Compute non-redundant set of uv-coordinates
        '''
        pass

    def initreg(self):
        '''
        Initialize regularization functions
        '''
        pass

    def imaging(niter=1000, nprint=500):
        pass
