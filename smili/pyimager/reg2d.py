class Imager2D(object):
    do_listuv  = True
    do_initreg = True

    def __init__(self):
        return self.param

    def init_params(self):
        def mkparam(key, value, help, type):
            item = {}
            item["key"]   = "key"
            item["value"] = ""
        params = []
        params.append({
            "key": "vistable",
            "value": None,
            "help": "Visibility Table (uvdata.VisTable)",
            "type": "object"
        })
        params.append = {
            "key": "amptable",
            "value": None,
            "help": "Visibility Amplitude Table (uvdata.VisTable)",
            "type": "object"
        }
        params.append = {
            "key": "bstable",
            "value": None,
            "help": "Bispectram Table (uvdata.BSTable)",
            "type": "object"
        }
        params.append = {
            "key": "catable",
            "value": None,
            "help": "Closure Amplitude Table (uvdata.CATable)",
            "type": "object"
        }

    def set_params(self, **argv):
        pass

    def print_params(self, **argv):
        pass

    def precompute(self):
        self.sanity_check()

        if do_listuv:
            pass

        if do_initreg:
            pass

    def sanity_check(self):
        pass

    def listuv(self):
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

class L1norm(object):
    '''
    flat/weighted smooothed L1 norm
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize l1")
        if prior is None:
            print("  use l1-norm")
            self.weight = 1./totalflux
        else:
            print("  use weighted l1-norm")
            self.weight = 1./(eps+smabs(prior))
            normfactor = self.weight*smabs(prior)
            normfactor = normfactor.sum()
            self.weight /= normfactor

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


class KLEnt(object):
    '''
    Smooothed KL entropy
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize Gull Skilling Entropy")
        if prior is None:
            print("  use flat prior")
            pass
        else:
            print("  use prior")
            pass

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


class GSEnt(object):
    '''
    Smooothed GS entropy
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize Gull Skilling Entropy")
        if prior is None:
            print("  use flat prior")
            pass
        else:
            print("  use prior")
            pass

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


class TV(object):
    '''
    KL entropy
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize Gull Skilling Entropy")
        if prior is None:
            print("  use flat prior")
            pass
        else:
            print("  use prior")
            pass

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


class TSV(object):
    '''
    Total Squared Variation
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize Gull Skilling Entropy")
        if prior is None:
            print("  use flat prior")
            pass
        else:
            print("  use prior")
            pass

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


class TotalFlux(object):
    '''
    Total Squared Variation
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize Gull Skilling Entropy")
        if prior is None:
            print("  use flat prior")
            pass
        else:
            print("  use prior")
            pass

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


class Centroid(object):
    '''
    Centroid Regularization
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize Gull Skilling Entropy")
        if prior is None:
            print("  use flat prior")
            pass
        else:
            print("  use prior")
            pass

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


class SecMoment(object):
    '''
    Innertial Regularization
    '''
    def __init__(self,totalflux=1.,prior=None,eps=1e-12):
        print("Initialize Gull Skilling Entropy")
        if prior is None:
            print("  use flat prior")
            pass
        else:
            print("  use prior")
            pass

    def reg(self,x):
        return self.weight * smabs(x).sum()

    def regmap(self,x):
        return self.weight * smabs(x)

    def grad(self,x):
        return self.weight * dsmabs(x)


def smabs(x, eps=1e-12):
    '''
    Smoothed absolute operator |x|~sqrt(x**2 + eps)
    '''
    return sqrt(x**2+eps)


def dsmabs(x, eps=1e-12):
    '''
    derivertive of smooth absolute operator |x|~sqrt(x**2 + eps)
    '''
    return x/(x**2+eps)
