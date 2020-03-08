#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
from collections import OrderedDict

# numpy
from numpy import sqrt, where, asarray, ones, zeros, float64, complex128, dtype
from numpy import nan

# astropy
from astropy.coordinates import EarthLocation
from astropy.table import QTable

# internal
from ..util.units import DIMLESS, M, JY, DEG

# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class Array(object):
    '''A class to handle an interferometric array

    Attributes: 
        name (str): the array name
        columns (list): the list of the table columns
        table (astropy.table.QTable): the array parameter
    
    Methods:
        load_ehtim_array: loading the array object or file of the eht-imaging library
    '''
    # name
    name = "MyArray"

    # ground table
    columns = "name,location,sefd1,sefd2,tau1,tau2,elmin,elmax,fr_pa_coeff,fr_el_coeff,fr_offset,d1,d2,type".split(",")
    table = None

    def __repr__(self):
        output = ""
        if self.table is not None:
            output += self.table.__repr__()
        return output
    
    def _repr_html_(self):
        output = ""
        if self.table is not None:
            output += self.table._repr_html_()
        return output
    
    def __len__(self):
        return len(self.table)

    @classmethod
    def load_ehtim_array(cls, arrayobj, name="myarray", **args_load_txt):
        import ehtim
        from copy import deepcopy
        
        # Units

        if type(arrayobj) == type(""):
            array = ehtim.array.load_txt(arrayobj, **args_load_txt)
        elif type(arrayobj) == type(ehtim.array.Array):
            array = deepcopy(arrayobj)
        else:
            raise ValueError("Invalid data type of arrayobj: %s"%(type(arrayobj)))
        
        tarr = array.tarr.copy()
        xyz_dist = sqrt(tarr["x"]**2+tarr["y"]**2+tarr["z"]**2)
        idx_tle = where(xyz_dist<1)
        tarr["x"][idx_tle] = nan
        tarr["y"][idx_tle] = nan
        tarr["z"][idx_tle] = nan

        Nant = tarr.size
        data = dict(
            name = asarray(tarr["site"], dtype="U8"),
            location = EarthLocation.from_geocentric(
                x = tarr["x"],
                y = tarr["y"],
                z = tarr["z"],
                unit = M
            ),
            sefd1 = tarr["sefdr"] * JY,
            sefd2 = tarr["sefdl"] * JY,
            tau1 = zeros(Nant)*DIMLESS,
            tau2 = zeros(Nant)*DIMLESS,
            elmin = ones(Nant)*0.*DEG,
            elmax = ones(Nant)*90.*DEG,
            fr_pa_coeff = tarr["fr_par"]*DIMLESS,
            fr_el_coeff = tarr["fr_elev"]*DIMLESS,
            fr_offset = tarr["fr_off"]*DEG,
            d1 = tarr["dr"]*DIMLESS,
            d2 = tarr["dl"]*DIMLESS,
            type = asarray(["ground" for i in range(Nant)], dtype="U8"),
        )

        data["elmin"][idx_tle] = -90.*DEG
        data["elmax"][idx_tle] =  90.*DEG
        data["type"][idx_tle] = "tle"
        
        outtable = cls()
        outtable.name = name
        outtable.table = QTable(
            data = data,
            names = cls.columns
        )
        return outtable