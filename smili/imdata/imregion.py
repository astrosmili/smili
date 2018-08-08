#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a sub-module of smili handling image fits data.
'''
__author__ = "Smili Developer Team"
#-------------------------------------------------------------------------
# Modules
#-------------------------------------------------------------------------
# standard modules
import copy
import os
from collections import Counter

# numerical packages
import numpy as np
import pandas as pd

# matplotlib
import matplotlib.pyplot as plt

# ds9
import pyds9

# internal
from smili import util

#-------------------------------------------------------------------------
# IMAGEFITS (Manupulating FITS FILES)
#-------------------------------------------------------------------------

class IMRegion(pd.DataFrame):
    '''
    This class is for handling two dimentional tables of region. The class
    inherits pandas.DataFrame class, so you can use this class like
    pandas.DataFrame. The class also has additional methods to edit,
    visualize, and handle ds9.
    '''

    imreg_columns = ["shape", "xc", "yc", "width", "height",
                     "radius", "maja", "mina", "angle", "angunit"]
    imreg_types = [None, np.float64, np.float64, np.float64, np.float64,
                   np.float64, np.float64, np.float64, np.float64, None]

    @property
    def _constructor(self):
        return IMRegion

    @property
    def _constructor_sliced(self):
        return ImRegSeries

#    def __init__(self):
#        self.initialize()

    def initialize(self):
        '''
        Initialize region table (empty table with column names).
        '''

        if len(self.keys())==0:
            for i in xrange(len(IMRegion.imreg_columns)):
                column = IMRegion.imreg_columns[i]
                self[column] = []
        else:
            self.drop(np.arange(len(self)),inplace=True)

    # region1 + region2
    def __add__(self,reg2):
        reg3 = self.append(reg2,ignore_index=True)
        return reg3

    ## Plot region
    def plot(self,angunit=None,**pltargs):
        '''
        Plot region.
        This method uses matplotlib.pyplot.plot().

        Args:
            angunit (str, default = None):
                The angular unit of plot. If none, it will use the most frequent
                angular unit in the region list
            **plotargs:
                You can set parameters of matplotlib.pyplot.plot.
        '''
        if angunit is None:
            raise ValueError("No input angunit")
            #cnt = Counter(self["angunit"])
            #angunit = cnt.most_common()[0][0]

        for index, row in self.iterrows():
            if row["shape"] == "box":
                plot_box(row,angunit=angunit,**pltargs)
            if row["shape"] == "circle":
                plot_circle(row,angunit=angunit,**pltargs)
            if row["shape"] == "ellipse":
                plot_ellipse(row,angunit=angunit,**pltargs)

    ## csv
    # save to csv file
    def to_csv(self, filename, index=False, index_label=False, **args):
        '''
        Output table into csv files using pd.DataFrame.to_csv().
        Default parameter will be
          index=False
          index_label=False.
        see DocStrings for pd.DataFrame.to_csv().

        Args:
            filename (string or filehandle): output filename
            **args: other arguments of pd.DataFrame.to_csv()
        '''
        super(IMRegion, self).to_csv(filename, index=False, index_label=False, **args)

    # read csv file as IMRegion
    def read_region(self, filename):
        '''
        Read a csv file as IMRegion using pd.DataFrame.read_csv().

        Args:
            filename (string or filehandle): output filename
        '''
        # empty region table
        table = IMRegion()
        table.initialize()

        # read csv file
        region = pd.read_csv(filename)

        # append pd.DataFrame table to empty region table
        table = table.append(region)

        return table

    ## DS9
    # Start DS9
    def open_pyds9(self,image,wait=10):
        '''
        Open pyds9 and plot the region.
        This method uses pyds9.DS9().

        Args:
            image (IMFITS)
            wait (float, default = 10):
                seconds to wait for ds9 to start.
        '''
        try:
            d = pyds9.DS9(wait=wait)
        except ValueError:
            print("ValueError: try longer 'wait' time or installation of XPA.")
        except:
            print("Unexpected Error!")
            raise
        else:
            d.set_pyfits(image.hdulist)
            d.set('zoom to fit')
            d.set('cmap heat')
            for index, row in self.iterrows():
                ds9reg = reg_to_ds9reg(row,image)
                d.set("region","image; %s" % ds9reg)

    # Load DS9 region
    def load_pyds9(self,image,angunit=None,wait=10,overwrite=True):
        '''
        Load DS9 region to IMRegion.
        This method uses pyds9.DS9().

        Args:
            image (IMFITS)
            angunit (str, default = None):
                The angular unit of region. If None, it will take from
                the default angunit of the input image
            wait (float, default = 10):
                seconds to wait for ds9 to start.
            overwrite (boolean, default = True):
                If overwrite=True, previous region is removed.
        Returns:
            IMRegion.
        '''
        if angunit is None:
            angunit = image.angunit

        try:
            d = pyds9.DS9(wait=wait)
        except ValueError:
            print("ValueError: try longer 'wait' time or installation of XPA.")
        except:
            print("Unexpected Error!")
            raise
        else:
            ds9reg = d.get("regions -system image")
            region = ds9reg_to_reg(ds9reg=ds9reg,image=image,angunit=angunit)
            if overwrite:
                return region
            else:
                return self + region

    # Read DS9 region file
    def load_ds9reg(self,regfile,image,angunit=None):
        '''
        Load DS9 region file to IMRegion.

        Args:
            regfile (str):
                Region file name to read.
            image (IMFITS)
            angunit (str, default = None):
                The angular unit of region. If None, it will take from
                the default angunit of the input image
        Returns:
            IMRegion.
        '''
        if angunit is None:
            angunit = image.angunit

        f = open(regfile,'r')
        ds9reg = f.read()
        f.close()
        region = ds9reg_to_reg(ds9reg=ds9reg,image=image,angunit=angunit)
        return region

    # Write DS9 region file
    def to_ds9reg(self,regfile,image,overwrite=True):
        '''
        Write DS9 region file from IMRegion.

        Args:
            regfile (str):
                Region file name to write.
            image (IMFITS)
            overwrite (boolean, default = True):
                if overwrite=True and the region file already exists,
                the file will be replaced.
        '''
        if overwrite:
            if os.path.isfile(regfile):
                os.remove(regfile)
                print("Old region file is removed.\n")
            else:
                pass
            f = open(regfile, "a+")
            f.write("image\n")
            for index, row in self.iterrows():
                ds9reg = reg_to_ds9reg(row,image)
                ds9reg = ds9reg.split(" ")[1]
                f.write(ds9reg+"\n")
            f.close()
        else:
            if os.path.isfile(regfile):
                print("Error: %s already exists. New region file is not produced.\n" % regfile)
                pass
            else:
                f = open(regfile, "a+")
                f.write("image\n")
                for index, row in self.iterrows():
                    ds9reg = reg_to_ds9reg(row,image)
                    ds9reg = ds9reg.split(" ")[1]
                    f.write(ds9reg+"\n")
                f.close()


    ## Difmap
    # Load and save difmap window
    #def load_difmapwin(self,winname,overwrite=True):
    #    pass
    #
    #def to_difmapwin(self,winname):
    #    # もしも box (angle=0) 以外の shape があれば、エラーを返す。
    #    pass


    ## Edit region
    # Add region
    def add_box(self,xc=0,yc=0,width=0.05,height=None,angle=0,angunit="mas"):
        '''
        Add box window.

        Args:
            xc (float, default=0):
                Center position of window relative to the referrence position in x-axis.
            yc (float, default=0):
                Center position of window relative to referrence position in y-axis.
            width (float, default=0.05):
                Width of the box.
            height (float, default=None):
                Height of the box.
            angle (float, default=0):
                Position angle of the box (counter clockwise).
            angunit (str, default=mas):
                Anguler unit of xc, yc, width, height.
        Returns:
            IMRegion.
        '''
        if height is None:
            height = width
        else:
            pass

        s = ImRegSeries(['box',xc,yc,width,height,np.nan,np.nan,np.nan,angle,angunit],
                       index=['shape','xc','yc','width','height','radius',
                              'maja','mina','angle','angunit'])

        region = self.copy()
        if len(region.keys())==0:
            region.initialize()
        else:
            pass

        region = region.append(s,ignore_index=True)

        for i in xrange(len(IMRegion.imreg_columns)):
            column = IMRegion.imreg_columns[i]
            if IMRegion.imreg_types[i] is None:
                pass
            else:
                region[column] = IMRegion.imreg_types[i](region[column])

        return region

    def add_circle(self,xc=0,yc=0,radius=0.05,angunit="mas"):
        '''
        Add circle window.

        Args:
            xc (float, default=0):
                Center position of window relative to the referrence position in x-axis.
            yc (float, default=0):
                Center position of window relative to referrence position in y-axis.
            radius (float, default=0.05):
                radius of the circle.
            angunit (str, default=mas):
                Anguler unit of xc, yc, radius.
        Returns:
            IMRegion.
        '''
        s = ImRegSeries(['circle',xc,yc,np.nan,np.nan,radius,np.nan,np.nan,np.nan,angunit],
                       index=['shape','xc','yc','width','height','radius',
                              'maja','mina','angle','angunit'])

        region = self.copy()
        if len(region.keys())==0:
            region.initialize()
        else:
            pass

        region = region.append(s,ignore_index=True)

        for i in xrange(len(IMRegion.imreg_columns)):
            column = IMRegion.imreg_columns[i]
            if IMRegion.imreg_types[i] is None:
                pass
            else:
                region[column] = IMRegion.imreg_types[i](region[column])

        return region

    def add_ellipse(self,xc=0,yc=0,maja=0.05,mina=0.02,angle=0,angunit="mas"):
        '''
        Add ellipse window.

        Args:
            xc (float, default=0):
                Center position of window relative to the referrence position in x-axis.
            yc (float, default=0):
                Center position of window relative to referrence position in y-axis.
            maja (float, default=0.05):
                Length of major axis.
            mina (float, default=0.02):
                Length of minor axis.
            angle (float, default=0):
                Position angle of the ellipse (counter clockwise).
            angunit (str, default=mas):
                Anguler unit of xc, yc, maja, mina.
        Returns:
            IMRegion.
        '''
        s = ImRegSeries(['ellipse',xc,yc,np.nan,np.nan,np.nan,maja,mina,angle,angunit],
                       index=['shape','xc','yc','width','height','radius',
                              'maja','mina','angle','angunit'])

        region = self.copy()
        if len(region.keys())==0:
            region.initialize()
        else:
            pass

        region = region.append(s,ignore_index=True)

        for i in xrange(len(IMRegion.imreg_columns)):
            column = IMRegion.imreg_columns[i]
            if IMRegion.imreg_types[i] is None:
                pass
            else:
                region[column] = IMRegion.imreg_types[i](region[column])

        return region

    # Shift region
    def shift(self,dx=0,dy=0,angunit="mas"):
        '''
        Shift region.

        Args:
            dx (float, default=0):
                Shift in x-axis.
            dy (float, default=0):
                Shift in y-axis.
            angunit (str, default=mas):
                Anguler unit of dx and dy.
        '''
        for index, row in self.iterrows():
            angconv = util.angconv(angunit,row["angunit"])
            self.loc[index,"xc"] += dx * angconv
            self.loc[index,"yc"] += dy * angconv

    # Zoom region
    def zoom(self,fx=1.0,fy=None):
        '''
        Zoom region.

        Args:
            fx (float, default=1.0):
                Zoom fraction of width, radius, and minor axis for box,
                circle, and ellipse windows, respectively.
            fy (float, default=None):
                Zoom fraction of width and major axis for box and ellipse
                windows, respectively.
            angunit (str, default=mas):
                Anguler unit of dx and dy.
        '''
        if fy is None:
            fy = fx
        else:
            pass

        for index, row in self.iterrows():
            if row["shape"] == "box":
                self.loc[index,"width"] *= fx
                self.loc[index,"height"] *= fy
            if row["shape"] == "circle":
                self.loc[index,"radius"] *= fx
            if row["shape"] == "ellipse":
                self.loc[index,"mina"] *= fx
                self.loc[index,"maja"] *= fy

    # Delete duplicated region
    def remove_duplicates(self):
        '''
        Remove duplicated regions.

        Returns:
            IMRegion.
        '''
        region = self.copy()
        outregion = self.copy()

        for index, row in region.iterrows():
            # Align angular unit
            angconv = util.angconv(row.angunit, "mas")
            region.loc[index,"xc"] *= angconv
            region.loc[index,"yc"] *= angconv
            region.loc[index,"width"] *= angconv
            region.loc[index,"height"] *= angconv
            region.loc[index,"radius"] *= angconv
            region.loc[index,"maja"] *= angconv
            region.loc[index,"mina"] *= angconv
            region.loc[index,"angunit"] = "mas"

            # Make "length" column for characteristic length
            if row["shape"] == "box":
                width = region.loc[index,"width"]
                height = region.loc[index,"height"]
                region.loc[index,"length"] = np.sqrt(width*width + height*height)
            elif row["shape"] == "circle":
                radius = region.loc[index,"radius"]
                region.loc[index,"length"] = radius*2.0
            elif row["shape"] == "ellipse":
                maja = region.loc[index,"maja"]
                mina = region.loc[index,"mina"]
                region.loc[index,"length"] = max(maja,mina)

        mesh = np.nanmin([region["width"].min(),region["height"].min(),
                       region["radius"].min()*2.0,region["maja"].min(),
                       region["mina"].min()])
        mesh /= 10.0
        length = region["length"].max()
        xmin = region["xc"].min() - length
        xmax = region["xc"].max() + length
        ymin = region["yc"].min() - length
        ymax = region["yc"].max() + length
        nx = int(round((xmax - xmin) / mesh)) + 1
        ny = int(round((ymax - ymin) / mesh)) + 1
        xgrid = np.linspace(xmax,xmin,nx)
        ygrid = np.linspace(ymin,ymax,ny)
        X, Y = np.meshgrid(xgrid, ygrid)

        # Make area panel
        items = len(region)
        area = np.zeros((items,ny,nx))
        for index, row in region.iterrows():
            if row["shape"] == "box":
                area[index] = region_box(X,Y,row.xc,row.yc,row.width,row.height,row.angle)
            elif row["shape"] == "circle":
                area[index] = region_circle(X,Y,row.xc,row.yc,row.radius)
            elif row["shape"] == "ellipse":
                area[index] = region_ellipse(X,Y,row.xc,row.yc,row.mina/2.0,row.maja/2.0,row.angle)

        # Search duplicated area
        droplist = []
        for index, row in region.iterrows():
            area1 = area[index]
            area1 = area1 > 0
            area2 = area[0:index].sum(axis=0) + area[index+1:].sum(axis=0)
            area2 = area2 > 0
            merge = area1 + area2
            merge = merge > 0
            if np.allclose(merge,area2):
                # area_index is duplicated.
                area[index] = np.zeros((ny,nx))
                droplist.append(index)
            else:
                pass

        print("Duplicated region is:")
        print(droplist)
        # Remove duplicated area
        outregion = outregion.drop(droplist)
        outregion = outregion.reset_index(drop=True)

        # return region
        return outregion


    ## Edit image
    def winmod(self,image,save_totalflux=False):
        '''
        Trim the image with image regions.

        Args:
            image (IMFITS)
            save_totalflux (boolean, default = False):
                If save_totalflux=True, the trimmed have the same total flux
                as before editting image.
        Returns:
            IMFITS object.
        '''
        xgrid = np.arange(image.header["nx"])
        ygrid = np.arange(image.header["ny"])
        X, Y = np.meshgrid(xgrid, ygrid)

        area = np.zeros(X.shape, dtype="Bool")
        for index, row in self.iterrows():
            angconv = util.angconv(row["angunit"],"deg")
            x0 = row["xc"]*angconv/image.header["dx"]+image.header["nxref"]-1
            y0 = row["yc"]*angconv/image.header["dy"]+image.header["nyref"]-1
            if row["shape"] == "box":
                width = row["width"]*angconv/image.header["dx"]
                height = row["height"]*angconv/image.header["dy"]
                angle = row["angle"]
                tmparea = region_box(X,Y,x0,y0,width,height,angle)
            elif row["shape"] == "circle":
                radius = row["radius"]*angconv/image.header["dx"]
                tmparea = region_circle(X,Y,x0,y0,radius)
            elif row["shape"] == "ellipse":
                radius1 = row["mina"]/2.0*angconv/image.header["dx"]
                radius2 = row["maja"]/2.0*angconv/image.header["dy"]
                angle = row["angle"]
                tmparea = region_ellipse(X,Y,x0,y0,radius1,radius2,angle)
            else:
                print("[WARNING] The shape %s is not available." % (row["shape"]))
            area += tmparea

        editimage = copy.deepcopy(image)
        for idxs in np.arange(image.header["ns"]):
            for idxf in np.arange(image.header["nf"]):
                matrix = editimage.data[idxs, idxf]
                editimage.data[idxs, idxf][~area] = 0.0
                if save_totalflux:
                    totalflux = editimage.totalflux(istokes=idxs, ifreq=idxf)
                    editimage.data[idxs, idxf] *= totalflux / matrix.sum()
        # Update and Return
        editimage.update_fits()
        return editimage

    ## Mask Image
    # 1.0 or 0.0
    def maskimage(self, image):
        '''
        Make a mask image. Each pixel of the output image is stored as a
        single bit—i.e., a 0 or 1.

        Args:
            image (IMFITS)
        Returns:
            IMFITS object.
        '''
        xgrid = np.arange(image.header["nx"])
        ygrid = np.arange(image.header["ny"])
        X, Y = np.meshgrid(xgrid, ygrid)

        area = np.zeros(X.shape, dtype="Bool")
        for index, row in self.iterrows():
            angconv = util.angconv(row["angunit"],"deg")
            x0 = row["xc"]*angconv/image.header["dx"]+image.header["nxref"]-1
            y0 = row["yc"]*angconv/image.header["dy"]+image.header["nyref"]-1
            if row["shape"] == "box":
                width = row["width"]*angconv/image.header["dx"]
                height = row["height"]*angconv/image.header["dy"]
                angle = row["angle"]
                tmparea = region_box(X,Y,x0,y0,width,height,angle)
            elif row["shape"] == "circle":
                radius = row["radius"]*angconv/image.header["dx"]
                tmparea = region_circle(X,Y,x0,y0,radius)
            elif row["shape"] == "ellipse":
                radius1 = row["mina"]/2.0*angconv/image.header["dx"]
                radius2 = row["maja"]/2.0*angconv/image.header["dy"]
                angle = row["angle"]
                tmparea = region_ellipse(X,Y,x0,y0,radius1,radius2,angle)
            else:
                print("[WARNING] The shape %s is not available." % (row["shape"]))
            area += tmparea

        maskimage = copy.deepcopy(image)
        for idxs in np.arange(image.header["ns"]):
            for idxf in np.arange(image.header["nf"]):
                maskimage.data[idxs, idxf][area] = 1.0
                maskimage.data[idxs, idxf][~area] = 0.0

        # Update and Return
        maskimage.update_fits()
        return maskimage

    # True or False
    def imagewin(self,image,istokes=0, ifreq=0):
        '''
        Make a mask image. Each pixel of the output image is stored as a
        True/False.

        Args:
            image (IMFITS)
        Returns:
            numpy.array.
        '''
        maskimage = self.maskimage(image)
        imagewin = maskimage.data[istokes,ifreq] == 1.0
        return imagewin

class ImRegSeries(pd.Series):

    @property
    def _constructor(self):
        return ImRegSeries

    @property
    def _constructor_expanddim(self):
        return IMRegion

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def read_imregion(filename, **args):
    '''
    This fuction loads imdata.IMRegion from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)

    Returns:
      imdata.IMRegion object
    '''
    region = IMRegion(pd.read_csv(filename, **args))

    return region


#-------------------------------------------------------------------------
# Fllowings are subfunctions for plot region.
#-------------------------------------------------------------------------

def plot_box(row,angunit,**pltargs):
    angconv = util.angconv(row.angunit, angunit)
    xc = row.xc*angconv
    yc = row.yc*angconv

    width = row.width*angconv
    height = row.height*angconv

    x0 = xc + width/2.0
    x1 = xc - width/2.0
    y0 = yc - height/2.0
    y1 = yc + height/2.0

    X0 = x0
    X1 = x1
    X2 = x1
    X3 = x0
    Y0 = y0
    Y1 = y0
    Y2 = y1
    Y3 = y1

    cosa = np.cos(np.deg2rad(row.angle))
    sina = np.sin(np.deg2rad(row.angle))
    X0 = (x0-xc)*cosa + (y0-yc)*sina + xc
    Y0 = - (x0-xc)*sina + (y0-yc)*cosa + yc
    X1 = (x1-xc)*cosa + (y0-yc)*sina + xc
    Y1 = - (x1-xc)*sina + (y0-yc)*cosa + yc
    X2 = (x1-xc)*cosa + (y1-yc)*sina + xc
    Y2 = - (x1-xc)*sina + (y1-yc)*cosa + yc
    X3 = (x0-xc)*cosa + (y1-yc)*sina + xc
    Y3 = - (x0-xc)*sina + (y1-yc)*cosa + yc

    plt.plot([X0,X1],[Y0,Y1],**pltargs)
    plt.plot([X1,X2],[Y1,Y2],**pltargs)
    plt.plot([X2,X3],[Y2,Y3],**pltargs)
    plt.plot([X3,X0],[Y3,Y0],**pltargs)


def plot_circle(row,angunit="mas",**pltargs):
    angconv = util.angconv(row.angunit, angunit)
    xc = row.xc*angconv
    yc = row.yc*angconv

    radius = row.radius*angconv
    theta = np.linspace(0.,360.,200)
    cost = np.cos(np.deg2rad(theta))
    sint = np.sin(np.deg2rad(theta))
    X = radius*cost + xc
    Y = radius*sint + yc

    plt.plot(X,Y,**pltargs)


def plot_ellipse(row,angunit="mas",**pltargs):
    angconv = util.angconv(row.angunit, angunit)
    xc = row.xc*angconv
    yc = row.yc*angconv

    maja = row.maja*angconv
    mina = row.mina*angconv
    a = mina/2.
    b = maja/2.
    theta = np.linspace(0.,360.,200)
    cost = np.cos(np.deg2rad(theta))
    sint = np.sin(np.deg2rad(theta))
    X = a*cost + xc
    Y = b*sint + yc

    cosa = np.cos(np.deg2rad(row.angle))
    sina = np.sin(np.deg2rad(row.angle))
    XX = (X-xc)*cosa - (Y-yc)*sina + xc
    YY = (X-xc)*sina + (Y-yc)*cosa + yc

    plt.plot(XX,YY,**pltargs)


#-------------------------------------------------------------------------
# Fllowings are subfunctions for make_imagewin and winmod.
#-------------------------------------------------------------------------

def region_box(X, Y, x0, y0, width, height, angle):
    cosa = np.cos(np.deg2rad(angle))
    sina = np.sin(np.deg2rad(angle))
    dX = X - x0
    dY = Y - y0
    X1 = dX * cosa + dY * sina
    Y1 = -dX * sina + dY * cosa
    region = (Y1 >= -np.abs(height) / 2.)
    region *= (Y1 <= np.abs(height) / 2.)
    region *= (X1 >= -np.abs(width) / 2.)
    region *= (X1 <= np.abs(width) / 2.)
    return region


def region_circle(X, Y, x0, y0, radius):
    return (X - x0) * (X - x0) + (Y - y0) * (Y - y0) <= radius * radius


def region_ellipse(X, Y, x0, y0, radius1, radius2, angle):
    cosa = np.cos(np.deg2rad(angle))
    sina = np.sin(np.deg2rad(angle))
    dX = X - x0
    dY = Y - y0
    X1 = dX * cosa - dY * sina
    Y1 = dX * sina + dY * cosa
    return X1 * X1 / radius1 / radius1 + Y1 * Y1 / radius2 / radius2 <= 1


#-------------------------------------------------------------------------
# Fllowings are subfunctions for open_pyds9, load_pyds9, load_ds9reg, and to_ds9reg.
#-------------------------------------------------------------------------

# Conversion from region to DS9 region
def reg_to_ds9reg(row,image):
    angconv = util.angconv("deg",row["angunit"])
    nxref = image.header["nxref"]
    nyref = image.header["nyref"]
    dx = image.header["dx"]*angconv
    dy = image.header["dy"]*angconv
    if not round(-dx,9) == round(dy,9):
        print("Warning: image |dx| is not equal to |dy|.")
        print("|dx| = %f, |dy| = %f" % (round(-dx,9),round(dy,9)))
    else:
        pass
    x = row["xc"]/dx + nxref
    y = row["yc"]/dy + nyref

    if row["shape"] == "box":
        width = -row["width"]/dx
        height = row["height"]/dy
        angle = row["angle"]
        return "image; box(%f,%f,%f,%f,%f)" % (x,y,width,height,angle)

    if row["shape"] == "circle":
        r = -row["radius"]/dx
        return "image; circle(%f,%f,%f)" % (x,y,r)

    if row["shape"] == "ellipse":
        a = -row["mina"]/2.0/dx
        b = row["maja"]/2.0/dy
        angle = row["angle"]
        return "image; ellipse(%f,%f,%f,%f,%f)" % (x,y,a,b,angle)

# Conversion from DS9 region to region
def ds9reg_to_reg(ds9reg,image,angunit="mas"):
    angconv = util.angconv("deg",angunit)
    nxref = image.header["nxref"]
    nyref = image.header["nyref"]
    dx = image.header["dx"]*angconv
    dy = image.header["dy"]*angconv
    if not round(-dx,9) == round(dy,9):
        print("Warning: image |dx| is not equal to |dy|.")
        print("|dx| = %f, |dy| = %f" % (round(-dx,9),round(dy,9)))
    else:
        pass

    ds9reg = ds9reg.split("\n")
    region = IMRegion()
    region.initialize()
    for reg in ds9reg:
        if reg[0:3] == "box":
            list = map(np.float64,reg[4:-1].split(","))
            x = list[0]
            y = list[1]
            width = list[2]
            height = list[3]
            angle = list[4]
            width *= (-dx)
            height *= dy
            xc = (x - nxref) * dx
            yc = (y - nyref) * dy
            region = region.add_box(xc=xc,yc=yc,
                                 width=width,height=height,angle=angle,
                                 angunit=angunit)

        if reg[0:6] == "circle":
            list = map(np.float64,reg[7:-1].split(","))
            x = list[0]
            y = list[1]
            r = list[2]
            radius = r * (-dx)
            xc = (x - nxref) * dx
            yc = (y - nyref) * dy
            region = region.add_circle(xc=xc,yc=yc,
                                          radius=radius,angunit=angunit)

        if reg[0:7] == "ellipse":
            list = map(np.float64,reg[8:-1].split(","))
            x = list[0]
            y = list[1]
            a = list[2]
            b = list[3]
            angle = list[4]-90.0
            maja = a * 2.0 * dy
            mina = b * 2.0 * (-dx)
            xc = (x - nxref) * dx
            yc = (y - nyref) * dy
            region = region.add_ellipse(xc=xc,yc=yc,
                                 maja=maja,mina=mina,angle=angle,angunit=angunit)

        if reg[0:7] == "polygon":
            print ("Error: polygon cannot be used.")

    return region
