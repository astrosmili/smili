#!/usr/bin/env python
# -*- coding: utf-8 -*-




'''
This module describes data formats and related functions of uv data tables
'''
__author__ = "Smili Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import numpy as np

def get_uvlist(fcvtable=None, amptable=None, bstable=None, catable=None):
    '''
    '''
    if ((fcvtable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    def create_arrays(u,v):
        sign = np.asarray([1 for i in range(u.size)])
        idx = np.where(v<0)
        u[idx] *= -1
        v[idx] *= -1
        sign[idx] *= -1
        del idx
        return u+1j*v, sign

    # Stack uv coordinates
    uvstack = None
    sgstack = None
    if fcvtable is not None:
        u = fcvtable.u.values.copy()
        v = fcvtable.v.values.copy()
        uvstack, sgstack = create_arrays(u,v)
        Nfcv = len(u)
        del u,v
    else:
        Nfcv = 0

    if amptable is not None:
        u = amptable.u.values.copy()
        v = amptable.v.values.copy()
        uv, sg = create_arrays(u,v)
        Namp = len(u)
        if uvstack is None:
            uvstack = uv.copy()
            sgstack = sg.copy()
        else:
            uvstack = np.concatenate((uvstack, uv))
            sgstack = np.concatenate((sgstack, sg))
        del u,v,uv,sg
    else:
        Namp = 0

    if bstable is not None:
        u1 = bstable.u12.values.copy()
        v1 = bstable.v12.values.copy()
        u2 = bstable.u23.values.copy()
        v2 = bstable.v23.values.copy()
        u3 = bstable.u31.values.copy()
        v3 = bstable.v31.values.copy()
        uv1, sg1 = create_arrays(u1,v1)
        uv2, sg2 = create_arrays(u2,v2)
        uv3, sg3 = create_arrays(u3,v3)
        Ncp = len(u1)
        if uvstack is None:
            uvstack = np.concatenate((uv1, uv2, uv3))
            sgstack = np.concatenate((sg1, sg2, sg3))
        else:
            uvstack = np.concatenate((uvstack, uv1, uv2, uv3))
            sgstack = np.concatenate((sgstack, sg1, sg2, sg3))
        del u1,u2,u3,v1,v2,v3,uv1,uv2,uv3,sg1,sg2,sg3
    else:
        Ncp = 0

    if catable is not None:
        u1 = catable.u1.values.copy()
        v1 = catable.v1.values.copy()
        u2 = catable.u2.values.copy()
        v2 = catable.v2.values.copy()
        u3 = catable.u3.values.copy()
        v3 = catable.v3.values.copy()
        u4 = catable.u4.values.copy()
        v4 = catable.v4.values.copy()
        uv1, sg1 = create_arrays(u1,v1)
        uv2, sg2 = create_arrays(u2,v2)
        uv3, sg3 = create_arrays(u3,v3)
        uv4, sg4 = create_arrays(u4,v4)
        Nca = len(u1)
        if uvstack is None:
            uvstack = np.concatenate((uv1, uv2, uv3, uv4))
            sgstack = np.concatenate((sg1, sg2, sg3, sg4))
        else:
            uvstack = np.concatenate((uvstack, uv1, uv2, uv3, uv4))
            sgstack = np.concatenate((sgstack, sg1, sg2, sg3, sg4))
        del u1,u2,u3,u4,v1,v2,v3,v4,uv1,uv2,uv3,uv4,sg1,sg2,sg3,sg4
    else:
        Nca = 0

    # make non-redundant u,v lists and index arrays for uv coordinates.
    Nstack = Nfcv + Namp + 3 * Ncp + 4 * Nca

    outputs = np.unique(uvstack, return_index=True, return_inverse=True, return_counts=True)
    u = np.real(outputs[0])
    v = np.imag(outputs[0])

    uvidx = (outputs[2]+1) * sgstack

    # distribute index information into each data
    if fcvtable is None:
        uvidxfcv = np.zeros(1, dtype=np.int32)
    else:
        uvidxfcv = uvidx[0:Nfcv]

    if amptable is None:
        uvidxamp = np.zeros(1, dtype=np.int32)
    else:
        uvidxamp = uvidx[Nfcv:Nfcv + Namp]

    if bstable is None:
        uvidxcp = np.zeros([3, 1], dtype=np.int32, order="F")
    else:
        uvidxcp = uvidx[Nfcv + Namp:Nfcv + Namp + 3 *
                        Ncp].reshape([Ncp, 3], order="F").transpose()

    if catable is None:
        uvidxca = np.zeros([4, 1], dtype=np.int32, order="F")
    else:
        uvidxca = uvidx[Nfcv + Namp + 3 * Ncp:Nfcv + Namp + 3 *
                        Ncp + 4 * Nca].reshape([Nca, 4], order="F").transpose()
    return (u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca)


def get_uvlist_old(fcvtable=None, amptable=None, bstable=None, catable=None, thres=1e-10):
    '''
    '''
    if ((fcvtable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Stack uv coordinates
    ustack = None
    vstack = None
    if fcvtable is not None:
        ustack = np.array(fcvtable["u"], dtype=np.float64)
        vstack = np.array(fcvtable["v"], dtype=np.float64)
        Nfcv = len(ustack)
    else:
        Nfcv = 0

    if amptable is not None:
        utmp = np.array(amptable["u"], dtype=np.float64)
        vtmp = np.array(amptable["v"], dtype=np.float64)
        Namp = len(utmp)
        if ustack is None:
            ustack = utmp
            vstack = vtmp
        else:
            ustack = np.concatenate((ustack, utmp))
            vstack = np.concatenate((vstack, vtmp))
    else:
        Namp = 0

    if bstable is not None:
        utmp1 = np.array(bstable["u12"], dtype=np.float64)
        vtmp1 = np.array(bstable["v12"], dtype=np.float64)
        utmp2 = np.array(bstable["u23"], dtype=np.float64)
        vtmp2 = np.array(bstable["v23"], dtype=np.float64)
        utmp3 = np.array(bstable["u31"], dtype=np.float64)
        vtmp3 = np.array(bstable["v31"], dtype=np.float64)
        Ncp = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3))
    else:
        Ncp = 0

    if catable is not None:
        utmp1 = np.array(catable["u1"], dtype=np.float64)
        vtmp1 = np.array(catable["v1"], dtype=np.float64)
        utmp2 = np.array(catable["u2"], dtype=np.float64)
        vtmp2 = np.array(catable["v2"], dtype=np.float64)
        utmp3 = np.array(catable["u3"], dtype=np.float64)
        vtmp3 = np.array(catable["v3"], dtype=np.float64)
        utmp4 = np.array(catable["u4"], dtype=np.float64)
        vtmp4 = np.array(catable["v4"], dtype=np.float64)
        Nca = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3, vtmp4))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3, vtmp4))
    else:
        Nca = 0

    # make non-redundant u,v lists and index arrays for uv coordinates.
    Nstack = Nfcv + Namp + 3 * Ncp + 4 * Nca
    uvidx = np.zeros(Nstack, dtype=np.int32)
    maxidx = 1
    u = []
    v = []
    uvstack = np.sqrt(np.square(ustack) + np.square(vstack))
    #uvthres = np.max(uvstack) * thres
    for i in range(Nstack):
        if uvidx[i] == 0:
            dist1 = np.sqrt(
                np.square(ustack - ustack[i]) + np.square(vstack - vstack[i]))
            dist2 = np.sqrt(
                np.square(ustack + ustack[i]) + np.square(vstack + vstack[i]))
            #uvdist = np.sqrt(np.square(ustack[i])+np.square(vstack[i]))

            #t = np.where(dist1<uvthres)
            t = np.where(dist1 < thres * (uvstack[i] + 1))
            uvidx[t] = maxidx
            #t = np.where(dist2<uvthres)
            t = np.where(dist2 < thres * (uvstack[i] + 1))
            uvidx[t] = -maxidx
            u.append(ustack[i])
            v.append(vstack[i])
            maxidx += 1
    u = np.asarray(u)  # Non redundant u coordinates
    v = np.asarray(v)  # Non redundant v coordinates

    # distribute index information into each data
    if fcvtable is None:
        uvidxfcv = np.zeros(1, dtype=np.int32)
    else:
        uvidxfcv = uvidx[0:Nfcv]

    if amptable is None:
        uvidxamp = np.zeros(1, dtype=np.int32)
    else:
        uvidxamp = uvidx[Nfcv:Nfcv + Namp]

    if bstable is None:
        uvidxcp = np.zeros([3, 1], dtype=np.int32, order="F")
    else:
        uvidxcp = uvidx[Nfcv + Namp:Nfcv + Namp + 3 *
                        Ncp].reshape([Ncp, 3], order="F").transpose()

    if catable is None:
        uvidxca = np.zeros([4, 1], dtype=np.int32, order="F")
    else:
        uvidxca = uvidx[Nfcv + Namp + 3 * Ncp:Nfcv + Namp + 3 *
                        Ncp + 4 * Nca].reshape([Nca, 4], order="F").transpose()
    return (u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca)


def get_uvlist_loop(Nt, fcvconcat=None, ampconcat=None, bsconcat=None, caconcat=None):
    '''
    '''
    if ((fcvconcat is None) and (ampconcat is None) and
            (bsconcat is None) and (caconcat is None)):
        print("Error: No data are input.")
        return -1

    u, v = [], []
    uvidxfcv, uvidxamp, uvidxcp, uvidxca = [], [], [], []
    Nuvs = []

    idxcon = 0
    for i in range(Nt):
        fcvsingle, ampsingle, bssingle, casingle = None, None, None, None
        if fcvconcat is not None:
            frmid = fcvconcat["frmidx"] == i
            idx = np.where(frmid == True)
            if idx[0] != []:
                fcvsingle = fcvconcat.loc[idx[0], :]

        if ampconcat is not None:
            frmid = ampconcat["frmidx"] == i
            idx = np.where(frmid == True)
            if idx[0] != []:
                ampsingle = ampconcat.loc[idx[0], :]

        if bsconcat is not None:
            frmid = bsconcat["frmidx"] == i
            idx = np.where(frmid == True)
            if idx[0] != []:
                bssingle = bsconcat.loc[idx[0], :]

        if caconcat is not None:
            frmid = caconcat["frmidx"] == i
            idx = np.where(frmid == True)
            if idx[0] != []:
                casingle = caconcat.loc[idx[0], :]

        if ((fcvsingle is None) and (ampsingle is None) and
            (bssingle is None)  and (casingle is None)):
            Nuvs.append(0)
        else:
            u0, v0, uvidxfcv0, uvidxamp0, uvidxcp0, uvidxca0 = get_uvlist(
                fcvtable=fcvsingle, amptable=ampsingle, bstable=bssingle, catable=casingle)
            u.append(u0)
            v.append(v0)
            Nuvs.append(len(u0))
            if fcvsingle is not None:
                uvidxfcv0 = np.sign(uvidxfcv0) * (np.abs(uvidxfcv0)+idxcon)
                uvidxfcv.append(uvidxfcv0)
            if ampsingle is not None:
                uvidxamp0 = np.sign(uvidxamp0) * (np.abs(uvidxamp0)+idxcon)
                uvidxamp.append(uvidxamp0)
            if bssingle is not None:
                uvidxcp0 = np.sign(uvidxcp0) * (np.abs(uvidxcp0)+idxcon)
                uvidxcp.append(uvidxcp0)
            if casingle is not None:
                uvidxca0 = np.sign(uvidxca0) * (np.abs(uvidxca0)+idxcon)
                uvidxca.append(uvidxca0)
            idxcon += len(u0)

    u = np.concatenate(u)
    v = np.concatenate(v)

    if fcvconcat is not None:
        uvidxfcv = np.concatenate(uvidxfcv)
    else:
        uvidxfcv = np.zeros(1, dtype=np.int32)

    if ampconcat is not None:
        uvidxamp = np.concatenate(uvidxamp)
    else:
        uvidxamp = np.zeros(1, dtype=np.int32)

    if bsconcat is not None:
        uvidxcp = np.hstack(uvidxcp)
    else:
        uvidxcp = np.zeros([3, 1], dtype=np.int32, order="F")

    if caconcat is not None:
        uvidxca = np.hstack(uvidxca)
    else:
        uvidxca = np.zeros([4, 1], dtype=np.int32, order="F")

    return (u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs)
