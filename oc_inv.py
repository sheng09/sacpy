#!/usr/bin/env python3

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import sacpy
import sys
np.set_printoptions(precision=23, suppress=True)
#### auxiliary functions
@jit(nopython=True, nogil=True)
def denser_xy(x, y, step):       # denser x, y so that abs(x_step) < abs(step)
    step = np.abs(step) if (x[1] > x[0]) else -np.abs(step)
    #####
    n = x.size
    # -------- First pass: count output length --------
    total_points = 1  # final point always added
    for i in range(n - 1):
        x0, x1 = x[i], x[i+1]
        if x1 != x0:
            seg_len = int(np.ceil((x1 - x0) / step))
            total_points += seg_len
        else:
            total_points += 1
    # -------- Allocate arrays --------
    u = np.empty(total_points, dtype=np.float64)
    v = np.empty(total_points, dtype=np.float64)
    idx = 0
    for i in range(n - 1):
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        if x1 != x0:
            slope = (y1 - y0) / (x1 - x0)
            # number of steps in this segment
            seg_len = int(np.ceil((x1 - x0) / step))
            for k in range(seg_len):
                xv = x0 + k * step
                u[idx] = xv
                v[idx] = y0 + slope * (xv - x0)
                idx += 1
        else:
            u[idx] = x0
            v[idx] = y0
            idx += 1
    # Add the final point
    u[idx] = x[-1]
    v[idx] = y[-1]
    return u, v
@jit(nopython=True, nogil=True)
def flatten(r, vr, R0):          # r,vr --> z,vz (increasing z is upward)
    z = R0*np.log(r/R0)
    vz= R0*vr/r
    return z, vz
@jit(nopython=True, nogil=True)
def unflatten(z, vz, R0):        # z,vz --> r,vr (increasing z is upward)
    r = R0*np.exp(z/R0)
    vr = vz*r/R0
    return r, vr

#### Given 1/p, compute two-way x and t (and gradients if needed)
@jit(nopython=True, nogil=True)
def p2xt(inv_p, z, v, dz, dv): # input inv_p the sametime for numerical issue
    if inv_p <= v[0]:
        return 0.0, 0.0
    #### get the layer where the ray turns
    ib = z.size-1 # in maximum case, the ray will penerate all layers and reach the z[z.size-1]
    for ilayer in range(z.size-1):
        if v[ilayer]< inv_p and inv_p <= v[ilayer+1]:
        #if inv_v[ilayer] > p and p >= inv_v[ilayer+1]:
            ib = ilayer
            break
    ####
        # The ray will penetrate layers with index: 0,1,2,...,ib-1. In other words,
        # the ray will cross the interfaces at: z[0],z[1],...,z[ib].
        #
        # So,
        # if 0<=ib< z.size-1, the ray will penetrate layers: 0,1,2,...,ib-1 and turn in ib. (0,1,2,...,ib-1 penerations + 1 turn)
        # if    ib==z.size-1, the ray will penetrate all layers: 0,1,2,...,ib-1(z.size-2).  (0,1,2,...,ib-1 penerations only)
    ####
    x = 0.0
    t = 0.0
    ####
    sin_theta_i0 = v[0]/inv_p
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    ln_v0 = np.log(v[0]/(1.0+cos_theta_i0) ) # ln( v0/(1+cos0) )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = v[i+1]/inv_p
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        ln_v1 = np.log(v[i+1]/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if np.abs(dv[i]) > 1e-10 and dz[i] < -1e-10:
            local_dzdv = dz[i]/dv[i]
            t += local_dzdv * (ln_v0 - ln_v1)
            x += local_dzdv * (inv_p *(cos_theta_i1 - cos_theta_i0))
        elif dz[i] < -1e-10:  # dv[i] ==0.  Note. dz is always negative
            t -= dz[i] / (v[i]*cos_theta_i0)
            x -= dz[i] * (sin_theta_i0/cos_theta_i0)
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( 1/p/(1+cos1) )
        if dz[ib] < -1e-10 and np.abs(dv[ib]) > 1e-10:  # dz is always negative
            local_dzdv = dz[ib]/dv[ib]
            t += local_dzdv * (ln_v0-ln_v1)
            x += local_dzdv * (inv_p * (-cos_theta_i0))
    return x*2.0, t*2.0
@jit(nopython=True, nogil=True)
def p2xt_grad(inv_p, z, v, dz, dv, buf_six_by_zsize):
    if inv_p <= v[0]:
        return 0.0, 0.0, np.zeros(z.size, dtype=np.float64), 0.0, np.zeros(z.size, dtype=np.float64), 0.0, np.zeros(z.size, dtype=np.float64)
    #### get the layer where the ray turns
    ib = z.size-1 # in maximum case, the ray will penerate all layers and reach the z[z.size-1]
    for ilayer in range(z.size-1):
        if v[ilayer]< inv_p and inv_p <= v[ilayer+1]:
        #if inv_v[ilayer] > p and p >= inv_v[ilayer+1]:
            ib = ilayer
            break
    #### buffer
    dist = 0.0
    trvt = 0.0
    buf_six_by_zsize.fill(0.0) #np.zeros((6, z.size), dtype=np.float64)
    par_ti_v0 = buf_six_by_zsize[0, :ib+1]
    par_ti_v1 = buf_six_by_zsize[1, :ib+1]
    par_xi_v0 = buf_six_by_zsize[2, :ib+1]
    par_xi_v1 = buf_six_by_zsize[3, :ib+1]
    par_dist_v = buf_six_by_zsize[4, :z.size]
    par_trvt_v = buf_six_by_zsize[5, :z.size]
    par_trvt_p = 0.0 # single values
    par_dist_p = 0.0 # single values
    #### critical angles
    sin_theta_i0 = v[0]/inv_p
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    ln_v0 = np.log(v[0]/(1.0+cos_theta_i0) ) # ln( v0/(1+cos0) )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = v[i+1]/inv_p
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        ln_v1 = np.log(v[i+1]/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if np.abs(dv[i]) > 1e-10 and dz[i] < -1e-10:
            local_dzdv = dz[i]/dv[i]
            t_i= local_dzdv * (ln_v0 - ln_v1)
            x_i = local_dzdv * (inv_p *(cos_theta_i1 - cos_theta_i0))
            dist += x_i
            trvt += t_i
            #
            tmp1 = sin_theta_i0/(cos_theta_i0*(1.0+cos_theta_i0))
            tmp2 = sin_theta_i1/(cos_theta_i1*(1.0+cos_theta_i1))
            tmp3 = t_i/dv[i] #
            par_ti_v0[i] =  tmp3 + local_dzdv*(1/v[i  ] + tmp1/inv_p)
            par_ti_v1[i] = -tmp3 - local_dzdv*(1/v[i+1] + tmp2/inv_p)
            par_trvt_p += local_dzdv*v[i  ]*tmp1
            par_trvt_p -= local_dzdv*v[i+1]*tmp2
            #
            tmp1 = local_dzdv*sin_theta_i0/cos_theta_i0
            tmp2 = local_dzdv*sin_theta_i1/cos_theta_i1
            tmp3 = x_i*local_dzdv/dz[i]
            #
            par_xi_v0[i] =  tmp3 + tmp1
            par_xi_v1[i] = -tmp3 - tmp2
            #
            par_dist_p -= x_i*inv_p
            par_dist_p += v[  i]*inv_p*tmp1
            par_dist_p -= v[i+1]*inv_p*tmp2
        elif dz[i] < -1e-10: # dv[i] == 0.0
            t_i = -dz[i] / (v[i]*cos_theta_i0)
            x_i = -dz[i] * (sin_theta_i0/cos_theta_i0)
            dist += x_i
            trvt += t_i
            #
            par_ti_v0[i] = -t_i/(v[i]+v[i+1]) + t_i/inv_p*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_ti_v1[i] = par_ti_v0[i]
            #
            tmp1 = t_i*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_trvt_p += v[i  ]*tmp1
            par_trvt_p += v[i+1]*tmp1
            #
            par_xi_v0[i] = 0.5/inv_p*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_xi_v1[i] = par_xi_v0[i]
            #
            tmp2 = 0.5*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_dist_p += v[i  ]*tmp2
            par_dist_p += v[i+1]*tmp2
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( 1/p/(1+cos1) )
        if np.abs(dz[ib]) > 1e-10 and np.abs(dv[ib]) > 1e-10:
            local_dzdv = dz[ib]/dv[ib]
            t_ib = local_dzdv * (ln_v0-ln_v1)
            x_ib = local_dzdv * (inv_p * (-cos_theta_i0))
            dist += x_ib
            trvt += t_ib
            # cos_theta[ib] is zero for distance=0 case.
            par_ti_v0[ib] = t_ib*local_dzdv/dz[ib] + local_dzdv*(1.0/v[ib] + (sin_theta_i0)/(inv_p*cos_theta_i0*(1+cos_theta_i0) ) )
            par_ti_v1[ib] =-t_ib*local_dzdv/dz[ib]
            par_xi_v0[ib] =  x_ib*local_dzdv/dz[ib] + local_dzdv*sin_theta_i0/cos_theta_i0
            par_xi_v1[ib] = -x_ib*local_dzdv/dz[ib]
            #
            par_trvt_p += v[ib]*local_dzdv/(sin_theta_i0*cos_theta_i0)
            par_dist_p -= x_ib*inv_p
            par_dist_p += v[ib]*inv_p*local_dzdv*sin_theta_i0/cos_theta_i0
    ####
    dist *= 2.0
    trvt *= 2.0
    ####
    par_dist_v[0] = par_xi_v0[0]
    par_trvt_v[0] = par_ti_v0[0]
    for j in range(1, ib+1):
        par_dist_v[j] = par_xi_v0[j] + par_xi_v1[j-1]
        par_trvt_v[j] = par_ti_v0[j] + par_ti_v1[j-1]
    par_dist_v[ib+1] = par_xi_v1[ib]
    par_trvt_v[ib+1] = par_ti_v1[ib]
    ###
    par_dist_v *= 2.0
    par_trvt_v *= 2.0
    par_dist_p *= 2.0
    par_trvt_p *= 2.0
    ###
    # the Delta T / Delta V if Delta X is zero
    d_trvt_v = par_trvt_v # where par_trvt_p and par_dist_p are Zeros
    if np.abs(par_dist_p) > 1e-10:
        d_trvt_v = par_trvt_v - par_trvt_p * (par_dist_v/par_dist_p)
    ###
    return dist, trvt, par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v
# v2
@jit(nopython=True, nogil=True)
def p2xt_v2(inv_p, z, v, dz, dv, ib):
    ib = ib if ib < z.size else z.size-1 # fix ib if out of range
    ####
    t = 0.0
    x = 0.0
    ####
    sin_theta_i0 = v[0]/inv_p
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    ln_v0 = np.log(v[0]/(1.0+cos_theta_i0) ) # ln( v0/(1+cos0) )
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = v[i+1]/inv_p
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        ln_v1 = np.log(v[i+1]/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if np.abs(dv[i]) > 1e-10 and dz[i] < -1e-10:
            local_dzdv = dz[i]/dv[i]
            t += local_dzdv * (ln_v0 - ln_v1)
            x += local_dzdv * (inv_p *(cos_theta_i1 - cos_theta_i0))
        elif dz[i] < -1e-10: # dv[i] ==0.  Note. dz is always negative
            t -= dz[i] / (v[i]*cos_theta_i0)
            x -= dz[i] * (sin_theta_i0/cos_theta_i0)
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( 1/p/(1+cos1) )
        if dz[ib] < -1e-10 and np.abs(dv[ib]) > 1e-10: # dz is always negative
            local_dzdv = dz[ib]/dv[ib]
            t += local_dzdv * (ln_v0-ln_v1)
            x += local_dzdv * (inv_p * (-cos_theta_i0))
    x *= 2.0
    t *= 2.0
    return x, t
@jit(nopython=True, nogil=True)
def p2xt_grad_v2(inv_p, z, v, dz, dv, ib, buf_six_by_zsize):
    ####
    #for ilayer in range(z.size-1):
    #    if 1.0/v[ilayer] > inv_p >= 1.0/v[ilayer+1]:
    #        #ib = ilayer
    #        #break
    #        if ib != ilayer:
    #            print("ib=", ib, ilayer)
    #ib = ilayer
    ####
    ib = ib if ib < z.size else z.size-1 # fix ib if out of range
    #### buffer
    dist = 0.0
    trvt = 0.0
    buf_six_by_zsize.fill(0.0) #np.zeros((6, z.size), dtype=np.float64)
    par_ti_v0 = buf_six_by_zsize[0, :ib+1]
    par_ti_v1 = buf_six_by_zsize[1, :ib+1]
    par_xi_v0 = buf_six_by_zsize[2, :ib+1]
    par_xi_v1 = buf_six_by_zsize[3, :ib+1]
    par_dist_v = buf_six_by_zsize[4, :z.size]
    par_trvt_v = buf_six_by_zsize[5, :z.size]
    par_trvt_p = 0.0 # single values
    par_dist_p = 0.0 # single values
    #### critical angles
    sin_theta_i0 = v[0]/inv_p
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    ln_v0 = np.log(v[0]/(1.0+cos_theta_i0) ) # ln( v0/(1+cos0) )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = v[i+1]/inv_p
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        #junk = 100.0/cos_theta_i0
        #junk = 100.0/cos_theta_i1 # test if error can happen
        #if cos_theta_i0 < 1e-20:
        #    print("i0", i, cos_theta_i0, p, 1/v[i],     v[i], v[0])
        #    traceback.print_stack()
        #if cos_theta_i1 < 1e-20:
        #    print(f'i1={i+1} cos_theta_i1={cos_theta_i1} p={p} 1/v[i+1]={1/v[i+1]} v[i+1]={v[i+1]} v[0]={v[0]}')
        #    print(f'\t v[i]={v[i]} v[i+1]={v[i+1]} inv_p={inv_p} ib={ib}')
        #    traceback.print_stack()
        #    #print("i1", i+1, "cos_theta_i1=",cos_theta_i1, "p=", p, 1/v[i+1], v[i+1], v[0])
        ln_v1 = np.log(v[i+1]/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if np.abs(dv[i]) > 1e-10 and dz[i] < -1e-10:
            local_dzdv = dz[i]/dv[i]
            t_i= local_dzdv * (ln_v0 - ln_v1)
            x_i = local_dzdv * (inv_p *(cos_theta_i1 - cos_theta_i0))
            dist += x_i
            trvt += t_i
            #
            tmp1 = sin_theta_i0/(cos_theta_i0*(1.0+cos_theta_i0))
            tmp2 = sin_theta_i1/(cos_theta_i1*(1.0+cos_theta_i1))
            tmp3 = t_i/dv[i] #
            par_ti_v0[i] =  tmp3 + local_dzdv*(1/v[i  ] + tmp1/inv_p)
            par_ti_v1[i] = -tmp3 - local_dzdv*(1/v[i+1] + tmp2/inv_p)
            par_trvt_p += local_dzdv*v[i  ]*tmp1
            par_trvt_p -= local_dzdv*v[i+1]*tmp2
            #
            tmp1 = local_dzdv*sin_theta_i0/cos_theta_i0
            tmp2 = local_dzdv*sin_theta_i1/cos_theta_i1
            tmp3 = x_i*local_dzdv/dz[i]
            #
            par_xi_v0[i] =  tmp3 + tmp1
            par_xi_v1[i] = -tmp3 - tmp2
            #
            par_dist_p -= x_i*inv_p
            par_dist_p += v[  i]*inv_p*tmp1
            par_dist_p -= v[i+1]*inv_p*tmp2
        elif dz[i] < -1e-10: # dv[i] == 0.0
            t_i = -dz[i] / (v[i]*cos_theta_i0)
            x_i = -dz[i] * (sin_theta_i0/cos_theta_i0)
            dist += x_i
            trvt += t_i
            #
            par_ti_v0[i] = -t_i/(v[i]+v[i+1]) + t_i/inv_p*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_ti_v1[i] = par_ti_v0[i]
            #
            tmp1 = t_i*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_trvt_p += v[i  ]*tmp1
            par_trvt_p += v[i+1]*tmp1
            #
            par_xi_v0[i] = 0.5/inv_p*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_xi_v1[i] = par_xi_v0[i]
            #
            tmp2 = 0.5*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_dist_p += v[i  ]*tmp2
            par_dist_p += v[i+1]*tmp2
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( 1/p/(1+cos1) )
        if np.abs(dz[ib]) > 1e-10 and np.abs(dv[ib]) > 1e-10:
            local_dzdv = dz[ib]/dv[ib]
            #junk = 100.0/cos_theta_i0
            t_ib = local_dzdv * (ln_v0-ln_v1)
            x_ib = local_dzdv * (inv_p * (-cos_theta_i0))
            dist += x_ib
            trvt += t_ib
            # cos_theta[ib] is zero for distance=0 case.
            par_ti_v0[ib] = t_ib*local_dzdv/dz[ib] + local_dzdv*(1.0/v[ib] + (sin_theta_i0)/(inv_p*cos_theta_i0*(1+cos_theta_i0) ) )
            par_ti_v1[ib] =-t_ib*local_dzdv/dz[ib]
            par_xi_v0[ib] =  x_ib*local_dzdv/dz[ib] + local_dzdv*sin_theta_i0/cos_theta_i0
            par_xi_v1[ib] = -x_ib*local_dzdv/dz[ib]
            #
            par_trvt_p += v[ib]*local_dzdv/(sin_theta_i0*cos_theta_i0)
            par_dist_p -= x_ib*inv_p
            par_dist_p += v[ib]*inv_p*local_dzdv*sin_theta_i0/cos_theta_i0
    ####
    dist *= 2.0
    trvt *= 2.0
    ####
    par_dist_v[0] = par_xi_v0[0]
    par_trvt_v[0] = par_ti_v0[0]
    for j in range(1, ib+1):
        par_dist_v[j] = par_xi_v0[j] + par_xi_v1[j-1]
        par_trvt_v[j] = par_ti_v0[j] + par_ti_v1[j-1]
    par_dist_v[ib+1] = par_xi_v1[ib]
    par_trvt_v[ib+1] = par_ti_v1[ib]
    ###
    par_dist_v *= 2.0
    par_trvt_v *= 2.0
    par_dist_p *= 2.0
    par_trvt_p *= 2.0
    ###
    # the Delta T / Delta V if Delta X is zero
    d_trvt_v = par_trvt_v # where par_trvt_p and par_dist_p are Zeros
    if np.abs(par_dist_p) > 1e-10:
        d_trvt_v = par_trvt_v - par_trvt_p * (par_dist_v/par_dist_p)
    ###
    return dist, trvt, par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v
# v4
@jit(nopython=True, nogil=True) #, fastmath=True)
def p2xt_v4(inv_p, z, v, dz, dv, dzdv, layer_type, ib):
    """
    layer_type is a list of 2 (non-constant v layer), 1 (constant v layer), 0 (zero thickness layer)
    """
    ib = ib if ib < z.size else z.size-1 # fix ib if out of range
    ####
    t = 0.0
    x = 0.0
    ####
    sin_theta_i0 = v[0]/inv_p
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    ln_v0 = np.log(v[0]/(1.0+cos_theta_i0) ) # ln( v0/(1+cos0) )
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = v[i+1]/inv_p
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        ln_v1 = np.log(v[i+1]/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if layer_type[i] == 2:    #if np.abs(dv[i]) > 1e-10 and dz[i] < -1e-10:
            t += dzdv[i] * (ln_v0 - ln_v1)
            x += dzdv[i] * (inv_p *(cos_theta_i1 - cos_theta_i0))
        elif layer_type[i]: # ==1 #  #elif dz[i] < -1e-10: # dv[i] ==0.  Note. dz is always negative
            t -= dz[i] / (v[i]*cos_theta_i0)
            x -= dz[i] * (sin_theta_i0/cos_theta_i0)
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( 1/p/(1+cos1) )
        if layer_type[ib] == 2: #if dz[ib] < -1e-10 and np.abs(dv[ib]) > 1e-10: # dz is always negative
            t += dzdv[ib] * (ln_v0-ln_v1) # Note inv_k[ib] is finite automatically!
            x += dzdv[ib] * (inv_p * (-cos_theta_i0))
    x *= 2.0
    t *= 2.0
    return x, t
@jit(nopython=True, nogil=True)
def p2xt_grad_v4(inv_p, z, v, dz, dv, dzdv, layer_type, ib, buf_six_by_zsize):
    ####
    #for ilayer in range(z.size-1):
    #    if 1.0/v[ilayer] > inv_p >= 1.0/v[ilayer+1]:
    #        #ib = ilayer
    #        #break
    #        if ib != ilayer:
    #            print("ib=", ib, ilayer)
    #ib = ilayer
    ####
    ib = ib if ib < z.size else z.size-1 # fix ib if out of range
    #### buffer
    dist = 0.0
    trvt = 0.0
    buf_six_by_zsize.fill(0.0) #np.zeros((6, z.size), dtype=np.float64)
    par_ti_v0 = buf_six_by_zsize[0, :ib+1]
    par_ti_v1 = buf_six_by_zsize[1, :ib+1]
    par_xi_v0 = buf_six_by_zsize[2, :ib+1]
    par_xi_v1 = buf_six_by_zsize[3, :ib+1]
    par_dist_v = buf_six_by_zsize[4, :z.size]
    par_trvt_v = buf_six_by_zsize[5, :z.size]
    par_trvt_p = 0.0 # single values
    par_dist_p = 0.0 # single values
    #### critical angles
    sin_theta_i0 = v[0]/inv_p
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    ln_v0 = np.log(v[0]/(1.0+cos_theta_i0) ) # ln( v0/(1+cos0) )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = v[i+1]/inv_p
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        ####junk = 100.0/cos_theta_i0
        ####junk = 100.0/cos_theta_i1 # test if error can happen
        #if cos_theta_i0 < 1e-20:
        #    print("i0", i, cos_theta_i0, p, 1/v[i],     v[i], v[0])
        #    traceback.print_stack()
        #if cos_theta_i1 < 1e-20:
        #    print(f'i1={i+1} cos_theta_i1={cos_theta_i1} p={p} 1/v[i+1]={1/v[i+1]} v[i+1]={v[i+1]} v[0]={v[0]}')
        #    print(f'\t v[i]={v[i]} v[i+1]={v[i+1]} inv_p={inv_p} ib={ib}')
        #    traceback.print_stack()
        #    #print("i1", i+1, "cos_theta_i1=",cos_theta_i1, "p=", p, 1/v[i+1], v[i+1], v[0])
        ln_v1 = np.log(v[i+1]/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if layer_type[i] == 2: #if np.abs(dv[i]) > 1e-10 and dz[i] < -1e-10:
            local_dzdv = dzdv[i] #dz[i]/dv[i]
            t_i= local_dzdv * (ln_v0 - ln_v1)
            x_i = local_dzdv * (inv_p *(cos_theta_i1 - cos_theta_i0))
            dist += x_i
            trvt += t_i
            #
            tmp1 = sin_theta_i0/(cos_theta_i0*(1.0+cos_theta_i0))
            tmp2 = sin_theta_i1/(cos_theta_i1*(1.0+cos_theta_i1))
            tmp3 = t_i/dv[i] #
            par_ti_v0[i] =  tmp3 + local_dzdv*(1/v[i  ] + tmp1/inv_p)
            par_ti_v1[i] = -tmp3 - local_dzdv*(1/v[i+1] + tmp2/inv_p)
            par_trvt_p += local_dzdv*v[i  ]*tmp1
            par_trvt_p -= local_dzdv*v[i+1]*tmp2
            #
            tmp1 = local_dzdv*sin_theta_i0/cos_theta_i0
            tmp2 = local_dzdv*sin_theta_i1/cos_theta_i1
            tmp3 = x_i*local_dzdv/dz[i]
            #
            par_xi_v0[i] =  tmp3 + tmp1
            par_xi_v1[i] = -tmp3 - tmp2
            #
            par_dist_p -= x_i*inv_p
            par_dist_p += v[  i]*inv_p*tmp1
            par_dist_p -= v[i+1]*inv_p*tmp2
        elif layer_type[i] == 1: #elif dz[i] < -1e-10: # dv[i] ==0.  Note. dz is always negative
            t_i = -dz[i] / (v[i]*cos_theta_i0)
            x_i = -dz[i] * (sin_theta_i0/cos_theta_i0)
            dist += x_i
            trvt += t_i
            #
            par_ti_v0[i] = -t_i/(v[i]+v[i+1]) + t_i/inv_p*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_ti_v1[i] = par_ti_v0[i]
            #
            tmp1 = t_i*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_trvt_p += v[i  ]*tmp1
            par_trvt_p += v[i+1]*tmp1
            #
            par_xi_v0[i] = 0.5/inv_p*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_xi_v1[i] = par_xi_v0[i]
            #
            tmp2 = 0.5*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_dist_p += v[i  ]*tmp2
            par_dist_p += v[i+1]*tmp2
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( 1/p/(1+cos1) )
        if layer_type[ib] == 2: #if np.abs(dz[ib]) > 1e-10 and np.abs(dv[ib]) > 1e-10:
            local_dzdv = dzdv[ib] #dz[ib]/dv[ib]
            ####junk = 100.0/cos_theta_i0
            t_ib = local_dzdv * (ln_v0-ln_v1)
            x_ib = local_dzdv * (inv_p * (-cos_theta_i0))
            dist += x_ib
            trvt += t_ib
            # cos_theta[ib] is zero for distance=0 case.
            par_ti_v0[ib] = t_ib*local_dzdv/dz[ib] + local_dzdv*(1.0/v[ib] + (sin_theta_i0)/(inv_p*cos_theta_i0*(1+cos_theta_i0) ) )
            par_ti_v1[ib] =-t_ib*local_dzdv/dz[ib]
            par_xi_v0[ib] =  x_ib*local_dzdv/dz[ib] + local_dzdv*sin_theta_i0/cos_theta_i0
            par_xi_v1[ib] = -x_ib*local_dzdv/dz[ib]
            #
            par_trvt_p += v[ib]*local_dzdv/(sin_theta_i0*cos_theta_i0)
            par_dist_p -= x_ib*inv_p
            par_dist_p += v[ib]*inv_p*local_dzdv*sin_theta_i0/cos_theta_i0
    ####
    dist *= 2.0
    trvt *= 2.0
    ####
    par_dist_v[0] = par_xi_v0[0]
    par_trvt_v[0] = par_ti_v0[0]
    for j in range(1, ib+1):
        par_dist_v[j] = par_xi_v0[j] + par_xi_v1[j-1]
        par_trvt_v[j] = par_ti_v0[j] + par_ti_v1[j-1]
    par_dist_v[ib+1] = par_xi_v1[ib]
    par_trvt_v[ib+1] = par_ti_v1[ib]
    ###
    par_dist_v *= 2.0
    par_trvt_v *= 2.0
    par_dist_p *= 2.0
    par_trvt_p *= 2.0
    ###
    # the Delta T / Delta V if Delta X is zero
    d_trvt_v = par_trvt_v # where par_trvt_p and par_dist_p are Zeros
    if np.abs(par_dist_p) > 1e-10:
        d_trvt_v = par_trvt_v - par_trvt_p * (par_dist_v/par_dist_p)
    ###
    return dist, trvt, par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v
@jit(nopython=True, nogil=True)
def p2xt_grad_ponly_v4(inv_p, z, v, dz, dv, dzdv, layer_type, ib):
    ####
    #for ilayer in range(z.size-1):
    #    if 1.0/v[ilayer] > inv_p >= 1.0/v[ilayer+1]:
    #        #ib = ilayer
    #        #break
    #        if ib != ilayer:
    #            print("ib=", ib, ilayer)
    #ib = ilayer
    ####
    ib = ib if ib < z.size else z.size-1 # fix ib if out of range
    #### buffer
    dist = 0.0
    trvt = 0.0
    par_dist_p = 0.0 # single values
    #### critical angles
    sin_theta_i0 = v[0]/inv_p
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    ln_v0 = np.log(v[0]/(1.0+cos_theta_i0) ) # ln( v0/(1+cos0) )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = v[i+1]/inv_p
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        ####junk = 100.0/cos_theta_i0
        ####junk = 100.0/cos_theta_i1 # test if error can happen
        #if cos_theta_i0 < 1e-20:
        #    print("i0", i, cos_theta_i0, p, 1/v[i],     v[i], v[0])
        #    traceback.print_stack()
        #if cos_theta_i1 < 1e-20:
        #    print(f'i1={i+1} cos_theta_i1={cos_theta_i1} p={p} 1/v[i+1]={1/v[i+1]} v[i+1]={v[i+1]} v[0]={v[0]}')
        #    print(f'\t v[i]={v[i]} v[i+1]={v[i+1]} inv_p={inv_p} ib={ib}')
        #    traceback.print_stack()
        #    #print("i1", i+1, "cos_theta_i1=",cos_theta_i1, "p=", p, 1/v[i+1], v[i+1], v[0])
        ln_v1 = np.log(v[i+1]/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if layer_type[i] == 2: #if np.abs(dv[i]) > 1e-10 and dz[i] < -1e-10:
            local_dzdv = dzdv[i] #dz[i]/dv[i]
            t_i= local_dzdv * (ln_v0 - ln_v1)
            x_i = local_dzdv * (inv_p *(cos_theta_i1 - cos_theta_i0))
            dist += x_i
            trvt += t_i
            #
            tmp1 = local_dzdv*sin_theta_i0/cos_theta_i0
            tmp2 = local_dzdv*sin_theta_i1/cos_theta_i1
            #
            par_dist_p -= x_i*inv_p
            par_dist_p += v[  i]*inv_p*tmp1
            par_dist_p -= v[i+1]*inv_p*tmp2
        elif layer_type[i] == 1: #elif dz[i] < -1e-10: # dv[i] ==0.  Note. dz is always negative
            t_i = -dz[i] / (v[i]*cos_theta_i0)
            x_i = -dz[i] * (sin_theta_i0/cos_theta_i0)
            dist += x_i
            trvt += t_i
            #
            tmp2 = 0.5*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_dist_p += v[i  ]*tmp2
            par_dist_p += v[i+1]*tmp2
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( 1/p/(1+cos1) )
        if layer_type[ib] == 2: #if np.abs(dz[ib]) > 1e-10 and np.abs(dv[ib]) > 1e-10:
            local_dzdv = dzdv[ib] #dz[ib]/dv[ib]
            ####junk = 100.0/cos_theta_i0
            t_ib = local_dzdv * (ln_v0-ln_v1)
            x_ib = local_dzdv * (inv_p * (-cos_theta_i0))
            dist += x_ib
            trvt += t_ib
            # cos_theta[ib] is zero for distance=0 case.
            #
            par_dist_p -= x_ib*inv_p
            par_dist_p += v[ib]*inv_p*local_dzdv*sin_theta_i0/cos_theta_i0
    ####
    dist *= 2.0
    trvt *= 2.0
    ####
    ###
    par_dist_p *= 2.0
    ###
    return dist, trvt, par_dist_p


#### Given z, vz, compute 1/p, x, and t (and their legs and ib)
@jit(nopython=True, nogil=True)
def split_pxt_legs(inv_ps, xs, ts): # split the p-x-t into legs so that each leg has monotonic p and x.
    dx = np.diff(xs)
    ### ps must be decreasing
    #if ps[0] < ps[1]:
    #    ps = ps[::-1]
    #    xs = xs[::-1]
    #    ts = ts[::-1]
    #    dx = dx[::-1]*-1 ## reverse and change sign
    #print('ps=', ps)
    for ind in range(1, len(dx)): # take care of the zeros in dx
        if dx[ind] == 0:
            dx[ind] = dx[ind-1]  # if two xs are the same, use the previous one
    tmp = dx[:-1] * dx[1:]
    idxs_minial_maximal = [0]
    idxs_minial_maximal.extend( np.where( tmp < 0 )[0]+1 )
    idxs_minial_maximal.append(len(xs)-1)
    inv_p_legs = [inv_ps[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    x_legs = [xs[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    t_legs = [ts[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    return inv_p_legs, x_legs, t_legs
@jit(nopython=True, nogil=True)
def zv2pxt(z, vz, theta_step_deg=0.1):
    theta_step_rad = theta_step_deg * (np.pi/180.0)
    v0 = vz[0]
    nlayers = vz.size - 1
    inv_rp_legs, dist_legs, trvt_legs = list(), list(), list()
    ##############################################################
    vtop = v0
    dz = np.diff(z)
    dvz = np.diff(vz)
    #inv_vz = 1.0/vz
    for ilayer in range(nlayers):
        # Before the below if-else, vtop is the max velocity above current layer (does not include the top of this layer)
        # In other words, vtop = max(v[:ilayer]) = max(v[0], v[1], ..., v[ilayer-1]) when ilayer>=1
        if vtop >= vz[ilayer]:
            # current layer's top is slowner than the max above. Hence, use the previous max as vtop.
            # Besides, we need to add a very small value to avoid numerical issues, we apply that later when setting rps[0] by np.nextafter(...)
            flag_next_after = True
        else:
            # current layer's top is faster than the max above. Hence, use the current layer's top as vtop.
            vtop = vz[ilayer]
            flag_next_after = False
        vbot = vz[ilayer+1]
        if vtop < vbot: # make sure ray can turn in this layer
            t0, t1 = np.arcsin(v0/vtop), np.arcsin(v0/vbot)
            nrp    = int(np.abs(t0-t1)/theta_step_rad) + 2
            thetas = np.linspace( t0, t1, nrp)
            inv_rps    = v0/np.sin(thetas)
            inv_rps[0] = vtop # force to set this to avoid numerical issues by sin(arcsin(...))
            inv_rps[-1]= vbot # the same
            if flag_next_after:
                inv_rps[0] = np.nextafter( inv_rps[0], inv_rps[-1] )
            ##
            dist= np.zeros(nrp, dtype=np.float64)
            trvt= np.zeros(nrp, dtype=np.float64)
            for irp in range(nrp):
                dist[irp], trvt[irp] = p2xt(inv_rps[irp], z, vz, dz, dvz)
            tmp1, tmp2, tmp3 = split_pxt_legs(inv_rps, dist, trvt)
            inv_rp_legs.extend(tmp1)
            dist_legs.extend(tmp2)
            trvt_legs.extend(tmp3)
            ####
            #vmax_above_current_layer = vbot ##### note vbot is alreay the max until now!
    ########
    # flatten the list of list
    inv_rp= np.array([it for leg in inv_rp_legs for it in leg])
    dist = np.array([it for leg in dist_legs for it in leg])
    trvt = np.array([it for leg in trvt_legs for it in leg])
    idx_sort = np.argsort(inv_rp)
    inv_rp = inv_rp[idx_sort]
    dist = dist[idx_sort]
    trvt = trvt[idx_sort]
    #########
    return (inv_rp, dist, trvt), (inv_rp_legs, dist_legs, trvt_legs)
# v2
@jit(nopython=True, nogil=True)
def split_pxt_legs_v2(xs): # split the p-x-t into legs so that each leg has monotonic p and x.
    dx = np.diff(xs) # ps must be decreasing
    for ind in range(1, len(dx)): # take care of the zeros in dx
        if dx[ind] == 0:
            dx[ind] = dx[ind-1]  # if two xs are the same, use the previous one
    tmp = dx[:-1] * dx[1:]
    idxs_boundary_exclude_start_end = np.where( tmp < 0 )[0]+1
    #idxs_minial_maximal = [0]
    #idxs_minial_maximal.extend( np.where( tmp < 0 )[0]+1 )
    #idxs_minial_maximal.append(len(xs)-1)
    return idxs_boundary_exclude_start_end
@jit(nopython=True, nogil=True)
def zv2pxt_v2(z, vz, theta_step_deg=0.1):
    theta_step_rad = theta_step_deg * (np.pi/180.0)
    v0 = vz[0]
    nlayers = vz.size - 1
    ##############################################################
    vtop = v0
    vel_ranges     = np.zeros((nlayers, 2), dtype=np.float64)# + np.nan
    ang_ranges     = np.zeros((nlayers, 2), dtype=np.float64)# + np.nan
    nrp_each_range = np.zeros(nlayers, dtype=np.int64)
    bottom_layer_ind_each_range = np.zeros(nlayers, dtype=np.int64)
    flag_next_after = np.zeros(nlayers, dtype=np.int8)
    n_ang_range = 0
    for ilayer in range(nlayers):
        # Before the below if-else, vtop is the max velocity above current layer (does not include the top of this layer)
        # In other words, vtop = max(v[:ilayer]) = max(v[0], v[1], ..., v[ilayer-1]) when ilayer>=1
        if vtop >= vz[ilayer]:
            # current layer's top is slowner than the max above. Hence, use the previous max as vtop.
            # Besides, we need to add a very small value to avoid numerical issues, we apply that later when setting rps[0] by np.nextafter(...)
            flag_next_after[n_ang_range] = 1 # Note, the index is n_ang_range, not ilayer!!!
        else:
            # current layer's top is faster than the max above. Hence, use the current layer's top as vtop.
            vtop = vz[ilayer]
            flag_next_after[ilayer] = 0 # default is alreay 0
        vbot = vz[ilayer+1]
        if vtop < vbot: # make sure ray can turn in this layer
            vel_ranges[n_ang_range][0] = vtop
            vel_ranges[n_ang_range][1] = vbot
            atop = np.arcsin(v0/vtop)
            abot = np.arcsin(v0/vbot)
            ang_ranges[n_ang_range][0] = atop
            ang_ranges[n_ang_range][1] = abot
            nrp_each_range[n_ang_range] = int((atop-abot)/theta_step_rad) + 2
            #
            bottom_layer_ind_each_range[n_ang_range] = ilayer # these rays turn in the layer of this index, also the bottom layer index
            #
            n_ang_range += 1
            #######
    ##############################################################
    total_nrp = np.sum(nrp_each_range[:n_ang_range])
    buf_pxt = np.zeros((3, total_nrp), dtype=np.float64)
    inv_rps = buf_pxt[0]
    dist= buf_pxt[1]
    trvt= buf_pxt[2]
    ibs = np.zeros(total_nrp, dtype=np.int64)
    ####
    leg_start_end_pairs_gind = np.zeros((2, total_nrp), dtype=np.int64)
    leg_start_gind = leg_start_end_pairs_gind[0] # the index points to the data point in (rps, dist, trvt) for the start of each leg
    leg_end_gind   = leg_start_end_pairs_gind[1] # the index points to the data point in (rps, dist, trvt) for the end   of each leg
    n_legs =0
    ####
    g_i0 = 0
    dz = np.diff(z)
    dv = np.diff(vz)
    for idx_range in range(n_ang_range):
        nrp = nrp_each_range[idx_range]
        g_i1 = g_i0 + nrp # g_i0 and g_i1 are the index for rps, dist, trvt
        atop = ang_ranges[idx_range][0]
        abot = ang_ranges[idx_range][1]
        vtop = vel_ranges[idx_range][0]
        vbot = vel_ranges[idx_range][1]
        thetas = np.linspace( atop, abot, nrp)
        inv_rps[g_i0:g_i1] = v0/np.sin(thetas)
        inv_rps[g_i0]   = vtop
        inv_rps[g_i1-1] = vbot
        if flag_next_after[idx_range]:
            inv_rps[g_i0] = np.nextafter( inv_rps[g_i0], inv_rps[g_i1-1] ) # make sure the first ray can enter the layer
            dist[g_i0], trvt[g_i0] = p2xt_v2(inv_rps[g_i0], z, vz, dz, dv, bottom_layer_ind_each_range[idx_range])
            ibs[g_i0] = bottom_layer_ind_each_range[idx_range]
        else:
            # for this case, rps[g_i0] is exactly the critical ray that turns at the top of the layer.
            # Hence, the ib should minus 1
            dist[g_i0], trvt[g_i0] = p2xt_v2(inv_rps[g_i0], z, vz, dz, dv, bottom_layer_ind_each_range[idx_range]-1)
            ibs[g_i0] = bottom_layer_ind_each_range[idx_range] -1
        for tmp_i in range(g_i0+1, g_i1):
            dist[tmp_i], trvt[tmp_i] = p2xt_v2(inv_rps[tmp_i], z, vz, dz, dv, bottom_layer_ind_each_range[idx_range])
        ibs[g_i0+1:g_i1] = bottom_layer_ind_each_range[idx_range]
        ########################################################
        # Split into legs with x! p is already monotonic
        b_inds = split_pxt_legs_v2(dist[g_i0:g_i1] )
        # b_inds are the local indices where many legs should be split
        # It does not include the start and end indices of the whole.
        # Thus,
        # the ind for the starts of the legs will be 0,           b_inds[0],   b_inds[1],...   b_inds[N-2], b_inds[N-1]
        # the ind for the ends   of the legs will be b_inds[0]+1, b_inds[1]+1, b_inds[2]+1,... b_inds[N-1], nrp
        # all these index need to be added by g_i0 to get global index
        #
        b_inds += g_i0 # adjust to global index
        leg_start_gind[n_legs]                         = g_i0
        leg_start_gind[n_legs+1: n_legs+1+b_inds.size] = b_inds
        leg_end_gind[  n_legs:   n_legs+b_inds.size]   = b_inds+1
        leg_end_gind[               n_legs+b_inds.size]= g_i1 # also (g_i0 + nrp)
        #leg_ib[n_legs: n_legs+1+b_inds.size]           = bottom_layer_ind_each_range[idx_range]
        #if not flag_next_after[idx_range]:
        #    # for this case, rps[g_i0] is exactly the critical ray that turns at the top of the layer.
        #    # Hence, the ib should minus 1
        #    leg_ib[n_legs] = bottom_layer_ind_each_range[idx_range]-1
        n_legs += (b_inds.size +1)
        ########################################################
        g_i0 = g_i1
    ############################################################
    # the leg_start_end_pairs_gind.T defines the start index, end index, and ib of each leg.
    return inv_rps, dist, trvt, ibs, leg_start_end_pairs_gind[:,:n_legs].T
# v3
@jit(nopython=True, nogil=True)
def zv2p_reflect_v3(z, vz, theta_step_deg=0.1):
    theta_step_rad = np.deg2rad(theta_step_deg)
    v0 = vz[0]
    vmax = np.max(vz)
    inv_rp_min = vmax
    theta0 = np.arcsin(v0/inv_rp_min)
    theta1 = 1e-3
    nrp = int( np.abs(theta1 - theta0) / theta_step_rad ) + 2
    inv_rps= v0/np.sin( np.linspace(theta0, theta1, nrp) )
    ###
    nblocks = 1
    bb = np.array((0, inv_rps.size)).reshape((nblocks, 2 ))
    ibs = np.zeros(inv_rps.size, dtype=np.int64) + (z.size -1) # all rays turn in the bottom layer
    return nblocks, bb, inv_rps, ibs
@jit(nopython=True, nogil=True)
def zv2p_v3(z, vz, theta_step_deg=0.1):
    theta_step_rad = np.deg2rad(theta_step_deg)
    #######################################################
    # Step 1
    # Split layers into blocks. Each block has continuous increasing velocities.
    v0 = vz[0]
    block_vmax_exclude_bottom = vz[0]
    block_boundary = np.zeros(z.size, dtype=np.int64) + z.size
    mono_vz = np.zeros(z.size, dtype=np.float64)
    nbb = 0
    flag_active_block = False
    #layer_flags = np.zeros(z.size-1, dtype=np.int8)
    for ilayer in range(z.size-1):
        vtop = vz[ilayer]
        vbot = vz[ilayer+1]
        if (vtop <= block_vmax_exclude_bottom < vbot) and (not flag_active_block):
            # A new block starts from the top of this layer
            block_boundary[nbb] = ilayer
            nbb += 1
            flag_active_block = True
            mono_vz[ilayer] = block_vmax_exclude_bottom
        elif block_vmax_exclude_bottom < vtop < vbot:
            # Append this layer into current layer block
            block_boundary[nbb] = ilayer + 1
            block_vmax_exclude_bottom = vtop
            mono_vz[ilayer] = vtop
        elif flag_active_block:
            # vtop <= max (vtop, block_vmax_exclude_bottom)
            # So,a shadow zone (discontinuous distance) occurs here
            # Current block (if an active exist) ends at the top interface of this layer
            block_boundary[nbb] = ilayer + 1 # note that something[:ix+1] will include ix!!!
            nbb += 1
            if vtop > block_vmax_exclude_bottom:
                block_vmax_exclude_bottom = vtop
            flag_active_block = False
            mono_vz[ilayer] = block_vmax_exclude_bottom
    # the last interface
    if block_vmax_exclude_bottom < vz[-1]:
        block_boundary[nbb] = vz.size
        mono_vz[-1] = vz[-1]
        nbb += 1
    # With the obtained block_boundary, the block will be
    # [bb[0]: bb[1]+1], [bb[2]: bb[3]+1], [bb[4]: bb[5]+1], ...,
    block_boundary = block_boundary[:nbb]
    # debug plot
    #if debug:
    #    cmap = plt.get_cmap('tab10')
    #    print(block_boundary.size )
    #    fig = plt.figure(figsize=(20, 10))
    #    gs  = fig.add_gridspec(200, 100)
    #    ax1 = fig.add_subplot(gs[:90, :30])
    #    ax2 = fig.add_subplot(gs[:90, 40:70])
    #    ax3 = fig.add_subplot(gs[110:, :30])
    #    ax4 = fig.add_subplot(gs[110:, 40:70])
    #    ax5 = fig.add_subplot(gs[110:, 80:])
    #    #fig, (ax1, ax2, ax3), (ax4, ax5) = plt.subplots(1,5, figsize=(20, 10))
    #    ax1.plot(vz, z, '.-', color='k')
    #    ax2.plot(1.0/vz, z, '.-', color='k')
    #    for istart, iend in zip(block_boundary[::2], block_boundary[1::2]):
    #        block_z = z[istart:iend]
    #        block_vz = vz[istart:iend]
    #        block_mono_vz = mono_vz[istart:iend]
    #        ax1.plot(block_vz, block_z, 's-', linewidth=4, color='r', alpha=0.8)
    #        ax1.plot(block_mono_vz, block_z, 'o-', linewidth=4, color='b', alpha=0.8)
    #        #
    #        ax2.plot(1.0/block_vz, block_z, 's-', linewidth=4, color='r', alpha=0.8)
    #        ax2.plot(1.0/block_mono_vz, block_z, 'o-', linewidth=4, color='b', alpha=0.8)
    #################################################################################
    # Step 2
    # Generate a list of rps, and ibs for each of the rp.
    # Generate a list of indexs declaring when the block_boundary is in rps.
    nblocks = nbb // 2
    #print('nblocks=', nblocks)
    #theta_range_each_effective_layer = np.zeros((z.size-1, 2), dtype=np.float64)
    #nrp_each_effective_layer         = np.zeros(z.size-1, dtype=np.int64)
    #################################################################################
    total_nrp = 0
    total_nlayer_effective = 0
    for iblock in range(nblocks):
        ilayer_start  = block_boundary[iblock*2]
        ilayer_end    = block_boundary[iblock*2+1]
        block_mono_vz = mono_vz[ilayer_start:ilayer_end]
        local_nlayers = block_mono_vz.size -1
        for local_ilayer in range(local_nlayers):
            vtop = block_mono_vz[local_ilayer]
            vbot = block_mono_vz[local_ilayer+1]
            theta_top = np.arcsin(v0/vtop)
            theta_bot = np.arcsin(v0/vbot)
            nrp       = int( np.abs(theta_top - theta_bot) / theta_step_rad ) + 2
            #theta_range_each_effective_layer[total_nlayer_effective, 0] = theta_top
            #theta_range_each_effective_layer[total_nlayer_effective, 1] = theta_bot
            #nrp_each_effective_layer[total_nlayer_effective] = nrp
            total_nlayer_effective += 1
            #
            total_nrp += nrp
    #theta_range_each_effective_layer = theta_range_each_effective_layer[:total_nlayer_effective]
    #nrp_each_effective_layer         = nrp_each_effective_layer[:total_nlayer_effective]
    #################################################################################
    #rps    = np.zeros(total_nrp, dtype=np.float64) # pre-allocate large space
    inv_rps= np.zeros(total_nrp, dtype=np.float64)
    ibs    = np.zeros(total_nrp, dtype=np.int64)
    #
    bb     = np.zeros((nblocks, 2), dtype=np.int64)
    bb_base= 0
    gidx_rp= 0
    #idx_nlayer_effective = 0
    for iblock in range(nblocks):
        ilayer_start  = block_boundary[iblock*2]
        ilayer_end    = block_boundary[iblock*2+1]
        block_mono_vz = mono_vz[ilayer_start:ilayer_end]
        #######
        lidx_rp       = 0
        local_nlayers = ilayer_end-ilayer_start-1
        #local_nlayers = block_mono_vz.size -1
        for local_ilayer in range(local_nlayers):
            vtop = block_mono_vz[local_ilayer]
            vbot = block_mono_vz[local_ilayer+1]
            theta_top = np.arcsin(v0/vtop)
            theta_bot = np.arcsin(v0/vbot)
            nrp     = int( np.abs(theta_top - theta_bot) / theta_step_rad ) + 2
            #theta_top = theta_range_each_effective_layer[idx_nlayer_effective, 0]
            #theta_bot = theta_range_each_effective_layer[idx_nlayer_effective, 1]
            #nrp       = nrp_each_effective_layer[idx_nlayer_effective]
            #idx_nlayer_effective += 1
            ##### alias
            #local_rps     = rps[    gidx_rp+lidx_rp : gidx_rp+lidx_rp+nrp]
            local_inv_rps = inv_rps[gidx_rp+lidx_rp : gidx_rp+lidx_rp+nrp]
            #
            #local_rps[:]    = np.sin( np.linspace(theta_top, theta_bot, nrp) )/v0
            local_inv_rps[:]= v0/np.sin( np.linspace(theta_top, theta_bot, nrp) )
            #
            #local_rps[0]     = 1.0/vtop
            local_inv_rps[0] = vtop
            #local_rps[-1]     = 1.0/vbot
            local_inv_rps[-1] = vbot
            # The critical ray at the top interface of this layer belongs to the previous layer
            # Do not worry about the first-most ray of this block. We will fix that after this loop.
            ibs[    gidx_rp+lidx_rp ] = ilayer_start + local_ilayer-1
            # All other rays belongs to this layer
            ibs[    gidx_rp+lidx_rp+1 : gidx_rp+lidx_rp+nrp] = ilayer_start + local_ilayer
            ######
            lidx_rp += nrp
        # Take care of the first ray enter this block.
        # In the loop above, the ray assigned with p=1/v (v is the topmost interface of this block.)
        # To make sure the ray can enter the block, we need to decrease the p a little bit!
        #rps[    gidx_rp] = np.nextafter(rps[gidx_rp],         rps[    gidx_rp+lidx_rp-1] )
        inv_rps[gidx_rp] = np.nextafter(inv_rps[gidx_rp],     inv_rps[gidx_rp+lidx_rp-1] )
        # After decreasing the p a little bit, the ray now can enter the block,
        # so its ib should be the first layer of this block!
        # Hence, Fix the ib for the first ray of this block, as mentioned above.
        ibs[    gidx_rp] = ilayer_start
        #####
        # Store index for this leg due to this block.
        bb[iblock, 0] = bb_base
        bb[iblock, 1] = bb_base + lidx_rp
        bb_base += lidx_rp
        #####
        gidx_rp += lidx_rp
        #
    #rps = rps[:gidx_rp]
    #inv_rps = inv_rps[:gidx_rp]
    #ibs = ibs[:gidx_rp]
    #if debug:
    #    old_rp0, old_rp1 = 0.0, 0.0
    #    for iblock in range(nblocks):
    #        istart = bb[iblock, 0]
    #        iend   = bb[iblock, 1]
    #        tmp_vz = inv_rps[istart:iend]
    #        tmp_rp = np.array(rps[istart:iend])
    #        tmp_ib = ibs[istart:iend]
    #        tmp_z  = z[np.array(tmp_ib)]
    #        ax1.plot(tmp_vz,     tmp_z, 'o', color='c', lw=1, markersize=4)
    #        ax1.plot(1.0/tmp_rp, tmp_z, '.', color='C1', lw=1, markersize=4)
    #        #
    #        ax2.semilogx(1.0/tmp_vz,     tmp_z, 'o', color='c', lw=1, markersize=4)
    #        ax2.semilogx(tmp_rp, tmp_z, '.', color='C1', lw=1, markersize=4)
    #        rp0, rp1 = tmp_rp[0], tmp_rp[-1]
    #        if iblock >0  and old_rp1 <= rp0:
    #            print('wrong', rp0, old_rp1)
    #        old_rp0 = rp0
    #        old_rp1 = rp1
    #######################################################
    return nblocks, bb, inv_rps, ibs
@jit(nopython=True, nogil=True)
def zv2pxt_v3(z, vz, theta_step_deg=0.1):
    nblocks, bb, inv_rps, ibs = zv2p_v3(z, vz, theta_step_deg)
    #
    total_nrp = inv_rps.size
    dist   = np.zeros(total_nrp, dtype=np.float64)
    trvt   = np.zeros(total_nrp, dtype=np.float64)
    leg_start_end_pairs_gind = np.zeros((2, total_nrp), dtype=np.int64)
    leg_start_gind = leg_start_end_pairs_gind[0]
    leg_end_gind   = leg_start_end_pairs_gind[1]
    # Step pxt
    dz = np.diff(z)
    dv = np.diff(vz)
    n_legs =0
    for iblock in range(nblocks):
        g_i0 = bb[iblock, 0]
        g_i1 = bb[iblock, 1]
        for irp in range(g_i0, g_i1):
            dist[irp], trvt[irp] = p2xt_v2(inv_rps[irp], z, vz, dz, dv, ibs[irp])
        ###
        b_inds = split_pxt_legs_v2(dist[g_i0:g_i1])
        ###
        b_inds += g_i0 # adjust to global index
        leg_start_gind[n_legs]                         = g_i0
        leg_start_gind[n_legs+1: n_legs+1+b_inds.size] = b_inds
        leg_end_gind[  n_legs:   n_legs+b_inds.size]   = b_inds+1
        leg_end_gind[               n_legs+b_inds.size]= g_i1
        n_legs += (b_inds.size +1)
        ###
        #print('iblock=', iblock, 'b_inds=', b_inds)
    #print(n_legs, nblocks)
    #print('n_legs=', n_legs )
    #print('leg_start_end_pairs_gind[:,:n_legs]=', leg_start_end_pairs_gind[:,:n_legs] )
    #leg_start_end_pairs_gind = leg_start_end_pairs_gind[:,:n_legs]
    ##
    #if debug:
    #    for (istart, iend) in leg_start_end_pairs_gind.T:
    #        tmp_rp = rps[istart:iend]
    #        tmp_dist = dist[istart:iend]
    #        tmp_trvt = trvt[istart:iend]
    #        ax3.plot(tmp_dist, tmp_trvt, '-', lw=2, markersize=6)
    #        ax4.semilogx(tmp_rp, tmp_dist, '-', lw=2, markersize=6)
    #        ax5.semilogx(tmp_rp, tmp_trvt, '-', lw=2, markersize=6)
    #    ax4.set_xlim(ax2.get_xlim())
    #######################################################
    #if debug:
    #    plt.show()
    return inv_rps, dist, trvt, ibs, leg_start_end_pairs_gind[:,:n_legs].T
# v4
@jit(nopython=True, nogil=True)
def zv2pxt_v4(z, vz, theta_step_deg=0.1):
    nblocks, bb, inv_rps, ibs = zv2p_v3(z, vz, theta_step_deg)
    #
    total_nrp = inv_rps.size
    dist   = np.zeros(total_nrp, dtype=np.float64)
    trvt   = np.zeros(total_nrp, dtype=np.float64)
    leg_start_end_pairs_gind = np.zeros((2, total_nrp), dtype=np.int64)
    leg_start_gind = leg_start_end_pairs_gind[0]
    leg_end_gind   = leg_start_end_pairs_gind[1]
    # Step pxt
    dz = np.diff(z)
    dv = np.diff(vz)
    layer_type = np.zeros(z.size-1, dtype=np.int64)
    for ilayer in range(layer_type.size):
        if np.abs(dv[ilayer]) > 1e-10 and np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 2  # non-constant v layer
        elif np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 1  # constant v layer
    dzdv = np.where(dv!=0.0, dz/dv, 0.0)
    n_legs =0
    for iblock in range(nblocks):
        g_i0 = bb[iblock, 0]
        g_i1 = bb[iblock, 1]
        for irp in range(g_i0, g_i1):
            dist[irp], trvt[irp] = p2xt_v4(inv_rps[irp], z, vz, dz, dv, dzdv, layer_type, ibs[irp])
        ###
        b_inds = split_pxt_legs_v2(dist[g_i0:g_i1])
        ###
        b_inds += g_i0 # adjust to global index
        leg_start_gind[n_legs]                         = g_i0
        leg_start_gind[n_legs+1: n_legs+1+b_inds.size] = b_inds
        leg_end_gind[  n_legs:   n_legs+b_inds.size]   = b_inds+1
        leg_end_gind[               n_legs+b_inds.size]= g_i1
        n_legs += (b_inds.size +1)
        ###
        #print('iblock=', iblock, 'b_inds=', b_inds)
    #print(n_legs, nblocks)
    #print('n_legs=', n_legs )
    #print('leg_start_end_pairs_gind[:,:n_legs]=', leg_start_end_pairs_gind[:,:n_legs] )
    #leg_start_end_pairs_gind = leg_start_end_pairs_gind[:,:n_legs]
    ##
    #if debug:
    #    for (istart, iend) in leg_start_end_pairs_gind.T:
    #        tmp_rp = rps[istart:iend]
    #        tmp_dist = dist[istart:iend]
    #        tmp_trvt = trvt[istart:iend]
    #        ax3.plot(tmp_dist, tmp_trvt, '-', lw=2, markersize=6)
    #        ax4.semilogx(tmp_rp, tmp_dist, '-', lw=2, markersize=6)
    #        ax5.semilogx(tmp_rp, tmp_trvt, '-', lw=2, markersize=6)
    #    ax4.set_xlim(ax2.get_xlim())
    #######################################################
    #if debug:
    #    plt.show()
    return inv_rps, dist, trvt, ibs, leg_start_end_pairs_gind[:,:n_legs].T


#### Ray tracing given distance
@jit(nopython=True, nogil=True)
def single_dist2trvt(target_single_dist, z, vz, inv_rp_legs, dist_legs, trvt_legs, xerr=1e-20, niter=1000):
    if target_single_dist <= 0.0:
        return vz[0], 0.0, 0.0
    s_inv_rp, s_dist, s_trvt = np.nan, np.nan, np.nan
    flag_none_found=True
    dz = np.diff(z)
    dvz = np.diff(vz)
    #inv_vz = 1.0/vz
    #m_rp, m_dist, m_trvt = list(), list(), list()
    for ileg in range(len(inv_rp_legs)):
        inv_rps = inv_rp_legs[ileg]
        dists = dist_legs[ileg]
        trvts = trvt_legs[ileg]
        if (dists[0] <= target_single_dist <= dists[-1]) or (dists[-1] <= target_single_dist <= dists[0]): # within the range of this leg
            if np.abs(target_single_dist-dists[0]) < xerr:
                inv_rp_found   = inv_rps[0]
                dist_found = dists[0]
                trv_found  = trvts[0]
            elif np.abs(target_single_dist-dists[-1]) < xerr:
                inv_rp_found   = inv_rps[-1]
                dist_found = dists[-1]
                trv_found  = trvts[-1]
            else:
                if dists[0] > dists[-1]:
                    inv_rps   = inv_rps[::-1]
                    dists = dists[::-1]
                    trvts = trvts[::-1]
                i1 = np.searchsorted(dists, target_single_dist)
                inv_rp_left, inv_rp_right = inv_rps[i1-1], inv_rps[i1]
                d_left            = dists[i1-1]
                #### start ray tracing with bisection method
                for idx_iter in range(niter):
                    inv_rp_mid = 0.5 * (inv_rp_left + inv_rp_right)
                    d_mid, t_mid = p2xt(inv_rp_mid, z, vz, dz, dvz)
                    #print('  iter:', idx_iter, d_left, d_mid, d_right, '|', rp_left, rp_mid, rp_right, )
                    if np.abs(d_mid-target_single_dist) < xerr:
                        ### found!!!
                        break
                    elif (d_left <= target_single_dist <= d_mid) or (d_mid <= target_single_dist <= d_left):
                        inv_rp_right = inv_rp_mid
                        #d_right = d_mid
                    else:
                        inv_rp_left = inv_rp_mid
                        d_left = d_mid
                inv_rp_found   = inv_rp_mid
                dist_found = d_mid
                trv_found  = t_mid
            ###########
            #print(target_single_dist, dist_found, rp_found, trv_found)
            #m_rp.append(rp_found)
            #m_dist.append(dist_found)
            #m_trvt.append(trv_found)
            if flag_none_found:
                s_inv_rp, s_dist, s_trvt = inv_rp_found, dist_found, trv_found
                flag_none_found = False
            elif trv_found < s_trvt:
                s_inv_rp, s_dist, s_trvt = inv_rp_found, dist_found, trv_found
    return s_inv_rp, s_dist, s_trvt #, (np.array(m_rp), np.array(m_dist), np.array(m_trvt) )
@jit(nopython=True, nogil=True)
def many_dist2trvt(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found and trvt_found correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
    """
    inv_rp_found   = np.zeros(dist.size, dtype=np.float64)
    dist_found = np.zeros(dist.size, dtype=np.float64)
    trvt_found = np.zeros(dist.size, dtype=np.float64)
    _, (inv_rp_legs, dist_legs, trvt_legs) = zv2pxt(z, vz, theta_step_deg=theta_step_deg)
    #### feasible dist ranges
    #raw_dist_ranges = np.sort( [(np.min(it), np.max(it)) for it in dist_legs] )
    #raw_dist_ranges = sorted(raw_dist_ranges)
    #valid_dist_ranges = [raw_dist_ranges[0] ]
    #for it3, it4 in raw_dist_ranges[1:]:
    #    it1, it2 = valid_dist_ranges[-1]   # it1 <= it2 as sorted before
    #    if it3 <= it2 and it2 < it4: # overlap
    #        valid_dist_ranges[-1] = (it1, it4)
    #    else:
    #        valid_dist_ranges.append( (it3, it4) )
    for idx in range(dist.size):
        tmp = single_dist2trvt(dist[idx], z, vz, inv_rp_legs, dist_legs, trvt_legs, xerr, niter)
        inv_rp_found[idx]   = tmp[0]
        dist_found[idx] = tmp[1]
        trvt_found[idx] = tmp[2]
    return inv_rp_found, dist_found, trvt_found
@jit(nopython=True, nogil=True)
def many_dist2trvt_jac(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found, d_trvt_v
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found, trvt_found, d_trvt_v correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
               zeros will be used for the corresponding rows in the d_trvt_v.
    """
    inv_rp_found, dist_found, trvt_found = many_dist2trvt(dist, z, vz, theta_step_deg=theta_step_deg, xerr=xerr, niter=niter)
    pxpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    pxpp     = np.zeros(dist.size, dtype=np.float64)
    ptpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    ptpp     = np.zeros(dist.size, dtype=np.float64)
    d_trvt_v = np.zeros((dist.size, vz.size), dtype=np.float64)
    buf = np.zeros((6, z.size), dtype=np.float64)
    dz = np.diff(z)
    dv = np.diff(vz)
    #inv_vz = 1.0/vz
    for idx in range(inv_rp_found.size):
        inv_rp = inv_rp_found[idx]
        if not np.isnan(inv_rp): ### Nan means no ray exist for this distance
            _, _, pxpv[idx], pxpp[idx], ptpv[idx], ptpp[idx], d_trvt_v[idx] = p2xt_grad(inv_rp, z, vz, dz, dv, buf)
    return inv_rp_found, dist_found, trvt_found, pxpv, pxpp, ptpv, ptpp, d_trvt_v
# v2
@jit(nopython=True, nogil=True)
def single_dist2trvt_v2(target_single_dist, z, vz, dz, dv, many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs, xerr=1e-20, niter=1000):
    s_inv_rp   = np.nan
    s_dist = np.nan
    s_trvt = np.inf # will be set to nan if not found latter
    s_ib   = -1
    nleg = many_ray_ind_pairs.shape[0]
    #### for debug purposes
    #junk_inv_rp_trials = np.zeros(niter, dtype=np.float64)
    #junk_dist_trials   = np.zeros(niter, dtype=np.float64)
    for ileg in range(nleg):
        jdx_start = many_ray_ind_pairs[ileg, 0]
        jdx_end   = many_ray_ind_pairs[ileg, 1]
        inv_rps= many_ray_inv_rps[jdx_start:jdx_end]
        dists = many_ray_dist[jdx_start:jdx_end]
        trvts = many_ray_trvt[jdx_start:jdx_end]
        ibs   = many_ray_ibs[jdx_start:jdx_end]
        if (dists[0] <= target_single_dist <= dists[-1]) or (dists[-1] <= target_single_dist <= dists[0]): # within the range of this leg
            if np.abs(target_single_dist-dists[0]) < xerr:
                inv_rp_found   = inv_rps[0]
                dist_found = dists[0]
                trv_found  = trvts[0]
                ib_found   = ibs[0]
            elif np.abs(target_single_dist-dists[-1]) < xerr:
                inv_rp_found   = inv_rps[-1]
                dist_found = dists[-1]
                trv_found  = trvts[-1]
                ib_found   = ibs[-1]
            else:
                if dists[0] <= dists[-1]:
                    i1 = np.searchsorted(dists, target_single_dist)
                else:
                    i1 = np.searchsorted(dists[::-1], target_single_dist)
                    i1 = dists.size - i1
                ##### Note, inv_rps is always increasing!!!   ibs is also not decreasing!!!
                i0 = i1-1
                # the search range by i0, i1
                inv_rp0 = inv_rps[i0] # so, inv_rp0 <= inv_rp1
                inv_rp1 = inv_rps[i1]
                d0      = dists[i0]
                d1      = dists[i1]
                ib0     = ibs[i0]     # also, ib0 <= ib1
                ib1     = ibs[i1]
                ###### init the bisection search, each step range is left, right
                inv_rp_left = inv_rp0 # also, inv_rp_left <= inv_rp_right at all step
                inv_rp_right= inv_rp1
                d_left  = d0
                d_right = d1
                for idx_iter in range(niter):
                    inv_rp_mid = 0.5 * (inv_rp_left + inv_rp_right)
                    ib_mid = ib1 if inv_rp0 < inv_rp_mid else ib0 # clever fix!!! (should I use  abs(inv_rp0-inv_rp_mid)~0.0? )
                    d_mid, t_mid = p2xt_v2(inv_rp_mid, z, vz, dz, dv, ib_mid)
                    ###
                    #junk_inv_rp_trials[idx_iter] = inv_rp_mid
                    #junk_dist_trials[idx_iter]   = d_mid
                    ###
                    if  (np.abs(d_mid-target_single_dist) < xerr) or (np.abs(d_left - d_right) < xerr) or (inv_rp_left >= inv_rp_right):
                        ### found!!!
                        break
                    elif (d_left <= target_single_dist <= d_mid) or (d_mid <= target_single_dist <= d_left):
                        inv_rp_right = inv_rp_mid
                        d_right      = d_mid
                    else:
                        inv_rp_left = inv_rp_mid
                        d_left = d_mid
                inv_rp_found= inv_rp_mid
                ib_found    = ib_mid
                dist_found  = d_mid
                trv_found   = t_mid
                ####
                #### for debug purposes
                vtop = vz[ib_found]
                vbot = vz[ib_found+1]
                if not (vtop < inv_rp_found <= vbot):
                    ####
                    # 1. The cos issue disappear if the ib_found is smaller than its correct value.
                    #    Which means vtop < vbot <= inv_rp_found
                    # 2. The cos issue appear if the ib_found is larger than its correct value.
                    #    Which means inv_rp_found <= vtop < vbot
                    ####
                    print('Error in single_dist2trvt_v2:')
                    print('Original search range')
                    print('inv_rp0=', inv_rp0,   'inv_rp1=  ', inv_rp1)
                    print('dist0=  ', dists[i0], 'dists[i1]=', dists[i1])
                    print('ib0=', ib0, 'ib1=', ib1)
                    ####
                    print('After bisection search:')
                    print('ib_found=    ', ib_found)
                    print('vtop=        ', vz[ib_found],   '=vz[ib_found]')
                    print('inv_rp_found=', inv_rp_found)
                    print('vbot=        ', vz[ib_found+1], '=vz[ib_found+1]')
                    ##
                    #print('junk_inv_rp_trials=', junk_inv_rp_trials[:idx_iter+1])
                    #
                    print('target_single_dist=', target_single_dist)
                    #print('junk_dist_trials  =', junk_dist_trials[:idx_iter+1])
                    pass
            ###########
            if trv_found < s_trvt:
                s_inv_rp   = inv_rp_found
                s_dist = dist_found
                s_trvt = trv_found
                s_ib   = ib_found
    if np.isinf(s_trvt):
        s_trvt = np.nan
    return s_inv_rp, s_dist, s_trvt, s_ib
@jit(nopython=True, nogil=True)
def many_dist2trvt_v2(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found and trvt_found correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
    """
    inv_rp_found   = np.zeros(dist.size, dtype=np.float64)
    dist_found = np.zeros(dist.size, dtype=np.float64)
    trvt_found = np.zeros(dist.size, dtype=np.float64)
    ib_found   = np.zeros(dist.size, dtype=np.int64)
    many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs = zv2pxt_v2(z, vz, theta_step_deg=theta_step_deg)
    dz = np.diff(z)
    dv = np.diff(vz)
    for idx in range(dist.size):
        inv_rp_found[idx], dist_found[idx], trvt_found[idx], ib_found[idx] = single_dist2trvt_v2(dist[idx], z, vz, dz, dv, many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs, xerr, niter)
    #print('v2 ', np.sum(many_ray_rps))
    #print('v2 ', np.sum(many_ray_dist))
    #print('v2 ', np.sum(many_ray_trvt))
    #print('v2 ', np.sum(many_ray_ibs))
    #print('v2 ', dist, dist_found)
    return inv_rp_found, dist_found, trvt_found, ib_found
@jit(nopython=True, nogil=True)
def many_dist2trvt_jac_v2(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found, d_trvt_v
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found, trvt_found, d_trvt_v correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
               zeros will be used for the corresponding rows in the d_trvt_v.
    """
    inv_rp_found, dist_found, trvt_found, ib_found = many_dist2trvt_v2(dist, z, vz, theta_step_deg=theta_step_deg, xerr=xerr, niter=niter)
    pxpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    pxpp     = np.zeros(dist.size, dtype=np.float64)
    ptpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    ptpp     = np.zeros(dist.size, dtype=np.float64)
    d_trvt_v = np.zeros((dist.size, vz.size), dtype=np.float64)
    dz = np.diff(z)
    dv = np.diff(vz)
    buf = np.zeros((6, z.size), dtype=np.float64)
    for idx in range(inv_rp_found.size):
        inv_rp = inv_rp_found[idx]
        if (not np.isnan(inv_rp)) and (ib_found[idx] !=-1): ### Nan means no ray exist for this distance
            _, _, pxpv[idx], pxpp[idx], ptpv[idx], ptpp[idx], d_trvt_v[idx] = p2xt_grad_v2(inv_rp, z, vz, dz, dv, ib_found[idx], buf)
    return inv_rp_found, dist_found, trvt_found, pxpv, pxpp, ptpv, ptpp, d_trvt_v
# v3
@jit(nopython=True, nogil=True)
def many_dist2trvt_v3(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found and trvt_found correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
    """
    inv_rp_found   = np.zeros(dist.size, dtype=np.float64)
    dist_found = np.zeros(dist.size, dtype=np.float64)
    trvt_found = np.zeros(dist.size, dtype=np.float64)
    ib_found   = np.zeros(dist.size, dtype=np.int64)
    many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs = zv2pxt_v3(z, vz, theta_step_deg=theta_step_deg)
    dz = np.diff(z)
    dv = np.diff(vz)
    for idx in range(dist.size):
        inv_rp_found[idx], dist_found[idx], trvt_found[idx], ib_found[idx] = single_dist2trvt_v2(dist[idx], z, vz, dz, dv, many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs, xerr, niter)
    #print('v3 ', np.sum(many_ray_rps))
    #print('v3 ', np.sum(many_ray_dist))
    #print('v3 ', np.sum(many_ray_trvt))
    #print('v3 ', np.sum(many_ray_ibs))
    #print('v3 ', dist, dist_found)
    return inv_rp_found, dist_found, trvt_found, ib_found
@jit(nopython=True, nogil=True)
def many_dist2trvt_jac_v3(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found, d_trvt_v
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found, trvt_found, d_trvt_v correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
               zeros will be used for the corresponding rows in the d_trvt_v.
    """
    inv_rp_found, dist_found, trvt_found, ib_found = many_dist2trvt_v3(dist, z, vz, theta_step_deg=theta_step_deg, xerr=xerr, niter=niter)
    pxpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    pxpp     = np.zeros(dist.size, dtype=np.float64)
    ptpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    ptpp     = np.zeros(dist.size, dtype=np.float64)
    d_trvt_v = np.zeros((dist.size, vz.size), dtype=np.float64)
    dz = np.diff(z)
    dv = np.diff(vz)
    buf = np.zeros((6, z.size), dtype=np.float64)
    for idx in range(inv_rp_found.size):
        inv_rp = inv_rp_found[idx]
        if (not np.isnan(inv_rp)) and (ib_found[idx] !=-1): ### Nan means no ray exist for this distance
            _, _, pxpv[idx], pxpp[idx], ptpv[idx], ptpp[idx], d_trvt_v[idx] = p2xt_grad_v2(inv_rp, z, vz, dz, dv, ib_found[idx], buf)
    return inv_rp_found, dist_found, trvt_found, pxpv, pxpp, ptpv, ptpp, d_trvt_v
# v4
@jit(nopython=True, nogil=True) #, fastmath=True)
def single_dist2trvt_v4(target_single_dist, z, vz, dz, dv, dzdv, layer_type, many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs, xerr=1e-20, niter=1000):
    s_inv_rp   = np.nan
    s_dist = np.nan
    s_trvt = np.inf # will be set to nan if not found latter
    s_ib   = -1
    nleg = many_ray_ind_pairs.shape[0]
    #### for debug purposes
    #junk_inv_rp_trials = np.zeros(niter, dtype=np.float64)
    #junk_dist_trials   = np.zeros(niter, dtype=np.float64)
    #n_total_trials = 0
    for ileg in range(nleg):
        jdx_start = many_ray_ind_pairs[ileg, 0]
        jdx_end   = many_ray_ind_pairs[ileg, 1]
        inv_rps= many_ray_inv_rps[jdx_start:jdx_end]
        dists = many_ray_dist[jdx_start:jdx_end]
        trvts = many_ray_trvt[jdx_start:jdx_end]
        ibs   = many_ray_ibs[jdx_start:jdx_end]
        if (dists[0] <= target_single_dist <= dists[-1]) or (dists[-1] <= target_single_dist <= dists[0]): # within the range of this leg
            if np.abs(target_single_dist-dists[0]) < xerr:
                inv_rp_found   = inv_rps[0]
                dist_found = dists[0]
                trv_found  = trvts[0]
                ib_found   = ibs[0]
            elif np.abs(target_single_dist-dists[-1]) < xerr:
                inv_rp_found   = inv_rps[-1]
                dist_found = dists[-1]
                trv_found  = trvts[-1]
                ib_found   = ibs[-1]
            else:
                if dists[0] <= dists[-1]:
                    i1 = np.searchsorted(dists, target_single_dist)
                else:
                    i1 = np.searchsorted(dists[::-1], target_single_dist)
                    i1 = dists.size - i1
                ##### Note, inv_rps is always increasing!!!   ibs is also not decreasing!!!
                i0 = i1-1
                # the search range by i0, i1
                inv_rp0 = inv_rps[i0] # so, inv_rp0 <= inv_rp1
                inv_rp1 = inv_rps[i1]
                d0      = dists[i0]
                d1      = dists[i1]
                ib0     = ibs[i0]     # also, ib0 <= ib1
                ib1     = ibs[i1]
                ###### init the bisection search, each step range is left, right
                inv_rp_left = inv_rp0 # also, inv_rp_left <= inv_rp_right at all step
                inv_rp_right= inv_rp1
                d_left  = d0
                d_right = d1
                ######################################################################################################
                #inv_rp_current = 0.5 * (inv_rp_left + inv_rp_right)
                #ib_current = ib1 if inv_rp0 < inv_rp_current else ib0
                #d_current, _, grad_p_current = p2xt_p_grad_v4(inv_rp_current, z, vz, dz, dv, dzdv, layer_type, ib_current)
                #grad_invp_current = -grad_p_current / (inv_rp_current * inv_rp_current) # Derived value
                #for idx_iter in range(niter):
                #    if grad_invp_current != 0.0:
                #        inv_rp_new = inv_rp_current + (target_single_dist - d_current) / grad_invp_current
                #        if inv_rp_new < inv_rp_left or inv_rp_new > inv_rp_right:
                #            inv_rp_new = 0.5*(inv_rp_left + inv_rp_right) # Default to safe step
                #        #if inv_rp_new < inv_rp_left:
                #        #    print(f'iter={idx_iter}: going out of bound, reset to left')
                #        #    inv_rp_new = inv_rp_left
                #        #elif inv_rp_new > inv_rp_right:
                #        #    print(f'iter={idx_iter}: going out of bound, reset to right')
                #        #    inv_rp_new = inv_rp_right
                #    else:
                #        inv_rp_new = 0.5*(inv_rp_left + inv_rp_right) # Default to safe step
                #    ib_new = ib1 if inv_rp0 < inv_rp_new else ib0
                #    d_new, t_new, grad_p_new = p2xt_p_grad_v4(inv_rp_new, z, vz, dz, dv, dzdv, layer_type, ib_new)
                #    if (np.abs(d_new-target_single_dist) < xerr) or (np.abs(d_left - d_right) < xerr) or (inv_rp_left >= inv_rp_right):
                #        ### found!!!
                #        break
                #    elif (d_left <= target_single_dist <= d_new) or (d_new <= target_single_dist <= d_left):
                #        inv_rp_right = inv_rp_new
                #        d_right      = d_new
                #    else:
                #        inv_rp_left = inv_rp_new
                #        d_left = d_new
                #    ####
                #    inv_rp_current = inv_rp_new
                #    d_current = d_new
                #    grad_p_current = grad_p_new
                #    grad_invp_current = -grad_p_current / (inv_rp_current * inv_rp_current)
                #    ib_current = ib_new
                #inv_rp_found= inv_rp_new
                #ib_found    = ib_new
                #dist_found  = d_new
                #trv_found   = t_new
                ######################################################################################################
                inv_rp_next = 0.5*(inv_rp_left + inv_rp_right)
                for idx_iter in range(niter):
                    inv_rp_cur = inv_rp_next
                    ib_cur = ib1 if inv_rp0 < inv_rp_cur else ib0 # clever fix!!! (should I use  abs(inv_rp0-inv_rp_mid)~0.0? )
                    d_cur, t_cur, grad_p = p2xt_grad_ponly_v4(inv_rp_cur, z, vz, dz, dv, dzdv, layer_type, ib_cur)
                    ###########
                    if  (np.abs(d_cur-target_single_dist) < xerr) or (np.abs(d_left - d_right) < xerr) or (inv_rp_left >= inv_rp_right):
                        ### found!!!
                        break
                    elif (d_left <= target_single_dist <= d_cur) or (d_cur <= target_single_dist <= d_left):
                        inv_rp_right = inv_rp_cur
                        d_right      = d_cur
                    else:
                        inv_rp_left = inv_rp_cur
                        d_left = d_cur
                    ########### newton update
                    grad_invp = -grad_p/( inv_rp_cur * inv_rp_cur )
                    if grad_invp != 0.0:
                        inv_rp_next =  inv_rp_cur + (target_single_dist-d_cur)/grad_invp
                        if inv_rp_next < inv_rp_left or inv_rp_next > inv_rp_right: # safe guard
                            #print(f'iter={idx_iter}: going out of bound, reset to mid')
                            inv_rp_next = 0.5 * (inv_rp_left + inv_rp_right)
                    else:
                        inv_rp_next = 0.5 * (inv_rp_left + inv_rp_right)
                inv_rp_found= inv_rp_cur
                ib_found    = ib_cur
                dist_found  = d_cur
                trv_found   = t_cur
                ######################################################################################################
                #print('inv_p left and right', inv_rp_left, inv_rp_right)
                #print('dist left and right ', d_left, d_right)
                #print('target dist         ', target_single_dist)
                #inv_rp_new = inv_rp_left #0.5*(inv_rp_left + inv_rp_right)
                #for idx_iter in range(niter):
                #    #print('iter', idx_iter)
                #    ib_new = ib1 if inv_rp0 < inv_rp_new else ib0 # clever fix!!! (should I use  abs(inv_rp0-inv_rp_mid)~0.0? )
                #    d_new, t_new, grad_p = p2xt_p_grad_v4(inv_rp_new, z, vz, dz, dv, dzdv, layer_type, ib_new)
                #    grad_invp = -grad_p/( inv_rp_new * inv_rp_new )
                #    #print('\tgrad=', grad)
                #    #####
                #    if grad_invp != 0.0:
                #        #print('\told=', inv_rp_new)
                #        inv_rp_new = inv_rp_new + (target_single_dist-d_new)/grad_invp
                #        #print('\tnew=', inv_rp_new)
                #    else:
                #        inv_rp_new = 0.5 * (inv_rp_left + inv_rp_right)
                #    #####
                #    if inv_rp_new < inv_rp_left:
                #        #print('\tgoint out of left bound, reset to left bound')
                #        inv_rp_new = inv_rp_left
                #    elif inv_rp_new > inv_rp_right:
                #        #print('\tgoint out of right bound, reset to right bound')
                #        inv_rp_new = inv_rp_right
                #    ##########################
                #    ib_new = ib1 if inv_rp0 < inv_rp_new else ib0 # clever fix!!! (should I use  abs(inv_rp0-inv_rp_mid)~0.0? )
                #    d_new, t_new, _ = p2xt_p_grad_v4(inv_rp_new, z, vz, dz, dv, dzdv, layer_type, ib_new)
                #    if  (np.abs(d_new-target_single_dist) < xerr) or (np.abs(d_left - d_right) < xerr) or (inv_rp_left >= inv_rp_right):
                #        ### found!!!
                #        break
                #inv_rp_found= inv_rp_new
                #ib_found    = ib_new
                #dist_found  = d_new
                #trv_found   = t_new
                ######################################################################################################
                #for idx_iter in range(niter):
                #    # n_total_trials +=1
                #    inv_rp_mid = 0.5 * (inv_rp_left + inv_rp_right)
                #    ib_mid = ib1 if inv_rp0 < inv_rp_mid else ib0 # clever fix!!! (should I use  abs(inv_rp0-inv_rp_mid)~0.0? )
                #    #d_mid, t_mid = p2xt_v4(inv_rp_mid, z, vz, dz, dv, dzdv, layer_type, ib_mid)
                #    d_mid, t_mid, _ = p2xt_p_grad_v4(inv_rp_mid, z, vz, dz, dv, dzdv, layer_type, ib_mid)
                #    ###
                #    #junk_inv_rp_trials[idx_iter] = inv_rp_mid
                #    #junk_dist_trials[idx_iter]   = d_mid
                #    ###
                #    if  (np.abs(d_mid-target_single_dist) < xerr) or (np.abs(d_left - d_right) < xerr) or (inv_rp_left >= inv_rp_right):
                #        ### found!!!
                #        break
                #    elif (d_left <= target_single_dist <= d_mid) or (d_mid <= target_single_dist <= d_left):
                #        inv_rp_right = inv_rp_mid
                #        d_right      = d_mid
                #    else:
                #        inv_rp_left = inv_rp_mid
                #        d_left = d_mid
                #inv_rp_found= inv_rp_mid
                #ib_found    = ib_mid
                #dist_found  = d_mid
                #trv_found   = t_mid
                ######################################################################################################
                ####
                #### for debug purposes
                #vtop = vz[ib_found]
                #vbot = vz[ib_found+1]
                #if not (vtop < inv_rp_found <= vbot):
                #    ####
                #    # 1. The cos issue disappear if the ib_found is smaller than its correct value.
                #    #    Which means vtop < vbot <= inv_rp_found
                #    # 2. The cos issue appear if the ib_found is larger than its correct value.
                #    #    Which means inv_rp_found <= vtop < vbot
                #    ####
                #    print('Error in single_dist2trvt_v2:')
                #    print('Original search range')
                #    print('inv_rp0=', inv_rp0,   'inv_rp1=  ', inv_rp1)
                #    print('dist0=  ', dists[i0], 'dists[i1]=', dists[i1])
                #    print('ib0=', ib0, 'ib1=', ib1)
                #    ####
                #    print('After bisection search:')
                #    print('ib_found=    ', ib_found)
                #    print('vtop=        ', vz[ib_found],   '=vz[ib_found]')
                #    print('inv_rp_found=', inv_rp_found)
                #    print('vbot=        ', vz[ib_found+1], '=vz[ib_found+1]')
                #    ##
                #    #print('junk_inv_rp_trials=', junk_inv_rp_trials[:idx_iter+1])
                #    #
                #    print('target_single_dist=', target_single_dist)
                #    #print('junk_dist_trials  =', junk_dist_trials[:idx_iter+1])
                #    pass
            ###########
            if trv_found < s_trvt:
                s_inv_rp   = inv_rp_found
                s_dist = dist_found
                s_trvt = trv_found
                s_ib   = ib_found
    #print('n_total_trials=', n_total_trials )
    if np.isinf(s_trvt):
        s_trvt = np.nan
    return s_inv_rp, s_dist, s_trvt, s_ib
@jit(nopython=True, nogil=True)
def many_dist2trvt_v4(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found and trvt_found correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
    """
    array_inv_rp_found   = np.zeros(dist.size, dtype=np.float64)
    array_dist_found = np.zeros(dist.size, dtype=np.float64)
    array_trvt_found = np.zeros(dist.size, dtype=np.float64)
    array_ib_found   = np.zeros(dist.size, dtype=np.int64)
    many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs = zv2pxt_v4(z, vz, theta_step_deg=theta_step_deg)
    nleg = many_ray_ind_pairs.shape[0]
    #### for debug purposes
    dz = np.diff(z)
    dv = np.diff(vz)
    layer_type = np.zeros(z.size-1, dtype=np.int64)
    for ilayer in range(layer_type.size):
        if np.abs(dv[ilayer]) > 1e-10 and np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 2  # non-constant v layer
        elif np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 1  # constant v layer
    dzdv = np.where(dv!=0.0, dz/dv, 0.0)
    for idx in range(dist.size):
        array_inv_rp_found[idx], array_dist_found[idx], array_trvt_found[idx], array_ib_found[idx] = single_dist2trvt_v4(dist[idx], z, vz, dz, dv, dzdv, layer_type, many_ray_inv_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs, xerr, niter)
    return array_inv_rp_found, array_dist_found, array_trvt_found, array_ib_found
@jit(nopython=True, nogil=True)
def many_dist2trvt_jac_v4(dist, z, vz, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found, d_trvt_v
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found, trvt_found, d_trvt_v correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
               zeros will be used for the corresponding rows in the d_trvt_v.
    """
    inv_rp_found, dist_found, trvt_found, ib_found = many_dist2trvt_v4(dist, z, vz, theta_step_deg=theta_step_deg, xerr=xerr, niter=niter)
    pxpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    pxpp     = np.zeros(dist.size, dtype=np.float64)
    ptpv     = np.zeros((dist.size, vz.size), dtype=np.float64)
    ptpp     = np.zeros(dist.size, dtype=np.float64)
    d_trvt_v = np.zeros((dist.size, vz.size), dtype=np.float64)
    dz = np.diff(z)
    dv = np.diff(vz)
    layer_type = np.zeros(z.size-1, dtype=np.int64)
    for ilayer in range(layer_type.size):
        if np.abs(dv[ilayer]) > 1e-10 and np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 2  # non-constant v layer
        elif np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 1  # constant v layer
    dzdv = np.where(dv!=0.0, dz/dv, 0.0)
    buf = np.zeros((6, z.size), dtype=np.float64)
    for idx in range(inv_rp_found.size):
        inv_rp = inv_rp_found[idx]
        if (not np.isnan(inv_rp)) and (ib_found[idx] !=-1): ### Nan means no ray exist for this distance
            _, _, pxpv[idx], pxpp[idx], ptpv[idx], ptpp[idx], d_trvt_v[idx] = p2xt_grad_v4(inv_rp, z, vz, dz, dv, dzdv, layer_type, ib_found[idx], buf)
    return inv_rp_found, dist_found, trvt_found, pxpv, pxpp, ptpv, ptpp, d_trvt_v




#### Get OC reference models. Return R(km), Vp(km/s) with decreasing R!
__ak135_oc_fnm = '%s/dataset/models/oc_model_ak135.txt' % sacpy.__path__[0]
def rd_ak135_OC_model(fnm=__ak135_oc_fnm):
    tab = np.loadtxt(fnm, comments='#')
    depth = tab[:, 0]
    r = 6371.0 - depth
    vp = tab[:, 2]
    return r, vp

__prem_oc_fnm = '%s/dataset/models/oc_model_prem.txt' % sacpy.__path__[0]
def rd_prem_OC_model(fnm=__prem_oc_fnm):
    tab = np.loadtxt(fnm, comments='#')
    depth = tab[:, 0]
    r = 6371.0 - depth
    vp = tab[:, 2]
    return r, vp

__prem_fnm = '%s/dataset/models/taup_prem.txt' % sacpy.__path__[0]
def rd_prem_model(fnm=__prem_fnm): # return r, vp, vs, icmb_index, iicb_index # mantle is [:icmb], OC is [icmb:iicb], IC is [iicb:]
    tab = np.loadtxt(fnm, comments='#')
    r  = tab[:, 1]
    vp = tab[:, 2]
    vs = tab[:, 3]
    ###
    # find CMB and ICB
    for ir in range(r.size-1):
        if vs[ir] < 1e-10:
            icmb = ir # mantle is [:icmb], core is [icmb:]
            break
    for ir in range(icmb, r.size-1):
        if vs[ir] > 1e-10:
            iicb = ir # outer core is [icmb:iicb], inner core is [iicb:]
            break
    ###
    return r, vp, vs, icmb, iicb

#### OBJECTIVE function and gradient function for optimization
def get_obj_and_grad_func(dist, trvt_obs, std, model_z, model_vz_ref,
                          alpha=0.0, beta=0.0, theta_step_deg=0.1, xerr=1e-20, niter=1000):
    """
    Generate two functions: `obj_func(m)` and `obj_grad(m)` for optimization.
    dist:         the distances where the data are observed (in km)
    trvt_obs:     the observed traveltimes at the distances (in s)
    std:          the standard deviation of the traveltime observations (in s)
    model_z:      the model depth grid from surface to depth (in km, positive upward)
    model_vz_ref: the reference model velocities at model_z (in km/s)
    """
    #### Test functions
    def test_my_model(dvz):
        vz = dvz + model_vz_ref
        a, b = vz
        y = a*a*dist + b
        return y
    def test_my_model_jac(dvz):
        vz = dvz + model_vz_ref
        a, b = vz
        jac = np.zeros((dist.size, dvz.size), dtype=np.float64)
        jac[:,0] = 2.0*a*dist
        jac[:,1] = 1.0
        return jac
    ####
    @jit(nopython=True, nogil=True)
    def my_model(dvz):
        tmp     = many_dist2trvt_v4(dist, model_z, dvz+model_vz_ref, theta_step_deg=theta_step_deg, xerr=xerr, niter=niter)
        d_syn   = tmp[2]
        idx_nan = np.where( np.isnan(d_syn) )[0]
        d_syn[idx_nan] = 1e-2
        return d_syn
    @jit(nopython=True, nogil=True)
    def my_model_jac(dvz):
        tmp     = many_dist2trvt_jac_v4(dist, model_z, dvz+model_vz_ref, theta_step_deg=theta_step_deg, xerr=xerr, niter=niter)
        d_syn   = tmp[2]
        jac     = tmp[7]
        # fix nan for none-exist distance given this dvz model
        idx_nan = np.where( np.isnan(d_syn) )[0]
        d_syn[idx_nan] = 1e-2
        jac[idx_nan,:] = 0.0
        return d_syn, jac
    inv_var = 1.0/(std*std)
    model_sz = len(model_z)
    ######### objective functions #########
    @jit(nopython=True, nogil=True)
    def obj_data_diff(dvz): # dvz is the perturbation from model_vz_ref
        trvt_syn = my_model(dvz)
        idx_nan = np.where( np.isnan(trvt_syn) )[0]
        trvt_syn[idx_nan] = -1e2 # set a never impossible value so that the misfit is very large
        tmp = (trvt_obs-trvt_syn)
        return np.sum(tmp*tmp*inv_var)
    @jit(nopython=True, nogil=True)
    def obj_model_d0(dvz):  # dvz is the perturbation from model_vz_ref
        return np.sum(dvz*dvz)
    @jit(nopython=True, nogil=True)
    def obj_model_d1(dvz):  # dvz is the perturbation from model_vz_ref
        tmp = np.diff(dvz)
        return np.sum(tmp*tmp)
    #########
    @jit(nopython=True, nogil=True)
    def obj_func(dvz):
        v = obj_data_diff(dvz)
        if alpha > 0.0:
            v += alpha * obj_model_d0(dvz)
        if beta > 0.0:
            v += beta  * obj_model_d1(dvz)
        return v
    ######### objective gradient #########
    @jit(nopython=True, nogil=True)
    def obj_grad(dvz):
        trvt_syn, jac = my_model_jac(dvz)
        tmp = 2*(trvt_syn-trvt_obs)*inv_var
        #grad = np.zeros(model_sz, dtype=np.float64)
        #for j in range(model_sz):
        #    grad[j] = np.sum(tmp * jac[:,j])
        grad = tmp @ jac
        #####
        if alpha > 0.0:
            grad2 = 2*alpha*dvz
            grad += grad2
        #####
        if beta > 0.0:
            tmp = np.diff( dvz )
            grad3 = np.zeros(model_sz, dtype=np.float64)
            grad3[1:-1] =  tmp[:-1] - tmp[1:]
            grad3[0]    = -tmp[0]
            grad3[-1]   =  tmp[-1]
            grad3      *= (2*beta)
            grad += grad3
        return grad
    return obj_func, obj_grad, (obj_data_diff, obj_model_d0, obj_model_d1)

########### above this are black box functions ###########
########### below is an example to use the above functions ###########
def benchmark_bfgs_inv():
    from scipy.optimize import minimize
    z      = np.array([0.,  -200])
    vz_ref = np.array([6.0, 8.0])
    #(inv_p1, x1, t1), (leg_p1, leg_x1, leg_t1) = zv2pxt(z, vz_ref, theta_step_deg=1)
    dist = np.array( (50.0, 1000))
    a, b = vz_ref
    trvt_obs = a*a*dist + b
    _,_,trvt_obs,_ = many_dist2trvt_v4(dist, z, vz_ref, 0.1, 1e-3, 1000)
    std = 1.0
    #####
    dvz_init = np.zeros(z.size, dtype=np.float64) + 0.5
    dvz_lower = -np.ones(z.size, dtype=np.float64)- 1.0
    dvz_upper = np.ones(z.size, dtype=np.float64) + 1.0
    bounds = np.array((dvz_lower, dvz_upper)).T
    #####
    tmp = get_obj_and_grad_func(dist, trvt_obs, std, z, vz_ref,
                                alpha=1.0, beta=1.0,
                                theta_step_deg=0.1, xerr=1e-20, niter=1000)
    obj_func, obj_grad, _ = tmp
    res = minimize(obj_func, dvz_init, method='L-BFGS-B', jac=obj_grad, bounds=bounds)
                        #options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': False, 'maxiter': 1000})
    print(res.x)
    pass
def benchmark_speedup():
    from sacpy import utils
    r, vr = rd_prem_OC_model()
    ####
    #print('1/vz=', 1.0/vz[:10])
    #z  =  np.array([0., -100, -200, -300, -400])
    #vz =  np.array([6.0, 7.0, 6.5, 9.0, 10.0])
    xerr = 1e-6
    niter = 1000
    if True: # speed up zv2pxt vs. zv2pxt_v2
        r_tmp, vr_tmp = denser_xy(r, vr, 50.0)  # make the model denser
        vr_tmp += (np.random.random(vr_tmp.size)-0.5)#*10  # add some noise
        z, vz   = flatten(r_tmp, vr_tmp, 6371.0)
        print()
        (p1, x1, t1), (leg_p1, leg_x1, leg_t1) = zv2pxt(z, vz, theta_step_deg=0.1)  # warm up numba jit
        with utils.Timer("zv2pxt: "):
            for _ in range(1):
                zv2pxt(z, vz, theta_step_deg=0.1)
        ####
        p2, x2, t2, _, ind_pairs = zv2pxt_v2(z, vz, theta_step_deg=0.1)  # warm up numba jit
        with utils.Timer("zv2pxt_v2: "):
            for _ in range(1):
                zv2pxt_v2(z, vz, theta_step_deg=0.1)
        leg_p2 = [p2[i0:i1] for (i0, i1) in ind_pairs]
        leg_x2 = [x2[i0:i1] for (i0, i1) in ind_pairs]
        leg_t2 = [t2[i0:i1] for (i0, i1) in ind_pairs]
        ####
        p3, x3, t3, _, ind_pairs = zv2pxt_v3(z, vz, theta_step_deg=0.1)  # warm up numba jit
        with utils.Timer("zv2pxt_v3: "):
            for _ in range(1):
                zv2pxt_v3(z, vz, theta_step_deg=0.1)
        leg_p3 = [p3[i0:i1] for (i0, i1) in ind_pairs]
        leg_x3 = [x3[i0:i1] for (i0, i1) in ind_pairs]
        leg_t3 = [t3[i0:i1] for (i0, i1) in ind_pairs]
        ####
        p4, x4, t4, _, ind_pairs = zv2pxt_v4(z, vz, theta_step_deg=0.1)  # warm up numba jit
        with utils.Timer("zv2pxt_v4: "):
            for _ in range(1):
                zv2pxt_v4(z, vz, theta_step_deg=0.1)
        leg_p4 = [p4[i0:i1] for (i0, i1) in ind_pairs]
        leg_x4 = [x4[i0:i1] for (i0, i1) in ind_pairs]
        leg_t4 = [t4[i0:i1] for (i0, i1) in ind_pairs]
        ####
        print('min,max, mean dist step=', np.max(np.abs(np.diff(x1))), np.min(np.abs(np.diff(x1))), np.mean(np.abs(np.diff(x1))) )
        ####
        sum_leg_p1 = np.sum( np.unique(p1))
        sum_leg_p2 = np.sum( np.unique(p2))
        sum_leg_p3 = np.sum( np.unique(p3))
        sum_leg_x1 = np.sum( np.unique(x1))
        sum_leg_x2 = np.sum( np.unique(x2))
        sum_leg_x3 = np.sum( np.unique(x3))
        sum_leg_t1 = np.sum( np.unique(t1))
        sum_leg_t2 = np.sum( np.unique(t2))
        sum_leg_t3 = np.sum( np.unique(t3))
        sum_leg_p4 = np.sum( np.unique(p4))
        sum_leg_x4 = np.sum( np.unique(x4))
        sum_leg_t4 = np.sum( np.unique(t4))
        print('sum_leg_p=', sum_leg_p1-sum_leg_p2, sum_leg_p3-sum_leg_p2, sum_leg_p4-sum_leg_p2)
        print('sum_leg_x=', sum_leg_x1-sum_leg_x2, sum_leg_x3-sum_leg_x2, sum_leg_x4-sum_leg_x2)
        print('sum_leg_t=', sum_leg_t1-sum_leg_t2, sum_leg_t3-sum_leg_t2, sum_leg_t4-sum_leg_t2)
        ####
        #nlegs = len(leg_p1)
        #for ileg in range(nlegs):
        #    idx_dif = np.where( (leg_x1[ileg] != leg_x2[ileg]) | (leg_t1[ileg] != leg_t2[ileg]) | (leg_p1[ileg] != leg_p2[ileg]) )[0]
        #    if idx_dif.size >0:
        #        print('ileg=', ileg)
        #        print(leg_p1[ileg][idx_dif], leg_x1[ileg][idx_dif])
        #        print(leg_p2[ileg][idx_dif], leg_x2[ileg][idx_dif])
    if True:
        print()
        ####
        for _ in range(1):
            r_tmp, vr_tmp = denser_xy(r, vr, 50.0)  # make the model denser
            vr_tmp += (np.random.random(vr_tmp.size)-0.5)  # add some noise
            z, vz   = flatten(r_tmp, vr_tmp, 6371.0)
            #print(vz.size )
            #####
            #z  = z[ :20]
            #vz = vz[:20]
            #np.set_printoptions(formatter={'float': '{},'.format})
            #print('z=', z)
            #print('vz=', vz)
            if False:
                z= [-3852.697555495905, -3944.898875244877, -4000.8670243444526,
                -4095.253935881583, -4191.06025257107, -4288.329317516324,
                -4387.1064899369585, -4487.439272180973, -4589.37744689915,
                -4692.973225373261, -4798.2814081044235, -4905.35955889809,
                -5014.26819383019, -5125.070986647509, -5237.834992347792,
                -5352.630890905355, -5469.533253360436, -5588.620832780859,
                -5709.976882939154, -5833.689507934748,]
                vz= [15.9642797106931, 16.104160087773842, 14.428167245426037,
                14.088599608649544, 17.714183238178162, 16.603439854709396,
                18.415389848187186, 17.942232863452272, 16.817860846322016,
                17.528723525208527, 20.020772624242746, 20.542904333799978,
                18.14329491573749, 18.1894034724067, 18.995305353324767,
                21.0653837297101, 23.74491004899037, 24.309326223596013,
                25.05557961461338, 21.666028903590306,]
                z = np.array(z)
                vz= np.array(vz)
            ####
            _,x,_,_,_= zv2pxt_v2(z, vz, theta_step_deg=0.1)  # warm up numba jit
            xmin, xmax = np.min(x), np.max(x)
            xmin += (xmax - xmin)*0.1
            xmax -= (xmax - xmin)*0.1
            dist = np.arange(xmin, xmax, 10.0) #* (np.pi/180.0) * 6371.0  # in km
            ####
            rp_found1, dist_found1, trvt_found1 = many_dist2trvt(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            idx_nan = np.where( np.isnan(rp_found1) )[0]
            rp_found1[idx_nan] = 0.0
            dist_found1[idx_nan] = 0.0
            trvt_found1[idx_nan] = 0.0
            with utils.Timer("many_dist2trvt: "):
                for _ in range(1):
                    many_dist2trvt(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            ####
            rp_found2, dist_found2, trvt_found2, _ = many_dist2trvt_v2(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            idx_nan = np.where( np.isnan(rp_found2) )[0]
            rp_found2[idx_nan] = 0.0
            dist_found2[idx_nan] = 0.0
            trvt_found2[idx_nan] = 0.0
            with utils.Timer("many_dist2trvt_v2: "):
                for _ in range(1):
                    many_dist2trvt_v2(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            ####
            rp_found3, dist_found3, trvt_found3, _ = many_dist2trvt_v3(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            idx_nan = np.where( np.isnan(rp_found3) )[0]
            rp_found3[idx_nan] = 0.0
            dist_found3[idx_nan] = 0.0
            trvt_found3[idx_nan] = 0.0
            with utils.Timer("many_dist2trvt_v3: "):
                for _ in range(1):
                    many_dist2trvt_v3(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            ###
            rp_found4, dist_found4, trvt_found4, _ = many_dist2trvt_v4(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            idx_nan = np.where( np.isnan(rp_found4) )[0]
            rp_found4[idx_nan] = 0.0
            dist_found4[idx_nan] = 0.0
            trvt_found4[idx_nan] = 0.0
            with utils.Timer("many_dist2trvt_v4: "):
                for _ in range(1):
                    many_dist2trvt_v4(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
            ###
            idx_dif = np.where((rp_found1 != rp_found2) )[0]
            if idx_dif.size >0:
                print(dist[idx_dif])
                print(dist_found1[idx_dif])
                print(dist_found2[idx_dif])
        ###
        print('rp difference=',   np.mean(np.abs(rp_found1 - rp_found2)),      np.sum(np.abs(rp_found3 - rp_found2)),     np.sum(np.abs(rp_found4 - rp_found2)),    )
        print('dist difference=', np.mean(np.abs(dist_found1 - dist_found2)),  np.sum(np.abs(dist_found3 - dist_found2)), np.sum(np.abs(dist_found4 - dist_found2)), )
        print('trvt difference=', np.mean(np.abs(trvt_found1 - trvt_found2)),  np.sum(np.abs(trvt_found3 - trvt_found2)), np.sum(np.abs(trvt_found4 - trvt_found2)), )
    if True:
        print()
        r_tmp, vr_tmp = denser_xy(r, vr, 5.0)  # make the model denser
        vr_tmp += (np.random.random(vr_tmp.size)-0.5)#*10  # add some noise
        z, vz   = flatten(r_tmp, vr_tmp, 6371.0)
        _,x,_,_,_= zv2pxt_v2(z, vz, theta_step_deg=0.1)  # warm up numba jit
        xmin, xmax = np.min(x), np.max(x)
        xmin += (xmax - xmin)*0.1
        xmax -= (xmax - xmin)*0.1
        dist = np.arange(xmin, xmax, 10.0) #* (np.pi/180.0) * 6371.0  # in km
        #########################
        rp_found1, dist_found1, trvt_found1, pxpv1, pxpp1, ptpv1, ptpp1, d_trvt_v1 = many_dist2trvt_jac(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)  # warm up numba jit
        idx_nan = np.where( np.isnan(rp_found1) )[0]
        rp_found1[idx_nan] = 0.0
        dist_found1[idx_nan] = 0.0
        trvt_found1[idx_nan] = 0.0
        with utils.Timer("many_dist2trvt_jac: "):
            for _ in range(1):
                many_dist2trvt_jac(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
        #########################
        rp_found2, dist_found2, trvt_found2, pxpv2, pxpp2, ptpv2, ptpp2, d_trvt_v2 = many_dist2trvt_jac_v2(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)  # warm up numba jit
        idx_nan = np.where( np.isnan(rp_found2) )[0]
        rp_found2[idx_nan] = 0.0
        dist_found2[idx_nan] = 0.0
        trvt_found2[idx_nan] = 0.0
        with utils.Timer("many_dist2trvt_jac_v2: "):
            for _ in range(1):
                many_dist2trvt_jac_v2(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
        #########################
        rp_found3, dist_found3, trvt_found3, pxpv3, pxpp3, ptpv3, ptpp3, d_trvt_v3 = many_dist2trvt_jac_v3(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)  # warm up numba jit
        idx_nan = np.where( np.isnan(rp_found3) )[0]
        rp_found3[idx_nan] = 0.0
        dist_found3[idx_nan] = 0.0
        trvt_found3[idx_nan] = 0.0
        with utils.Timer("many_dist2trvt_jac_v3: "):
            for _ in range(1):
                many_dist2trvt_jac_v3(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
        #########################
        rp_found4, dist_found4, trvt_found4, pxpv4, pxpp4, ptpv4, ptpp4, d_trvt_v4 = many_dist2trvt_jac_v4(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)  # warm up numba jit
        idx_nan = np.where( np.isnan(rp_found4) )[0]
        rp_found4[idx_nan] = 0.0
        dist_found4[idx_nan] = 0.0
        trvt_found4[idx_nan] = 0.0
        with utils.Timer("many_dist2trvt_jac_v4: "):
            for _ in range(1):
                many_dist2trvt_jac_v4(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
        #########################
        idx_dif = np.where((rp_found1 != rp_found2) | (rp_found1 != rp_found3) | (rp_found1 != rp_found4))[0]
        if idx_dif.size >0:
            print('rp', rp_found1[idx_dif], rp_found2[idx_dif], rp_found3[idx_dif], rp_found4[idx_dif])
        print('rp difference=  ', np.sum(np.abs(rp_found1 - rp_found2)),     np.sum(np.abs(rp_found2 - rp_found3)),     np.sum(np.abs(rp_found2 - rp_found4)),     )
        print('dist difference=', np.sum(np.abs(dist_found1 - dist_found2)), np.sum(np.abs(dist_found2 - dist_found3)), np.sum(np.abs(dist_found2 - dist_found4)), )
        print('trvt difference=', np.sum(np.abs(trvt_found1 - trvt_found2)), np.sum(np.abs(trvt_found2 - trvt_found3)), np.sum(np.abs(trvt_found2 - trvt_found4)), )
        print('pxpv difference=', np.sum(np.abs(pxpv1 - pxpv2)), np.sum(np.abs(pxpv2 - pxpv3)), np.sum(np.abs(pxpv2 - pxpv4)), )
        print('pxpp difference=', np.sum(np.abs(pxpp1 - pxpp2)), np.sum(np.abs(pxpp2 - pxpp3)), np.sum(np.abs(pxpp2 - pxpp4)), )
        print('ptpv difference=', np.sum(np.abs(ptpv1 - ptpv2)), np.sum(np.abs(ptpv2 - ptpv3)), np.sum(np.abs(ptpv2 - ptpv4)), )
        print('ptpp difference=', np.sum(np.abs(ptpp1 - ptpp2)), np.sum(np.abs(ptpp2 - ptpp3)), np.sum(np.abs(ptpp2 - ptpp4)), )
        print('d_trvt_v difference=', np.sum(np.abs(d_trvt_v1 - d_trvt_v2)), np.sum(np.abs(d_trvt_v2 - d_trvt_v3)),     np.sum(np.abs(d_trvt_v2 - d_trvt_v4)), )
def benchmark_single_pxt_gradient():
    r, vr = rd_prem_OC_model()
    r_tmp, vr_tmp = denser_xy(r, vr, 50.0)
    #vr_tmp += (np.random.random(vr_tmp.size)-0.5)#*10  # add some noise
    z, vz   = flatten(r_tmp, vr_tmp, 6371.0)
    ####
    inv_rp = vz[15] - 0.001
    ib     = 14
    ####
    buf_six_by_zsize = np.zeros((6, z.size), dtype=np.float64)
    dz = np.diff(z)
    dv = np.diff(vz)
    layer_type = np.zeros(z.size-1, dtype=np.int64)
    for ilayer in range(layer_type.size):
        if np.abs(dv[ilayer]) > 1e-10 and np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 2  # non-constant v layer
        elif np.abs(dz[ilayer]) >1e-10:
            layer_type[ilayer] = 1  # constant v layer
    dzdv = np.where(dv!=0.0, dz/dv, 0.0)
    ####
    dist1, trvt1, par_dist_v1, par_dist_p1, par_trvt_v1, par_trvt_p1, d_trvt_v1 = p2xt_grad_v4(inv_rp, z, vz, dz, dv, dzdv, layer_type, ib, buf_six_by_zsize)
    dist1, trvt1, par_dist_p2, = p2xt_grad_ponly_v4(inv_rp, z, vz, dz, dv, dzdv, layer_type, ib)
    print(f'par_dist_p1= {par_dist_p1}, par_trvt_p1= {par_trvt_p1}')
    print(f'par_dist_p2= {par_dist_p2}')
def benchmark_my_trvt_gradient():
    ####
    #z, vz = flatten(r, vr, 6371.0)
    #z  = np.linspace(0, -3000, 3)
    #vz = np.linspace(5.0, 8.0, 3)
    #print(z, vz)
    #1. postive Z is upward
    #2. the array z and vz should go from surface to depth
    #3. e.g., z=0, -100, -200, -300,
    #4.      vz[0], v[-100], vz[-200], vz[-300]...
    ####
    r  = np.array([3500., 3000, 2800, 2900, 2500, 2500, 500] )
    vr = np.array([  5.0, 5.1,  6.0,  13, 13.5,   12,  14] )
    z, vz = flatten(r, vr, 6371.0)
    ####
    theta_step_deg = 0.1
    niter = 1000
    ####
    #dist_deg = np.linspace(5, 10, 30) # where the data are
    #dist = dist_deg * (np.pi/180.0) * 6371.0  # in km
    _,x,_,_,_ = zv2pxt_v4(z, vz, theta_step_deg=0.1)  # warm up numba jit
    xmin, xmax = np.min(x), np.max(x)
    xmax_minus_xmin = xmax - xmin
    xmin += xmax_minus_xmin*0.1
    xmax -= xmax_minus_xmin*0.1
    dist = np.linspace(xmin, xmax, 10)
    ####
    #_, _, _, par_trvt_v = many_dist2trvt_jac_v4(dist, z, vz, theta_step_deg=theta_step_deg, xerr=1e-3, niter=niter)
    tmp = many_dist2trvt_jac(dist, z, vz, theta_step_deg=theta_step_deg, xerr=1e-3, niter=niter)
    par_trvt_v = tmp[7]
    par_trvt_v2 = np.zeros(par_trvt_v.shape)
    for iz in range(vz.size):
        vz1 = vz.copy()
        vz2 = vz.copy()
        vz1[iz] *= (1-1e-6)
        vz2[iz] *= (1+1e-6)
        #_, _, trvt1 = many_dist2trvt(dist, z,  vz1, theta_step_deg=theta_step_deg)
        #_, _, trvt2 = many_dist2trvt(dist, z,  vz2, theta_step_deg=theta_step_deg)
        _, _, trvt1, _ = many_dist2trvt_v4(dist, z, vz1, theta_step_deg=theta_step_deg, xerr=1e-20, niter=niter)
        _, _, trvt2, _ = many_dist2trvt_v4(dist, z, vz2, theta_step_deg=theta_step_deg, xerr=1e-20, niter=niter)
        junk = (trvt2 - trvt1) / (2e-6*vz[iz])
        par_trvt_v2[:, iz] = junk
        #print(trvt1, trvt2)
    np.set_printoptions(formatter={'float': '{: .6e}'.format})
    print('#### Analytical')
    print(par_trvt_v[:,:5])
    print('#### FD')
    print(par_trvt_v2[:,:5])
    print('#### Analytical vs. FD')
    print((par_trvt_v2[:,:5]-par_trvt_v[:,:5])/ par_trvt_v[:,:5])
    ####
    #print("dist:", np.sum(dist))
    #print("trvt: ", trvt) #np.sum(trvt))
    #print("trvt_:", trvt_)
    #print("par_trvt_v:", np.sum(par_trvt_v))
    #print("junk:", np.sum(junk))
    return 0
    #### below is only for plotting
    fig, ((ax1, ax2, ax3), (cax1, cax2, cax3)) = plt.subplots(2, 3, figsize=(20, 4), gridspec_kw={'height_ratios': [3, 0.1], 'wspace': 0.4, 'hspace': 0.4})
    ax1.plot(dist, trvt, 'o', markersize=2, color='r', label='My XT')
    ax1.legend()
    #
    par_trvt_v = np.transpose(par_trvt_v)
    #
    im2 = ax2.imshow(par_trvt_v,   interpolation='none', aspect='auto', extent=[0, dist.size, vz.size, 0], cmap='RdBu')
    plt.colorbar(im2, cax=cax2, orientation='horizontal')
    for ax in [ax2, ]:
        ax.set_xlabel('Distance ($\degree$)')
        ax.set_ylabel('Index of model parameters')
    for cax in [cax2, ]:
        cax.set_xlabel('Gradient')
    #
    plt.show()
def plot_benchmark_my_trvt_gradient():
    ##########################################################################################
    # Define model
    #   option 1: use r, vr and then flatten to z, vz
    #   option 2: use z, vz directly
    # Note:
    #   1. positive Z is upward
    #   2. the array z and vz should go from surface to depth
    #   3. e.g., if using z, vz:
    #             z=0, -100, -200, -300,...               in km
    #             vz(0), vz(-100), vz(-200), vz(-300)...  in km/s
    #   4. e.g., if using r, vr:
    #            r =3000, 2000, 1000,...                  in km
    #            vr(3000), vr(2000), vr(1000)...          in km/s
    r  = [4000, 3100,  3000, 3000, 2500, 2000, 2000, 1500, 1000, 500]
    #vr = [  6.0,  6.1,  10,  10.5, 11,   11,    16,  16.1, 16.2]
    vr = [5.0,  6.2,  6.1,  10,  10.5, 11,   15.5,    16,  16.5, 19]
    #vr = [  6.0,  6.1,  10,  10.5, 11,   15.5,    16,  16.5, 17]
    r, vr = np.array(r), np.array(vr)
    z, vz = flatten(r, vr, 6371.0)
    #
    (rp1, x1, t1), (rp1_legs, x1_legs, t1_legs)  = zv2pxt(z, vz, theta_step_deg=0.1)
    print(x1[1:].min(), x1[1:].max() )
    ####   gradient
    dist_deg = np.linspace(0.001, 90, 30)
    dist = dist_deg * (np.pi/180.0) * 6371.0  # in km
    print(dist)
    theta_step_deg = 0.1
    _, _, syn_trvt, par_syn_trvt_v = many_dist2trvt_jac(dist, z, vz, theta_step_deg=theta_step_deg)
    ######
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
    ax1.plot(vr, r)
    #ax2.plot(x1, t1, '.-', label='zv2pxt')
    for leg_x, leg_t in zip(x1_legs, t1_legs):
        ax2.plot(leg_x, leg_t, '-', markersize=2, label='zv2pxt')
    ax2.plot(dist, syn_trvt, 'o', markersize=2, color='r', label='many_dist2trvt_jac')
    ax3.imshow(par_syn_trvt_v, interpolation='none', aspect='auto', extent=[0, vz.size, dist[0], dist[-1]], cmap='RdBu', origin='lower')
    plt.show()
def debug():
    from sacpy import utils
    r, vr = rd_prem_OC_model()
    r, vr = denser_xy(r, vr, 100.0)  # make the model denser
    vr_tmp =  vr #+ (np.random.random(vr.size)-0.5)*0.5 # add some noise
    vr_tmp = np.abs(vr_tmp)  # make sure velocity is positive
    z, vz = flatten(r, vr_tmp, 6371.0)
    ####
    #z = np.array([0., -100, -200, -300, -400])
    #vz= np.array([6.0, 6.5,  10.0 , 7.0,  12.0])
    ###
    print('model nlayer=', z.size-1)
    ###
    p2, x2, t2, ib2, ind_pairs2 = zv2pxt_v2(z, vz, theta_step_deg=0.1)  # warm up numba jit
    leg_p2 = [p2[i0:i1] for (i0, i1) in ind_pairs2]
    leg_x2 = [x2[i0:i1] for (i0, i1) in ind_pairs2]
    leg_t2 = [t2[i0:i1] for (i0, i1) in ind_pairs2]
    ####
    p3, x3, t3, ib3, ind_pairs3 = zv2pxt_v3(z, vz, theta_step_deg=0.1)
    leg_p3 = [p3[i0:i1] for (i0, i1) in ind_pairs3]
    leg_x3 = [x3[i0:i1] for (i0, i1) in ind_pairs3]
    leg_t3 = [t3[i0:i1] for (i0, i1) in ind_pairs3]
    ####
    print('ind shape:', ind_pairs2.shape, ind_pairs3.shape)
    print('nlegs:', len(leg_p2), len(leg_p3))
    print('size of rps=', p2.size, p3.size)
    print('rps diff=', 1000*np.sum(np.abs(p2 - p3)))
    print('x   diff=', 1000*np.sum(np.abs(x2 - x3)))
    print('t   diff=', 1000*np.sum(np.abs(t2 - t3)))
    print('ib  diff=', np.sum(np.abs(ib2 - ib3)))

    #print(len(leg_p2),  len(leg_p3))
    ####
    ####
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(15,5))
    ax0.plot(vz, z, 's-')
    #ax0.plot(-10000*np.diff(vz)/np.diff(z), z[:-1], 's-')
    for ileg in range(len(leg_p2))[::-1]:
        ax1.plot(    leg_p2[ileg], leg_x2[ileg], 's-', lw=2, markersize=6, alpha=0.5)
    for ileg in range(len(leg_p3))[::-1]:
        #print(len(leg_p3[ileg]) )
        ax2.plot(    leg_p3[ileg], leg_x3[ileg], 's-', lw=2, markersize=4, alpha=0.5)
        ax3.plot(    leg_x3[ileg], leg_t3[ileg], 's-', lw=2, markersize=6, alpha=0.5)
        #ax2.semilogx(leg_p3[ileg], leg_x3[ileg], '-', color='C2', lw=0.5, markersize=4)
        #ax3.semilogx(leg_p3[ileg], leg_t3[ileg], '-', color='C2', lw=0.5, markersize=4)
    plt.show()
def debug2():
    from sacpy import utils
    r, vr = rd_prem_OC_model()
    r, vr = denser_xy(r, vr, 100.0)  # make the model denser
    vr_tmp =  vr #+ (np.random.random(vr.size)-0.5)*0.5 # add some noise
    vr_tmp = np.abs(vr_tmp)  # make sure velocity is positive
    z, vz = flatten(r, vr_tmp, 6371.0)
    #
    xerr = 1e-3
    niter = 1000
    ####
    #z = np.array([0., -100, -200, -300, -400])
    #vz= np.array([6.0, 6.5,  10.0 , 7.0,  12.0])
    ###
    print('model nlayer=', z.size-1)
    ###
    _,x,_,_,_= zv2pxt_v2(z, vz, theta_step_deg=0.1)  # warm up numba jit
    xmin, xmax = np.min(x), np.max(x)
    xmin += (xmax - xmin)*0.1
    xmax -= (xmax - xmin)*0.1
    dist = np.arange(xmin, xmax, 10.0) #* (np.pi/180.0) * 6371.0  # in km
    ####
    print('dist.size=', dist.size)
    #dist = dist[567:568]
    p1, x1, t1      = many_dist2trvt(     dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
    idx_nan = np.where( np.isnan(p1) )[0]
    p1[idx_nan] = 0.0
    x1[idx_nan] = 0.0
    t1[idx_nan] = 0.0
    print()
    p2, x2, t2, ib2 = many_dist2trvt_v2(dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
    idx_nan = np.where( np.isnan(p2) )[0]
    p2[idx_nan] = 0.0
    x2[idx_nan] = 0.0
    t2[idx_nan] = 0.0
    ib2[idx_nan] = 0
    print()
    p3, x3, t3, ib3 = many_dist2trvt_v3( dist, z, vz, theta_step_deg=0.1, xerr=xerr, niter=niter)
    idx_nan = np.where( np.isnan(p3) )[0]
    p3[idx_nan] = 0.0
    x3[idx_nan] = 0.0
    t3[idx_nan] = 0.0
    ib3[idx_nan] = 0
    ####
    print()
    print('p   idff=', np.sum(np.abs(p2  - p3)),  np.sum(np.abs(p2  - p1)), )
    print('x   idff=', np.sum(np.abs(x2  - x3)),  np.sum(np.abs(x2  - x1)), )
    print('t   idff=', np.sum(np.abs(t2  - t3)),  np.sum(np.abs(t2  - t1)), )
    print('ibs idff=', np.sum(np.abs(ib2 - ib3)))
def debug3():
    z = np.array([0., -100, -200, -300, -400])
    vz= np.array([3.0, 4.0,  5.0 , 6.0,  7.0])
    z = np.array(z)
    vz= np.array(vz)
    ###
    rp = 0.2
    dz = np.diff(z)
    dv = np.diff(vz)
    buf = np.zeros((6, z.size), dtype=np.float64)
    p2xt_grad_v2(rp, z, vz, dz, dv, 2, buf)
    ###
if __name__ == "__main__":
    benchmark_bfgs_inv()
    #plot_benchmark_my_trvt_gradient()
    #benchmark_my_trvt_gradient()
    #benchmark_speedup()
    #benchmark_single_pxt_gradient()
    #debug2()
    #debug3()
    pass
