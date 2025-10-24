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
            seg_len = int(np.floor((x1 - x0) / step))
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
            seg_len = int(np.floor((x1 - x0) / step))
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

#### Calculate X-T(-gradient) for a single ray with a single ray parameter
@jit(nopython=True, nogil=True)
def single_ray_xt(p, z, v, dz, dv, inv_v):
    inv_p = 1.0/p
    #### get the layer where the ray turns
    for ilayer in range(z.size-1):
        #if v[ilayer]< inv_p and inv_p <= v[ilayer+1]:
        if inv_v[ilayer] > p and p >= inv_v[ilayer+1]:
            ib = ilayer
            break
    ib = ilayer
    ####
    x = 0.0
    t = 0.0
    ####
    sin_theta_i0 = p*v[0]
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    tan_half_theta_i0 = (1.0-cos_theta_i0) / sin_theta_i0
    ln_tan_half_theta_i0 = np.log( tan_half_theta_i0 )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = p*v[i+1]
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        tan_half_theta_i1 = (1.0-cos_theta_i1) / sin_theta_i1
        ln_tan_half_theta_i1 = np.log( tan_half_theta_i1 )
        if np.abs(dv[i]) > 1e-10:
            local_dzdv = dz[i]/dv[i]
            t += local_dzdv * (ln_tan_half_theta_i0 - ln_tan_half_theta_i1)
            x += inv_p * local_dzdv * (cos_theta_i1 - cos_theta_i0)
        elif dz[i] < -1e-10:  # dv[i] ==0.  Note. dz is always negative
            t -= dz[i] / (v[i]*cos_theta_i0)
            x -= dz[i] * (sin_theta_i0/cos_theta_i0)
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        tan_half_theta_i0 = tan_half_theta_i1
        ln_tan_half_theta_i0 = ln_tan_half_theta_i1
    if dz[ib] < -1e-10:
        local_dzdv = dz[ib]/dv[ib]
        t += local_dzdv * ln_tan_half_theta_i0 # Note inv_k[ib] is finite automatically!
        x += inv_p * local_dzdv * (-cos_theta_i0)
    return x*2.0, t*2.0
@jit(nopython=True, nogil=True)
def single_ray_xt_fast(p, z, v, dz, dv, ib):
    inv_p = 1.0/p
    t = 0.0
    x = 0.0
    ####
    sin_theta_i0 = p*v[0]
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    tan_half_theta_i0 = (1.0-cos_theta_i0) / sin_theta_i0
    ln_tan_half_theta_i0 = np.log( tan_half_theta_i0 )
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = p*v[i+1]
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        tan_half_theta_i1 = (1.0-cos_theta_i1) / sin_theta_i1
        ln_tan_half_theta_i1 = np.log( tan_half_theta_i1 )
        if np.abs(dv[i]) > 1e-10:
            local_dzdv = dz[i]/dv[i]
            t += local_dzdv * (ln_tan_half_theta_i0 - ln_tan_half_theta_i1 )
            x += inv_p * local_dzdv * (cos_theta_i1 - cos_theta_i0)
        elif dz[i] < -1e-10: # dv[i] ==0.  Note. dz is always negative
            t -= dz[i] / (v[i]*cos_theta_i0)
            x -= dz[i] * (sin_theta_i0/cos_theta_i0)
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        tan_half_theta_i0 = tan_half_theta_i1
        ln_tan_half_theta_i0 = ln_tan_half_theta_i1
    if dz[ib] < -1e-10: # dz is always negative
        local_dzdv = dz[ib]/dv[ib]
        t += local_dzdv * ln_tan_half_theta_i0 # Note inv_k[ib] is finite automatically!
        x += inv_p * local_dzdv * (-cos_theta_i0)
    x *= 2.0
    t *= 2.0
    return x, t
@jit(nopython=True, nogil=True)
def single_ray_xt_grad(p, z, v, dz, dv, inv_v, buf_six_by_zsize):
    inv_p = 1.0/p
    #### get the layer where the ray turns
    for ilayer in range(z.size-1):
        #if v[ilayer]< inv_p and inv_p <= v[ilayer+1]:
        if inv_v[ilayer] > p and p >= inv_v[ilayer+1]:
            ib = ilayer
            break
    ib = ilayer
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
    sin_theta_i0 = p*v[0]
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    tan_half_theta_i0 = (1.0-cos_theta_i0) / sin_theta_i0
    ln_tan_half_theta_i0 = np.log( tan_half_theta_i0 )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = p*v[i+1]
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        tan_half_theta_i1 = (1.0-cos_theta_i1) / sin_theta_i1
        ln_tan_half_theta_i1 = np.log( tan_half_theta_i1 )
        if np.abs(dv[i]) > 1e-10:
            local_dzdv = dz[i]/dv[i]
            t_i = local_dzdv * (ln_tan_half_theta_i0 - ln_tan_half_theta_i1)
            x_i = inv_p * local_dzdv * (cos_theta_i1 - cos_theta_i0)
            dist += x_i
            trvt += t_i
            #
            tmp1 = local_dzdv/(tan_half_theta_i0*(1.0+cos_theta_i0)*cos_theta_i0)
            tmp2 = local_dzdv/(tan_half_theta_i1*(1.0+cos_theta_i1)*cos_theta_i1)
            tmp3 = t_i*local_dzdv/dz[i]
            #
            par_ti_v0[i] =  tmp3 + p*tmp1
            par_ti_v1[i] = -tmp3 - p*tmp2
            #
            par_trvt_p += v[i  ]*tmp1
            par_trvt_p -= v[i+1]*tmp2
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
            par_ti_v0[i] = -t_i/(v[i]+v[i+1]) + p*t_i*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_ti_v1[i] = par_ti_v0[i]
            #
            tmp1 = t_i*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_trvt_p += v[i  ]*tmp1
            par_trvt_p += v[i+1]*tmp1
            #
            par_xi_v0[i] = 0.5*p*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_xi_v1[i] = par_xi_v0[i]
            #
            tmp2 = 0.5*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_dist_p += v[i  ]*tmp2
            par_dist_p += v[i+1]*tmp2
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        tan_half_theta_i0 = tan_half_theta_i1
        ln_tan_half_theta_i0 = ln_tan_half_theta_i1
    #
    if np.abs(dz[ib]) > 1e-10:
        local_dzdv = dz[ib]/dv[ib]
        t_ib = local_dzdv * ln_tan_half_theta_i0 # Note inv_k[ib] is finite automatically! ? really?
        x_ib = inv_p * local_dzdv * (-cos_theta_i0)
        dist += x_ib
        trvt += t_ib
        # cos_theta[ib] is zero for distance=0 case.
        par_ti_v0[ib] =  t_ib*local_dzdv/dz[ib] + p*local_dzdv/(tan_half_theta_i0*(1.0+cos_theta_i0)*cos_theta_i0)
        par_ti_v1[ib] = -t_ib*local_dzdv/dz[ib]
        par_xi_v0[ib] =  x_ib*local_dzdv/dz[ib] + local_dzdv*sin_theta_i0/cos_theta_i0
        par_xi_v1[ib] = -x_ib*local_dzdv/dz[ib]
        #
        par_trvt_p += v[ib]*local_dzdv/(tan_half_theta_i0*(1.0+cos_theta_i0)*cos_theta_i0)
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
    d_trvt_v = par_trvt_v - par_trvt_p * (par_dist_v/par_dist_p)
    ###
    return dist, trvt, par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v
@jit(nopython=True, nogil=True)
def single_ray_xt_grad_fast(p, z, v, dz, dv, ib, buf_six_by_zsize):
    inv_p = 1.0/p
    ####
    #for ilayer in range(z.size-1):
    #    if 1.0/v[ilayer] > inv_p >= 1.0/v[ilayer+1]:
    #        #ib = ilayer
    #        #break
    #        if ib != ilayer:
    #            print("ib=", ib, ilayer)
    #ib = ilayer
    ####
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
    sin_theta_i0 = p*v[0]
    cos_theta_i0 = np.sqrt(1.0 - sin_theta_i0*sin_theta_i0)
    tan_half_theta_i0 = (1.0-cos_theta_i0) / sin_theta_i0
    ln_tan_half_theta_i0 = np.log( tan_half_theta_i0 )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        sin_theta_i1 = p*v[i+1]
        cos_theta_i1 = np.sqrt(1.0 - sin_theta_i1*sin_theta_i1)
        #if cos_theta_i0 < 1e-20:
        #    print("i0", i, cos_theta_i0, p, 1/v[i],     v[i], v[0])
        #if cos_theta_i1 < 1e-20:
        #    print("i1", i+1, cos_theta_i1, p, 1/v[i+1], v[i+1], v[0])
        tan_half_theta_i1 = (1.0-cos_theta_i1) / sin_theta_i1
        ln_tan_half_theta_i1 = np.log( tan_half_theta_i1 )
        if np.abs(dv[i]) > 1e-10:
            local_dzdv = dz[i]/dv[i]
            t_i = local_dzdv * (ln_tan_half_theta_i0 - ln_tan_half_theta_i1)
            x_i = inv_p * local_dzdv * (cos_theta_i1 - cos_theta_i0)
            dist += x_i
            trvt += t_i
            #
            tmp1 = local_dzdv/(tan_half_theta_i0*(1.0+cos_theta_i0)*cos_theta_i0)
            tmp2 = local_dzdv/(tan_half_theta_i1*(1.0+cos_theta_i1)*cos_theta_i1)
            tmp3 = t_i*local_dzdv/dz[i]
            #
            par_ti_v0[i] =  tmp3 + p*tmp1
            par_ti_v1[i] = -tmp3 - p*tmp2
            #
            par_trvt_p += v[i  ]*tmp1
            par_trvt_p -= v[i+1]*tmp2
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
            par_ti_v0[i] = -t_i/(v[i]+v[i+1]) + p*t_i*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_ti_v1[i] = par_ti_v0[i]
            #
            tmp1 = t_i*sin_theta_i0/(cos_theta_i0*(cos_theta_i0+cos_theta_i1))
            par_trvt_p += v[i  ]*tmp1
            par_trvt_p += v[i+1]*tmp1
            #
            par_xi_v0[i] = 0.5*p*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_xi_v1[i] = par_xi_v0[i]
            #
            tmp2 = 0.5*dz[i]/(cos_theta_i0*cos_theta_i0*cos_theta_i0)
            par_dist_p += v[i  ]*tmp2
            par_dist_p += v[i+1]*tmp2
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        tan_half_theta_i0 = tan_half_theta_i1
        ln_tan_half_theta_i0 = ln_tan_half_theta_i1
    #
    if np.abs(dz[ib]) > 1e-10:
        local_dzdv = dz[ib]/dv[ib]
        t_ib = local_dzdv * ln_tan_half_theta_i0 # Note inv_k[ib] is finite automatically! ? really?
        x_ib = inv_p * local_dzdv * (-cos_theta_i0)
        dist += x_ib
        trvt += t_ib
        # cos_theta[ib] is zero for distance=0 case.
        par_ti_v0[ib] =  t_ib*local_dzdv/dz[ib] + p*local_dzdv/(tan_half_theta_i0*(1.0+cos_theta_i0)*cos_theta_i0)
        par_ti_v1[ib] = -t_ib*local_dzdv/dz[ib]
        par_xi_v0[ib] =  x_ib*local_dzdv/dz[ib] + local_dzdv*sin_theta_i0/cos_theta_i0
        par_xi_v1[ib] = -x_ib*local_dzdv/dz[ib]
        #
        par_trvt_p += v[ib]*local_dzdv/(tan_half_theta_i0*(1.0+cos_theta_i0)*cos_theta_i0)
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
    d_trvt_v = par_trvt_v - par_trvt_p * (par_dist_v/par_dist_p)
    ###
    return dist, trvt, par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v
####
@jit(nopython=True, nogil=True)
def split_pxt_legs(ps, xs, ts): # split the p-x-t into legs so that each leg has monotonic p and x.
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
    p_legs = [ps[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    x_legs = [xs[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    t_legs = [ts[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    return p_legs, x_legs, t_legs
@jit(nopython=True, nogil=True)
def split_pxt_legs_simple(xs): # split the p-x-t into legs so that each leg has monotonic p and x.
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
def many_rays_pxt(z, vz, theta_step_deg=0.1):
    theta_step_rad = theta_step_deg * (np.pi/180.0)
    v0 = vz[0]
    nlayers = vz.size - 1
    rp_legs, dist_legs, trvt_legs = list(), list(), list()
    ##############################################################
    vtop = v0
    dz = np.diff(z)
    dvz = np.diff(vz)
    inv_vz = 1.0/vz
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
            rps    = np.sin(thetas)/v0
            rps[0] = 1.0/vtop # force to set this to avoid numerical issues by sin(arcsin(...))
            rps[-1]= 1.0/vbot # the same
            if flag_next_after:
                rps[0] = np.nextafter( rps[0], rps[-1] )
            ##
            dist= np.zeros(nrp, dtype=np.float64)
            trvt= np.zeros(nrp, dtype=np.float64)
            for irp in range(nrp):
                dist[irp], trvt[irp] = single_ray_xt(rps[irp], z, vz, dz, dvz, inv_vz)
            tmp1, tmp2, tmp3 = split_pxt_legs(rps, dist, trvt)
            rp_legs.extend(tmp1)
            dist_legs.extend(tmp2)
            trvt_legs.extend(tmp3)
            ####
            #vmax_above_current_layer = vbot ##### note vbot is alreay the max until now!
    ########
    # flatten the list of list
    rp   = np.array([it for leg in rp_legs for it in leg])
    dist = np.array([it for leg in dist_legs for it in leg])
    trvt = np.array([it for leg in trvt_legs for it in leg])
    idx_sort = np.argsort(rp)[::-1]
    rp = rp[idx_sort]
    dist = dist[idx_sort]
    trvt = trvt[idx_sort]
    #########
    return (rp, dist, trvt), (rp_legs, dist_legs, trvt_legs)
@jit(nopython=True, nogil=True)
def many_rays_pxt_simple(z, vz, theta_step_deg=0.1):
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
    rps = buf_pxt[0]
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
        rps[g_i0:g_i1] = np.sin(thetas)/v0
        rps[g_i0]   = 1.0/vtop
        rps[g_i1-1] = 1.0/vbot
        if flag_next_after[idx_range]:
            rps[g_i0] = np.nextafter( rps[g_i0], rps[g_i1-1] ) # make sure the first ray can enter the layer
            dist[g_i0], trvt[g_i0] = single_ray_xt_fast(rps[g_i0], z, vz, dz, dv, bottom_layer_ind_each_range[idx_range])
            ibs[g_i0] = bottom_layer_ind_each_range[idx_range]
        else:
            # for this case, rps[g_i0] is exactly the critical ray that turns at the top of the layer.
            # Hence, the ib should minus 1
            dist[g_i0], trvt[g_i0] = single_ray_xt_fast(rps[g_i0], z, vz, dz, dv, bottom_layer_ind_each_range[idx_range]-1)
            ibs[g_i0] = bottom_layer_ind_each_range[idx_range] -1
        for tmp_i in range(g_i0+1, g_i1):
            dist[tmp_i], trvt[tmp_i] = single_ray_xt_fast(rps[tmp_i], z, vz, dz, dv, bottom_layer_ind_each_range[idx_range])
        ibs[g_i0+1:g_i1] = bottom_layer_ind_each_range[idx_range]
        ########################################################
        # Split into legs with x! p is already monotonic
        b_inds = split_pxt_legs_simple(dist[g_i0:g_i1] )
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
    return rps, dist, trvt, ibs, leg_start_end_pairs_gind[:,:n_legs].T
####
@jit(nopython=True, nogil=True)
def single_dist2trvt(target_single_dist, z, vz, rp_legs, dist_legs, trvt_legs, critical_dist_err=1e-20, niter=1000):
    s_rp, s_dist, s_trvt = np.nan, np.nan, np.nan
    flag_none_found=True
    dz = np.diff(z)
    dvz = np.diff(vz)
    inv_vz = 1.0/vz
    #m_rp, m_dist, m_trvt = list(), list(), list()
    for ileg in range(len(rp_legs)):
        rps = rp_legs[ileg]
        dists = dist_legs[ileg]
        trvts = trvt_legs[ileg]
        if (dists[0] <= target_single_dist <= dists[-1]) or (dists[-1] <= target_single_dist <= dists[0]): # within the range of this leg
            if np.abs(target_single_dist-dists[0]) < critical_dist_err:
                rp_found   = rps[0]
                dist_found = dists[0]
                trv_found  = trvts[0]
            elif np.abs(target_single_dist-dists[-1]) < critical_dist_err:
                rp_found   = rps[-1]
                dist_found = dists[-1]
                trv_found  = trvts[-1]
            else:
                if dists[0] > dists[-1]:
                    rps   = rps[::-1]
                    dists = dists[::-1]
                    trvts = trvts[::-1]
                i1 = np.searchsorted(dists, target_single_dist)
                rp_left, rp_right = rps[i1-1], rps[i1]
                d_left            = dists[i1-1]
                #### start ray tracing with bisection method
                for idx_iter in range(niter):
                    rp_mid = 0.5 * (rp_left + rp_right)
                    d_mid, t_mid = single_ray_xt(rp_mid, z, vz, dz, dvz, inv_vz)
                    #print('  iter:', idx_iter, d_left, d_mid, d_right, '|', rp_left, rp_mid, rp_right, )
                    if np.abs(d_mid-target_single_dist) < critical_dist_err:
                        ### found!!!
                        break
                    elif (d_left <= target_single_dist <= d_mid) or (d_mid <= target_single_dist <= d_left):
                        rp_right = rp_mid
                        #d_right = d_mid
                    else:
                        rp_left = rp_mid
                        d_left = d_mid
                rp_found   = rp_mid
                dist_found = d_mid
                trv_found  = t_mid
            ###########
            #print(target_single_dist, dist_found, rp_found, trv_found)
            #m_rp.append(rp_found)
            #m_dist.append(dist_found)
            #m_trvt.append(trv_found)
            if flag_none_found:
                s_rp, s_dist, s_trvt = rp_found, dist_found, trv_found
                flag_none_found = False
            elif trv_found < s_trvt:
                s_rp, s_dist, s_trvt = rp_found, dist_found, trv_found
    return s_rp, s_dist, s_trvt #, (np.array(m_rp), np.array(m_dist), np.array(m_trvt) )
@jit(nopython=True, nogil=True)
def many_dist2trvt(dist, z, vz, theta_step_deg=0.1, critical_dist_err=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found and trvt_found correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
    """
    rp_found   = np.zeros(dist.size, dtype=np.float64)
    dist_found = np.zeros(dist.size, dtype=np.float64)
    trvt_found = np.zeros(dist.size, dtype=np.float64)
    _, (rp_legs, dist_legs, trvt_legs) = many_rays_pxt(z, vz, theta_step_deg=theta_step_deg)
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
        tmp = single_dist2trvt(dist[idx], z, vz, rp_legs, dist_legs, trvt_legs, critical_dist_err, niter)
        rp_found[idx]   = tmp[0]
        dist_found[idx] = tmp[1]
        trvt_found[idx] = tmp[2]
    return rp_found, dist_found, trvt_found
@jit(nopython=True, nogil=True)
def many_dist2trvt_jac(dist, z, vz, theta_step_deg=0.1, critical_dist_err=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found, d_trvt_v
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found, trvt_found, d_trvt_v correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
               zeros will be used for the corresponding rows in the d_trvt_v.
    """
    rp_found, dist_found, trvt_found = many_dist2trvt(dist, z, vz, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
    d_trvt_v = np.zeros((dist.size, vz.size), dtype=np.float64)
    buf = np.zeros((6, z.size), dtype=np.float64)
    dz = np.diff(z)
    dv = np.diff(vz)
    inv_vz = 1.0/vz
    for idx in range(rp_found.size):
        rp = rp_found[idx]
        if not np.isnan(rp): ### Nan means no ray exist for this distance
            _, _, _, _, _, _, d_trvt_v[idx] = single_ray_xt_grad(rp, z, vz, dz, dv, inv_vz, buf)
    return rp_found, dist_found, trvt_found, d_trvt_v
####
@jit(nopython=True, nogil=True)
def single_dist2trvt_fast(target_single_dist, z, vz, dz, dv, many_ray_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs, critical_dist_err=1e-20, niter=1000):
    s_rp   = np.nan
    s_dist = np.nan
    s_trvt = np.inf # will be set to nan if not found latter
    nleg = many_ray_ind_pairs.shape[0]
    for ileg in range(nleg):
        jdx_start = many_ray_ind_pairs[ileg, 0]
        jdx_end   = many_ray_ind_pairs[ileg, 1]
        rps   = many_ray_rps[jdx_start:jdx_end]
        dists = many_ray_dist[jdx_start:jdx_end]
        trvts = many_ray_trvt[jdx_start:jdx_end]
        ibs   = many_ray_ibs[jdx_start:jdx_end]
        if (dists[0] <= target_single_dist <= dists[-1]) or (dists[-1] <= target_single_dist <= dists[0]): # within the range of this leg
            if np.abs(target_single_dist-dists[0]) < critical_dist_err:
                rp_found   = rps[0]
                dist_found = dists[0]
                trv_found  = trvts[0]
                ib_found   = ibs[0]
            elif np.abs(target_single_dist-dists[-1]) < critical_dist_err:
                rp_found   = rps[-1]
                dist_found = dists[-1]
                trv_found  = trvts[-1]
                ib_found   = ibs[-1]
            else:
                if dists[0] > dists[-1]:
                    rps   = rps[::-1]
                    dists = dists[::-1]
                    trvts = trvts[::-1]
                ib_found = ibs[0] if ibs[0]>=ibs[-1] else ibs[-1] # if a bisector is needed, use the larger ib to make sure ray can enter the layer
                i1 = np.searchsorted(dists, target_single_dist)
                rp_left = rps[i1-1]
                rp_right= rps[i1]
                d_left  = dists[i1-1]
                #### start ray tracing with bisection method
                for idx_iter in range(niter):
                    rp_mid = 0.5 * (rp_left + rp_right)
                    d_mid, t_mid = single_ray_xt_fast(rp_mid, z, vz, dz, dv, ib_found)
                    if np.abs(d_mid-target_single_dist) < critical_dist_err:
                        ### found!!!
                        break
                    elif (d_left <= target_single_dist <= d_mid) or (d_mid <= target_single_dist <= d_left):
                        rp_right = rp_mid
                    else:
                        rp_left = rp_mid
                        d_left = d_mid
                rp_found   = rp_mid
                dist_found = d_mid
                trv_found  = t_mid
            ###########
            if trv_found < s_trvt:
                s_rp   = rp_found
                s_dist = dist_found
                s_trvt = trv_found
                s_ib   = ib_found
    if np.isinf(s_trvt):
        s_trvt = np.nan
    return s_rp, s_dist, s_trvt, s_ib
@jit(nopython=True, nogil=True)
def many_dist2trvt_fast(dist, z, vz, theta_step_deg=0.1, critical_dist_err=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found and trvt_found correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
    """
    rp_found   = np.zeros(dist.size, dtype=np.float64)
    dist_found = np.zeros(dist.size, dtype=np.float64)
    trvt_found = np.zeros(dist.size, dtype=np.float64)
    ib_found   = np.zeros(dist.size, dtype=np.int64)
    many_ray_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs = many_rays_pxt_simple(z, vz, theta_step_deg=theta_step_deg)
    dz = np.diff(z)
    dv = np.diff(vz)
    for idx in range(dist.size):
        rp_found[idx], dist_found[idx], trvt_found[idx], ib_found[idx] = single_dist2trvt_fast(dist[idx], z, vz, dz, dv, many_ray_rps, many_ray_dist, many_ray_trvt, many_ray_ibs, many_ray_ind_pairs, critical_dist_err, niter)
    return rp_found, dist_found, trvt_found, ib_found
@jit(nopython=True, nogil=True)
def many_dist2trvt_jac_fast(dist, z, vz, theta_step_deg=0.1, critical_dist_err=1e-20, niter=1000):
    """
    Return: rp_found, dist_found, trvt_found, d_trvt_v
        The dist_found will be very close to the input dist, but may not be exactly the same due to numerical errors.
        The rp_found, trvt_found, d_trvt_v correspond to the dist_found.

        Note: `np.nan` will used for elements in rp_found, dist_found, and trvt_found
               for any distances that do not exist given the model.
               zeros will be used for the corresponding rows in the d_trvt_v.
    """
    rp_found, dist_found, trvt_found, ib_found = many_dist2trvt_fast(dist, z, vz, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
    d_trvt_v = np.zeros((dist.size, vz.size), dtype=np.float64)
    dz = np.diff(z)
    dv = np.diff(vz)
    buf = np.zeros((6, z.size), dtype=np.float64)
    for idx in range(rp_found.size):
        rp = rp_found[idx]
        if not np.isnan(rp): ### Nan means no ray exist for this distance
            _, _, _, _, _, _, d_trvt_v[idx] = single_ray_xt_grad_fast(rp, z, vz, dz, dv, ib_found[idx], buf)
    return rp_found, dist_found, trvt_found, d_trvt_v
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

#### OBJECTIVE function and gradient function for optimization
def get_obj_and_grad_func(dist, trvt_obs, std, model_z, model_vz_ref,
                          alpha=0.0, beta=0.0, theta_step_deg=0.1, critical_dist_err=1e-20, niter=1000):
    """
    Generate two functions: `obj_func(m)` and `obj_grad(m)` for optimization.
    dist:         the distances where the data are observed (in km)
    trvt_obs:     the observed traveltimes at the distances (in s)
    std:          the standard deviation of the traveltime observations (in s)
    model_z:      the model depth grid from surface to depth (in km, positive upward)
    model_vz_ref: the reference model velocities at model_z (in km/s)
    """
    inv_var = 1.0/(std*std)
    model_sz = len(model_z)
    ######### objective functions #########
    @jit(nopython=True, nogil=True)
    def obj_data_diff(dvz): # dvz is the perturbation from model_vz_ref
        _,_,trvt_syn = many_dist2trvt(dist, model_z, dvz+model_vz_ref, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
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
        _, _, d, jac = many_dist2trvt_jac(dist, model_z, dvz+model_vz_ref, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
        tmp = 2*(d-trvt_obs)*inv_var
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
def benchmark_speedup():
    from sacpy import utils
    r, vr = rd_prem_OC_model()
    r, vr = denser_xy(r, vr, 5.0)  # make the model denser
    vr_tmp =  vr + (np.random.random(vr.size)-0.5)  # add some noise
    z, vz = flatten(r, vr_tmp, 6371.0)
    ####
    #print('1/vz=', 1.0/vz[:10])
    #z  =  np.array([0., -100, -200, -300, -400])
    #vz =  np.array([6.0, 7.0, 6.5, 9.0, 10.0])
    critical_dist_err = 1e1
    niter = 100
    if True: # speed up many_rays_pxt vs. many_rays_pxt_simple
        print()
        (p1, x1, t1), (leg_p1, leg_x1, leg_t1) = many_rays_pxt(z, vz, theta_step_deg=0.1)  # warm up numba jit
        with utils.Timer("many_rays_pxt: "):
            for _ in range(1):
                many_rays_pxt(z, vz, theta_step_deg=0.1)
        ####
        p2, x2, t2, _, ind_pairs = many_rays_pxt_simple(z, vz, theta_step_deg=0.1)  # warm up numba jit
        with utils.Timer("many_rays_pxt_simple: "):
            for _ in range(1):
                many_rays_pxt_simple(z, vz, theta_step_deg=0.1)
        leg_p2 = [p2[i0:i1] for (i0, i1) in ind_pairs]
        leg_x2 = [x2[i0:i1] for (i0, i1) in ind_pairs]
        leg_t2 = [t2[i0:i1] for (i0, i1) in ind_pairs]
        ####
        #print('p difference=', np.sum(np.abs(p1 - p2)) )
        #print('x difference=', np.sum(np.abs(x1 - x2)) )
        #print('t difference=', np.sum(np.abs(t1 - t2)) )
        nlegs = len(leg_p1)
        for ileg in range(nlegs):
            idx_dif = np.where( (leg_x1[ileg] != leg_x2[ileg]) | (leg_t1[ileg] != leg_t2[ileg]) | (leg_p1[ileg] != leg_p2[ileg]) )[0]
            if idx_dif.size >0:
                print('ileg=', ileg)
                print(leg_p1[ileg][idx_dif], leg_x1[ileg][idx_dif])
                print(leg_p2[ileg][idx_dif], leg_x2[ileg][idx_dif])
            #if leg_p1[ileg].size != leg_p2[ileg].size:
            #    print('ileg=', ileg, leg_p1[ileg].size, leg_p2[ileg].size)
            #    print(leg_p1[ileg])
            #    print(leg_p2[ileg])
            #    print(leg_x1[ileg])
            #    print(leg_x2[ileg])
        #   #     break
            #p_dif = np.sum( np.abs( leg_p1[ileg] - leg_p2[ileg] ) )
            #x_dif = np.sum( np.abs( leg_x1[ileg] - leg_x2[ileg] ) )
            #t_dif = np.sum( np.abs( leg_t1[ileg] - leg_t2[ileg] ) )
            #if (p_dif > 1e-6) or (x_dif > 1e-6) or (t_dif > 1e-6):
            #    print('ileg=', ileg, p_dif, x_dif, t_dif)
    if True:
        print()
        ####
        for _ in range(1):
            vr_tmp =  vr + (np.random.random(vr.size)-0.5)  # add some noise
            z, vz = flatten(r, vr_tmp, 6371.0)
            _,x,_,_,_= many_rays_pxt_simple(z, vz, theta_step_deg=0.1)  # warm up numba jit
            xmin, xmax = np.min(x), np.max(x)
            xmin += (xmax - xmin)*0.1
            xmax -= (xmax - xmin)*0.1
            dist = np.arange(xmin, xmax, 10.0) #* (np.pi/180.0) * 6371.0  # in km
            ####
            rp_found1, dist_found1, trvt_found1 = many_dist2trvt(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)
            idx_nan = np.where( np.isnan(rp_found1) )[0]
            rp_found1[idx_nan] = 0.0
            dist_found1[idx_nan] = 0.0
            trvt_found1[idx_nan] = 0.0
            with utils.Timer("many_dist2trvt: "):
                for _ in range(1):
                    many_dist2trvt(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)
            rp_found2, dist_found2, trvt_found2, _ = many_dist2trvt_fast(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)
            idx_nan = np.where( np.isnan(rp_found2) )[0]
            rp_found2[idx_nan] = 0.0
            dist_found2[idx_nan] = 0.0
            trvt_found2[idx_nan] = 0.0
            with utils.Timer("many_dist2trvt_fast: "):
                for _ in range(1):
                    many_dist2trvt_fast(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)
            ###
            idx_dif = np.where(rp_found1 != rp_found2)[0]
            if idx_dif.size >0:
                print(dist[idx_dif])
                print(dist_found1[idx_dif])
                print(dist_found2[idx_dif])
        ###
        print('rp difference=', np.mean(np.abs(rp_found1 - rp_found2)) )
        print('dist difference=', np.mean(np.abs(dist_found1 - dist_found2)) )
        print('trvt difference=', np.mean(np.abs(trvt_found1 - trvt_found2)) )
    if True:
        print()
        vr_tmp =  vr + (np.random.random(vr.size)-0.5)  # add some noise
        z, vz = flatten(r, vr_tmp, 6371.0)
        _,x,_,_,_= many_rays_pxt_simple(z, vz, theta_step_deg=0.1)  # warm up numba jit
        xmin, xmax = np.min(x), np.max(x)
        xmin += (xmax - xmin)*0.1
        xmax -= (xmax - xmin)*0.1
        dist = np.arange(xmin, xmax, 10.0) #* (np.pi/180.0) * 6371.0  # in km
        #####
        rp_found1, dist_found1, trvt_found1, d_trvt_v1 = many_dist2trvt_jac(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)  # warm up numba jit
        idx_nan = np.where( np.isnan(rp_found1) )[0]
        rp_found1[idx_nan] = 0.0
        dist_found1[idx_nan] = 0.0
        trvt_found1[idx_nan] = 0.0
        with utils.Timer("many_dist2trvt_jac: "):
            for _ in range(1):
                many_dist2trvt_jac(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)
        ####
        rp_found2, dist_found2, trvt_found2, d_trvt_v2 = many_dist2trvt_jac_fast(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)  # warm up numba jit
        idx_nan = np.where( np.isnan(rp_found2) )[0]
        rp_found2[idx_nan] = 0.0
        dist_found2[idx_nan] = 0.0
        trvt_found2[idx_nan] = 0.0
        with utils.Timer("many_dist2trvt_jac_fast: "):
            for _ in range(1):
                many_dist2trvt_jac_fast(dist, z, vz, theta_step_deg=0.1, critical_dist_err=critical_dist_err, niter=niter)
        ###
        idx_dif = np.where(rp_found1 != rp_found2)[0]
        if idx_dif.size >0:
            print('rp', rp_found1[idx_dif], rp_found2[idx_dif])
        print('rp difference=', np.sum(np.abs(rp_found1 - rp_found2))   )
        print('dist difference=', np.sum(np.abs(dist_found1 - dist_found2)) )
        print('trvt difference=', np.sum(np.abs(trvt_found1 - trvt_found2)) )
        print('d_trvt_v difference=', np.sum(np.abs(d_trvt_v1 - d_trvt_v2)) )
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
    niter = 100
    critical_dist_err = 1e-20
    ####
    dist_deg = np.linspace(5, 10, 100) # where the data are
    dist = dist_deg * (np.pi/180.0) * 6371.0  # in km
    ####
    _, _, _, par_trvt_v = many_dist2trvt_jac(dist, z, vz, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
    par_trvt_v2 = np.zeros(par_trvt_v.shape)
    for iz in range(vz.size):
        vz1 = vz.copy()
        vz2 = vz.copy()
        vz1[iz] *= (1-1e-6)
        vz2[iz] *= (1+1e-6)
        #_, _, trvt1 = many_dist2trvt(dist, z,  vz1, theta_step_deg=theta_step_deg)
        #_, _, trvt2 = many_dist2trvt(dist, z,  vz2, theta_step_deg=theta_step_deg)
        _, _, trvt1, _ = many_dist2trvt_jac(dist, z, vz1, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
        _, _, trvt2, _ = many_dist2trvt_jac(dist, z, vz2, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
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
    (rp1, x1, t1), (rp1_legs, x1_legs, t1_legs)  = many_rays_pxt(z, vz, theta_step_deg=0.1)
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
    #ax2.plot(x1, t1, '.-', label='many_rays_pxt')
    for leg_x, leg_t in zip(x1_legs, t1_legs):
        ax2.plot(leg_x, leg_t, '-', markersize=2, label='many_rays_pxt')
    ax2.plot(dist, syn_trvt, 'o', markersize=2, color='r', label='many_dist2trvt_jac')
    ax3.imshow(par_syn_trvt_v, interpolation='none', aspect='auto', extent=[0, vz.size, dist[0], dist[-1]], cmap='RdBu', origin='lower')
    plt.show()

if __name__ == "__main__":
    #plot_benchmark_my_trvt_gradient()
    #benchmark_my_trvt_gradient()
    benchmark_speedup()
