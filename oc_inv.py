#!/usr/bin/env python3

import matplotlib.pyplot as plt
from numba import jit
import numpy as np

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
def single_ray_xt(p, z, v):
    inv_p = 1.0/p
    #### get the layer where the ray turns
    for ilayer in range(z.size-1):
        if v[ilayer]<= inv_p and inv_p <= v[ilayer+1]:
            ib = ilayer
            break
    ib = ilayer
    #### buffer
    t = np.zeros(ib+1, dtype=np.float64)
    x = np.zeros(ib+1, dtype=np.float64)
    ####
    dz = np.diff(z)
    dv = np.diff(v)
    inv_k = np.where(np.abs(dv) > 1e-10, dz/dv, 0.0)
    #### critical angles
    v = v[:ib+1]
    sin_theta         = p*v
    #sin_theta         = np.clip(sin_theta, 0.0, 1.0)
    cos_theta         = np.sqrt(1.0 - sin_theta*sin_theta)
    tan_half_theta    = (1-cos_theta) / sin_theta
    ln_tan_half_theta = np.log( tan_half_theta )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        if np.abs(dz[i]) < 1e-10:
            t[i], x[i] = 0.0, 0.0
        elif np.abs(dv[i]) > 1e-10:
            t[i] = inv_k[i] * (ln_tan_half_theta[i] - ln_tan_half_theta[i+1] )
            x[i] = inv_p * inv_k[i] * (cos_theta[i+1] - cos_theta[i])
        else: # dv[i] == 0.0
            t[i] = -dz[i] / (v[i]*cos_theta[i])
            x[i] = -dz[i] * (sin_theta[i]/cos_theta[i])
    if np.abs(dz[ib]) > 1e-10:
        t[ib] = inv_k[ib] * ln_tan_half_theta[ib] # Note inv_k[ib] is finite automatically!
        x[ib] = inv_p * inv_k[ib] * (-cos_theta[ib])
    dist = 2.0 * np.sum(x)
    trvt = 2.0 * np.sum(t)
    return dist, trvt
@jit(nopython=True, nogil=True)
def single_ray_gradient_xt(p, z, v):
    inv_p = 1.0/p
    #### get the layer where the ray turns
    for ilayer in range(z.size-1):
        if v[ilayer]<= inv_p and inv_p <= v[ilayer+1]:
            ib = ilayer
            break
    ib = ilayer
    #print('ib:', ib)
    #### buffer
    t = np.zeros(ib+1, dtype=np.float64)
    x = np.zeros(ib+1, dtype=np.float64)
    par_t_v0 = np.zeros(ib+1, dtype=np.float64)
    par_t_v1 = np.zeros(ib+1, dtype=np.float64)
    par_x_v0 = np.zeros(ib+1, dtype=np.float64)
    par_x_v1 = np.zeros(ib+1, dtype=np.float64)
    par_dist_v = np.zeros(z.size, dtype=np.float64)
    par_trvt_v = np.zeros(z.size, dtype=np.float64)
    ####
    dz = np.diff(z)
    dv = np.diff(v)
    inv_k  = np.where(np.abs(dv) > 1e-10, dz/dv, 0.0)
    #### critical angles
    sin_theta         = p*v[:ib+1]
    #sin_theta         = np.clip(sin_theta, 0.0, 1.0)
    cos_theta         = np.sqrt(1.0 - sin_theta*sin_theta)
    tan_half_theta    = (1-cos_theta) / sin_theta
    ln_tan_half_theta = np.log( tan_half_theta )
    #print("ln_tan_half_theta:", np.sum(ln_tan_half_theta) )
    ####
    for i in range(ib): # i = 0,1,2,...,ib-1
        if np.abs(dz[i]) < 1e-10:
            t[i], x[i] = 0.0, 0.0
        elif np.abs(dv[i]) > 1e-10:
            t[i] = inv_k[i] * (ln_tan_half_theta[i] - ln_tan_half_theta[i+1] )
            x[i] = inv_p * inv_k[i] * (cos_theta[i+1] - cos_theta[i])
            #
            par_t_v0[i] =  t[i]*inv_k[i]/dz[i] + p*inv_k[i]/(tan_half_theta[i  ]*(1.0+cos_theta[i  ])*cos_theta[i  ])
            par_t_v1[i] = -t[i]*inv_k[i]/dz[i] - p*inv_k[i]/(tan_half_theta[i+1]*(1.0+cos_theta[i+1])*cos_theta[i+1])
            #
            par_x_v0[i] =  x[i]*inv_k[i]/dz[i] + inv_k[i]*sin_theta[i  ]/cos_theta[i  ]
            par_x_v1[i] = -x[i]*inv_k[i]/dz[i] - inv_k[i]*sin_theta[i+1]/cos_theta[i+1]
        else: # dv[i] == 0.0
            t[i] = -dz[i] / (v[i]*cos_theta[i])
            x[i] = -dz[i] * (sin_theta[i]/cos_theta[i])
            #
            par_t_v0[i] = -t[i]/(v[i]+v[i+1]) + p*t[i]*sin_theta[i]/(cos_theta[i]*(cos_theta[i]+cos_theta[i+1]))
            par_t_v1[i] = par_t_v0[i]
            par_x_v0[i] = 0.5*p*dz[i]/(cos_theta[i]*cos_theta[i]*cos_theta[i])
            par_x_v1[i] = par_x_v0[i]
    t[ib] = inv_k[ib] * ln_tan_half_theta[ib] # Note inv_k[ib] is finite automatically!
    x[ib] = inv_p * inv_k[ib] * (-cos_theta[ib])
    #
    if np.abs(dv[ib]) > 1e-10 and cos_theta[ib] > 1e-10 and  np.abs(dz[ib]) > 1e-10:
        # cos_theta[ib] is zero for distance=0 case.
        par_t_v0[ib] =  t[ib]*inv_k[ib]/dz[ib] + p*inv_k[ib]/(tan_half_theta[ib]*(1.0+cos_theta[ib])*cos_theta[ib])
        par_t_v1[ib] = -t[ib]*inv_k[ib]/dz[ib]
        par_x_v0[ib] =  x[ib]*inv_k[ib]/dz[ib] + inv_k[ib]*sin_theta[ib]/cos_theta[ib]
        par_x_v1[ib] = -x[ib]*inv_k[ib]/dz[ib]
    ####
    dist = 2.0 * np.sum(x)
    trvt = 2.0 * np.sum(t)
    ####
    par_dist_v[0] = par_x_v0[0]
    par_trvt_v[0] = par_t_v0[0]
    for j in range(1, ib+1):
        par_dist_v[j] = par_x_v0[j] + par_x_v1[j-1]
        par_trvt_v[j] = par_t_v0[j] + par_t_v1[j-1]
    par_dist_v[ib+1] = par_x_v1[ib]
    par_trvt_v[ib+1] = par_t_v1[ib]#
    ###
    #print(p, par_dist_v, par_trvt_v)
    ###
    par_dist_v *= 2.0
    par_trvt_v *= 2.0
    return dist, trvt, par_dist_v, par_trvt_v

#### Calculate X-T(-gradient) for a Many rays (many ray parameters)
@jit(nopython=True, nogil=True)
def my_slowness(v, nrp):
    v0   = v[0]
    vmax = v.max()
    pmin = (1+1e-6)/vmax
    theta_min = np.arcsin(v0*pmin)
    theta_max = np.pi*(0.5-1e-6)
    theta = np.linspace(theta_max, theta_min, nrp)
    rps = np.sin(theta) / v0
    #vmin = v[0]
    #vmax = v.max()
    #imin= np.arcsin(vmin/vmax)
    #imax= np.pi*0.5
    #rps = np.linspace(imax, imin, nrp)
    #rps = np.sin(rps) * (1.0/vmin)
    return rps
@jit(nopython=True, nogil=True)
def many_rays_pxt(z, v, nrp):
    rps = my_slowness(v, nrp)
    trvt = np.zeros(nrp, dtype=np.float64)
    dist = np.zeros(nrp, dtype=np.float64)
    for irp, p in enumerate(rps):
        dist[irp], trvt[irp] = single_ray_xt(p, z, v)
    dist /= 6371.0
    dist = np.rad2deg(dist)
    return rps, dist, trvt
@jit(nopython=True, nogil=True)
def many_rays_pxt_gradient(z, v, nrp):
    rps = my_slowness(v, nrp)
    trvt = np.zeros(nrp, dtype=np.float64)
    dist = np.zeros(nrp, dtype=np.float64)
    par_trvt_v = np.zeros((nrp, v.size), dtype=np.float64)
    par_dist_v = np.zeros((nrp, v.size), dtype=np.float64)
    #print('rps:', np.sum(rps) )
    for irp, p in enumerate(rps):
        dist[irp], trvt[irp], par_dist_v[irp,:], par_trvt_v[irp,:] = single_ray_gradient_xt(p, z, v)
    #print('dist:', np.sum(dist))
    #print('trvt:', np.sum(trvt))
    #print('par_dist_v:', np.sum(par_dist_v))
    #print('par_trvt_v:', np.sum(par_trvt_v))
    dist /= 6371.0
    dist = np.rad2deg(dist)
    par_dist_v /= 6371.0
    par_dist_v = np.rad2deg(par_dist_v)
    return rps, dist, trvt, par_dist_v, par_trvt_v

#### interpolation functions for first arrivals
@jit(nopython=True, nogil=True)
def split_pxt_legs(decreasing_ps, xs, ts): # split the p-x-t into legs so that each leg has monotonic p and x.
    ps = decreasing_ps
    dx = np.diff(xs)
    for ind in range(1, len(dx)): # take care of the zeros in dx
        if dx[ind] == 0:
            dx[ind] = dx[ind-1]  # if two xs are the same, use the previous one
    tmp = dx[:-1] * dx[1:]
    idxs_minial_maximal = [0]
    idxs_minial_maximal.extend( np.where( tmp < 0 )[0]+1 )
    idxs_minial_maximal.append(len(xs)-1)
    leg_idx =[np.arange(i0, i1+1) for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    leg_ps = [ps[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    leg_xs = [xs[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    leg_ts = [ts[i0:i1+1] for i0, i1 in zip(idxs_minial_maximal[:-1], idxs_minial_maximal[1:]) ]
    return leg_idx, leg_ps, leg_xs, leg_ts
@jit(nopython=True, nogil=True)
def select_leg(leg_idx, leg_ps, leg_xs, leg_ts): # only select the leg with increasing x
    idx = [ileg for ileg, leg in enumerate(leg_xs) if leg[0] < leg[-1]]
    leg_idx = [leg_idx[it] for it in idx]
    leg_ps = [leg_ps[it] for it in idx]
    leg_xs = [leg_xs[it] for it in idx]
    leg_ts = [leg_ts[it] for it in idx]
    return leg_idx, leg_ps, leg_xs, leg_ts
@jit(nopython=True, nogil=True)
def interp_and_merge(increasing_dist, leg_idx, leg_xs, leg_ts, max_index, flag_interp_index_and_weight=False): # interpolate for each leg and select the minimal time
    """
    :param flag_interp_index_and_weight: true or false to return valid `ileft` and `pleft` which is useful for gradient computation.

    Return: trvt, ileft, pleft
            trvt:  the traveltime for each of the `dist`
            ileft: the index of the left point in the original p-x-t curve for linear interpolation for each of the `dist`
            pleft: the weight of the left point in the original p-x-t curve for linear interpolation for each of the `dist`
            #
            trvt = t[ileft]*pleft + t[ileft+1]*(1-pleft), where `t` is from the original p-x-t curve.
    """
    dist = increasing_dist
    nlegs = len(leg_idx)
    mat_t     = np.zeros((nlegs, dist.size), dtype=np.float64) + 1e100
    mat_ileft = np.zeros((nlegs, dist.size), dtype=np.int64) + max_index
    mat_pleft = np.zeros((nlegs, dist.size), dtype=np.float64)
    #### interp for each leg (each leg has decreasing P and increasing X)
    for irow, (ref_i, ref_x, ref_t) in enumerate(zip(leg_idx, leg_xs, leg_ts)):
        #
        xmin, xmax = ref_x[0], ref_x[-1]
        i0 = np.searchsorted(dist, xmin, side="left") # Note, dist is increasing!
        i1 = np.searchsorted(dist, xmax, side="right")
        if i0 == i1:
            continue
        x = dist[i0:i1]
        #
        mat_t[irow][i0:i1]  = np.interp(x, ref_x, ref_t)
        #print("mat_t:", np.sum(mat_t[irow][i0:i1]) )
        #
        if flag_interp_index_and_weight:
            mat_ileft[irow][i0:i1] = np.interp(x, ref_x, ref_i).astype(np.int64) # the index in the original p-x-t single curve
            #
            tmp = np.searchsorted(ref_x, x)
            idx_left  = np.clip(tmp - 1, 0, len(ref_x) - 2) # the index in this segment
            idx_right = idx_left + 1
            x0, x1    = ref_x[idx_left], ref_x[idx_right]
            mat_pleft[irow][i0:i1] = (x1-x)/(x1-x0)
            #print('float:', np.interp(x, ref_x, ref_i)+1)
            #print("mat_ileft:", mat_ileft[irow][i0:i1]+1 )
    #### get the minimal from all legs
    trvt  = np.zeros(dist.size, dtype=np.float64)
    ileft = np.zeros(dist.size, dtype=np.int64)
    pleft = np.zeros(dist.size, dtype=np.float64)
    if flag_interp_index_and_weight:
        for icol in range(dist.size):
            irow = np.argmin( mat_t[:,icol] )
            trvt[icol]  = mat_t[irow, icol]
            ileft[icol] = mat_ileft[irow, icol]
            pleft[icol] = mat_pleft[irow, icol]
    else:
        for icol in range(dist.size):
            trvt[icol]  = mat_t[:, icol].min()
    trvt = np.where(trvt<1e100, trvt, 0.0)
    return trvt, ileft, pleft

#### trvt(-gradient) for many distances using linear interpolation method
@jit(nopython=True, nogil=True)
def dist2trvt(dist, z, v, nrp):
    p, x, t = many_rays_pxt(z, v, nrp)
    max_index = p.size
    leg_idx, leg_ps, leg_xs, leg_ts = split_pxt_legs(p, x, t)
    leg_idx, leg_ps, leg_xs, leg_ts = select_leg(leg_idx, leg_ps, leg_xs, leg_ts)
    trvt, ileft, pleft = interp_and_merge(dist, leg_idx, leg_xs, leg_ts, max_index, False)
    #check
    trvt2 = t[ileft]*pleft + t[ileft+1]*(1-pleft)
    #print('err:', np.sum(np.abs(trvt-trvt2)))
    return trvt
@jit(nopython=True, nogil=True)
def dist2trvt_jac(dist, z, v, nrp):
    p, x, t, par_x_v, par_t_v = many_rays_pxt_gradient(z, v, nrp) # par_dist_vi and par_trvt_vi have shape (nrp, nlayer+1)
    max_index = p.size-2
    leg_idx, leg_ps, leg_xs, leg_ts = split_pxt_legs(p, x, t)
    leg_idx, leg_ps, leg_xs, leg_ts = select_leg(leg_idx, leg_ps, leg_xs, leg_ts)
    trvt, ileft, pleft = interp_and_merge(dist, leg_idx, leg_xs, leg_ts, max_index, True)
    #print('leg_idx:', np.sum(leg_idx[0]) + np.sum(leg_idx[1]) + np.sum(leg_idx[2]))
    #print('leg_ps:',  np.sum( leg_ps[0]) + np.sum( leg_ps[1]) + np.sum( leg_ps[2]))
    #print('leg_xs:',  np.sum( leg_xs[0]) + np.sum( leg_xs[1]) + np.sum( leg_xs[2]))
    #print('leg_ts:',  np.sum( leg_ts[0]) + np.sum( leg_ts[1]) + np.sum( leg_ts[2]))
    #print('ileft:', np.sum(ileft))
    #print('pleft:', np.sum(pleft))
    par_dist_v = np.zeros( (trvt.size, par_x_v.shape[1]), dtype=np.float64 )
    par_trvt_v = np.zeros( (trvt.size, par_t_v.shape[1]), dtype=np.float64 )
    for idx in range(trvt.size):
        #
        # t = p * tl + (1-p) * tr
        #   = p * (tl-tr) + tr
        # dt/dv = dp/dv * (tl-tr) + p * (dtl/dv - dtr/dv) + dtr/dv
        #       = dp/dv * (tl-tr) + p * dtl/dv + (1-p) * dtr/dv
        # part 1: p * dtl/dv + (1-p) * dtr/dv
        #
        # part 2: dp/dv * (tl-tr)
        # p     = (xr-x)/(xr-xl)
        #       = (xr-x) * (xr-xl)^-1
        # dp/dv = (dxr/dx) * (xr-xl)^-1 - (xr-x) * (xr-xl)^-2 * (dxr/dx-dxl/dx)
        #
        i0, i1 = ileft[idx], ileft[idx]+1
        c0, c1 = pleft[idx], (1-pleft[idx])
        par_dist_v[idx,:] = par_x_v[i0,:] * c0  +  par_x_v[i1,:] * c1
        par_trvt_v[idx,:] = par_t_v[i0,:] * c0  +  par_t_v[i1,:] * c1
        #####
        x0 = dist[idx]
        xl, xr = x[i0], x[i1]
        part_alpha_v = (par_x_v[i1, :]/(xr-xl) - (xr-x0)/(xr-xl)/(xr-xl) * (par_x_v[i1, :]-par_x_v[i0, :]))
        par_trvt_v[idx,:] += part_alpha_v * (t[i0]-t[i1])
        #####
        #print('internal:', idx, par_t_v[i0,:], par_t_v[i1,:], par_x_v[i0,:], par_x_v[i1,:])
    #print('par_x_v:', np.sum(par_x_v))
    #print('par_x_v:', np.sum(par_x_v))
    #check
    #trvt2 = t[ileft]*pleft + t[ileft+1]*(1-pleft)
    #print('here', np.sum(trvt-trvt2) )
    #print('err:', np.sum(np.abs(trvt-trvt2)))
    return trvt, par_dist_v, par_trvt_v

#### Get OC reference models. Return R(km), Vp(km/s) with decreasing R!
def rd_ak135_OC_model(fnm='model_ak135.txt'):
    tab = np.loadtxt(fnm, comments='#')
    depth = tab[:, 0]
    r = 6371.0 - depth
    vp = tab[:, 2]
    return r, vp
def rd_prem_OC_model(fnm='model_prem.txt'):
    tab = np.loadtxt(fnm, comments='#')
    depth = tab[:, 0]
    r = 6371.0 - depth
    vp = tab[:, 2]
    return r, vp

#### OBJECTIVE function and gradient function for optimization
def get_obj_and_grad_func(dist, trvt_obs, std, model_z, nrp=10000,
                          ref_model_vz=None, alpha=0.0, beta=0.0):
    """
    Generate two functions: `obj_func(m)` and `obj_grad(m)` for optimization.
    """
    inv_var = 1.0/(std*std)
    model_sz = len(model_z)
    if ref_model_vz is None:
        alpha = 0.0
        beta  = 0.0
        ref_model_vz = np.zeros(model_sz, dtype=np.float64)
    ######### objective functions #########
    @jit(nopython=True, nogil=True)
    def obj_data_diff(model_vz):
        d = dist2trvt(dist, model_z, model_vz, nrp)
        tmp = (d-trvt_obs)
        return np.sum(tmp*tmp*inv_var)
    @jit(nopython=True, nogil=True)
    def obj_model_d0(model_vz):
        v2 = model_vz - ref_model_vz
        return np.sum(v2*v2)
    @jit(nopython=True, nogil=True)
    def obj_model_d1(model_vz):
        v3 = np.diff(model_vz-ref_model_vz)
        return np.sum(v3*v3)
    #########
    @jit(nopython=True, nogil=True)
    def obj_func(model_vz):
        v = obj_data_diff(model_vz)
        if alpha > 0.0:
            v += alpha * obj_model_d0(model_vz)
        if beta > 0.0:
            v += beta  * obj_model_d1(model_vz)
        return v
    ######### objective gradient #########
    @jit(nopython=True, nogil=True)
    def obj_grad(model_vz):
        d, _, jac = dist2trvt_jac(dist, model_z, model_vz, nrp)
        tmp = 2*(d-trvt_obs)*inv_var
        #grad = np.zeros(model_sz, dtype=np.float64)
        #for j in range(model_sz):
        #    grad[j] = np.sum(tmp * jac[:,j])
        grad = tmp @ jac
        #####
        if alpha > 0.0:
            grad2 = 2*alpha*(model_vz - ref_model_vz)
            grad += grad2
        #####
        if beta > 0.0:
            tmp = np.diff( model_vz-ref_model_vz )
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
def benchmark_my_trvt_gradient():
    ####
    #r = np.linspace(3500, 500, 3)
    #vr = np.linspace(5.0, 8.0, 3)
    ####
    #z, vz = flatten(r, vr, 6371.0)
    z  = np.linspace(0, -3000, 3)
    vz = np.linspace(5.0, 8.0, 3)
    #print(z, vz)
    #1. postive Z is upward
    #2. the array z and vz should go from surface to depth
    #3. e.g., z=0, -100, -200, -300,
    #4.      vz[0], v[-100], vz[-200], vz[-300]...
    ####
    nrp = 10000
    p, x, t = many_rays_pxt(z, vz, nrp)
    ####
    dist = np.linspace(5, 110, 20) # where the data are
    ####
    trvt_ = dist2trvt(dist, z, vz, nrp=nrp)
    trvt, junk, par_trvt_v = dist2trvt_jac(dist, z, vz, nrp=nrp)
    par_trvt_v2 = np.zeros(par_trvt_v.shape)
    for iz in range(vz.size):
        vz1 = vz.copy()
        vz2 = vz.copy()
        vz1[iz] *= (1-1e-6)
        vz2[iz] *= (1+1e-6)
        trvt1 = dist2trvt(dist, z,  vz1, nrp=nrp)
        trvt2 = dist2trvt(dist, z,  vz2, nrp=nrp)
        junk = (trvt2 - trvt1) / (2e-6*vz[iz])
        par_trvt_v2[:, iz] = junk
    np.set_printoptions(suppress=True, precision=6)
    print('#### Analytical')
    print(par_trvt_v[:20,])
    print('#### FD')
    print(par_trvt_v2[:20])
    print('#### Analytical vs. FD')
    print((par_trvt_v2[:20]-par_trvt_v[:20])/ par_trvt_v[:20])
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
    r  = [3500,  3000, 3000, 2500, 2000, 2000, 1500, 1000, 500]
    #vr = [  6.0,  6.1,  10,  10.5, 11,   11,    16,  16.1, 16.2]
    vr = [  12,  6.1,  10,  10.5, 11,   15.5,    16,  16.5, 19]
    #vr = [  6.0,  6.1,  10,  10.5, 11,   15.5,    16,  16.5, 17]
    r, vr = np.array(r), np.array(vr)
    z, vz = flatten(r, vr, 6371.0)
    #
    rp1, x1, t1       = many_rays_pxt(z, vz, 100)
    rp2, x2, t2, _, _ = many_rays_pxt_gradient(z, vz, 100)  # just to make sure the global variables are initialized
    ####   gradient
    dist = np.linspace(5, 110, 20)
    nrp = 1000
    syn_trvt, junk, par_syn_trvt_v = dist2trvt_jac(dist, z, vz, nrp)
    ######
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
    ax1.plot(vr, r)
    ax2.plot(x1, t1, label='many_rays_pxt')
    ax2.plot(x2, t2, label='many_rays_pxt_gradient')
    ax2.plot(dist, syn_trvt, 'o', markersize=2, color='r', label='dist2trvt_jac')
    ax3.imshow(par_syn_trvt_v, interpolation='none', aspect='auto', extent=[0, vz.size, dist[0], dist[-1]], cmap='RdBu', origin='lower')
    plt.show()

if __name__ == "__main__":
    plot_benchmark_my_trvt_gradient()
    #benchmark_my_trvt_gradient()
