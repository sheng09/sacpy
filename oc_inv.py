#!/usr/bin/env python3

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import sacpy

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
def single_ray_xt_grad(p, z, v):
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
    par_trvt_p = 0.0 # single values
    par_dist_p = 0.0 # single values
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
            par_trvt_p += v[i  ]*inv_k[i]/(tan_half_theta[i  ]*(1.0+cos_theta[i  ])*cos_theta[i  ])
            par_trvt_p -= v[i+1]*inv_k[i]/(tan_half_theta[i+1]*(1.0+cos_theta[i+1])*cos_theta[i+1])
            #
            par_x_v0[i] =  x[i]*inv_k[i]/dz[i] + inv_k[i]*sin_theta[i  ]/cos_theta[i  ]
            par_x_v1[i] = -x[i]*inv_k[i]/dz[i] - inv_k[i]*sin_theta[i+1]/cos_theta[i+1]
            #
            par_dist_p -= x[i]*inv_p
            par_dist_p += v[  i]*inv_p*inv_k[i]*sin_theta[i  ]/cos_theta[i  ]
            par_dist_p -= v[i+1]*inv_p*inv_k[i]*sin_theta[i+1]/cos_theta[i+1]
        else: # dv[i] == 0.0
            t[i] = -dz[i] / (v[i]*cos_theta[i])
            x[i] = -dz[i] * (sin_theta[i]/cos_theta[i])
            #
            par_t_v0[i] = -t[i]/(v[i]+v[i+1]) + p*t[i]*sin_theta[i]/(cos_theta[i]*(cos_theta[i]+cos_theta[i+1]))
            par_t_v1[i] = par_t_v0[i]
            #
            par_trvt_p += v[i  ]*t[i]*sin_theta[i]/(cos_theta[i]*(cos_theta[i]+cos_theta[i+1]))
            par_trvt_p += v[i+1]*t[i]*sin_theta[i]/(cos_theta[i]*(cos_theta[i]+cos_theta[i+1]))
            #
            par_x_v0[i] = 0.5*p*dz[i]/(cos_theta[i]*cos_theta[i]*cos_theta[i])
            par_x_v1[i] = par_x_v0[i]
            #
            par_dist_p += 0.5*v[i  ]*dz[i]/(cos_theta[i]*cos_theta[i]*cos_theta[i])
            par_dist_p += 0.5*v[i+1]*dz[i]/(cos_theta[i]*cos_theta[i]*cos_theta[i])
    #
    if np.abs(dz[ib]) > 1e-10:
        t[ib] = inv_k[ib] * ln_tan_half_theta[ib] # Note inv_k[ib] is finite automatically! ? really?
        x[ib] = inv_p * inv_k[ib] * (-cos_theta[ib])
        # cos_theta[ib] is zero for distance=0 case.
        par_t_v0[ib] =  t[ib]*inv_k[ib]/dz[ib] + p*inv_k[ib]/(tan_half_theta[ib]*(1.0+cos_theta[ib])*cos_theta[ib])
        par_t_v1[ib] = -t[ib]*inv_k[ib]/dz[ib]
        par_x_v0[ib] =  x[ib]*inv_k[ib]/dz[ib] + inv_k[ib]*sin_theta[ib]/cos_theta[ib]
        par_x_v1[ib] = -x[ib]*inv_k[ib]/dz[ib]
        #
        par_trvt_p += v[ib]*inv_k[ib]/(tan_half_theta[ib]*(1.0+cos_theta[ib])*cos_theta[ib])
        par_dist_p -= x[ib]*inv_p
        par_dist_p += v[ib]*inv_p*inv_k[ib]*sin_theta[ib]/cos_theta[ib]
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
    if ps[0] < ps[1]:
        ps = ps[::-1]
        xs = xs[::-1]
        ts = ts[::-1]
        dx = dx[::-1]*-1 ## reverse and change sign
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
def many_rays_pxt(z, vz, theta_step_deg=0.1):
    theta_step_rad = theta_step_deg * (np.pi/180.0)
    v0 = vz[0]
    nlayers = vz.size - 1
    rp_legs, dist_legs, trvt_legs = list(), list(), list()
    vmax_above_current_layer = v0 #
    for ilayer in range(nlayers):
        vtop = vz[ilayer]
        vbot = vz[ilayer+1]
        if vmax_above_current_layer < vbot and vtop < vbot: # make sure ray can turn in this layer
            #######
            if vmax_above_current_layer >= vtop:
                # make sure the ray can enter this layer!
                # this also avoids the zero distance case in the first layer
                #vtop = vmax_above_current_layer + (vbot - vmax_above_current_layer)*1e-15 #the 1e-12 may still cause numerical issue?
                vtop = np.nextafter(vmax_above_current_layer, vbot) # use this smart method
            #######
            t0, t1 = np.arcsin(v0/vtop), np.arcsin(v0/vbot)
            nrp    = int(np.abs(t0-t1)/theta_step_rad) + 2
            thetas = np.linspace( t0, t1, nrp)
            rps    = np.sin(thetas)/v0
            ##
            dist= np.zeros(nrp, dtype=np.float64)
            trvt= np.zeros(nrp, dtype=np.float64)
            for irp, p in enumerate(rps):
                dist[irp], trvt[irp] = single_ray_xt(p, z, vz)
            tmp1, tmp2, tmp3 = split_pxt_legs(rps, dist, trvt)
            rp_legs.extend(tmp1)
            dist_legs.extend(tmp2)
            trvt_legs.extend(tmp3)
            ####
            vmax_above_current_layer = vbot ##### note vbot is alreay the max until now!
    ########
    # flatten the list of list
    rp   = np.array([it for leg in rp_legs for it in leg])
    dist = np.array([it for leg in dist_legs for it in leg])
    trvt = np.array([it for leg in trvt_legs for it in leg])
    idx_sort = np.argsort(rp)
    rp = rp[idx_sort]
    dist = dist[idx_sort]
    trvt = trvt[idx_sort]
    #########
    return (rp, dist, trvt), (rp_legs, dist_legs, trvt_legs)
@jit(nopython=True, nogil=True)
def single_dist2trvt(target_single_dist, z, vz, rp_legs, dist_legs, trvt_legs, critical_dist_err=1e-20, niter=1000):
    s_rp, s_dist, s_trvt = np.nan, np.nan, np.nan
    flag_none_found=True
    m_rp, m_dist, m_trvt = list(), list(), list()
    for ileg, rps in enumerate(rp_legs):
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
                d_left,  d_right  = dists[i1-1], dists[i1]
                #### start ray tracing with bisection method
                for idx_iter in range(niter):
                    rp_mid = 0.5 * (rp_left + rp_right)
                    d_mid, t_mid = single_ray_xt(rp_mid, z, vz)
                    #print('  iter:', idx_iter, d_left, d_mid, d_right, '|', rp_left, rp_mid, rp_right, )
                    if np.abs(d_mid-target_single_dist) < critical_dist_err:
                        ### found!!!
                        break
                    elif (d_left <= target_single_dist <= d_mid) or (d_mid <= target_single_dist <= d_left):
                        rp_right = rp_mid
                        d_right = d_mid
                    else:
                        rp_left = rp_mid
                        d_left = d_mid
                rp_found   = rp_mid
                dist_found = d_mid
                trv_found  = t_mid
            ###########
            #print(target_single_dist, dist_found, rp_found, trv_found)
            m_rp.append(rp_found)
            m_dist.append(dist_found)
            m_trvt.append(trv_found)
            if flag_none_found:
                s_rp, s_dist, s_trvt = rp_found, dist_found, trv_found
                flag_none_found = False
            elif trv_found < s_trvt:
                s_rp, s_dist, s_trvt = rp_found, dist_found, trv_found
    return (s_rp, s_dist, s_trvt), (np.array(m_rp), np.array(m_dist), np.array(m_trvt) )
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
    for idx, single_dist in enumerate(dist):
        tmp = single_dist2trvt(single_dist, z, vz, rp_legs, dist_legs, trvt_legs, critical_dist_err, niter)
        rp_found[idx]   = tmp[0][0]
        dist_found[idx] = tmp[0][1]
        trvt_found[idx] = tmp[0][2]
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
    for idx, rp in enumerate(rp_found):
        if not np.isnan(rp): ### Nan means no ray exist for this distance
            _, _, _, _, _, _, d_trvt_v[idx] = single_ray_xt_grad(rp, z, vz)
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
def get_obj_and_grad_func(dist, trvt_obs, std, model_z, theta_step_deg=0.1,
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
        _,_,d = many_dist2trvt(dist, model_z, model_vz, theta_step_deg=theta_step_deg)
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
        _, _, d, jac = many_dist2trvt_jac(dist, model_z, model_vz, theta_step_deg=theta_step_deg)
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
    benchmark_my_trvt_gradient()
