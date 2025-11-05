#!/usr/bin/env python3

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import sacpy
import sys

from sacpy.oc_inv import denser_xy, flatten, unflatten, rd_prem_model, p2xt_grad, p2xt_v4, p2xt_grad_v4, zv2p_v3, zv2p_reflect_v3
from sacpy.oc_inv import split_pxt_legs_v2

@jit(nopython=True, nogil=True)
def p2xt(inv_p, z, v, dz, dv): # input inv_p the sametime for numerical issue
    """
    Given a linear layer vz= k*z+b, (for z0,v0, and z1,v1, z0!=z1)
        If k != 0:
            t = 1/k * [ln(tan(a0/2)) - ln(tan(a1/2)) ]
              = 1/k * [ln(sin0/(1+cos0)) - ln(sin1/(1+cos1)) ]
              = 1/k * [ln(p*v0/(1+cos0)) - ln(p*v1/(1+cos1)) ]
              = 1/k * [ln(v0/(1+cos0)) - ln(v1/(1+cos1)) ]
            x = 1/k/p * (cos1-cos0)
    """
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
            #if local_dzdv * (ln_v0 - ln_v1) < 0.0:
            #    print('Error in p2xt: local_dzdv * (ln_v0 - ln_v1) < 0.0')
        elif dz[i] < -1e-10:  # dv[i] ==0.  Note. dz is always negative
            t -= dz[i] / (v[i]*cos_theta_i0)
            x -= dz[i] * (sin_theta_i0/cos_theta_i0)
        #
        sin_theta_i0 = sin_theta_i1
        cos_theta_i0 = cos_theta_i1
        ln_v0 = ln_v1
    if ib != z.size-1: # the ray turns in layer ib, or skip this if the ray penetrate all layers
        cos_theta_i1 = 0.0
        ln_v1 = np.log(inv_p/(1.0+cos_theta_i1) ) # ln( v1/(1+cos1) )
        if dz[ib] < -1e-10 and np.abs(dv[ib]) > 1e-10:  # dz is always negative
            local_dzdv = dz[ib]/dv[ib]
            t += local_dzdv * (ln_v0-ln_v1)
            x += local_dzdv* (inv_p * (-cos_theta_i0))
    return x*2.0, t*2.0
@jit(nopython=True, nogil=True)
def z2inv_rp_turn_back(z, vz, debug=0):
    """
    Get the inverse ray parameters for rays that can going down from surface and then turn back to surface.
    #
    z:
    vz:
    #
    Returns: inv_rps, ibs, ibreaks
    inv_rps:   1D array of inv_rp associated with rays that can turn back to surface and also
               critically refract at an interface.
    ibs:       1D array of layer index for each inv_rp where the ray refracts.
               Note, a layer includes its bottom interface but exclude its top interface.
               So, ib means that a ray refracts at within the depth interval ( z[ib], z[ib+1] ].
    ibreaks:   1D array of indices in inv_rps where a new rps segments starts.
               The istart for all segments are ibreaks[:-1].
               The iend   for all segments are ibreaks[1:].
               #
               E.g., the 1st segment has: inv_rps[ibreaks[0]: ibreaks[1]], and iz[ibreaks[0]: ibreaks[1]],
                     the 2nd ...     has: inv_rps[ibreaks[1]: ibreaks[2]], and iz[ibreaks[1]: ibreaks[2]],
                     the 3rd ...     has: inv_rps[ibreaks[2]: ibreaks[3]], and iz[ibreaks[2]: ibreaks[3]],
                     ... and so on.
    """
    ########################################################################
    # debug plot
        #if debug:
        #    nz = 20
        #    z  = np.linspace(0, -1000, nz)
        #    vz_mono = np.linspace(10, 20, nz)
        #    dvz     = (np.random.random(nz)-0.5)*2
        #    vz      = vz_mono + dvz
        #    inv_rps, ind_zs, ibreaks = z2vp_turn_back(z, vz, debug=False)
        #    inv_rps_z = z[ind_zs]
        #    #####
        #    cmap = plt.get_cmap('tab10')
        #    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
        #    ax1.plot(vz, z, '.-', color='k')
        #    #ax2.plot(dvz, z, '.-', color='k')
        #    for ileg, (istart, iend) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
        #        local_rps = inv_rps[istart:iend+1]
        #        local_z   = inv_rps_z[istart:iend+1]
        #        ax1.plot(local_rps, local_z, 's-', lw=5, color=cmap(ileg%10), label=f'leg {ileg}')
        #        ax1.plot(local_rps*0+local_rps[0], local_z, 's-', lw=5, color=cmap(ileg%10), label=f'leg {ileg}')
        #        #ax1.plot(local_rps, local_z, 's-', lw=5) #, color=cmap(ileg%10), label=f'leg {ileg}')
        #        #for rp in local_rps:
        #        #    ax1.plot([rp, rp], [z[0], z[-1]], '--', color=cmap(ileg%10), alpha=0.5)
        #    ax1.plot(inv_rps, inv_rps_z, '-', color='k', lw=1)
        #    ##### check ind_zs
        #    for inv_rp, iz in zip(inv_rps, ind_zs):
        #        print(vz[iz] == inv_rp)
        #    plt.show()
    ###################################################################################################
    # Step 1. Get the interface indices with increasing vz excluding the 0th interface.
            # E.g. (1): [1, 2, 10, 11, 12, 20, 30]
            # E.g. (2): [      10, 11, 12, 20, 30]
            # E.g. (3): [1,    10, 11, 12, 20, 30]
            # The discontinuities correspond to seismic shadow zone (discontinous X for continuous P).
    # Step 2. Split the indices into segments due to the discontinuities.
            # Note, a Zero will be inserted at the starting of the first segment.
            # E.g. (1): [0, 1, 2], [2, 10, 11, 12], [12, 20], [20, 30]
            # E.g. (2):            [0, 10, 11, 12], [12, 20], [20, 30]
            # E.g. (3): [0, 1],    [1, 10, 11, 12], [12, 20], [20, 30]
    # Step 3. Convert the index segments to inv_rps, ibs.
            # Note1: for each interface index iz, the inv_rp = vz[iz] except the starting point of
            # each segment, where its inv_rp equals to vz[iz]+
            # Note2: for each interface index iz, the ib = iz-1, except the starting point of each
            # segment, where its ib equals the ib of the segment's second ib, as shown below.
            # E.g. (1): inv_rps = vz @ [0+, 1, 2], [2+, 10, 11, 12], [12+, 20], [20+, 30]
            #           ibs     =      [0,  0, 1], [9,   9, 10, 11], [19,  19], [29,  29]
            # E.g. (2): inv_rps = vz @             [0+, 10, 11, 12], [12+, 20], [20+, 30]
            #           ibs     =                  [9,   9, 10, 11], [19,  19], [29,  29]
            # E.g. (3): inv_rps = vz @ [0+, 1],    [1+, 10, 11, 12], [12+, 20], [20+, 30]
            #           ibs     =      [0,  0],    [9,   9, 10, 11], [19,  19], [29,  29]
    # Step 4. Avoid using LIST. Use numpy arrays. Instead, using
            # E.g. (1): inv_rps = vz @ [0+, 1, 2, | 2+, 10, 11, 12, | 12+, 20, | 20+, 30]
            #           ibs     =      [0,  0, 1, | 9,   9, 10, 11, | 19,  19, | 29,  29]
            #                     index 0   1  2  | 3   4   5   6   | 7    8   | 9    10
            #           ibreaks =      [0,        | 3,              | 7,       | 9      | 11]
            #
            # E.g. (2): inv_rps = vz @             [0+, 10, 11, 12  | 12+, 20  | 20+, 30]
            #           ibs     =                  [9,   9, 10, 11  | 19,  19  | 29,  29]
            #                     index             0   1   2   3   | 4    5   | 6    7
            #           ibreaks =                  [0               | 4        | 6      | 8 ]
            #
            # E.g. (3): inv_rps = vz @ [0+, 1   |   1+, 10, 11, 12  |  12+, 20 |  20+, 30]
            #           ibs     =      [0,  0   |   9,   9, 10, 11  |  19,  19 |  29,  29]
            #                     index 0   1   |   2   3   4   5   |  6    7  |  8    9
            #           ibreaks =      [0       |   2               |  6       |  8     | 10 ]
    ###################################################################################################
    ##if debug:
    ##    print('vz=', vz)
    # step 1:
    vmax = vz[0]
    ind_mono_vzs = np.zeros(vz.size, dtype=np.int64)
    n = 0
    for iz in range(1, vz.size):
        if vmax < vz[iz]:
            ind_mono_vzs[n] = iz
            n += 1
            vmax = vz[iz]
    ind_mono_vzs = ind_mono_vzs[:n]
    ##if debug:
    ##    print('ind_mono_vzs=', ind_mono_vzs)
    # step 2:
    ind_zs = np.zeros(vz.size*2, dtype=np.int64)
    nzs = 1 # the Zero inserted at the start
    ibreaks = np.zeros(ind_zs.size+1, dtype=np.int64)
    nbreaks = 1
    for im in range(ind_mono_vzs.size-1):
        if (ind_mono_vzs[im+1]-ind_mono_vzs[im]) == 1:
            ind_zs[nzs] = ind_mono_vzs[im]
            nzs += 1
        else:
            ind_zs[nzs] = ind_mono_vzs[im]
            nzs += 1
            #
            ibreaks[nbreaks] = nzs
            nbreaks += 1
            #
            ind_zs[nzs] = ind_mono_vzs[im]
            nzs += 1
    ind_zs[nzs] = ind_mono_vzs[-1]
    nzs += 1
    ibreaks[nbreaks] = nzs
    nbreaks += 1
    ind_zs = ind_zs[:nzs]
    ibreaks = ibreaks[:nbreaks]
    ##if debug:
    ##    print('ind_zs=', ind_zs)
    ##    print('ibreaks=', ibreaks)
    ##    #
    ##    for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
    ##        print('ileg=', ileg, ind_zs[i0:i1])
    # step 3
    inv_rps = vz[ind_zs] #np.zeros(ind_zs.size, dtype=np.float64)
    ibs     = ind_zs-1   #np.zeros(ind_zs.size, dtype=np.float64)
    for iseg in range(ibreaks.size-1):
        istart = ibreaks[iseg]
        #inv_rps[istart] = np.nextafter(inv_rps[istart], inv_rps[istart+1]) # inv_rps is mono increasing!
        inv_rps[istart] += 1e-6
        ibs[istart] = ibs[istart+1]
    ##if debug:
    ##    print('inv_rps=', inv_rps)
    ##    print('ibs=', ibs)
    ##    for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
    ##        print('ileg=', ileg, 'inv_rps=', inv_rps[i0:i1])
    ##    for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
    ##        print('ileg=', ileg, 'ibs=', ibs[i0:i1])
    #######
    return inv_rps, ibs, ibreaks
@jit(nopython=True, nogil=True)
def z2inv_rp_penetrate(z, vz, debug=0):
    """
    Get the inverse ray parameters for rays that can penetrate all interfaces, including
    the one that critically refract at the bottom-most interface.
    #
    z:
    vz:
    #
    Return: inv_rps, ibs, ibreaks
    Refer to z2inv_rp_turn_back(...)'s docstring for details about the return values.
    """
    vmax = np.max(vz)
    inv_rps = np.array((vmax, 1.0e100), dtype=np.float64) # 1e100 means infinite here for ray parameter =0.0
    ibs     = np.array((z.size-2, z.size-1), dtype=np.int64)
    ibreaks = np.array((0,2), dtype=np.int64)
    if inv_rps[0] != vz[-1]:
        # If the vmax is not at the bottom-most interface, need to increase the starting inv_rp a little bit.
        #inv_rps[0] = np.nextafter(inv_rps[0], inv_rps[0]+2.0)
        inv_rps[0] += 1e-6
        # Also, the ib for the starting ray is not critically refraction at the bottom-most interface (ib=z.size-2), but a penetration (ib=z.size-1).
        ibs[0] = z.size-1
    return inv_rps, ibs, ibreaks
@jit(nopython=True, nogil=True)
def z2inv_rp_both(z, vz, debug=0): # for both turn-back and penetrating rays
    inv_vps1, ibs1, ibreaks1 = z2inv_rp_turn_back(z, vz)
    inv_vps2, ibs2, ibreaks2 = z2inv_rp_penetrate(z, vz) # only two points and one segment
    inv_rps = np.zeros(inv_vps1.size + inv_vps2.size, dtype=np.float64)
    ibs     = np.zeros(ibs1.size     + ibs2.size,     dtype=np.int64)
    ibreaks = np.zeros(ibreaks1.size + 1, dtype=np.int64)
    inv_rps[:-2] = inv_vps1
    inv_rps[-2:] = inv_vps2
    ibs[:-2]     = ibs1
    ibs[-2:]     = ibs2
    ibreaks[:-1] = ibreaks1
    ibreaks[-1]  = inv_rps.size
    ####
    return inv_rps, ibs, ibreaks

@jit(nopython=True, nogil=True)
def denser_inv_rp_interval(inv_rp_min, inv_rp_max, ib_min, ib_max, v0, size, buf_inv_rps, buf_ibs):
    ang0 = np.arcsin(v0/inv_rp_min)
    ang1 = np.arcsin(v0/inv_rp_max)
    buf_inv_rps[:] = v0 / np.sin( np.linspace(ang0, ang1, size) )
    buf_inv_rps[0] = inv_rp_min
    buf_inv_rps[-1]= inv_rp_max
    buf_ibs[:] = ib_max
    buf_ibs[0] = ib_min
@jit(nopython=True, nogil=True)
def denser_inv_rps_old(inv_rps, ibs, ibreaks, v0, max_theta_step_rad=0.0017, max_denser_coef=-1):
    """
    Return a new inv_rps, ibs, ibreaks with denser sampling of incidence angles.
    ###################################################################################################
    # Algorithm:
    # Set the 1st break `denser_ibreaks[0] = 0`
    # For each segments:
        Get the inv_rp segment `seg`
        For each interval <-- two consecutive values in `seg`:
            Get `local_denser_inv_rps` and `local_denser_ibs` using denser_inv_rp_interval(..., n)
            Remove the last point of `local_denser_inv_rps` and `local_denser_ibs`.
            Extend the processed `local_denser_inv_rps` and `local_denser_ibs` to its global arrays.
            # (n - 1) points added for each interval.
        Append the last point of the segments to `denser_inv_rps` and `denser_ibs`.
        Append current length of `denser_inv_rps` to `denser_ibreaks`.
        #    (n - 1)* (len(seg)-1 ) + 1  added for each segment.
        #  = (n - 1)* len(seg) + 2 - n
    # To sum, the 0th seg brings: (n-1)*len(seg0)+2-n
    #         the 1st seg brings: (n-1)*len(seg1)+2-n
    #         ...
              SUM:                (n-1)* (len(seg0)+len(seg1)+...) + (2-n) * nsegs
                                = (n-1)* len(inv_rps) + (2-n) * nsegs
    """
    nsegs   = ibreaks.size - 1
    istarts = ibreaks[:-1]
    iends   = ibreaks[1:]
    #####
    theta = np.arcsin(v0/inv_rps)
    max_theta_jump = -np.min(np.diff(theta) ) # theta is mono descending
    n = int(max_theta_jump/max_theta_step_rad) + 2
    nrp = (n-1)*inv_rps.size + (2-n)*nsegs
    n_minus_one = n-1
    denser_inv_rps = np.zeros(nrp, dtype=np.float64)
    denser_ibs     = np.zeros(nrp, dtype=np.int64)
    denser_ibreaks = np.zeros(ibreaks.size, dtype=np.int64) # same size as ibreaks
    global_idx = 0
    for iseg in range(nsegs):
        i0 = istarts[iseg]
        i1 = iends[iseg]
        # the valid is inv_rps[i0:i1], or the index, i0,i0+1,...,i1-1
        for irp in range(i0, i1-1):
            inv_rp_min = inv_rps[irp]
            inv_rp_max = inv_rps[irp+1]
            ib_min     = ibs[irp]
            ib_max     = ibs[irp+1]
            buf_inv_rps = denser_inv_rps[global_idx:global_idx+n]
            buf_ibs     = denser_ibs[    global_idx:global_idx+n]
            denser_inv_rp_interval(inv_rp_min, inv_rp_max, ib_min, ib_max, v0, n, buf_inv_rps, buf_ibs)
            global_idx += n_minus_one # well, the last point will be removed here, except for the last one of the segment
        ## add the last point of this segment
        denser_inv_rps[global_idx] = inv_rps[i1-1]
        denser_ibs[    global_idx] = ibs[  i1-1]
        global_idx += 1
        denser_ibreaks[iseg+1] = global_idx
    return denser_inv_rps, denser_ibs, denser_ibreaks
@jit(nopython=True, nogil=True)
def denser_inv_rps(inv_rps, ibs, ibreaks, v0, max_theta_step_rad=0.0017, max_denser_coef=-1):
    """
    Return a new inv_rps, ibs, ibreaks with denser sampling of incidence angles.
    """
    nsegs   = ibreaks.size - 1
    istarts = ibreaks[:-1]
    iends   = ibreaks[1:]
    #####
    theta = np.arcsin(v0/inv_rps)
    n_per_layer = np.zeros(theta.size-1, dtype=np.int64)
    nrp = 0
    for iseg in range(nsegs):
        i0 = istarts[iseg]
        i1 = iends[iseg]
        # the valid is inv_rps[i0:i1], or the index, i0,i0+1,...,i1-1
        for ilayer in range(i0, i1-1):
            a0 = theta[ilayer]
            a1 = theta[ilayer+1]
            local_n = int( np.abs(a1 - a0)/max_theta_step_rad ) + 2
            n_per_layer[ilayer] = local_n
            nrp += local_n-1
        nrp += 1 # the last point of this segment
    #####
    denser_inv_rps = np.zeros(nrp, dtype=np.float64)
    denser_ibs     = np.zeros(nrp, dtype=np.int64)
    denser_ibreaks = np.zeros(ibreaks.size, dtype=np.int64) # same size as ibreaks
    global_idx = 0
    for iseg in range(nsegs):
        i0 = istarts[iseg]
        i1 = iends[iseg]
        # the valid is inv_rps[i0:i1], or the index, i0,i0+1,...,i1-1
        for irp in range(i0, i1-1):
            inv_rp_min = inv_rps[irp]
            inv_rp_max = inv_rps[irp+1]
            ib_min     = ibs[irp]
            ib_max     = ibs[irp+1]
            local_n    = n_per_layer[irp]
            buf_inv_rps = denser_inv_rps[global_idx:global_idx+local_n]
            buf_ibs     = denser_ibs[    global_idx:global_idx+local_n]
            denser_inv_rp_interval(inv_rp_min, inv_rp_max, ib_min, ib_max, v0, local_n, buf_inv_rps, buf_ibs)
            global_idx += local_n - 1 # well, the last point will be removed here, except for the last one of the segment
        ## add the last point of this segment
        denser_inv_rps[global_idx] = inv_rps[i1-1]
        denser_ibs[    global_idx] = ibs[  i1-1]
        global_idx += 1
        denser_ibreaks[iseg+1] = global_idx
    return denser_inv_rps, denser_ibs, denser_ibreaks
@jit(nopython=True, nogil=True)
def cut_and_interp_line(x, y, xstart, xend): # x must be increasing float, and y not-decreasing int
    """
    Given a line defined by (x, y), cut the line to the interval [xstart, xend], and
    the new line must start at xstart and end at xend, with linear interpolation if needed.
    """
    if xstart > xend or xstart > x[-1] or xend < x[0]:
        return x[:0], y[:0] # empty arrays
    xstart = max(xstart, x[0])
    xend   = min(xend,   x[-1])
    #####
    #if xstart == xend: # not needed
    #    y_value = np.interp(xstart, x, y)
    #    return np.array([xstart]), np.array([y_value])
    #####
    istart = np.searchsorted(x, xstart, side='left') # x[istart-1] <  xstart <= x[istart]
    iend   = np.searchsorted(x, xend, side='right')  # x[iend-1]   <= xend   <  x[iend]
    #####
    if xstart != x[istart]: # xstart < x[istart]
        #ystart = np.interp(xstart, x, y)
        ystart = y[istart] # the y is int, so use the value just after xstart
        if x[iend-1] != xend: #  x[iend-1] < xend < x[iend]
            #yend = np.interp(xend, x, y)
            yend = y[iend] # the y is int, so use the value just after xend
            new_x = x[istart-1:iend+1].copy()
            new_y = y[istart-1:iend+1].copy()
            new_x[0]   = xstart
            new_y[0]   = ystart
            new_x[-1]  = xend
            new_y[-1]  = yend
        else:
            new_x = x[istart-1:iend].copy()
            new_y = y[istart-1:iend].copy()
            new_x[0]   = xstart
            new_y[0]   = ystart
    else:
        if x[iend-1] != xend: #  x[iend-1] < xend < x[iend]
            #yend = np.interp(xend, x, y)
            yend = y[iend] # the y is int, so use the value just after xend
            new_x = x[istart:iend+1].copy()
            new_y = y[istart:iend+1].copy()
            new_x[-1]  = xend
            new_y[-1]  = yend
        else:
            new_x = x[istart:iend].copy()
            new_y = y[istart:iend].copy()
    return new_x, new_y
@jit(nopython=True, nogil=True)
def common_x_two_lines(x1, y1, x2, y2): # x must be increasing float, and y not-decreasing int
    """
    Get two new lines as subsets of the input two lines, and also the two new lines
    have the common x intervals as the intersection of x1 and x2.
    """
    x_start = max(x1[0], x2[0])
    x_end   = min(x1[-1], x2[-1])
    if x_start <= x_end:
        new_x1, new_y1 = cut_and_interp_line(x1, y1, x_start, x_end)
        new_x2, new_y2 = cut_and_interp_line(x2, y2, x_start, x_end)
        ######
        common_x  = np.union1d(new_x1, new_x2)
        ind1 = np.searchsorted(new_x1, common_x, side='left')
        common_y1 = new_y1[ind1]
        ind2 = np.searchsorted(new_x2, common_x, side='left')
        common_y2 = new_y2[ind2]
        return common_x, common_y1, common_y2
    else:
        return x1[:0], y1[:0], y2[:0] # empty arrays
@jit(nopython=True, nogil=True)
def common_x_two_segs(seg_x1, seg_y1, ibreaks1, seg_x2, seg_y2, ibreaks2):
    nmax = seg_x1.size + seg_x2.size
    common_seg_x    = np.zeros(nmax, dtype=seg_x1.dtype)
    common_seg_y1   = np.zeros(nmax, dtype=seg_y1.dtype)
    common_seg_y2   = np.zeros(nmax, dtype=seg_y2.dtype)
    common_ibreaks = np.zeros(nmax+1, dtype=ibreaks1.dtype)
    ####
    nleg1 = ibreaks1.size-1
    nleg2 = ibreaks2.size-1
    nx = 0
    nseg = 1
    for ileg1 in range(nleg1):
        x1 = seg_x1[ibreaks1[ileg1]: ibreaks1[ileg1+1] ]
        y1 = seg_y1[ibreaks1[ileg1]: ibreaks1[ileg1+1] ]
        for ileg2 in range(nleg2):
            x2 = seg_x2[ibreaks2[ileg2]: ibreaks2[ileg2+1] ]
            y2 = seg_y2[ibreaks2[ileg2]: ibreaks2[ileg2+1] ]
            ####
            common_x, common_y1, common_y2 = common_x_two_lines(x1, y1, x2, y2)
            if common_x.size > 0:
                common_seg_x[nx: nx + common_x.size] = common_x
                common_seg_y1[nx: nx + common_x.size] = common_y1
                common_seg_y2[nx: nx + common_x.size] = common_y2
                nx += common_x.size
                #######
                common_ibreaks[nseg] = nx
                nseg += 1
    common_seg_x  = common_seg_x[:nx]
    common_seg_y1 = common_seg_y1[:nx]
    common_seg_y2 = common_seg_y2[:nx]
    common_ibreaks = common_ibreaks[:nseg]
    return common_seg_x, common_seg_y1, common_seg_y2, common_ibreaks

def yield_lines(x, y, ibreaks):
    ileg = 0
    for i in range(len(ibreaks) - 1):
        i_start = ibreaks[i]
        i_end   = ibreaks[i+1]
        if i_start >= i_end:
            continue
        yield ileg, x[i_start:i_end], y[i_start:i_end]
        ileg += 1
def yield_triplets(x, y, z, ibreaks):
    ileg = 0
    for i in range(len(ibreaks) - 1):
        i_start = ibreaks[i]
        i_end   = ibreaks[i+1]
        if i_start >= i_end:
            continue
        yield ileg, x[i_start:i_end], y[i_start:i_end], z[i_start:i_end]
        ileg += 1

#### some specific rays
class FastTauP:
    MT_PHASE = 2
    OC_PHASE = 1
    IC_PHASE = 0
    def __init__(self, R0=6371.0, dr=None):
        r, vp, vs, icmb, iicb = rd_prem_model()
        self.proc_mod(r, vp, vs, icmb, iicb, R0=R0, dr=dr)
    def proc_mod(self, r, vp, vs, icmb, iicb, R0=6371.0, dr=None):
        self.R0 = R0
        ######
        mt_r, mt_vrp, mt_vrs = r[:icmb], vp[:icmb], vs[:icmb]
        oc_r, oc_vrp, oc_vrs = r[icmb:iicb], vp[icmb:iicb], vs[icmb:iicb]
        ic_r, ic_vrp, ic_vrs = r[iicb:], vp[iicb:], vs[iicb:]
        ######
        # Fix mantle
        # The mantle's layers must be denser because of the flattening.
        _,    mt_vrp = denser_xy(mt_r, mt_vrp, 20.0)
        mt_r, mt_vrs = denser_xy(mt_r, mt_vrs, 20.0)
        icmb = mt_r.size
        ######
        # Fix outer core
        # The outer core's layers must be denser because of the flattening.
        _,    oc_vrp = denser_xy(oc_r, oc_vrp, 5.0)
        oc_r, oc_vrs = denser_xy(oc_r, oc_vrs, 5.0)
        iicb = mt_r.size + oc_r.size
        ######
        # Fix inner core
        #The inner core's  layers must be denser because of the flattening near the center.
        _,    ic_vrp = denser_xy(ic_r, ic_vrp, 2.0)
        ic_r, ic_vrs = denser_xy(ic_r, ic_vrs, 2.0)
        #
        ic_r[-1] = ic_r[-2]-1e-3
        ic_vrp[-1]= ic_vrp[-2]+1e-3
        ci = -20
        _,     tmp_vrp = denser_xy(ic_r[ci:], ic_vrp[ci:], 0.5)
        tmp_r, tmp_vrs = denser_xy(ic_r[ci:], ic_vrs[ci:], 0.5)
        ic_r   = np.concatenate( (ic_r[  :ci],   tmp_r) )
        ic_vrp = np.concatenate( (ic_vrp[:ci], tmp_vrp) )
        ic_vrs = np.concatenate( (ic_vrs[:ci], tmp_vrs) )
        ######
        if dr is not None:
            _,    mt_vrp = denser_xy(mt_r, mt_vrp, dr)
            mt_r, mt_vrs = denser_xy(mt_r, mt_vrs, dr)
            _,    oc_vrp = denser_xy(oc_r, oc_vrp, dr)
            oc_r, oc_vrs = denser_xy(oc_r, oc_vrs, dr)
            _,    ic_vrp = denser_xy(ic_r, ic_vrp, dr)
            ic_r, ic_vrs = denser_xy(ic_r, ic_vrs, dr)
        ic_r[-1] = 1e-3 #ic_r[-2] # avoid zero radius, a nan for flattening
        #ic_vrp[-1]= ic_vrp[-2]
        #ic_vrs[-1]= ic_vrs[-2]
        ######
        _,    mt_vzp = flatten(mt_r, mt_vrp, R0)
        mt_z, mt_vzs = flatten(mt_r, mt_vrs, R0)
        _,    oc_vzp = flatten(oc_r, oc_vrp, R0)
        oc_z, oc_vzs = flatten(oc_r, oc_vrs, R0)
        _,    ic_vzp = flatten(ic_r, ic_vrp, R0)
        ic_z, ic_vzs = flatten(ic_r, ic_vrs, R0)
        ######
        mt_vzp_max, mt_vzs_max = np.max(mt_vzp), np.max(mt_vzs)
        oc_vzp_max, oc_vzs_max = np.max(oc_vzp), np.max(oc_vzs)
        ic_vzp_max, ic_vzs_max = np.max(ic_vzp), np.max(ic_vzs)
        ######
        mt_dz  = np.diff(mt_z)
        mt_dvzp= np.diff(mt_vzp)
        mt_dvzs= np.diff(mt_vzs)
        oc_dz  = np.diff(oc_z)
        oc_dvzp= np.diff(oc_vzp)
        oc_dvzs= np.diff(oc_vzs)
        ic_dz  = np.diff(ic_z)
        ic_dvzp= np.diff(ic_vzp)
        ic_dvzs= np.diff(ic_vzs)
        ######
        mt_dzdvzp = np.zeros_like(mt_dz, dtype=np.float64)
        mt_dzdvzs = np.zeros_like(mt_dz, dtype=np.float64)
        mt_dzdvzp = np.divide(mt_dz, mt_dvzp, out=mt_dzdvzp, where=mt_dvzp != 0.0)
        mt_dzdvzs = np.divide(mt_dz, mt_dvzs, out=mt_dzdvzs, where=mt_dvzs != 0.0)
        oc_dzdvzp = np.zeros_like(oc_dz, dtype=np.float64)
        oc_dzdvzs = np.zeros_like(oc_dz, dtype=np.float64)
        oc_dzdvzp = np.divide(oc_dz, oc_dvzp, out=oc_dzdvzp, where=oc_dvzp != 0.0)
        oc_dzdvzs = np.divide(oc_dz, oc_dvzs, out=oc_dzdvzs, where=oc_dvzs != 0.0)
        ic_dzdvzp = np.zeros_like(ic_dz, dtype=np.float64)
        ic_dzdvzs = np.zeros_like(ic_dz, dtype=np.float64)
        ic_dzdvzp = np.divide(ic_dz, ic_dvzp, out=ic_dzdvzp, where=ic_dvzp != 0.0)
        ic_dzdvzs = np.divide(ic_dz, ic_dvzs, out=ic_dzdvzs, where=ic_dvzs != 0.0)
        ######
        mt_p_layer_type = np.zeros(mt_z.size-1, dtype=np.int64)
        mt_p_layer_type = np.where( (np.abs(mt_dvzp) > 1e-10)  & (np.abs(mt_dz) > 1e-10), 2, mt_p_layer_type) # non-constant vp layer
        mt_p_layer_type = np.where( (np.abs(mt_dvzp) <=1e-10)  & (np.abs(mt_dz) > 1e-10), 1, mt_p_layer_type) # constant vp layer
        mt_s_layer_type = np.zeros(mt_z.size-1, dtype=np.int64)
        mt_s_layer_type = np.where( (np.abs(mt_dvzs) > 1e-10)  & (np.abs(mt_dz) > 1e-10), 2, mt_s_layer_type) # non-constant vs layer
        mt_s_layer_type = np.where( (np.abs(mt_dvzs) <=1e-10)  & (np.abs(mt_dz) > 1e-10), 1, mt_s_layer_type) # constant vs layer
        oc_p_layer_type = np.zeros(oc_z.size-1, dtype=np.int64)
        oc_p_layer_type = np.where( (np.abs(oc_dvzp) > 1e-10)  & (np.abs(oc_dz) > 1e-10), 2, oc_p_layer_type) # non-constant vp layer
        oc_p_layer_type = np.where( (np.abs(oc_dvzp) <=1e-10)  & (np.abs(oc_dz) > 1e-10), 1, oc_p_layer_type) # constant vp layer
        oc_s_layer_type = np.zeros(oc_z.size-1, dtype=np.int64)
        oc_s_layer_type = np.where( (np.abs(oc_dvzs) > 1e-10)  & (np.abs(oc_dz) > 1e-10), 2, oc_s_layer_type) # non-constant vs layer
        oc_s_layer_type = np.where( (np.abs(oc_dvzs) <=1e-10)  & (np.abs(oc_dz) > 1e-10), 1, oc_s_layer_type) # constant vs layer
        ic_p_layer_type = np.zeros(ic_z.size-1, dtype=np.int64)
        ic_p_layer_type = np.where( (np.abs(ic_dvzp) > 1e-10)  & (np.abs(ic_dz) > 1e-10), 2, ic_p_layer_type) # non-constant vp layer
        ic_p_layer_type = np.where( (np.abs(ic_dvzp) <=1e-10)  & (np.abs(ic_dz) > 1e-10), 1, ic_p_layer_type) # constant vp layer
        ic_s_layer_type = np.zeros(ic_z.size-1, dtype=np.int64)
        ic_s_layer_type = np.where( (np.abs(ic_dvzs) > 1e-10)  & (np.abs(ic_dz) > 1e-10), 2, ic_s_layer_type) # non-constant vs layer
        ic_s_layer_type = np.where( (np.abs(ic_dvzs) <=1e-10)  & (np.abs(ic_dz) > 1e-10), 1, ic_s_layer_type) # constant vs layer
        ######
        self.mt_z, self.mt_vzp, self.mt_vzs = mt_z, mt_vzp, mt_vzs
        self.oc_z, self.oc_vzp, self.oc_vzs = oc_z, oc_vzp, oc_vzs
        self.ic_z, self.ic_vzp, self.ic_vzs = ic_z, ic_vzp, ic_vzs
        #
        self.mt_vzp_max, self.mt_vzs_max = mt_vzp_max, mt_vzs_max
        self.oc_vzp_max, self.oc_vzs_max = oc_vzp_max, oc_vzs_max
        self.ic_vzp_max, self.ic_vzs_max = ic_vzp_max, ic_vzs_max
        #
        self.mt_dz, self.mt_dvzp, self.mt_dvzs = mt_dz, mt_dvzp, mt_dvzs
        self.oc_dz, self.oc_dvzp, self.oc_dvzs = oc_dz, oc_dvzp, oc_dvzs
        self.ic_dz, self.ic_dvzp, self.ic_dvzs = ic_dz, ic_dvzp, ic_dvzs
        #
        self.mt_dzdvzp, self.mt_dzdvzs = mt_dzdvzp, mt_dzdvzs
        self.oc_dzdvzp, self.oc_dzdvzs = oc_dzdvzp, oc_dzdvzs
        self.ic_dzdvzp, self.ic_dzdvzs = ic_dzdvzp, ic_dzdvzs
        #
        self.mt_vp_layer_type, self.mt_vs_layer_type = mt_p_layer_type, mt_s_layer_type
        self.oc_vp_layer_type, self.oc_vs_layer_type = oc_p_layer_type, oc_s_layer_type
        self.ic_vp_layer_type, self.ic_vs_layer_type = ic_p_layer_type, ic_s_layer_type
        #
        ######
        self.all_z = np.concatenate((mt_z, oc_z, ic_z) )
        self.all_vzp = np.concatenate((mt_vzp, oc_vzp, ic_vzp) )
        self.all_vzs = np.concatenate((mt_vzs, oc_vzs, ic_vzs) )
        #
        _,          self.all_vrp = unflatten(self.all_z, self.all_vzp, R0)
        self.all_r, self.all_vrs = unflatten(self.all_z, self.all_vzs, R0)
        ######
        self.all_icmb = mt_z.size
        self.all_iicb = mt_z.size + oc_z.size
        self.buf= np.zeros((10, self.all_z.size), dtype=np.float64)
        ######
    ################################################################################
    # Get inv_rps, ibs, ibreaks for specific ray types.
    #  or inv_rps, p_ibs, s_ibs, ibreaks for PS or IJ mixing.
    def get_P_turnback_inv_rps(self, max_theta_step_rad=0.0017): # For P, PP, PPP,...
        z  = self.mt_z
        vz = self.mt_vzp
        inv_rps, ibs, ibreaks = z2inv_rp_turn_back(z, vz)
        return denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
    def get_S_turnback_inv_rps(self, max_theta_step_rad=0.0017): # For S, SS, SSS,...
        z  = self.mt_z
        vz = self.mt_vzs
        inv_rps, ibs, ibreaks = z2inv_rp_turn_back(z, vz)
        return denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
    def get_PS_turnback_inv_rps(self, max_theta_step_rad=0.0017): # For PS, PSP, PSSP,...
        #    need to find the intersection of P and S turn-back inv_rp intervals.
        p_inv_rps, p_ibs, p_ibreaks = self.get_P_turnback_inv_rps(max_theta_step_rad)
        s_inv_rps, s_ibs, s_ibreaks = self.get_S_turnback_inv_rps(max_theta_step_rad)
        inv_rps, p_ibs, s_ibs, ibreaks = common_x_two_segs(p_inv_rps, p_ibs, p_ibreaks, s_inv_rps, s_ibs, s_ibreaks)
        return inv_rps, p_ibs, s_ibs, ibreaks
    def get_PcP_inv_rps(self, max_theta_step_rad=0.0017):   # For PcP, PcPPcP,...
        z = self.mt_z
        vz = self.mt_vzp
        inv_rps, ibs, ibreaks = z2inv_rp_penetrate(z, vz)
        return denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
    def get_ScS_inv_rps(self, max_theta_step_rad=0.0017):   # For ScS, ScSSScS,...
        z = self.mt_z
        vz = self.mt_vzs
        inv_rps, ibs, ibreaks = z2inv_rp_penetrate(z, vz)
        return denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
    def get_PcS_inv_rps(self, max_theta_step_rad=0.0017):   # For PcS, PcSPcP, PcSScS,...
        return self.get_PcP_inv_rps(max_theta_step_rad) # same as PcP as the intersection of PcP and ScS inv_rp intervals.
    def get_K_inv_rps(self,  mantle_P=True, mantle_S=True, max_theta_step_rad=0.0017): # For K(SKS-ScS), PKP, SKS, SKP, SKPPcS,...
        """
        mantle_P: True for the ray having mantle P legs, False for no mantle P legs.
        mantle_S: True for the ray having mantle S legs, False for no mantle S legs.
        """
        z  = self.oc_z
        vz = self.oc_vzp
        vz0_backup = vz[0]
        ######
        # Make sure the ray can cross the CMB if there must be some mantle legs.
        if mantle_P:
            mt_vp_max = np.max(self.mt_vzp)
            vz[0] = mt_vp_max if vz[0] < mt_vp_max else vz[0]
        if mantle_S:
            mt_vs_max = np.max(self.mt_vzs)
            vz[0] = mt_vs_max if vz[0] < mt_vs_max else vz[0]
        ######
        inv_rps, ibs, ibreaks = z2inv_rp_turn_back(z, vz)
        ibs += self.mt_z.size
        result = denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
        vz[0] = vz0_backup
        return result
    def get_KiK_inv_rps(self, mantle_P=True, mantle_S=True, max_theta_step_rad=0.0017): # For KiK(SKiKS-ScS), PKiKP, SKiS, SKiP,...
        z = self.oc_z
        vz= self.oc_vzp
        vz0_backup = vz[0]
        ######
        # Make sure the ray can cross the CMB if there must be some mantle legs.
        if mantle_P:
            mt_vp_max = np.max(self.mt_vzp)
            vz[0] = mt_vp_max if vz[0] < mt_vp_max else vz[0]
        if mantle_S:
            mt_vs_max = np.max(self.mt_vzs)
            vz[0] = mt_vs_max if vz[0] < mt_vs_max else vz[0]
        ######
        inv_rps, ibs, ibreaks = z2inv_rp_penetrate(z, vz)
        ibs += self.mt_z.size
        result = denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
        vz[0] = vz0_backup
        return result
    def get_I_inv_rps(self, mantle_P=True, mantle_S=True, oc_K=True, max_theta_step_rad=0.0017): # For PKIKP, PKIKS, SKIKS, KIK, I,...
        """
        mantle_P: True for the ray having mantle P legs, False for no mantle P legs.
        mantle_S: True for the ray having mantle S legs, False for no mantle S legs.
        oc_K: True for the ray having outer core K legs, False for no outer core K legs.
        """
        z  = self.ic_z
        vz = self.ic_vzp
        vz0_backup = vz[0]
        ######
        # Make sure the ray can cross the CMB and ICB if there must be some
        if mantle_P:
            mt_vp_max = np.max(self.mt_vzp)
            vz[0] = mt_vp_max if vz[0] < mt_vp_max else vz[0]
        if mantle_S:
            mt_vs_max = np.max(self.mt_vzs)
            vz[0] = mt_vs_max if vz[0] < mt_vs_max else vz[0]
        if oc_K:
            oc_vp_max = np.max(self.oc_vzp)
            vz[0] = oc_vp_max if vz[0] < oc_vp_max else vz[0]
        ######
        inv_rps, ibs, ibreaks = z2inv_rp_both(z, vz)
        inv_rps, ibs, ibreaks = z2inv_rp_turn_back(z, vz)
        ibs += (self.mt_z.size + self.oc_z.size)
        result = denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
        vz[0] = vz0_backup
        return result
    def get_J_inv_rps(self, mantle_P=True, mantle_S=True, oc_K=True, max_theta_step_rad=0.0017): # For PKJKP, PKJKS, SKJKS, KJK, J,...
        """
        mantle_P: True for the ray having mantle P legs, False for no mantle P legs.
        mantle_S: True for the ray having mantle S legs, False for no mantle S legs.
        oc_K: True for the ray having outer core K legs, False for no outer core K legs.
        """
        z  = self.ic_z
        vz = self.ic_vzs
        vz0_backup = vz[0]
        ######
        # Make sure the ray can cross the CMB and ICB if there must be some
        if mantle_P:
            mt_vp_max = np.max(self.mt_vzp)
            vz[0] = mt_vp_max if vz[0] < mt_vp_max else vz[0]
        if mantle_S:
            mt_vs_max = np.max(self.mt_vzs)
            vz[0] = mt_vs_max if vz[0] < mt_vs_max else vz[0]
        if oc_K:
            oc_vs_max = np.max(self.oc_vzs)
            vz[0] = oc_vs_max if vz[0] < oc_vs_max else vz[0]
        ######
        inv_rps, ibs, ibreaks = z2inv_rp_both(z, vz)
        ibs += (self.mt_z.size + self.oc_z.size)
        result = denser_inv_rps(inv_rps, ibs, ibreaks, vz[0], max_theta_step_rad)
        vz[0] = vz0_backup
        return result
    def get_IJ_inv_rps(self, mantle_P=True, mantle_S=True, oc_K=True, max_theta_step_rad=0.0017): # For JKIKJ, PKIKS, JKIKP,...
        p_inv_rps, p_ibs, p_ibreaks = self.get_I_inv_rps(max_theta_step_rad)
        s_inv_rps, s_ibs, s_ibreaks = self.get_J_inv_rps(max_theta_step_rad)
        inv_rps, p_ibs, s_ibs, ibreaks = common_x_two_segs(p_inv_rps, p_ibs, p_ibreaks, s_inv_rps, s_ibs, s_ibreaks)
        p_ibs += (self.mt_z.size + self.oc_z.size)
        s_ibs += (self.mt_z.size + self.oc_z.size)
        return inv_rps, p_ibs, s_ibs, ibreaks
    @staticmethod
    def benchmark_PS_inv_rps():
        app = FastTauP()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
        ###### P & S turn-back rays
        z = app.mt_z
        vzp = app.mt_vzp
        vzs = app.mt_vzs
        vzp[3] = 4.0 # for testing shadow zones
        vzs[3] = 1.0
        ax2.plot(vzp, z, '-', color='b', lw=10, label='P velocity')
        ax2.plot(vzs, z, '-', color='g', lw=10, label='S velocity')
        #
        inv_rps, ibs, ibreaks = app.get_P_turnback_inv_rps()
        for ileg, c, ib in yield_lines(inv_rps, ibs, ibreaks):
            ax1.plot(c, ib,    '-', color='r', lw=5)
            ax1.plot([c[0], c[-1]], [ib[0], ib[-1]], 'o', color='r', markersize=8)
            ax2.plot(c, z[ib], '-', color='r', lw=5)
        inv_rps, ibs, ibreaks = app.get_S_turnback_inv_rps()
        for ileg, c, ib in yield_lines(inv_rps, ibs, ibreaks):
            ax1.plot(c, ib,    '-', color='C1', lw=5)
            ax1.plot([c[0], c[-1]], [ib[0], ib[-1]], 'o', color='C1', markersize=8)
            ax2.plot(c, z[ib], '-', color='C1', lw=5)
        inv_rps, p_ibs, s_ibs, ibreaks = app.get_PS_turnback_inv_rps()
        for ileg, c, ib in yield_lines(inv_rps, p_ibs, ibreaks):
            ax1.plot(c, ib,    '-', color='k', label='P leg')
            ax1.plot([c[0], c[-1]], [ib[0], ib[-1]], 'o', color='k', markersize=8)
            ax2.plot(c, z[ib], '-', color='k', label='P leg')
        for ileg, c, ib in yield_lines(inv_rps, s_ibs, ibreaks):
            ax1.plot(c, ib,    '-', color='k', label='S leg')
            ax1.plot([c[0], c[-1]], [ib[0], ib[-1]], 'o', color='k', markersize=8)
            ax2.plot(c, z[ib], '-', color='k', label='S leg')
        ax1.invert_yaxis()
        plt.show()
    @staticmethod
    def benchmark_IJ_inv_rps():
        app = FastTauP()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18,6))
        ###### P & S turn-back rays
        z = app.ic_z
        vzp = app.ic_vzp
        vzs = app.ic_vzs
        #vzp[3] = 4.0 # for testing shadow zones
        #vzs[3] = 1.0
        ax2.semilogx(vzp, z, '-', color='k', lw=10, label='P velocity')
        ax4.semilogx(vzs, z, '-', color='k', lw=10, label='S velocity')
        #
        cmap = plt.get_cmap('tab10_r', 10)
        inv_rps, ibs, ibreaks = app.get_I_inv_rps()
        for ileg, c, ib in yield_lines(inv_rps, ibs, ibreaks):
            clr = cmap(ileg % 10)
            ax1.semilogx(c, ib,    '-',  lw=5, color=clr)
            ax1.semilogx([c[0], c[-1]], [ib[0], ib[-1]], 's',  markersize=12, color=clr)
            ax2.semilogx(c, z[ib], '-', lw=5, color=clr)
            ax2.semilogx([c[0], c[-1]], [z[ib][0], z[ib][-1]], 's', markersize=12, color=clr)
        inv_rps, ibs, ibreaks = app.get_J_inv_rps()
        for ileg, c, ib in yield_lines(inv_rps, ibs, ibreaks):
            clr = cmap(ileg % 10)
            ax3.semilogx(c, ib,    '-', color=clr, lw=5)
            ax3.semilogx([c[0], c[-1]], [ib[0], ib[-1]], 's', color=clr, markersize=12)
            ax4.semilogx(c, z[ib], '-', color=clr, lw=5)
            ax4.semilogx([c[0], c[-1]], [z[ib][0], z[ib][-1]], 's', color=clr, markersize=12)
        inv_rps, p_ibs, s_ibs, ibreaks = app.get_IJ_inv_rps()
        for ileg, c, ib in yield_lines(inv_rps, p_ibs, ibreaks):
            clr = 'r'
            ax1.semilogx(c, ib,    '-', color=clr, label='P leg')
            ax1.semilogx([c[0], c[-1]], [ib[0], ib[-1]], 'o', color=clr, markersize=8) #, zorder=0)
            ax2.semilogx(c, z[ib], '-', color=clr, label='P leg')
            ax2.semilogx([c[0], c[-1]], [z[ib][0], z[ib][-1]], 'o', color=clr, markersize=8) #, zorder=0)
        for ileg, c, ib in yield_lines(inv_rps, s_ibs, ibreaks):
            clr = 'b'
            ax3.semilogx(c, ib,    '-', color=clr, label='S leg')
            ax3.semilogx([c[0], c[-1]], [ib[0], ib[-1]], 'o', color=clr, markersize=8)
            ax4.semilogx(c, z[ib], '-', color=clr, label='S leg')
            ax4.semilogx([c[0], c[-1]], [z[ib][0], z[ib][-1]], 'o', color=clr, markersize=8)
        ax1.invert_yaxis()
        ax3.invert_yaxis()
        plt.show()
    ################################################################################
    # basic p2xt functions for P, S, K, I, J legs...
    def wrapper_p2xt_grad(self, inv_rp, z, vz, dz, dvz, grad):
        if grad:
            return p2xt_grad(inv_rp, z, vz, dz, dvz, self.buf)
        else:
            x, t = p2xt(inv_rp, z, vz, dz, dvz)
            return x, t, 0., 0., 0., 0., 0.
    def leg_p2xt_P(self, inv_rp, grad=False):
        return self.wrapper_p2xt_grad(inv_rp, self.mt_z, self.mt_vzp, self.mt_dz, self.mt_dvzp, grad)
    def leg_p2xt_S(self, inv_rp, grad=False):
        return self.wrapper_p2xt_grad(inv_rp, self.mt_z, self.mt_vzs, self.mt_dz, self.mt_dvzs, grad)
    def leg_p2xt_PcP(self, inv_rp, grad=False):
        return self.wrapper_p2xt_grad(inv_rp, self.mt_z, self.mt_vzp, self.mt_dz, self.mt_dvzp, grad)
    def leg_p2xt_ScS(self, inv_rp, grad=False):
        return self.wrapper_p2xt_grad(inv_rp, self.mt_z, self.mt_vzs, self.mt_dz, self.mt_dvzs, grad)
    def leg_p2xt_K(self, inv_rp, grad=False):
        return self.wrapper_p2xt_grad(inv_rp, self.oc_z, self.oc_vzp, self.oc_dz, self.oc_dvzp, grad)
    def leg_p2xt_KiK(self, inv_rp, grad=False):
        return self.wrapper_p2xt_grad(inv_rp, self.oc_z, self.oc_vzp, self.oc_dz, self.oc_dvzp, grad)
    def leg_p2xt_I(self, inv_rp, grad=False):
        x, t, pxpv, pxpp, ptpv, ptpp, dtdv = self.wrapper_p2xt_grad(inv_rp, self.ic_z, self.ic_vzp, self.ic_dz, self.ic_dvzp, grad)
        if inv_rp > self.ic_vzp_max:
            # The ray penetrates the deepest z layer. It is not a reflection on spherical Earth, but a transmission through the inner core.
            x = np.pi*self.R0 - x
            if grad:
                pxpv = -pxpv
                pxpp = -pxpp
                # dtdv = ptpv - ptpp * (pxpv/pxpp)  no need to change
        return x, t, pxpv, pxpp, ptpv, ptpp, dtdv
    def leg_p2xt_J(self, inv_rp, grad=False):
        x, t, pxpv, pxpp, ptpv, ptpp, dtdv = self.wrapper_p2xt_grad(inv_rp, self.ic_z, self.ic_vzs, self.ic_dz, self.ic_dvzs, grad)
        if inv_rp > self.ic_vzs_max:
            # The ray penetrates the deepest z layer. It is not a reflection on spherical Earth, but a transmission through the inner core.
            x = np.pi*self.R0 - x
            if grad:
                pxpv = -pxpv
                pxpp = -pxpp
                # dtdv = ptpv - ptpp * (pxpv/pxpp)  no need to change
        return x, t, pxpv, pxpp, ptpv, ptpp, dtdv
    ################################################################################
    # Decipher a phase name into phase_type, nP, nS, nK, nI, nJ
    # And validate inv_rp for the given phase name.
    def decipher_phase_name(self, phase_name):
        nP, nS, nK, nI, nJ = 0.0, 0.0, 0.0, 0.0, 0.0
        if ('I' in phase_name) or ('J' in phase_name) or ('i' in phase_name):
            # e.g., PKIKP,  PKJKP,  PKiKP
            phase_type = FastTauP.IC_PHASE
            nI = phase_name.count('I')*1.0 # Note, the p2xt contains both downgoing and upgoing paths.
            nJ = phase_name.count('J')*1.0
            nK = phase_name.count('K')*0.5
            nP = phase_name.count('P')*0.5
            nS = phase_name.count('S')*0.5
        elif ('K' in phase_name) or ('c' in phase_name):
            # e.g., PKP, PcP, PcS, ScS, SKS, SKP
            phase_type = FastTauP.OC_PHASE
            nK = phase_name.count('K')*1.0
            nP = phase_name.count('P')*0.5
            nS = phase_name.count('S')*0.5
        else:
            # e.g., P, PP, S, SS, PS
            phase_type = FastTauP.MT_PHASE
            nP = phase_name.count('P')*1.0
            nS = phase_name.count('S')*1.0
        return phase_type, nP, nS, nK, nI, nJ
    def is_inv_rp_valid_base(self, inv_rp, phase_type, nP, nS, nK, nI, nJ):
        valid = True
        if phase_type == FastTauP.MT_PHASE : #'MT':
            if nP > 0:
                # There must be >=1 P interface in the mantle that the ray cannot penetrate through
                valid &= (inv_rp <= self.mt_vzp_max)
            if nS > 0:
                # There must be >=1 S interface in the mantle that the ray cannot penetrate through
                valid &= (inv_rp <= self.mt_vzs_max)
        elif phase_type == FastTauP.OC_PHASE : #'OC':
            if nP > 0:
                # The ray must be able to penetrate all P interface of the mantle
                valid &= (inv_rp >= self.mt_vzp_max)
            if nS > 0:
                # The ray must be able to penetrate all S interface of the mantle
                valid &= (inv_rp >= self.mt_vzs_max)
            if nK > 0:
                # There must be >=1 P interface in the OC that the ray cannot penetrate through
                valid &= (inv_rp <= self.oc_vzp_max)
        elif phase_type == FastTauP.IC_PHASE: # IC
            if nP > 0:
                # The ray must be able to penetrate all P interface of the mantle
                valid &= (inv_rp >= self.mt_vzp_max)
            if nS > 0:
                # The ray must be able to penetrate all S interface of the mantle
                valid &= (inv_rp >= self.mt_vzs_max)
            if nK > 0:
                # The ray must be able to penetrate all P interface of the OC
                valid &= (inv_rp >= self.oc_vzp_max)
        else:
            raise ValueError(f'Unknown phase type: {phase_type}.')
        return valid
    def is_inv_rp_valid(self, inv_rp, phase_name):
        phase_type, nP, nS, nK, nI, nJ = self.decipher_phase_name(phase_name)
        return self.is_inv_rp_valid_base(inv_rp, phase_type, nP, nS, nK, nI, nJ)
    ################################################################################
    # Get  inv_rps, p_ibs, s_ibs, ibreaks given a phase name.
    #      p_ibs or s_ibs can be None if there is no P(I) or S(J) leg in the phase.
    def phase2rp_base(self, phase_type, nP, nS, nK, nI, nJ, max_theta_step_rad=0.0017):
        p_ibs = None
        s_ibs = None
        if phase_type == FastTauP.MT_PHASE : #'MT':
            if (nP>0) and (nS>0):     # PS
                inv_rps, p_ibs, s_ibs, ibreaks = self.get_PS_turnback_inv_rps(max_theta_step_rad=max_theta_step_rad)
            elif (nP>0) and (nS==0):  # P
                inv_rps, p_ibs, ibreaks = self.get_P_turnback_inv_rps(max_theta_step_rad=max_theta_step_rad)
            elif (nP==0) and (nS>0):  # S
                inv_rps, s_ibs, ibreaks = self.get_S_turnback_inv_rps(max_theta_step_rad=max_theta_step_rad)
            else:
                raise ValueError(f'Invalid phase name for MT phase: nP={nP}, nS={nS}.')
        elif phase_type == FastTauP.OC_PHASE : #'OC':
            if nK==0:
                if (nP>0) and (nS>0):    # PcS
                    inv_rps, p_ibs, ibreaks = self.get_PcS_inv_rps(max_theta_step_rad=max_theta_step_rad)
                elif (nP>0) and (nS==0): # PcP
                    inv_rps, p_ibs, ibreaks = self.get_PcP_inv_rps(max_theta_step_rad=max_theta_step_rad)
                elif (nP==0) and (nS>0): # ScS
                    inv_rps, s_ibs, ibreaks = self.get_ScS_inv_rps(max_theta_step_rad=max_theta_step_rad)
                else:
                    raise ValueError(f'Invalid phase name for OC phase without K legs: nP={nP}, nS={nS}.')
            else:                        # PKP, SKS, SKP, K(SKS-ScS)
                inv_rps, p_ibs, ibreaks = self.get_K_inv_rps(mantle_P=(nP>0), mantle_S=(nS>0), max_theta_step_rad=max_theta_step_rad)
        elif phase_type == FastTauP.IC_PHASE: #'IC':
            if (nI==0) and (nJ==0):    # PKiKP, PKiKS, SKiKS, KiK(SKiKS-ScS)
                inv_rps, p_ibs, ibreaks = self.get_KiK_inv_rps(mantle_P=(nP>0), mantle_S=(nS>0), max_theta_step_rad=max_theta_step_rad)
            elif (nI>0) and (nJ==0):   # PKIKP, PKIKS, SKIKS, KIK(SKIKS-ScS), I(SKIKS-SKiKS)
                inv_rps, p_ibs, ibreaks = self.get_I_inv_rps(mantle_P=(nP>0), mantle_S=(nS>0), oc_K=(nK>0), max_theta_step_rad=max_theta_step_rad)
            elif (nI==0) and (nJ>0):   # PKJKP, PKJKS, SKJKS, KJK(SKJKS-ScS), J(SKJKS-SKiKS)
                inv_rps, s_ibs, ibreaks = self.get_J_inv_rps(mantle_P=(nP>0), mantle_S=(nS>0), oc_K=(nK>0), max_theta_step_rad=max_theta_step_rad)
            elif (nI>0) and (nJ>0):    # PKIJKP, ...
                inv_rps, p_ibs, s_ibs, ibreaks = self.get_IJ_inv_rps(mantle_P=(nP>0), mantle_S=(nS>0), oc_K=(nK>0), max_theta_step_rad=max_theta_step_rad)
            else:
                raise ValueError(f'Invalid phase name for IC phase: nI={nI}, nJ={nJ}.')
        else:
            raise ValueError(f'Unknown phase type: {phase_type}.')
        return inv_rps, p_ibs, s_ibs, ibreaks
    def phase2rp(self, phase_name, max_theta_step_rad=0.0017):
        phase_type, nP, nS, nK, nI, nJ = self.decipher_phase_name(phase_name)
        return self.phase2rp_base(phase_type, nP, nS, nK, nI, nJ, max_theta_step_rad=max_theta_step_rad)
        # return inv_rps, p_ibs, s_ibs, ibreaks
        # in which p_ibs or s_ibs can be None if there is no P or S leg in the phase.
    ################################################################################
    # Single rp to x, t for a given phase name.
    def phase_p2xt_base(self, inv_rp, nP, nS, nK, nI, nJ, grad=False):
        icmb = self.all_icmb
        iicb = self.all_iicb
        x, t, par_x_p, par_t_p = 0.0, 0.0, 0.0, 0.0
        par_x_vzp = np.zeros(self.all_z.size, dtype=np.float64)
        par_x_vzs = np.zeros(self.all_z.size, dtype=np.float64)
        par_t_vzp = np.zeros(self.all_z.size, dtype=np.float64)
        par_t_vzs = np.zeros(self.all_z.size, dtype=np.float64)
        d_t_vzp   = np.zeros(self.all_z.size, dtype=np.float64)
        d_t_vzs   = np.zeros(self.all_z.size, dtype=np.float64)
        ###
        if nP > 0:
            dx, dt, pxpv, pxpp, ptpv, ptpp, dtdv = self.leg_p2xt_P(inv_rp, grad=grad)
            x += dx * nP
            t += dt * nP
            if grad:
                par_x_p += pxpp * nP
                par_t_p += ptpp * nP
                par_x_vzp[:icmb] += pxpv * nP
                par_t_vzp[:icmb] += ptpv * nP
                d_t_vzp[:icmb]   += dtdv * nP
        if nS > 0:
            dx, dt, pxpv, pxpp, ptpv, ptpp, dtdv = self.leg_p2xt_S(inv_rp, grad=grad)
            x += dx * nS
            t += dt * nS
            if grad:
                par_x_p += pxpp * nS
                par_t_p += ptpp * nS
                par_x_vzs[:icmb] += pxpv * nS
                par_t_vzs[:icmb] += ptpv * nS
                d_t_vzs[:icmb]   += dtdv * nS
        if nK > 0:
            dx, dt, pxpv, pxpp, ptpv, ptpp, dtdv = self.leg_p2xt_K(inv_rp, grad=grad)
            x += dx * nK
            t += dt * nK
            if grad:
                par_x_p += pxpp * nK
                par_t_p += ptpp * nK
                par_x_vzp[icmb:iicb] += pxpv * nK
                par_t_vzp[icmb:iicb] += ptpv * nK
                d_t_vzp[icmb:iicb]   += dtdv * nK
        if nI > 0:
            dx, dt, pxpv, pxpp, ptpv, ptpp, dtdv = self.leg_p2xt_I(inv_rp, grad=grad)
            x += dx * nI
            t += dt * nI
            if grad:
                par_x_p += pxpp * nI
                par_t_p += ptpp * nI
                par_x_vzp[iicb:] += pxpv * nI
                par_t_vzp[iicb:] += ptpv * nI
                d_t_vzp[iicb:]   += dtdv * nI
        if nJ > 0:
            dx, dt, pxpv, pxpp, ptpv, ptpp, dtdv = self.leg_p2xt_J(inv_rp, grad=grad)
            x += dx * nJ
            t += dt * nJ
            if grad:
                par_x_p += pxpp * nJ
                par_t_p += ptpp * nJ
                par_x_vzs[iicb:] += pxpv * nJ
                par_t_vzs[iicb:] += ptpv * nJ
                d_t_vzs[iicb:]   += dtdv * nJ
        return x, t, par_x_vzp, par_x_vzs, par_x_p, par_t_vzp, par_t_vzs, par_t_p, d_t_vzp, d_t_vzs
    def phase_p2xt(self, inv_rp, phase_name, grad=False):
        phase_type, nP, nS, nK, nI, nJ = self.decipher_phase_name(phase_name)
        valid = self.is_inv_rp_valid_base(inv_rp, phase_type, nP, nS, nK, nI, nJ)
        if not valid:
            raise ValueError(f'The given inv_rp={inv_rp} is invalid for phase_name={phase_name}!' )
        return self.phase_p2xt_base(inv_rp, nP, nS, nK, nI, nJ, grad=grad)
    @staticmethod
    def benchmark_phase_p2xt():
        from obspy.taup import TauPyModel
        fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(2, 4, figsize=(16, 10))
        app = FastTauP(R0=6371.0) #, dr=1.0)
        ######
        ax0.loglog(app.mt_vzp, 1-app.mt_z, )
        ax0.loglog(app.mt_vzs, 1-app.mt_z, )
        ax0.loglog(app.oc_vzp, 1-app.oc_z, )
        ax0.loglog(app.ic_vzp, 1-app.ic_z, )
        ax0.loglog(app.ic_vzs, 1-app.ic_z, )
        ax0.invert_yaxis()
        ######
        mod_prem = TauPyModel(model='prem')
        #for phase_name in ['P', 'PP', 'S', 'SS', 'SSS' ]:
        #for phase_name in ['P', 'PS', 'SP', 'SS', 'SSP' ]:
        #for phase_name in ['PcP', 'PcPPcP', 'ScS', 'ScSScSScS']:
        for phase_name in ['PcP', 'PcS', 'ScS', 'ScP', 'PcPPcS']:
        #for phase_name in ['PKP', 'PKKP', 'PKKKP', 'PKPPKP']:
        #for phase_name in ['SKS', 'SKKS', 'SKKKS', 'SKSSKKS']:
        #for phase_name in ['PKSPKP', 'PKSScS', 'PKKPScS', 'ScSSKS']:
        #for phase_name in ['PKIKP', 'PKiKPPcP', 'PKiKPScS', 'PKJKPPcS']:
            tmp  = [mod_prem.get_travel_times(0, it, [phase_name]) for it in range(1, 180)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1   = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1   = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1   = np.array([it.time for it in arrs], dtype=np.float64)
            p1   *= (180.0/(np.pi*6371.0))  # convert to s/deg
            x1   *= (6371.0 * np.pi / 180.0)  # convert to degree
            idx = np.argsort(p1)[::-1]
            p1  = p1[idx]
            x1  = x1[idx]
            t1  = t1[idx]
            ax1.plot(x1, t1,     'o-', label=f'ObsPy-{phase_name}', color='k')
            ax2.plot(x1, 1.0/p1, 'o-', label=f'ObsPy-{phase_name}', color='k')
            ax3.plot(t1, 1.0/p1, 'o-', label=f'ObsPy-{phase_name}', color='k')
            #####
            p2 = p1.copy()
            x2 = np.zeros_like(p2, dtype=np.float64)
            t2 = np.zeros_like(p2, dtype=np.float64)
            for ip in range(p2.size):
                tmp = app.phase_p2xt(1/p2[ip], phase_name)  # warm up
                x2[ip], t2[ip] = tmp[0], tmp[1]
            ax1.plot(x2, t2,     '.', label=f'FastTauP-{phase_name}', color='C3')
            ax2.plot(x2, 1.0/p2, '.', label=f'FastTauP-{phase_name}', color='C3')
            ax3.plot(t2, 1.0/p2, '.', label=f'FastTauP-{phase_name}', color='C3')
            #####
            ax6.plot(x2 - x1, 1.0/p2, '.', label=f'FastTauP-{phase_name}', color='C3')
            ax7.plot(t2 - t1, 1.0/p2, '.', label=f'FastTauP-{phase_name}', color='C3')
        plt.show()
    ################################################################################
    # Get Multiple p-x-t segments (and optional gradient) given phase name.
    def phase2xt_base(self, phase_type, nP, nS, nK, nI, nJ, grad=False, max_theta_step_rad=0.0017):
        inv_rps, p_ibs, s_ibs, ibreaks = self.phase2rp_base(phase_type, nP, nS, nK, nI, nJ, max_theta_step_rad=max_theta_step_rad)
        x = np.zeros(inv_rps.size, dtype=np.float64)
        t = np.zeros(inv_rps.size, dtype=np.float64)
        pxpvzp = np.zeros((inv_rps.size, self.all_z.size), dtype=np.float64)
        pxpvzs = np.zeros((inv_rps.size, self.all_z.size), dtype=np.float64)
        pxprp  = np.zeros((inv_rps.size), dtype=np.float64)
        ptpvzp = np.zeros((inv_rps.size, self.all_z.size), dtype=np.float64)
        ptpvzs = np.zeros((inv_rps.size, self.all_z.size), dtype=np.float64)
        ptprp  = np.zeros((inv_rps.size), dtype=np.float64)
        dtdvzp = np.zeros((inv_rps.size, self.all_z.size), dtype=np.float64)
        dtdvzs = np.zeros((inv_rps.size, self.all_z.size), dtype=np.float64)
        for irp in range(inv_rps.size):
            tmp = self.phase_p2xt_base(inv_rps[irp], nP, nS, nK, nI, nJ, grad=grad)
            x[irp], t[irp] = tmp[0], tmp[1]
            if grad:
                pxpvzp[irp, :] = tmp[2]
                pxpvzs[irp, :] = tmp[3]
                pxprp[irp]     = tmp[4]
                ptpvzp[irp, :] = tmp[5]
                ptpvzs[irp, :] = tmp[6]
                ptprp[irp]     = tmp[7]
                dtdvzp[irp, :] = tmp[8]
                dtdvzs[irp, :] = tmp[9]
        ####
        leg_start_end_pairs_gind = np.zeros((2, inv_rps.size), dtype=np.int64)
        leg_start_gind = leg_start_end_pairs_gind[0] # the index points to the data point in (rps, dist, trvt) for the start of each leg
        leg_end_gind   = leg_start_end_pairs_gind[1] # the index points to the data point in (rps, dist, trvt) for the end   of each leg
        n_legs =0
        ####
        g_i0 = 0
        for ileg, it_x, junk in yield_lines(x, t, ibreaks):
            g_i1 = g_i0 + it_x.size
            b_inds =  split_pxt_legs_v2(it_x)
            b_inds += g_i0
            leg_start_gind[n_legs]                         = g_i0
            leg_start_gind[n_legs+1: n_legs+1+b_inds.size] = b_inds
            leg_end_gind[  n_legs:   n_legs+b_inds.size]   = b_inds+1
            leg_end_gind[               n_legs+b_inds.size]= g_i1
            n_legs += (b_inds.size +1)
            g_i0 = g_i1
        leg_start_end_pairs_gind = leg_start_end_pairs_gind[:, :n_legs].T
        return inv_rps, x, t, leg_start_end_pairs_gind, pxpvzp, pxpvzs, ptpvzp, ptpvzs, dtdvzp, dtdvzs
    def phase2xt(self, phase_name, grad=False, max_theta_step_rad=0.0017):
        phase_type, nP, nS, nK, nI, nJ = self.decipher_phase_name(phase_name)
        return self.phase2xt_base(phase_type, nP, nS, nK, nI, nJ, grad=grad, max_theta_step_rad=max_theta_step_rad)
    @staticmethod
    def benchmark_phase2xt():
        from matplotlib.gridspec import GridSpec
        from obspy.taup import TauPyModel
        from matplotlib.ticker import AutoMinorLocator
        fig = plt.figure(figsize=(16, 8))
        gs  = GridSpec(100, 100, figure=fig)
        ax0 = fig.add_subplot(gs[0:100, 0:10])
        ax1 = fig.add_subplot(gs[0:100, 18:60])
        ax2 = fig.add_subplot(gs[0:100, 65:80])
        ax3 = fig.add_subplot(gs[0:100, 85:100])
        app = FastTauP(R0=6371.0, dr=5.0)
        ax0.loglog(app.all_vzp, -app.all_z, '-', color='k', lw=2, label='P velocity')
        ax0.loglog(app.all_vzs, -app.all_z, '-', color='r', lw=2, label='S velocity')
        ax0.invert_yaxis()
        #####################
        # add auto minor ticks to ax0
        mod_prem = TauPyModel(model='prem')
        #for phase_name in ['P', 'PP', 'S', 'SS', 'SSS' ]:
        #for phase_name in ['P', 'PS', 'SP', 'SS', 'SSP' ]:
        #for phase_name in ['PcP', 'PcPPcP', 'ScS', 'ScSScSScS']:
        #for phase_name in ['PcP', 'PcS', 'ScS', 'ScP', 'PcPPcS']:
        #for phase_name in ['PKP', 'PKKP', 'PKKKP', 'PKPPKP']:
        for phase_name in ['ScS', 'SKS', 'SKKS', 'SKKKS', 'SKKKKS']:
        #for phase_name in ['PKSPKP', 'PKSScS', 'PKKPScS', 'ScSSKS']:
        #for phase_name in ['PKIKP', 'PKiKPPcP', 'PKiKPScS', 'PKJKPPcS']:
        #for phase_name in ['PKIKP']:
            tmp  = [mod_prem.get_travel_times(0, it, [phase_name]) for it in range(0, 180)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1   = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1   = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1   = np.array([it.time for it in arrs], dtype=np.float64)
            p1   *= (180.0/(np.pi*6371.0))  # convert to s/km
            idx = np.argsort(p1)[::-1]
            p1  = p1[idx]
            x1  = x1[idx]
            t1  = t1[idx]
            ax1.plot(x1, t1, 'o-', color='gray', zorder=0)
            ax2.plot(x1, p1, 'o-', color='gray', zorder=0)
            ax3.plot(t1, p1, 'o-', color='gray', zorder=0)
            ##################
            tmp = app.phase2xt(phase_name, max_theta_step_rad=0.01)  # warm up
            inv_rp2= tmp[0]
            x2     = tmp[1]
            t2     = tmp[2]
            x2    *= (180.0/(np.pi*6371.0))  # convert to degree
            istart_ends = tmp[3]
            for i0, i1 in istart_ends:
                ax1.plot(x2[i0:i1], t2[i0:i1],          '.-', label=phase_name, markersize=2, zorder=1)
                ax2.plot(x2[i0:i1], 1.0/inv_rp2[i0:i1], '.-', label=phase_name, markersize=2, zorder=1)
                ax3.plot(t2[i0:i1], 1.0/inv_rp2[i0:i1], '.-', label=phase_name, markersize=2, zorder=1)
            #
        ax1.xaxis.set_major_locator(plt.MultipleLocator(50))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))
        ax1.yaxis.set_major_locator(plt.MultipleLocator(500))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(100))
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, which='major', linestyle='-', alpha=0.5, lw=0.6)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3, lw=0.6)
            ax.legend()
        ax1.set_xlabel('Purist distance (degree)')
        ax1.set_ylabel('Travel time (s)')
        ax2.set_xlabel('Purist distance (degree)')
        ax2.set_ylabel('Ray parameter (s/km)')
        ax3.set_xlabel('Travel time (s)')
        ax3.set_ylabel('Ray parameter (s/km)')
        plt.show()
    @staticmethod
    def benchmark_phase2xt_dist_range():
        from matplotlib.gridspec import GridSpec
        from obspy.taup import TauPyModel
        from matplotlib.ticker import AutoMinorLocator
        fig = plt.figure(figsize=(16, 8))
        gs  = GridSpec(100, 100, figure=fig)
        ax0 = fig.add_subplot(gs[0:100, 0:10])
        ax1 = fig.add_subplot(gs[0:100, 18:60])
        ax2 = fig.add_subplot(gs[0:100, 65:80])
        ax3 = fig.add_subplot(gs[0:100, 85:100])
        app = FastTauP(R0=6371.0, dr=5.0)
        ax0.loglog(app.all_vzp, -app.all_z, '-', color='k', lw=2, label='P velocity')
        ax0.loglog(app.all_vzs, -app.all_z, '-', color='r', lw=2, label='S velocity')
        ax0.invert_yaxis()
        #####################
        # add auto minor ticks to ax0
        mod_prem = TauPyModel(model='prem')
        for phase_name in ['S']:
            tmp = app.phase2xt(phase_name, max_theta_step_rad=0.01)  # warm up
            inv_rp2= tmp[0]
            x2     = tmp[1]
            t2     = tmp[2]
            x2    *= (180.0/(np.pi*6371.0))  # convert to degree
            istart_ends = tmp[3]
            for i0, i1 in istart_ends:
                ax1.plot(x2[i0:i1], t2[i0:i1],          color='k', label=phase_name, lw=1, zorder=1)
                ax2.plot(x2[i0:i1], 1.0/inv_rp2[i0:i1], color='k', label=phase_name, lw=1, zorder=1)
                ax3.plot(t2[i0:i1], 1.0/inv_rp2[i0:i1], color='k', label=phase_name, lw=1, zorder=1)
        for phase_name in ['ScS', 'SKS', 'SKKS', 'SKKKS', 'SKKKKS']:
            tmp = app.phase2xt(phase_name, max_theta_step_rad=0.01)  # warm up
            inv_rp2= tmp[0]
            x2     = tmp[1]
            t2     = tmp[2]
            x2    *= (180.0/(np.pi*6371.0))  # convert to degree
            istart_ends = tmp[3]
            for i0, i1 in istart_ends:
                ax1.plot(x2[i0:i1], t2[i0:i1],          '-', label=phase_name, zorder=1, alpha=0.8)
                ax2.plot(x2[i0:i1], 1.0/inv_rp2[i0:i1], '-', label=phase_name, zorder=1, alpha=0.8)
                ax3.plot(t2[i0:i1], 1.0/inv_rp2[i0:i1], '-', label=phase_name, zorder=1, alpha=0.8)
            #
        ax1.xaxis.set_major_locator(plt.MultipleLocator(50))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))
        ax1.yaxis.set_major_locator(plt.MultipleLocator(500))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(100))
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True, which='major', linestyle='-', alpha=0.5, lw=0.6)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3, lw=0.6)
            # remove duplicate labels in legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
        ax1.set_xlabel('Purist distance (degree)')
        ax1.set_ylabel('Travel time (s)')
        ax2.set_xlabel('Purist distance (degree)')
        ax2.set_ylabel('Ray parameter (s/km)')
        ax3.set_xlabel('Travel time (s)')
        ax3.set_ylabel('Ray parameter (s/km)')
        plt.savefig('benchmark_phase2xt_dist_range.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()
    # Ray tracing
    def dist2pxt_base(self, dist, phase_type, nP, nS, nK, nI, nJ, db_inv_rp, db_x, db_t, db_istart_end, xerr=1e-3, niter=1000):
        nsol = 0
        sol_inv_rps = np.zeros(1024, dtype=np.float64)
        sol_x       = np.zeros(1024, dtype=np.float64)
        sol_t       = np.zeros(1024, dtype=np.float64)
        for istart, iend in db_istart_end:
            inv_rps = db_inv_rp[istart:iend]
            x       = db_x[istart:iend]
            t       = db_t[istart:iend]
            if not ((x[0] <= dist <= x[-1]) or (x[-1] <= dist <= x[0])):
                continue
            ######
            if x[0] <= x[-1]:
                ir = np.searchsorted(x, dist)
            else:
                ir = np.searchsorted(x[::-1], dist)
                ir = x.size - ir
            ######
            ir = np.clip(ir, 1, x.size-1)
            il = ir-1
            if np.abs(x[ir] - dist) < xerr:
                inv_rp_found = inv_rps[ir]
                x_found      = x[ir]
                t_found      = t[ir]
            elif np.abs(x[il] - dist) < xerr:
                inv_rp_found = inv_rps[il]
                x_found      = x[il]
                t_found      = t[il]
            else:
                x_start = x[il]
                x_end   = x[ir]
                inv_rp_start = inv_rps[il]
                inv_rp_end   = inv_rps[ir]
                for it in range(niter):
                    inv_rp_mid = 0.5 * (inv_rp_start + inv_rp_end)
                    x_mid, t_mid = self.phase_p2xt_base(inv_rp_mid, nP, nS, nK, nI, nJ, grad=False)[0:2]
                    if np.abs(x_mid - dist) < xerr:
                        break
                    if (x_start <= dist <= x_mid) or (x_mid <= dist <= x_start):
                        inv_rp_end   = inv_rp_mid
                    else:
                        inv_rp_start = inv_rp_mid
                        x_start      = x_mid
                ##########
                inv_rp_found = inv_rp_mid
                x_found      = x_mid
                t_found      = t_mid
            ######
            sol_inv_rps[nsol] = inv_rp_found
            sol_x[nsol]       = x_found
            sol_t[nsol]       = t_found
            nsol += 1
        return sol_inv_rps[:nsol], sol_x[:nsol], sol_t[:nsol]
    def dist2pxt(self, distances, phase_name, xerr=1e-3, niter=1000, grad=False):
        phase_type, nP, nS, nK, nI, nJ = self.decipher_phase_name(phase_name)
        tmp = self.phase2xt_base(phase_type, nP, nS, nK, nI, nJ)  # warm up
        db_inv_rp, db_x, db_t, db_istart_end = tmp[:4]
        sol_inv_rps = list()
        sol_xs = list()
        sol_ts = list()
        for single_x in distances:
            inv_rps, xs, ts = self.dist2pxt_base(single_x, phase_type, nP, nS, nK, nI, nJ, db_inv_rp, db_x, db_t, db_istart_end, xerr=xerr, niter=niter)
            sol_inv_rps.extend(inv_rps)
            sol_xs.extend(xs)
            sol_ts.extend(ts)
        sol_inv_rps = np.array(sol_inv_rps)
        sol_xs = np.array(sol_xs)
        sol_ts = np.array(sol_ts)
        if not grad:
            return sol_inv_rps, sol_xs, sol_ts
        else:
            nrp = sol_inv_rps.size
            nz = self.all_z.size
            pxpvzp = np.zeros((nrp, nz), dtype=np.float64)
            pxpvzs = np.zeros((nrp, nz), dtype=np.float64)
            pxpp   = np.zeros(nrp, dtype=np.float64)
            ptpvzp = np.zeros((nrp, nz), dtype=np.float64)
            ptpvzs = np.zeros((nrp, nz), dtype=np.float64)
            ptpp   = np.zeros(nrp, dtype=np.float64)
            dtdvzp = np.zeros((nrp, nz), dtype=np.float64)
            dtdvzs = np.zeros((nrp, nz), dtype=np.float64)
            for irp in range(nrp):
                tmp = self.phase_p2xt_base(sol_inv_rps[irp], nP, nS, nK, nI, nJ, True)
                #par_x_vzp, par_x_vzs, par_x_p, par_t_vzp, par_t_vzs, par_t_p, d_t_vzp, d_t_vzs
                pxpvzp[irp] = tmp[2]
                pxpvzs[irp] = tmp[3]
                pxpp[irp]   = tmp[4]
                ptpvzp[irp] = tmp[5]
                ptpvzs[irp] = tmp[6]
                ptpp[irp]   = tmp[7]
                dtdvzp[irp] = tmp[8]
                dtdvzs[irp] = tmp[9]
            return sol_inv_rps, sol_xs, sol_ts, pxpvzp, pxpvzs, pxpp, ptpvzp, ptpvzs, ptpp, dtdvzp, dtdvzs
    @staticmethod
    def benchmark_dist2pxt_grad():
        import pickle
        from matplotlib.gridspec import GridSpec
        app = FastTauP(R0=6371.0, dr=0.5)
        mt_r = app.all_r[:app.all_icmb]
        oc_r = app.all_r[app.all_icmb: app.all_iicb]
        ic_r = app.all_r[app.all_iicb:]
        mt_dr = np.diff(mt_r)
        oc_dr = np.diff(oc_r)
        ic_dr = np.diff(ic_r)
        print('mt dr range=', mt_dr.min(), mt_dr.max() )
        print('oc dr range=', oc_dr.min(), oc_dr.max() )
        print('ic dr range=', ic_dr.min(), ic_dr.max() )
        vol_data = dict()
        for phase_name in ['SKS', 'SKKS', 'SKKKS', 'K', 'ScS']:
        #for phase_name in ['ScS']: #, 'SKKS', 'SKKKS']:
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(200, 100, figure=fig)
            ax1 = fig.add_subplot(gs[0:70,     0:20])
            ax2 = fig.add_subplot(gs[0:70,    30:50])
            ax3 = fig.add_subplot(gs[0:70,    60:80])
            ax4 = fig.add_subplot(gs[90:180,   0:80])
            cax4= fig.add_subplot(gs[195:200,  0:35])
            cax5= fig.add_subplot(gs[195:200, 45:80])
            ####
            tmp = app.phase2xt(phase_name, grad=False, max_theta_step_rad=0.0017)
            db_inv_rp, db_x_km, db_t = tmp[:3]
            db_x = db_x_km / (np.pi/180.0) / 6371.0
            ax1.plot(db_x, db_t, '.', color='k', markersize=2)
            ax2.plot(db_x, 1.0/np.array(db_inv_rp), '.', color='k', markersize=2)
            ax3.plot(db_t, 1.0/np.array(db_inv_rp), '.', color='k', markersize=2)
            ######
            #db_x_deg = db_x/6371.0 * (180.0/np.pi)
            #xmin, xmax = np.min(db_x_deg), np.max(db_x_deg)
            #print(xmin, xmax)
            #x1_deg = np.arange(xmin+1, xmax-1, 0.1)
            #x1 = x1_deg * (np.pi/180.0) * 6371.0
            ######
            x2 = np.arange(62.0, db_x.max()-1, 1)
            if phase_name in ['K', 'ScS']:
                x2 = np.arange(1.0, db_x.max()-1, 1)
            x2_km = x2*np.pi/180.0*6371.0
            result = app.dist2pxt(x2_km, phase_name, xerr=1e-3, niter=1000, grad=True)
            inv_rps, _, t2 = result[:3]
            #x2 = x2_km / (np.pi/180.0) / 6371.0
            dtdvzp, dtdvzs = result[-2], result[-1]
            #####
            vol_data[phase_name] = {
                'x_deg': x2,
                't': t2,
                'inv_rps': inv_rps,
                'dtdvzp': dtdvzp,
                'dtdvzs': dtdvzs,
                'r': app.all_r,
                'icmb': app.all_icmb,
                'iicb': app.all_iicb,
                'vzp': app.all_vzp,
                'vzs': app.all_vzs,
            }

            mt_rs = app.all_r[:app.all_icmb]
            oc_rs = app.all_r[app.all_icmb: app.all_iicb]
            #####
            ax1.plot(x2, t2,                    'o', color='C3', markersize=4, label=phase_name)
            ax2.plot(x2, 1.0/np.array(inv_rps), 'o', color='C3', markersize=4, label=phase_name)
            ax3.plot(t2, 1.0/np.array(inv_rps), 'o', color='C3', markersize=4, label=phase_name)
            for ax in [ax1, ax2, ax3]:
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.legend()
            ax1.set_xlabel('Purist distance (deg)')
            ax1.set_ylabel('Travel Time (s)')
            ax2.set_xlabel('Purist distance (deg)')
            ax2.set_ylabel('Ray Parameter (s/km)')
            ax3.set_xlabel('Travel Time (s)')
            ax3.set_ylabel('Ray Parameter (s/km)')
            ######
            mt_rs = app.all_r[:app.all_icmb]
            oc_rs = app.all_r[app.all_icmb: app.all_iicb]
            mt_dtdvzs = dtdvzs[:, :app.all_icmb]
            mt_extend = [mt_rs[0], mt_rs[-1], x2[0],x2[-1]]
            oc_dtdvzp = dtdvzp[:, app.all_icmb: app.all_iicb]
            oc_extend = [oc_rs[0], oc_rs[-1], x2[0],x2[-1]]
            ######
            if mt_dtdvzs.min() < mt_dtdvzs.max():
                vmin, vmax = np.percentile(mt_dtdvzs, 1.0), 0.0
                im4 = ax4.imshow(mt_dtdvzs, aspect='auto', cmap='Greys_r', origin='lower',
                                extent=mt_extend, vmin=vmin, vmax=vmax) #, vmax=0.0) #, vmin=vmin, vmax=vmax)
                fig.colorbar(im4, cax=cax4, orientation='horizontal', label='Mantle $dT/dV_P$ ($s^2/km$)')
            if oc_dtdvzp.min() < oc_dtdvzp.max():
                vmin, vmax = np.percentile(oc_dtdvzp, 1.0), 0.0
                im5 = ax4.imshow(oc_dtdvzp, aspect='auto', cmap='PuRd_r', origin='lower', extent=oc_extend, vmin=vmin, vmax=vmax)
                fig.colorbar(im5, cax=cax5, orientation='horizontal', label='Outer Core $dT/dV_S$ ($s^2/km$)')
            ax4.set_xlim((6371, 1300))
            ax4.set_xlabel('Radius (km)')
            ax4.set_ylabel('Purist distance (deg)')
            figname = 'benchmark_dist2pxt_grad_%s.png' % phase_name
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()
            print('Saved figure to %s' % figname)
        with open('benchmark_dist2pxt_grad_vol_data.pkl', 'wb') as f:
            pickle.dump(vol_data, f)
    @staticmethod
    def benchmark_dist2pxt():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        x1 = np.arange(61.0, 140.0, 1)*np.pi/180.0*6371.0
        for phase_name in ['SKS', 'SKKS', 'SKKKS']:
            app = FastTauP(R0=6371.0, dr=5.0)
            tmp = app.phase2xt(phase_name, grad=False, max_theta_step_rad=0.0017)
            db_inv_rp, db_x, db_t = tmp[:3]
            ax1.plot(db_x, db_t, '.', color='k', markersize=2)
            ax2.plot(db_x, 1.0/np.array(db_inv_rp), '.', color='k', markersize=2)
            ax3.plot(db_t, 1.0/np.array(db_inv_rp), '.', color='k', markersize=2)
            ######
            #db_x_deg = db_x/6371.0 * (180.0/np.pi)
            #xmin, xmax = np.min(db_x_deg), np.max(db_x_deg)
            #print(xmin, xmax)
            #x1_deg = np.arange(xmin+1, xmax-1, 0.1)
            #x1 = x1_deg * (np.pi/180.0) * 6371.0
            ######
            inv_rps, x2, t = app.dist2pxt(x1, phase_name, xerr=1e-3, niter=1000)
            ax1.plot(x2, t, 'o', color='C3', markersize=4)
            ax2.plot(x2, 1.0/np.array(inv_rps), 'o', color='C3', markersize=4)
            ax3.plot(t, 1.0/np.array(inv_rps), 'o', color='C3', markersize=4)
            for ax in [ax1, ax2, ax3]:
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.show()
    ################################################################################
class JUNK:
    def __init__(self):
        pass
    ################################################################################
    def make_model(self, r, vp, vs, icmb, iicb, R0=6371.0, dr=None):
        tmp = {
            'mantle':     { 'r': r[:icmb],     'vp': vp[:icmb],     'vs': vs[:icmb]},
            'outer_core': { 'r': r[icmb:iicb], 'vp': vp[icmb:iicb], 'vs': vs[icmb:iicb]},
            'inner_core': { 'r': r[iicb:],     'vp': vp[iicb:],     'vs': vs[iicb:]},
        }
        if dr is not None:
            for key in tmp.keys():
                old_r = tmp[key]['r']
                new_r, new_vp = denser_xy(old_r, tmp[key]['vp'], dr)
                _,     new_vs = denser_xy(old_r, tmp[key]['vs'], dr)
                tmp[key]['r']  = new_r
                tmp[key]['vp'] = new_vp
                tmp[key]['vs'] = new_vs
        tmp['inner_core']['r'][-1] = 0.01  # avoid zero radius
        #
        self.flattened_model = dict()
        for key in tmp.keys():
            z, vp = flatten(tmp[key]['r'], tmp[key]['vp'], R0)
            _, vs = flatten(tmp[key]['r'], tmp[key]['vs'], R0)
            dz = np.diff(z)
            dvp= np.diff(vp)
            dvs= np.diff(vs)
            #####
            dzdvp = np.zeros_like(dz, dtype=np.float64)
            dzdvs = np.zeros_like(dz, dtype=np.float64)
            dzdvp = np.divide(dz, dvp, out=dzdvp, where=dvp != 0.0)
            dzdvs = np.divide(dz, dvs, out=dzdvs, where=dvs != 0.0)
            #print('key=', key, 'dzdvs=', dzdvs, 'dzdvp=', dzdvp)
            #####
            #print('key=', key, ' nlayer=', z.size-1)
            vp_layer_type = np.zeros(z.size-1, dtype=np.int64)
            for ilayer in range(vp_layer_type.size):
                if np.abs(dvp[ilayer]) > 1e-10 and np.abs(dz[ilayer]) >1e-10:
                    vp_layer_type[ilayer] = 2  # non-constant vp layer
                elif np.abs(dz[ilayer]) >1e-10:
                    vp_layer_type[ilayer] = 1  # constant vp layer
            #####
            vs_layer_type = np.zeros(z.size-1, dtype=np.int64)
            for ilayer in range(vs_layer_type.size):
                if np.abs(dvs[ilayer]) > 1e-10 and np.abs(dz[ilayer]) >1e-10:
                    vs_layer_type[ilayer] = 2  # non-constant vs layer
                elif np.abs(dz[ilayer]) >1e-10:
                    vs_layer_type[ilayer] = 1  # constant vs layer
            #####
            self.flattened_model[key] = {
                'z': z,
                'vp': vp,
                'vs': vs,
                'vp_layer_type': vp_layer_type,
                'vs_layer_type': vs_layer_type,
                'dz': dz,
                'dvp': dvp,
                'dvs': dvs,
                'dzdvp': dzdvp,
                'dzdvs': dzdvs,
            }
        self.all_z = np.concatenate((self.flattened_model['mantle']['z'],
                                     self.flattened_model['outer_core']['z'],
                                     self.flattened_model['inner_core']['z'] ) )
        self.all_icmb = self.flattened_model['mantle']['z'].size
        self.all_iicb = self.all_icmb + self.flattened_model['outer_core']['z'].size
        self.buf= np.zeros((10, self.all_z.size), dtype=np.float64)
        ######
        self.mt_vp_max = np.max(self.flattened_model['mantle']['vp'])
        self.mt_vs_max = np.max(self.flattened_model['mantle']['vs'])
        self.oc_vp_max = np.max(self.flattened_model['outer_core']['vp'])
        self.ic_vp_max = np.max(self.flattened_model['inner_core']['vp'])
        self.ic_vs_max = np.max(self.flattened_model['inner_core']['vs'])
        #
        self.mt_vp_min = np.min(self.flattened_model['mantle']['vp'])
        self.mt_vs_min = np.min(self.flattened_model['mantle']['vs'])
        self.oc_vp_min = np.min(self.flattened_model['outer_core']['vp'])
        self.ic_vp_min = np.min(self.flattened_model['inner_core']['vp'])
        self.ic_vs_min = np.min(self.flattened_model['inner_core']['vs'])
    ################################################################################
    # Fundmental individual ray legs in mantle, OC, and IC.
    # Note: Each ray computation contain both downgoing and upgoing paths.
    # X pr XcX in the mantle
    def p2xt_mt_X(self, inv_rp, grad=False, wave_type='P'):
        if wave_type == 'S':
            z = self.flattened_model['mantle']['z']
            v = self.flattened_model['mantle']['vs']
            dz = self.flattened_model['mantle']['dz']
            dv = self.flattened_model['mantle']['dvs']
        elif wave_type == 'P':
            z = self.flattened_model['mantle']['z']
            v = self.flattened_model['mantle']['vp']
            dz = self.flattened_model['mantle']['dz']
            dv = self.flattened_model['mantle']['dvp']
        else:
            raise ValueError('wave_type must be P or S!')
        ########################################################################################################################
        if not grad:
            x, t = p2xt(inv_rp, z, v, dz, dv)
            ak = (x, t, None, None, None, None, None)
        else:
            ak = p2xt_grad(inv_rp, z, v, dz, dv, self.buf)
            #ak = (x, t, pxpv, pxpp, ptpv, ptpp, dtdv)
        return ak
    # K or KiK in the outer core
    def p2xt_oc_K(self, inv_rp, grad=False):
        z = self.flattened_model['outer_core']['z']
        v = self.flattened_model['outer_core']['vp']
        dz = self.flattened_model['outer_core']['dz']
        dv = self.flattened_model['outer_core']['dvp']
        ########################################################################################################################
        if not grad:
            x, t = p2xt(   inv_rp, z, v, dz, dv)
            ak = (x, t, None, None, None, None, None)
        else:
            x, t, tmp_pxpv, pxpp, tmp_ptpv, ptpp, tmp_dtdv = p2xt_grad(   inv_rp, z, v, dz, dv, self.buf)
            pxpv = np.zeros(self.all_z.size, dtype=np.float64)
            ptpv = np.zeros(self.all_z.size, dtype=np.float64)
            dtdv = np.zeros(self.all_z.size, dtype=np.float64)
            icmb = self.all_icmb
            iicb = self.all_iicb
            pxpv[icmb:iicb] = tmp_pxpv
            ptpv[icmb:iicb] = tmp_ptpv
            dtdv[icmb:iicb] = tmp_dtdv
            ak = (x, t, pxpv, pxpp, ptpv, ptpp, dtdv)
        return ak
    # X in the inner core
    def p2xt_ic_X(self, inv_rp, grad=False, wave_type='I'):
        if wave_type == 'J':
            z = self.flattened_model['inner_core']['z']
            v = self.flattened_model['inner_core']['vs']
            dz = self.flattened_model['inner_core']['dz']
            dv = self.flattened_model['inner_core']['dvs']
        elif wave_type == 'I':
            z = self.flattened_model['inner_core']['z']
            v = self.flattened_model['inner_core']['vp']
            dz = self.flattened_model['inner_core']['dz']
            dv = self.flattened_model['inner_core']['dvp']
        else:
            raise ValueError('wave_type must be I or J!')
        ########################################################################################################################
        if not grad:
            x, t = p2xt(inv_rp, z, v, dz, dv)
            ak = (x, t, None, None, None, None, None)
        else:
            ak = p2xt_grad(inv_rp, z, v, dz, dv, self.buf)
            #ak = (x, t, pxpv, pxpp, ptpv, ptpp, dtdv)
        return ak
    ################################################################################
    ################################################################################
    def p2xt_ScS(self, inv_rp, grad=False):
        return self.p2xt(inv_rp, 'ScS', grad=grad)
    def p2xt_SmKS(self, inv_rp, m=1, grad=False):
        phase_name = 'S%sS' % ('K'*int(m))
        return self.p2xt(inv_rp, phase_name, grad=grad)
    def deprecated_p2xt_ScS(self, inv_rp, grad=False):
        z = self.flattened_model['mantle']['z']
        v = self.flattened_model['mantle']['vs']
        dz = self.flattened_model['mantle']['dz']
        dv = self.flattened_model['mantle']['dvs']
        dzdv = self.flattened_model['mantle']['dzdvs']
        layer_type = self.flattened_model['mantle']['vs_layer_type']
        if not grad:
            x, t = p2xt_v4(inv_rp, z, v, dz, dv, dzdv, layer_type, z.size-1)
            ak = (x, t, None, None, None, None, None)
        else:
            buf = np.zeros((6, z.size), dtype=np.float64)
            x, t, tmp_pxpv, pxpp, tmp_ptpv, ptpp, tmp_dtdv = p2xt_grad_v4(inv_rp, z, v, dz, dv, dzdv, layer_type, z.size-1, buf)
            pxpv = np.zeros(self.all_z.size, dtype=np.float64)
            ptpv = np.zeros(self.all_z.size, dtype=np.float64)
            dtdv = np.zeros(self.all_z.size, dtype=np.float64)
            icmb = self.all_icmb
            pxpv[:icmb] = tmp_pxpv
            ptpv[:icmb] = tmp_ptpv
            dtdv[:icmb] = tmp_dtdv
            ak = (x, t, pxpv, pxpp, ptpv, ptpp, dtdv)
        return ak
    def deprecated_p2xt_SmKS(self, inv_rp, m=1, grad=False):
        part_ScS = self.deprecated_p2xt_ScS(inv_rp, grad=grad)
        part_K   = self.p2xt_oc_K(inv_rp,grad=grad)
        ###
        if not grad:
            x1, t1 = part_ScS[0], part_ScS[1]
            x2, t2 = part_K[0],   part_K[1]
            x = x1 + m * x2
            t = t1 + m * t2
            ak = (x, t, None, None, None, None, None)
        else:
            x1, t1, pxpv1, pxpp1, ptpv1, ptpp1, dtdv1 = part_ScS
            x2, t2, pxpv2, pxpp2, ptpv2, ptpp2, dtdv2 = part_K
            x = x1 + m * x2
            t = t1 + m * t2
            pxpv = pxpv1 + m*pxpv2
            pxpp = pxpp1 + m*pxpp2
            ptpv = ptpv1 + m*ptpv2
            ptpp = ptpp1 + m*ptpp2
            dtdv = dtdv1 + m*dtdv2
            ak = (x, t, pxpv, pxpp, ptpv, ptpp, dtdv)
        return ak
    ################################################################################
    def zv2pxt_ScS(self, theta_step_deg=0.1):
        ######
        z = self.flattened_model['mantle']['z']
        v = self.flattened_model['mantle']['vs']
        dz = self.flattened_model['mantle']['dz']
        dv = self.flattened_model['mantle']['dvs']
        dzdv = self.flattened_model['mantle']['dzdvs']
        layer_type = self.flattened_model['mantle']['vs_layer_type']
        ######
        nblocks, bb, inv_rps, ibs  = zv2p_reflect_v3(z, v, theta_step_deg)
        ######
        x = np.zeros(inv_rps.size, dtype=np.float64)
        t = np.zeros(inv_rps.size, dtype=np.float64)
        for irp in range(inv_rps.size):
            x[irp], t[irp] = p2xt_v4(inv_rps[irp], z, v, dz, dv, dzdv, layer_type, ibs[irp])
        return inv_rps, x, t
    def zv2pxt_K(self, theta_step_deg=0.1):
        z = self.flattened_model['outer_core']['z']
        v = self.flattened_model['outer_core']['vp']
        ######
        inv_rps, dist, trvt, ibs, leg_ind = zv2pxt_v4(z, v, theta_step_deg=theta_step_deg)
        return inv_rps, dist, trvt, ibs, leg_ind
    def zv2pxt_SmKS(self, m=1, theta_step_deg=0.1):
        inv_rps, oc_x, oc_t, oc_ibs, leg_ind = self.zv2pxt_K(theta_step_deg=theta_step_deg)
        ####
        mt_z = self.flattened_model['mantle']['z']
        mt_v = self.flattened_model['mantle']['vs']
        mt_dz = self.flattened_model['mantle']['dz']
        mt_dv = self.flattened_model['mantle']['dvs']
        mt_dzdv = self.flattened_model['mantle']['dzdvs']
        mt_layer_type = self.flattened_model['mantle']['vs_layer_type']
        ####
        x = np.zeros(inv_rps.size, dtype=np.float64)
        t = np.zeros(inv_rps.size, dtype=np.float64)
        for irp in range(inv_rps.size):
            x1, t1 = p2xt_v4(inv_rps[irp], mt_z, mt_v, mt_dz, mt_dv, mt_dzdv, mt_layer_type, mt_z.size-1)
            x2, t2 = oc_x[irp], oc_t[irp]
            x[irp] = x1 + m * x2
            t[irp] = t1 + m * t2
        return inv_rps, x, t, oc_ibs, leg_ind
    @staticmethod
    def benchmark3():
        from obspy.taup import TauPyModel
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
        model = TauPyModel("prem")
        #################################################################
        if False: # ScS
            tmp = [model.get_travel_times(0.0, x, ['ScS']) for x in np.arange(1, 60, 1.0)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1 = np.array([it.time     for it in arrs], dtype=np.float64)
            p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
            x1 *= (np.pi/180.0)*6371.0  # deg to km
            idx = np.argsort(p1)
            p1 = p1[idx]
            x1 = x1[idx]
            t1 = t1[idx]
            ax1.plot(x1, t1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax2.plot(x1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax3.plot(t1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ##################
            my_app = FastTauP(dr=0.1)
            x2 = np.zeros_like(x1)
            t2 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                tmp = my_app.deprecated_p2xt_ScS(1.0/rp, grad=True)
                x2[irp], t2[irp] = tmp[0], tmp[1]
            ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=3)
            ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=3)
            ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=3)
            ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
            ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
            ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
            ##################
            my_app = FastTauP(dr=0.1)
            x3 = np.zeros_like(x1)
            t3 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                tmp = my_app.p2xt_ScS(1.0/rp, grad=False)
                x3[irp], t3[irp] = tmp[0], tmp[1]
            ax1.plot(x3, t3, '.-', color='r', label='This work', markersize=1)
            ax2.plot(x3, p1, '.-', color='r', label='This work', markersize=1)
            ax3.plot(t3, p1, '.-', color='r', label='This work', markersize=1)
            ###
            ax4.plot(x3, t3-t2, 'o-', color='r', markersize=4)
            ax5.plot(x3-x2, p1, 'o-', color='r', markersize=4)
            ax6.plot(t3-t2, p1, 'o-', color='r', markersize=4)
        #################################################################
        if True: # SmKS
            phs = 'SKS', 'SKKS', 'SKKKS', 'SKKKKS'
            ms  = 1, 2, 3, 4
            for m, ph in zip(ms, phs):
                tmp = [model.get_travel_times(0.0, x, [ph]) for x in np.arange(1, 180, 1.0)]
                arrs = [it2 for it1 in tmp for it2 in it1]
                p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
                x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
                t1 = np.array([it.time     for it in arrs], dtype=np.float64)
                p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
                x1 *= (np.pi/180.0)*6371.0  # deg to km
                idx = np.argsort(p1)
                p1 = p1[idx]
                x1 = x1[idx]
                t1 = t1[idx]
                ax1.plot(x1, t1, 'o-', color='k', label='ObsPy TauP', markersize=6)
                ax2.plot(x1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
                ax3.plot(t1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
                ##################
                my_app = FastTauP(dr=0.1)
                x2 = np.zeros_like(x1)
                t2 = np.zeros_like(t1)
                for irp, rp in enumerate(p1):
                    tmp = my_app.deprecated_p2xt_SmKS(1.0/rp, m=m, grad=True)
                    x2[irp], t2[irp] = tmp[0], tmp[1]
                ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=3)
                ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=3)
                ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=3)
                ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
                ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
                ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
                ##################
                my_app = FastTauP(dr=0.1)
                x3 = np.zeros_like(x1)
                t3 = np.zeros_like(t1)
                for irp, rp in enumerate(p1):
                    tmp = my_app.p2xt_SmKS(1.0/rp, m=m, grad=False)
                    x3[irp], t3[irp] = tmp[0], tmp[1]
                ax1.plot(x3, t3, '.-', color='r', label='This work', markersize=1)
                ax2.plot(x3, p1, '.-', color='r', label='This work', markersize=1)
                ax3.plot(t3, p1, '.-', color='r', label='This work', markersize=1)
                ###
                ax4.plot(x3, t3-t2, 'o-', color='r', markersize=4)
                ax5.plot(x3-x2, p1, 'o-', color='r', markersize=4)
                ax6.plot(t3-t2, p1, 'o-', color='r', markersize=4)
        #################################################################
        plt.show()
    @staticmethod
    def benchmark2():
        from obspy.taup import TauPyModel
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
        model = TauPyModel("prem")
        #################################################################
        if True: # ScS
            tmp = [model.get_travel_times(0.0, x, ['ScS']) for x in np.arange(1, 60, 1.0)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1 = np.array([it.time     for it in arrs], dtype=np.float64)
            p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
            x1 *= (np.pi/180.0)*6371.0  # deg to km
            idx = np.argsort(p1)
            p1 = p1[idx]
            x1 = x1[idx]
            t1 = t1[idx]
            ax1.plot(x1, t1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax2.plot(x1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax3.plot(t1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ###
            my_app = FastTauP(dr=0.1)
            x2 = np.zeros_like(x1)
            t2 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                x2[irp], t2[irp], par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v = my_app.deprecated_p2xt_ScS(1.0/rp, grad=True)
            ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=3)
            ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=3)
            ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=3)
            ###
            ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
            ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
            ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
            ###
            inv_rp3, x3, t3 = my_app.zv2pxt_ScS()
            p3 = 1.0/inv_rp3
            ax1.plot(x3, t3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
            ax2.plot(x3, p3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
            ax3.plot(t3, p3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
        #################################################################
        if True: # SKS
            tmp = [model.get_travel_times(0.0, x, ['SKS']) for x in np.arange(0, 180, 1.0)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1 = np.array([it.time     for it in arrs], dtype=np.float64)
            p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
            x1 *= (np.pi/180.0)*6371.0  # deg to km
            idx = np.argsort(p1)[::-1]
            p1 = p1[idx]
            x1 = x1[idx]
            t1 = t1[idx]
            ax1.plot(x1, t1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax2.plot(x1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax3.plot(t1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ###
            my_app = FastTauP(dr=0.1)
            x2 = np.zeros_like(x1)
            t2 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                x2[irp], t2[irp], par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v = my_app.deprecated_p2xt_SmKS(1.0/rp, m=1, grad=True)
            ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=3)
            ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=3)
            ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=3)
            ###
            ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
            ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
            ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
            ###
            inv_rp3, x3, t3, oc_ibs, leg_ind= my_app.zv2pxt_SmKS(m=1)
            p3 = 1.0/inv_rp3
            ax1.plot(x3, t3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
            ax2.plot(x3, p3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
            ax3.plot(t3, p3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
        #################################################################
        if True: # SmKS
            phase = 'SKKKS'
            m     = 3
            tmp = [model.get_travel_times(0.0, x, [phase]) for x in np.arange(0, 180, 1.0)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1 = np.array([it.time     for it in arrs], dtype=np.float64)
            p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
            x1 *= (np.pi/180.0)*6371.0  # deg to km
            idx = np.argsort(p1)[::-1]
            p1 = p1[idx]
            x1 = x1[idx]
            t1 = t1[idx]
            ax1.plot(x1, t1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax2.plot(x1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ax3.plot(t1, p1, 'o-', color='k', label='ObsPy TauP', markersize=6)
            ###
            my_app = FastTauP(dr=0.1)
            x2 = np.zeros_like(x1)
            t2 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                x2[irp], t2[irp], par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v = my_app.deprecated_p2xt_SmKS(1.0/rp, m=m, grad=True)
            ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=3)
            ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=3)
            ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=3)
            ###
            ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
            ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
            ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
            ###
            inv_rp3, x3, t3, oc_ibs, leg_ind= my_app.zv2pxt_SmKS(m=m)
            p3 = 1.0/inv_rp3
            ax1.plot(x3, t3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
            ax2.plot(x3, p3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
            ax3.plot(t3, p3, '-', color='C0', lw=1, label='This work (zv2pxt)', markersize=6, zorder=100)
        ###
        plt.show()
    @staticmethod
    def benchmark1():
        from obspy.taup import TauPyModel
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
        model = TauPyModel("prem")
        #################################################################
        if True: # ScS
            tmp = [model.get_travel_times(0.0, x, ['ScS']) for x in np.arange(1, 60, 1.0)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1 = np.array([it.time     for it in arrs], dtype=np.float64)
            p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
            x1 *= (np.pi/180.0)*6371.0  # deg to km
            idx = np.argsort(p1)
            p1 = p1[idx]
            x1 = x1[idx]
            t1 = t1[idx]
            ax1.plot(x1, t1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ax2.plot(x1, p1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ax3.plot(t1, p1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ###
            my_app = FastTauP(dr=0.1)
            x2 = np.zeros_like(x1)
            t2 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                x2[irp], t2[irp], par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v = my_app.deprecated_p2xt_ScS(1.0/rp, grad=True)
            ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=6)
            ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=6)
            ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=6)
            ###
            ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
            ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
            ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
        #################################################################
        if True: # SKS
            tmp = [model.get_travel_times(0.0, x, ['SKS']) for x in np.arange(0, 180, 1.0)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1 = np.array([it.time     for it in arrs], dtype=np.float64)
            p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
            x1 *= (np.pi/180.0)*6371.0  # deg to km
            idx = np.argsort(p1)[::-1]
            p1 = p1[idx]
            x1 = x1[idx]
            t1 = t1[idx]
            ax1.plot(x1, t1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ax2.plot(x1, p1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ax3.plot(t1, p1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ###
            my_app = FastTauP(dr=0.1)
            x2 = np.zeros_like(x1)
            t2 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                x2[irp], t2[irp], par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v = my_app.deprecated_p2xt_SmKS(1.0/rp, m=1, grad=True)
            ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=6)
            ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=6)
            ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=6)
            ###
            ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
            ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
            ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
        #################################################################
        if True: # SmKS
            tmp = [model.get_travel_times(0.0, x, ['SKKKS']) for x in np.arange(0, 180, 1.0)]
            arrs = [it2 for it1 in tmp for it2 in it1]
            p1 = np.array([it.ray_param_sec_degree for it in arrs], dtype=np.float64)
            x1 = np.array([it.purist_distance for it in arrs], dtype=np.float64)
            t1 = np.array([it.time     for it in arrs], dtype=np.float64)
            p1 /= (np.pi/180.0)*6371.0  # sec/deg to sec/km
            x1 *= (np.pi/180.0)*6371.0  # deg to km
            idx = np.argsort(p1)[::-1]
            p1 = p1[idx]
            x1 = x1[idx]
            t1 = t1[idx]
            ax1.plot(x1, t1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ax2.plot(x1, p1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ax3.plot(t1, p1, 's-', color='C0', label='ObsPy TauP', markersize=4)
            ###
            my_app = FastTauP(dr=0.1)
            x2 = np.zeros_like(x1)
            t2 = np.zeros_like(t1)
            for irp, rp in enumerate(p1):
                x2[irp], t2[irp], par_dist_v, par_dist_p, par_trvt_v, par_trvt_p, d_trvt_v = my_app.deprecated_p2xt_SmKS(1.0/rp, m=3, grad=True)
            ax1.plot(x2, t2, '.-', color='C1', label='This work', markersize=6)
            ax2.plot(x2, p1, '.-', color='C1', label='This work', markersize=6)
            ax3.plot(t2, p1, '.-', color='C1', label='This work', markersize=6)
            ###
            ax4.plot(x2, t2 - t1, 'o-', color='k', markersize=4)
            ax5.plot(x2-x1, p1, 'o-', color='k', markersize=4)
            ax6.plot(t2-t1, p1, 'o-', color='k', markersize=4)
        ###
        plt.show()

if __name__ == '__main__':
    if True:
        #FastTauP.benchmark_IJ_inv_rps()
        #FastTauP.benchmark_phase_p2xt()
        #FastTauP.benchmark_phase2xt_dist_range()
        FastTauP.benchmark_dist2pxt_grad()
    if False:
        np.set_printoptions(precision=6, suppress=True)
        z = np.arange(31)
        # E.g. (1): [1, 2, 10, 11, 12, 20, 30]
        v = np.arange(31) + 10
        v[3:10] = 1.0
        v[13:20]= 1.0
        v[21:30]= 1.0
        ## E.g. (2): [      10, 11, 12, 20, 30]
        #v = np.arange(31) + 10
        #v[1:10] = 1.0
        #v[13:20]= 1.0
        #v[21:30]= 1.0
        ## E.g. (3): [1,    10, 11, 12, 20, 30]
        #v = np.arange(31) + 10
        #v[2:10] = 1.0
        #v[13:20]= 1.0
        #v[21:30]= 1.0
        z = np.array(z, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        print('v=', v)
        print()
        inv_rps, ibs, ibreaks = z2inv_rp_turn_back(z, v)
        print('Turn back')
        print('inv_rps=', inv_rps)
        print('ibs=', ibs)
        print('ibreaks=', ibreaks)
        for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
            print('ileg=', ileg, 'inv_rp=', inv_rps[i0:i1], 'ibs=', ibs[i0:i1])
        print()
        print('Penetrate')
        inv_rps, ibs, ibreaks = z2inv_rp_penetrate(z, v)
        print('inv_rps=', inv_rps)
        print('ibs=', ibs)
        print('ibreaks=', ibreaks)
        for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
            print('ileg=', ileg, 'inv_rp=', inv_rps[i0:i1], 'ibs=', ibs[i0:i1])
        print()
        print('Turn back + Penetrate')
        inv_rps, ibs, ibreaks = z2inv_rp_both(z, v)
        print('inv_rps=', inv_rps)
        print('ibs=', ibs)
        print('ibreaks=', ibreaks)
        for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
            print('ileg=', ileg, 'inv_rp=', inv_rps[i0:i1], 'ibs=', ibs[i0:i1])
    if False:
        np.set_printoptions(precision=6, suppress=True)
        z = np.arange(31)
        # E.g. (1): [1, 2, 10, 11, 12, 20, 30]
        v = np.arange(31) + 10
        v[3:10] = 1.0
        v[13:20]= 1.0
        v[21:30]= 1.0
        ## E.g. (2): [      10, 11, 12, 20, 30]
        #v = np.arange(31) + 10
        #v[1:10] = 1.0
        #v[13:20]= 1.0
        #v[21:30]= 1.0
        ## E.g. (3): [1,    10, 11, 12, 20, 30]
        #v = np.arange(31) + 10
        #v[2:10] = 1.0
        #v[13:20]= 1.0
        #v[21:30]= 1.0
        z = np.array(z, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        inv_rps, ibs, ibreaks = z2inv_rp_turn_back(z, v)
        print('v=', v)
        print('inv_rps=', inv_rps)
        print('ibs=', ibs)
        print('ibreaks=', ibreaks)
        for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
            print('ileg=', ileg, 'inv_rp=', inv_rps[i0:i1], 'ibs=', ibs[i0:i1])
        #########
        inv_rps, ibs, ibreaks = denser_inv_rps(inv_rps, ibs, ibreaks, v0=v[0], max_theta_step_rad=0.1)
        print()
        print('Denser')
        print('inv_rps=', inv_rps)
        print('ibs=', ibs)
        print('ibreaks=', ibreaks)
        for ileg, (i0, i1) in enumerate(zip(ibreaks[:-1], ibreaks[1:])):
            print('ileg=', ileg, 'inv_rp=', inv_rps[i0:i1], 'ibs=', ibs[i0:i1])
    if False:
        nz = 30
        z  = np.linspace(0, -1000, nz)
        vz_mono = np.linspace(10, 20, nz)
        dvz     = (np.random.random(nz)-0.5)*10
        vz      = vz_mono + dvz
        inv_rps, ind_zs, ibs, ibreaks = z2inv_rp_turn_back(z, vz, True)
        print(inv_rps)
        print(ind_zs)
        print(ibs)
        print(ibreaks)
        #
        inv_rps, ind_zs, ibs, ibreaks = z2inv_rp_penetrate(z, vz, True)
        print(inv_rps)
        print(ind_zs)
        print(ibs)
        print(ibreaks)