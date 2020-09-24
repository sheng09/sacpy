#!/usr/bin/env python3

import numpy as np
import sys
import sacpy.sac as sac
import getopt
import matplotlib.pyplot as plt

def rd_sf_output(nrcv, ux_fnm, uz_fnm, data_type):
    """
    Read specfem2D output
    """
    ux = np.fromfile(ux_fnm, dtype= data_type)
    uz = np.fromfile(uz_fnm, dtype= data_type)
    npts = int(ux.size/nrcv)
    ux.shape = (nrcv, npts)
    uz.shape = (nrcv, npts)
    return ux, uz
def plot_vr(figname, uv_mat, ur_mat, extent, t_range, d_range, title_v='Vertical', title_r='Radial'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10) )
    for ax, mat, title in zip([ax1, ax2], [uv_mat, ur_mat], [title_v, title_r]):
        vmax = mat.max() * 0.1
        ax.imshow(mat, extent=extent, aspect='auto', cmap='seismic', vmax= vmax, vmin=-vmax, origin='lower')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Receiver longitude ($\degree$)')
        ax.set_xlim(t_range)
        ax.set_ylim(d_range)
        ax.set_title(title)
    plt.tight_layout()
    print(figname)
    plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.2)

def plot_xz(figname, ux_mat, uz_mat, extend, t_range, d_range, title_x='X', title_z='Z'):
    plot_vr(figname, ux_mat, uz_mat, extend, t_range, d_range, title_x, title_z)

def run_out2sac(src_xz, rcv_fnm, delta, t0, ux_fnm, uz_fnm, data_type, prefilename, plot_t_range, plot_stlo_range, cmp):
    sx, sz = src_xz
    s_lon = np.arctan2(sz, sx)
    rcv_vol = np.loadtxt(rcv_fnm, dtype={'names': ['stnm', 'netwk', 'x', 'z', 'v4', 'v5' ], 
                                         'formats': ['S32', 'S32', np.float32, np.float32, np.float32, np.float32] } )
    ux_mat, uz_mat = rd_sf_output(len(rcv_vol), ux_fnm, uz_fnm, data_type)
    npts = ux_mat.shape[1]
    uv_mat = np.zeros((len(rcv_vol), npts), dtype=data_type)
    ur_mat = np.zeros((len(rcv_vol), npts), dtype=data_type)
    ####
    for idx, rcv in enumerate(rcv_vol):
        ux = ux_mat[idx, :]
        uz = uz_mat[idx, :]
        rx, rz = rcv['x'], rcv['z']
        r_lon = np.arctan2(rz, rx)
        gcarc_360 = np.rad2deg(r_lon - s_lon) % 360.0
        ##
        sac_x = sac.make_sactrace_v(ux, delta, t0, stlo=np.rad2deg(r_lon), stla=0.0, evlo=np.rad2deg(s_lon), evla=0.0 )
        sac_z = sac.make_sactrace_v(uz, delta, t0, stlo=np.rad2deg(r_lon), stla=0.0, evlo=np.rad2deg(s_lon), evla=0.0 )
        ## rotate
        u_vertical =  np.cos(r_lon)*ux + np.sin(r_lon)*uz
        u_radial   = -np.sin(r_lon)*ux + np.cos(r_lon)*uz
        uv_mat[idx, :] = u_vertical
        ur_mat[idx, :] = u_radial
        if gcarc_360 > 180.0:
            u_radial *= -1.0
        sac_vertical = sac.make_sactrace_v(u_vertical, delta, t0, stlo=np.rad2deg(r_lon), stla=0.0, evlo=np.rad2deg(s_lon), evla=0.0)
        sac_radial   = sac.make_sactrace_v(u_radial,   delta, t0, stlo=np.rad2deg(r_lon), stla=0.0, evlo=np.rad2deg(s_lon), evla=0.0)
        ## output
        cmp_out = 'all'
        sac_cmp_out = [sac_x, sac_z, sac_vertical, sac_radial]
        if cmp == 'XZ':
            cmp_out = ['X', 'Z']
            sac_cmp_out = [sac_x, sac_z]
        elif cmp == 'RV':
            cmp_out = ['V', 'R']
            sac_cmp_out = [sac_vertical, sac_radial]
        else:
            cmp_out = ['X', 'Z', 'V', 'R']
            sac_cmp_out = [sac_x, sac_z, sac_vertical, sac_radial]
        ###
        for append, st in zip( cmp_out, sac_cmp_out ):
            fnm = '%s.%s.%s.%s.sac' % (prefilename, rcv['netwk'].decode('utf8'), rcv['stnm'].decode('utf8'), append)
            st.write(fnm)
    ####
    t1 = t0 + npts*delta
    lon0 = np.rad2deg(np.arctan2(rcv_vol[0]['z'], rcv_vol[0]['x']) ) % 360.0
    lon1 = np.rad2deg(np.arctan2(rcv_vol[-1]['z'], rcv_vol[-1]['x']) ) % 360.0
    if plot_t_range != None and plot_stlo_range != None and (cmp == 'all' or cmp == 'XZ') :
        plot_xz(prefilename +'_xz.png', ux_mat, uz_mat, [t0, t1, lon0, lon1], plot_t_range, plot_stlo_range )
    if plot_t_range != None and plot_stlo_range != None and (cmp == 'all' or cmp == 'RV') :
        plot_vr(prefilename +'_vr.png', uv_mat, ur_mat, [t0, t1, lon0, lon1], plot_t_range, plot_stlo_range )



if __name__ == "__main__":
    HMSG = '%s -S source_file -R receiver_file -D delta -B t_start -O pre_fnm [-P t0/t1/stlo0/stlo1] --ux ux.file --uz uz.file --accuracy=single --cmp=[XZ|RV|all]'
    src_x, src_z = None, None
    rcv_fnm = None
    t0 = 0.0
    ux_fnm, uz_fnm = None, None
    delta = None
    output_sac_prefnm = ''
    plot_t_range = None #[0.0, 3000.0]
    plot_stlo_range = None #[0.0, 180.0]
    data_type = np.float32
    cmp = 'all'
    options, remainder = getopt.getopt(sys.argv[1:], 'S:R:D:B:O:P:H', ['ux=', 'uz=', 'accuracy=', 'cmp='] )
    for opt, arg in options:
        if opt in ('-S'):
            with open(arg, 'r') as fid:
                for line in fid:
                    if line[:2] == 'xs':
                        src_x = float(line.split('#')[0].split('=')[1])
                    elif line[:2] == 'zs':
                        src_z = float(line.split('#')[0].split('=')[1])
        elif opt in ('-R'):
            rcv_fnm = arg
        elif opt in ('-D'):
            delta = float(arg)
        elif opt in ('-B'):
            t0 = float(arg)
        elif opt in ('--ux'):
            ux_fnm = arg
        elif opt in ('--uz'):
            uz_fnm = arg
        elif opt in ('--accuracy'):
            if arg =='single':
                data_type = np.float32
            elif arg == 'double':
                data_type = np.float64
        elif opt in ('--cmp'):
            if arg in ('RV', 'VR'):
                cmp = 'RV'
            elif arg in ('XZ', 'ZX'):
                cmp = 'XZ'
            else:
                cmp = 'all'
        elif opt in ('-O'):
            output_sac_prefnm = arg
        elif opt in ('-P'):
            junk = [float(it) for it in arg.split('/') ]
            plot_t_range = junk[:2]
            plot_stlo_range = junk[2:]
        elif opt in ('-H'):
            print(HMSG % (sys.argv[0]) )
            sys.exit(0)
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG % (sys.argv[0]) )
            sys.exit(0)
    run_out2sac((src_x, src_z), rcv_fnm, delta, t0, ux_fnm, uz_fnm, data_type, output_sac_prefnm, plot_t_range, plot_stlo_range, cmp)
