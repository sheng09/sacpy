#!/usr/bin/env python3
import numpy as np
import scipy
import os, os.path
from glob import glob
import h5py
from matplotlib import pyplot as plt

########################################################################################################################
# Prepare model file for Yspec
########################################################################################################################
def modify_velocity(iicb, icmb, radius, v, radius_range, ratio, modify_mantle=True, modify_oc=True, modify_ic=True):
    rmin, rmax = radius_range
    ic_r = radius[:iicb]
    oc_r = radius[iicb:icmb]
    mt_r = radius[icmb:]
    ic_v = v[:iicb]
    oc_v = v[iicb:icmb]
    mt_v = v[icmb:]
    for it_r, it_v, flag in zip((ic_r, oc_r, mt_r), (ic_v, oc_v, mt_v), (modify_ic, modify_oc, modify_mantle)):
        if flag:
            idx = np.where((it_r>=rmin) & (it_r<=rmax) )[0]
            it_v[idx] *= (1.0+ratio)
def modify_cmb(iicb, icmb, table, new_rcmb):
    r = table[:,0]
    old_rcmb = r[icmb]
    riicb = r[iicb]
    R0 = np.max(r)
    if (new_rcmb <= riicb) or (new_rcmb>=R0):
        raise ValueError(f'Err: the new rcmb {new_rcmb} is below icb or above free surface.  ({riicb}, {R0})')
    ######
    mt_tab= table[icmb:, :].copy()
    cr_tab= table[:icmb, :].copy()
    ######
    if new_rcmb == old_rcmb:
        pass
    elif new_rcmb > old_rcmb: # shallow RCMB
        ####
        mt_r = mt_tab[:,0]
        idx = np.searchsorted(mt_r, new_rcmb, side='left')
        ####
        one_row = [np.interp(new_rcmb, mt_r, mt_tab[:,icol]) for icol in (0,1,2,3,4,5) ]
        mt_tab  = np.concatenate(([one_row], mt_tab[idx+1:]) )
        ####
        tmp_rs = np.concatenate( (mt_r[:idx], [new_rcmb] ) )
        tmp_tab_t = [tmp_rs]
        cr_r = cr_tab[:,0]
        for icol in (1,2,3,4,5):
            f_linear = scipy.interpolate.interp1d(cr_r, cr_tab[:,icol], fill_value='extrapolate')
            tmp_tab_t.append( f_linear(tmp_rs) )
        tmp_tab = np.transpose(np.array(tmp_tab_t))
        cr_tab = np.concatenate((cr_tab, tmp_tab))
    else: # deeper RCMB
        ###
        cr_r = cr_tab[:,0]
        idx  = np.searchsorted(cr_r, new_rcmb, side='left')
        ###
        one_row = [np.interp(new_rcmb, cr_r, cr_tab[:,icol]) for icol in (0,1,2,3,4,5) ]
        cr_tab  = np.concatenate((cr_tab[:idx], [one_row]) )
        ###
        tmp_rs  = np.concatenate(([new_rcmb], cr_r[idx+1:] ) )
        tmp_tab_t = [tmp_rs]
        mt_r = mt_tab[:,0]
        for icol in (1,2,3,4,5):
            f_linear = scipy.interpolate.interp1d(mt_r, mt_tab[:,icol], fill_value='extrapolate')
            tmp_tab_t.append( f_linear(tmp_rs) )
        tmp_tab = np.transpose(np.array(tmp_tab_t))
        mt_tab = np.concatenate((tmp_tab, mt_tab))
    ####
    iicb = iicb
    icmb = cr_tab.shape[0]
    new_table = np.concatenate((cr_tab, mt_tab))
    return iicb, icmb, new_table
def modify_icb(iicb, icmb, table, new_ricb):
    r = table[:,0]
    old_ricb = r[iicb]
    rcmb = r[icmb]
    if (new_ricb <= 0) or (new_ricb>=rcmb):
        raise ValueError(f'Err: the new ricb {new_ricb} is above CMB or negative.  (0, {rcmb})')
    return modify_cmb(0, iicb, table, new_ricb) # cheat here!
def wrt_model(filename, iicb, icmb, table):
    """
    The table has rows of r, rho, vp, vs, qk, qm, with increasing r!
    """
    radius_m, rho_kg_m3, vp_m_s, vs_m_s, qk, qm = np.transpose(table)
    n = len(radius_m)
    with open(filename, 'w') as fid:
        print(filename.split('/')[-1], file=fid)
        print('0 1 1', file=fid)
        print(f'{n} {iicb} {icmb}', file=fid)
        for (r, rho, vp, vs, q1, q2) in zip(radius_m, rho_kg_m3, vp_m_s, vs_m_s, qk, qm):
            print(f'{r:13.4f} {rho:13.4f} {vp:13.4f} {vs:13.4f} {q1:13.4f} {q2:13.4f}', file=fid)
def rd_model(filename):
    with open(filename, 'r') as fid:
        name  = fid.readline().strip()
        line2 = fid.readline()
        n, iicb, icmb = [int(it) for it in fid.readline().strip().split()]
    table = np.loadtxt(filename, skiprows=3)
    if n != table.shape[0]:
        raise ValueError(f'Number of rows does not match: {filename}')
    return iicb, icmb, table
def plot_model(iicb, icmb, table, ax1=None, color='k', highlight_icb=True, highlight_cmb=True, plot_rho=True, plot_vp=True, plot_vs=True, plot_q=True, **kwargs):
    if (ax1 is None):
        fig, ax1 = plt.subplots(1, 1)
    r, rho, vp, vs, qk, qm = np.transpose(table)
    if plot_rho:
        ax1.plot(rho, r, color=color, **kwargs)
    if plot_vp:
        ax1.plot(vp,  r, color=color, **kwargs)
    if plot_vs:
        ax1.plot(vs,  r, color=color, **kwargs)
    if plot_q:
        ax1.plot(qk,  r, color=color, **kwargs)
        ax1.plot(qm,  r, color=color, **kwargs)
    #####################
    for y, flag in zip((rho, vp, vs), (plot_rho, plot_vp, plot_vs) ):
        if not flag:
            continue
        if highlight_icb:
            ax1.plot(y[iicb],   r[iicb],   's', color=color, zorder=0)
            ax1.plot(y[iicb-1], r[iicb-1], 's', color=color, zorder=0)
        if highlight_cmb:
            ax1.plot(y[icmb],   r[icmb],   's', color=color, zorder=0)
            ax1.plot(y[icmb-1], r[icmb-1], 's', color=color, zorder=0)
    if plot_q:
        for y in (qk, qm):
            if highlight_icb:
                ax1.plot(y[iicb],   r[iicb],    's', color=color, zorder=0)
                ax1.plot(y[iicb-1], r[iicb-1],  's', color=color, zorder=0)
            if highlight_cmb:
                ax1.plot(y[icmb],   r[icmb],    's', color=color, zorder=0)
                ax1.plot(y[icmb-1], r[icmb-1],  's', color=color, zorder=0)
    return ax1

########################################################################################################################
# Prepare configure files and folders for running Yspec 
########################################################################################################################
def wrt_yspec_config(filename, 
                     output_prefix, earth_model_fnm,
                     flag_attenuation=1, flag_gravitation=0,
                     output_unit=0, flag_tilt_correction=0,
                     lmin=0, lmax=3000,
                     fmin=0.2, fmax=160., # in mHz
                     length_min=550.,     # in minutes
                     delta=1.,            # in seconds
                     f11=0.5, f12=1., f21=155., f22=160., # in mhz
                     evdp=50., evla=0., evlo=0., # in km and degree
                     m_rr=1e26, m_rt=0., m_rp=0., m_tt=1e26, m_tp=0., m_pp=1e26, # in Nm
                     stdp=0., stla=[0.], stlo=[0.]): # in km and degree
    """
    """
    folder = '/'.join(filename.split('/')[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename, 'w') as fid:
        print("# This file contains the parameters needed to run the", file=fid) # 0
        print("# program yspec",                                       file=fid) # 1
        print("",                                                      file=fid) # 2
        print("# prefix for output files", file=fid) # 3
        print(f"{output_prefix}",          file=fid) # 4
        print("",                          file=fid) # 5
        print("# Earth Model",       file=fid) # 6
        print(f"{earth_model_fnm}",  file=fid) # 7
        print("",                    file=fid) # 8
        print("# attenuation switch: 1 = on, 0 = off", file=fid) # 9
        print(f"{flag_attenuation:1d}",                file=fid) # 10
        print("",                                      file=fid) # 11
        print("# gravitation: 0 = none, 1 = cowling, 2 = self", file=fid) # 12
        print(f"{flag_gravitation:1d}",                         file=fid) # 13
        print("",                                               file=fid) # 14
        print("# output: 0 = displacement, 1 = velocity, 2 = acceleration", file=fid) # 15
        print(f"{output_unit:1d}",                                          file=fid) # 16
        print("",                                                           file=fid) # 17
        print("# potential and tilt corrections: 0 = no, 1 = yes",  file=fid) # 18
        print(f"{flag_tilt_correction:1d}",                         file=fid) # 19
        print("",                                                   file=fid) # 20
        print("# lmin",         file=fid) # 21
        print(f"{lmin}",        file=fid) # 22
        print("",               file=fid) # 23
        print("# lmax",     file=fid) # 24
        print(f"{lmax}",    file=fid) # 25
        print("",           file=fid) # 26
        print("# fmin (mHz) (this should always be greater than 0)", file=fid) # 27
        print(f"{fmin}",                                         file=fid) # 28
        print("",                                                    file=fid) # 29
        print("# fmax (mHz)",                               file=fid) # 30
        print(f"{fmax}",                                file=fid) # 31
        print("",                                           file=fid) # 32
        print("# length of time series (min)",  file=fid) # 33
        print(f"{length_min}",                  file=fid) # 34
        print("",                               file=fid) # 35
        print("# time step (sec)",  file=fid) # 36
        print(f"{delta:.5f}",       file=fid) # 37
        print("",                   file=fid) # 38
        print("# f11 filter (mHz)",         file=fid) # 39
        print(f"{f11}",                     file=fid) # 40
        print("",                           file=fid) # 41
        print("# f12 filter (mHz)",     file=fid) # 42
        print(f"{f12}",                 file=fid) # 43
        print("",                       file=fid) # 44
        print("# f21 filter (mHz)",         file=fid) # 45
        print(f"{f21}",                     file=fid) # 46
        print("",                           file=fid) # 47
        print("# f22 filter (mHz)",     file=fid) # 48
        print(f"{f22}",                 file=fid) # 49
        print("",                       file=fid) # 50
        print("# source depth (km)",        file=fid) # 51
        print(f"{evdp}",                    file=fid) # 52
        print("",                           file=fid) # 53
        print("# source latitude (deg)",        file=fid) # 54
        print(f"{evla}",                        file=fid) # 55
        print("",                               file=fid) # 56
        print("# source longitude (deg)",   file=fid) # 57
        print(f"{evlo}",                    file=fid) # 58
        print("",                           file=fid) # 59
        print("# M_{r,r} (Nm)", file=fid) # 60
        print(f"{m_rr:e}",      file=fid) # 61
        print("",               file=fid) # 62
        print("# M_{r,theta} (Nm)", file=fid) # 63
        print(f"{m_rt:e}",          file=fid) # 64
        print("",                   file=fid) # 65
        print("# M_{r,phi} (Nm)",           file=fid) # 66
        print(f"{m_rp}",                    file=fid) # 67
        print("",                           file=fid) # 68
        print("# M_{theta,theta} (Nm)", file=fid) # 69
        print(f"{m_tt:e}",              file=fid) # 70
        print("",                       file=fid) # 71
        print("# M_{theta,phi} (Nm)",       file=fid) # 72
        print(f"{m_tp:e}",                  file=fid) # 73
        print("",                           file=fid) # 74
        print("# M_{phi,phi} (Nm)", file=fid) # 75
        print(f"{m_pp:e}",          file=fid) # 76
        print("",                   file=fid) # 77
        print("# receiver depth (km)",  file=fid) # 78
        print(f"{stdp}",                file=fid) # 79
        print("",                       file=fid) # 80
        print("# number of receivers",      file=fid) # 81
        print(f"{len(stla)}",               file=fid) # 82
        print("",                           file=fid) # 83
        print("# receiver latitudes and longitudes", file=fid) # 84
        print("",                                    file=fid) # 85
        for idx in range(len(stla)):
            print(f"{stla[idx]:11.6f}  {stlo[idx]:11.6f}", file=fid)
def wrt_pbs_file(pbs_fnm, job_name='yspec', email="u6510109@anu.edu.au", project='em78', mem_gb=150, walltime_hr=1, ncpus=48,
                 storage="gdata/gb32+gdata/em78+scratch/em78+scratch/gb32"):
    with open(pbs_fnm, 'w') as fid:
        print(f"#!/bin/bash",                                                      file=fid)
        print(f"#PBS -N {job_name}",                                                  file=fid)
        print(f"#PBS -M {email}",                                                  file=fid)
        print(f"#PBS -P {project}",                                                file=fid)
        print(f"#PBS -l mem={mem_gb}G",                                           file=fid)
        print(f"#PBS -l walltime={walltime_hr:02d}:00:00",                         file=fid)
        print(f"#PBS -l jobfs=0.05GB",                                             file=fid)
        print(f"#PBS -l ncpus={ncpus}",                                            file=fid)
        print(f"#PBS -l storage={storage}",                                        file=fid)
        print(f"#PBS -l wd",                                                       file=fid)
        print(f"#PBS -o stdout.txt",                                              file=fid)
        print(f"#PBS -e stderr.txt",                                              file=fid)
        print(f"",                                                                 file=fid)
        print(f"# running command",                                                file=fid)
        print(f"mpirun -n {ncpus} ./yspec_mpi yspec.in > yspec_output",           file=fid)
        print(f"",                                                                 file=fid)
        print(f"mv out.* yspec_out/",                                        file=fid)
        print(f"# delete unnecessary output files",                                file=fid)
        print(f"rm -f yspec_cal.* yspec_pre.*",                                    file=fid)
def mk_working_directory(wd, model_filename, yspec_home, yspec_config_dict=dict(), pbs_config_dict=dict(), nsta_per_grp=200):
    all_stla = yspec_config_dict.pop('stla')
    all_stlo = yspec_config_dict.pop('stlo')
    job_name = pbs_config_dict.pop('job_name')
    nsta = len(all_stla)
    if nsta > nsta_per_grp:
        #ind = np.arange(nsta_per_grp, nsta, nsta_per_grp)
        #groups_stla = np.split(all_stla, ind)
        #groups_stlo = np.split(all_stlo, ind)
        ngrp = int(np.ceil(nsta/nsta_per_grp) )
        groups_stla = np.array_split(all_stla, ngrp)
        groups_stlo = np.array_split(all_stlo, ngrp)
    else:
        groups_stla = [all_stla]
        groups_stlo = [all_stlo]
    #
    cum_nsta = 0
    lst_local_wd = list()
    for (stla, stlo) in zip(groups_stla, groups_stlo):
        ########################################################################
        if len(groups_stla) > 1:
            local_wd = f'{wd}/ircv_{cum_nsta:05d}'
        else:
            local_wd = wd
        lst_local_wd.append(local_wd)
        ########################################################################
        if not os.path.exists(local_wd):
            os.makedirs(local_wd)
        ########################################################################
        output_dir = f'{local_wd}/yspec_out'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ########################################################################
        dest = f'{local_wd}/mod'
        os.symlink(model_filename, dest)
        ########################################################################
        os.symlink(f'{yspec_home}/bin/yspec_cal', f'{local_wd}/yspec_cal')
        os.symlink(f'{yspec_home}/bin/yspec_mpi', f'{local_wd}/yspec_mpi')
        os.symlink(f'{yspec_home}/bin/yspec_pre', f'{local_wd}/yspec_pre')
        os.symlink(f'{yspec_home}/bin/yspec_pro', f'{local_wd}/yspec_pro')        
        ########################################################################
        yspec_fnm = f'{local_wd}/yspec.in'
        yspec_config_dict.pop('output_prefix',   None)
        yspec_config_dict.pop('earth_model_fnm', None)
        wrt_yspec_config(yspec_fnm,
                        output_prefix='out', earth_model_fnm='mod',
                        stlo=stlo, stla=stla,
                        **yspec_config_dict)
        ########################################################################
        pbs_fnm = f'{local_wd}/run_pbs.sh'
        if len(groups_stla) > 1:
            local_job_name = f'{job_name}_{cum_nsta:05d}'
        else:
            local_job_name = job_name
        wrt_pbs_file(pbs_fnm, job_name=local_job_name, **pbs_config_dict)
        ########################################################################
        cum_nsta += len(stlo)
    ####
    if len(lst_local_wd) > 1:
        script_fnm = f'{wd}/qsub.sh'
        with open(script_fnm, 'w') as fid:
            print('#!/usr/bin/env bash', file=fid)
            for local_wd in lst_local_wd:
                tmp = os.path.realpath(local_wd)
                print(f'cd {tmp}',   file=fid)
                print(f'qsub run_pbs.sh\n', file=fid)

########################################################################################################################
# Read and save/transform Yspec running results
########################################################################################################################
def rd_yspec_config(filename):
    """
    Return a dictionary
    """
    with open(filename, 'r') as fid:
        tmp = fid.readlines()
        ####
        #output_prefix    = tmp[4]
        #earth_model_fnm  = tmp[7]
        ####
        flag_attenuation = int( tmp[10] )
        flag_gravitation = int( tmp[13] )
        output_unit      = int( tmp[16] )
        flag_tilt_correction = int( tmp[19] )
        ####
        lmin = float( tmp[22] )
        lmax = float( tmp[25] )
        fmin = float( tmp[28] )
        fmax = float( tmp[31] )
        ####
        length_min = float( tmp[34]  )
        delta      = float( tmp[37] )
        ####
        f11 = float( tmp[40] )
        f12 = float( tmp[43] )
        f21 = float( tmp[46] )
        f22 = float( tmp[49] )
        ####
        evdp = float(tmp[52] )
        evla = float(tmp[55] )
        evlo = float(tmp[58] )
        ####
        m_rr = float(tmp[61] )
        m_rt = float(tmp[64] )
        m_rp = float(tmp[67] )
        m_tt = float(tmp[70] )
        m_tp = float(tmp[73] )
        m_pp = float(tmp[76] )
        ####
        stdp = float(tmp[79] )
        nst  = int( tmp[82] )
        ####
        tmp2 = [line.strip().split() for line in tmp[86:86+nst] ]
        tmp2 = [it for it in tmp2 if len(it) == 2]
        stla = np.array([float(it[0]) for it in tmp2], dtype=np.float64)
        stlo = np.array([float(it[1]) for it in tmp2], dtype=np.float64)
        ####
        yspec_config = dict()
        yspec_config = {
            #####
            #'output_prefix':    output_prefix,
            #'earth_model_fnm':  earth_model_fnm,
            #####
            'flag_attenuation': flag_attenuation,
            'flag_gravitation': flag_gravitation,
            'output_unit': output_unit,
            'flag_tilt_correction': flag_tilt_correction,
            #####
            'lmin': lmin, 'lmax': lmax,
            'fmin': fmin, 'fmax': fmax,
            #####
            'length_min': length_min,
            'delta': delta,
            #####
            'f11': f11, 'f12': f12,
            'f21': f21, 'f22': f22,
            #####
            'evdp': evdp,
            'evla': evla,
            'evlo': evlo,
            #####
            'm_rr': m_rr, 'm_rt': m_rt,
            'm_rp': m_rp, 'm_tt': m_tt,
            'm_tp': m_tp, 'm_pp': m_pp,
            #####
            'stdp': stdp,
            'stla': stla,
            'stlo': stlo
        }
        return yspec_config
def rd_yspec_data_single(wd):
    yspec_config_fnm = f'{wd}/yspec.in'
    yspec_config     = rd_yspec_config(yspec_config_fnm)
    nsta = yspec_config['stlo'].size
    ############################################################
    ascii_fnms = sorted( glob(f'{wd}/yspec_out/out*') )
    tmp  = np.loadtxt(ascii_fnms[0])
    ncol = tmp.shape[0]
    yspec_config['tstart'] = tmp[0,0]
    zne_mat = np.zeros( (nsta*3, ncol) )
    for idx, (ascii_fnm) in enumerate(ascii_fnms):
        tmp = np.loadtxt(ascii_fnm)
        zne_mat[idx*3,   :] = tmp[:,1]
        zne_mat[idx*3+1, :] = tmp[:,2]
        zne_mat[idx*3+2, :] = tmp[:,3]
    return yspec_config, zne_mat
def rd_yspec_data_mult(wd):
    folders = sorted(glob(f'{wd}/ircv_*') )
    stlo, stla = list(), list()
    zne_mat = list()
    for it in folders:
        yc, single_zne_mat = rd_yspec_data_single(it)
        stlo.extend( yc['stlo'] )
        stla.extend( yc['stla'] )
        zne_mat.append(single_zne_mat)
    #########
    stlo = np.array(stlo)
    stla = np.array(stla)
    zne_mat = np.vstack(zne_mat).astype(np.float32)
    #########
    stdp = np.zeros( stlo.size ) + yc['stdp']
    evlo = np.zeros( stlo.size ) + yc['evlo'] 
    evla = np.zeros( stlo.size ) + yc['evla']
    evdp = np.zeros( stlo.size ) + yc['evdp']
    #########
    yc['stlo'] = stlo
    yc['stla'] = stla
    yc['stdp'] = stdp
    #########
    return yc, zne_mat
def wrt_h5(ofnm, zne_mat, sampling_interval, stlo, stla, stdp, evlo, evla, evdp, **kwargs):
    with h5py.File(ofnm, 'w') as fid:
        fid.create_dataset('dat', data=zne_mat, dtype=np.float32)
        ######
        hdr = fid.create_group('hdr')
        hdr.create_dataset('stlo',  data=stlo)
        hdr.create_dataset('stla',  data=stla)
        hdr.create_dataset('stdp',  data=stdp)
        hdr.create_dataset('evlo',  data=evlo)
        hdr.create_dataset('evla',  data=evla)
        hdr.create_dataset('evdp',  data=evdp)
        ######
        fid.attrs['nch']  = zne_mat.shape[0]
        fid.attrs['nsta'] = stlo.size
        fid.attrs['nt']   = zne_mat.shape[1]
        fid.attrs['sampling_interval'] = sampling_interval
        for key in kwargs:
            fid.attrs[key] = kwargs[key]
def yspec2h5(wd, ofnm):
    if os.path.exists(f'{wd}/yspec.in'):
        yc, zne_mat = rd_yspec_data_single(wd)
    else:
        yc, zne_mat = rd_yspec_data_mult(wd)
    #################
    sampling_interval = yc.pop('delta')
    stlo = yc.pop('stlo')
    stla = yc.pop('stla')
    stdp = yc.pop('stdp')
    evlo = yc.pop('evlo')
    evla = yc.pop('evla')
    evdp = yc.pop('evdp')
    wrt_h5(ofnm, zne_mat, sampling_interval, stlo, stla, stdp, evlo, evla, evdp, channels='ZNE', **yc)


def example_modify_model():
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fnm = 'dataset/models/prem.200noiso'
    iicb, icmb, table = rd_model(fnm)
    plot_model(iicb, icmb, table, ax1=ax1, ax2=ax2, color='k')
    plot_model(iicb, icmb, table, ax1=ax3, ax2=ax4, color='k')
    ################################################################################################
    # modify CMB
    #fnm = 'dataset/models/prem.200noiso'
    #iicb, icmb, table = rd_model(fnm)
    #iicb, icmb, table = modify_cmb(iicb, icmb, table, new_rcmb=table[icmb-5,0])
    #plot_model(iicb, icmb, table, ax1=ax1, ax2=ax2, color='r')
    #####
    #fnm = 'dataset/models/prem.200noiso'
    #iicb, icmb, table = rd_model(fnm)
    #iicb, icmb, table = modify_cmb(iicb, icmb, table, new_rcmb=table[icmb+15,0])
    #plot_model(iicb, icmb, table, ax1=ax1, ax2=ax2, color='C0')
    ################################################################################################
    # modify CMB and IICB
    #fnm = 'dataset/models/prem.200noiso'
    #iicb, icmb, table = rd_model(fnm)
    #iicb, icmb, table = modify_cmb(iicb, icmb, table, new_rcmb=table[icmb-5,0])
    #iicb, icmb, table = modify_icb(iicb, icmb, table, new_ricb=table[iicb-15,0])
    #plot_model(iicb, icmb, table, ax1=ax3, ax2=ax4, color='g')
    ######
    #fnm = 'dataset/models/prem.200noiso'
    #iicb, icmb, table = rd_model(fnm)
    #iicb, icmb, table = modify_cmb(iicb, icmb, table, new_rcmb=table[icmb-5,0])
    #iicb, icmb, table = modify_icb(iicb, icmb, table, new_ricb=table[iicb+15,0])
    #plot_model(iicb, icmb, table, ax1=ax3, ax2=ax4, color='b')
    ################################################################################################
    # modify CMB and IICB and Velfnm = 'dataset/models/prem.200noiso'
    iicb, icmb, table = rd_model(fnm)
    #iicb, icmb, table = modify_cmb(iicb, icmb, table, new_rcmb=table[icmb-5,0])
    #iicb, icmb, table = modify_icb(iicb, icmb, table, new_ricb=table[iicb-15,0])
    rcmb, ricb = table[icmb, 0], table[iicb, 0]
    modify_velocity(iicb, icmb, table[:,0], table[:,2], (ricb, rcmb), -0.01, modify_mantle=False, modify_ic=False)
    plot_model(iicb, icmb, table, ax1=ax3, ax2=ax4, color='r')
    plt.show()

def example_pre_yspec():
    stlo = np.arange(0, 180.00001, 0.1)
    stla = np.zeros(stlo.size)
    wd = './junk'
    model_filename='/mod'
    yspec_home='/yspec_home'
    yspec_config_dict={ 
                        'flag_attenuation': 1,
                        'flag_gravitation': 0,
                        'output_unit': 0,
                        'flag_tilt_correction': 0,
                        #####
                        'lmin': 0, 'lmax': 3000, 'fmin': 0.2, 'fmax': 160.,
                        #####
                        'length_min': 10., 'delta': 1.,
                        #####
                        'f11': 0.5, 'f12': 1., 'f21': 155., 'f22': 160.,
                        #####
                        'evdp': 50., 'evla': 0., 'evlo': 0.,
                        #####
                        'm_rr': 1e26, 'm_rt': 0., 'm_rp': 0.,
                        'm_tt': 1e26, 'm_tp': 0., 'm_pp': 1e26,
                        #####
                        'stdp': 0., 'stla': stla, 'stlo': stlo
                    }
    pbs_config_dict={
                        'job_name':     'yspec',
                        'email':        'email_address',
                        'project':      'em78',
                        'mem_gb':       150,
                        'walltime_hr':  1,
                        'ncpus':        48,
                        'storage':      'gdata/gb32+gdata/em78+scratch/em78+scratch/gb32',
    }
    mk_working_directory(wd=wd, model_filename=model_filename, yspec_home=yspec_home,
                         yspec_config_dict=yspec_config_dict,
                         pbs_config_dict=pbs_config_dict)

def example_post_yspec():
    wd = './junk'
    ofnm = './junk/junk.h5'
    yspec2h5(wd, ofnm)


if __name__ == '__main__':
    example_modify_model()
    #example_pre_yspec()
    