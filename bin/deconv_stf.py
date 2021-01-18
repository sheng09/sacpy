#!/usr/bin/env python3


from mpi4py import MPI
import sacpy
from sacpy.sac import c_rd_sac, c_rd_sachdr, plot_sac_lst
import h5py
import sys, getopt
from glob import glob
from datetime import date, datetime
import numpy as np
from numpy.fft import rfft, irfft
from obspy.signal.interpolation import lanczos_interpolation
import os.path, os
from scipy.signal import tukey

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(fnm_wildcard, out_root_dir, log_prefnm,
         plot_stf_flag=False, plot_freq_range=(0, 1),
         resample_t_start=-0.5, water_level_ratio= 0.005):
    """
    """
    ### MPI
    mpi_comm = MPI.COMM_WORLD.Dup()
    mpi_rank = mpi_comm.Get_rank()
    mpi_ncpu = mpi_comm.Get_size()
    mpi_log_file = '%s_%03d.txt' % (log_prefnm, mpi_rank)
    mpi_log_fid = open(mpi_log_file, 'w')
    ### prepare all stf keys
    stf_fnm = '%s/bin/dataset/STFs_SCARDEC/stfs_scardec.h5' % sacpy.__path__[0]
    stf_fid = h5py.File(stf_fnm, 'r')
    stf_keys_lst = list( stf_fid.keys() )
    stf_keys_set = set(stf_keys_lst)
    stf_ots_lst = [datetime.strptime(it, '%Y-%m-%d-%H-%M-%S.%f') for it in stf_keys_lst]
    #print(stf_keys_lst[0], stf_ots_lst[0] )
    ### loop over all SAC
    directories   = sorted( glob('/'.join(fnm_wildcard.split('/')[:-1] ) ) )
    sac_wildcards = fnm_wildcard.split('/')[-1]
    local_directories = directories[mpi_rank::mpi_ncpu]
    for it_direc in local_directories:
        ###### prepare the out direc
        out_dir = out_root_dir + '/' + '/'.join( it_direc.split('/')[1:] )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        ######
        print(it_direc, out_dir, file=mpi_log_fid, flush=True )
        ######
        sacfnms = sorted( glob(it_direc + '/' + sac_wildcards) )
        hdr0 = c_rd_sachdr(sacfnms[0] )
        resample_dt = hdr0.delta
        year, jday, h, m, s, ms = hdr0.nzyear, hdr0.nzjday, hdr0.nzhour, hdr0.nzmin, hdr0.nzsec, hdr0.nzmsec
        time_str = '%04d-%03d-%02d-%02d-%02d-%03d' % (year, jday, h, m, s, ms)
        ot = datetime.strptime(time_str, '%Y-%j-%H-%M-%S-%f')
        search_str = '%04d-%02d-%02d-%02d-%02d-%06.3f' % (year, ot.month, ot.day, h, m, s )
        ######
        if search_str not in stf_keys_set:
            tmp = [ot-it for it in stf_ots_lst]
            idx = np.argmin( [abs(it.total_seconds()) for it in tmp] )
            search_str = stf_keys_lst[idx]
        ###### obtain and resample the STF
        grp = stf_fid[search_str]
        ns = grp.attrs['ns']
        dt = grp.attrs['dt']
        t0 = grp.attrs['t0']
        ts = grp['ts']
        stf = grp['stf']
        n = int( np.ceil( (ns*dt+t0 - resample_t_start)/resample_dt) ) - 1
        wavelet = lanczos_interpolation(stf, t0, dt, resample_t_start, resample_dt, n, 20)
        new_ts = np.arange(n)*resample_dt + resample_t_start
        ###### obtain the deconvolution-inverse and the crosscorrelatio inverse of the wavelet
        spec = rfft(wavelet)
        spec *= (1.0/np.max(np.abs(spec) ) )
        nu = np.conj(spec)
        deno_raw = spec.real*spec.real + spec.imag*spec.imag
        deno = deno_raw + np.max(deno_raw)*water_level_ratio
        deconv_inv_spec = nu/deno
        deconv_inv_spec *= (1.0/np.max(np.abs(deconv_inv_spec) ) )
        window = tukey(wavelet.size, 0.05)
        deconv_inv = window*irfft(deconv_inv_spec, wavelet.size ) # deconvolution-inverse
        deconv_inv *= (1.0/np.sum(deconv_inv) )
        cc_inv = window*irfft(nu, wavelet.size) # cc-inverse
        cc_inv *= (1.0/np.sum(cc_inv) )
        ###### plot the STF and its inverse
        if plot_stf_flag:
            fig_name = '%s/%s.png' % (out_dir, search_str)
            fig, (ax_stf, ax_deconv, ax_cc) = plt.subplots(3, 3, figsize=(14, 9) )
            #######
            ### STF
            #######
            ax_stf[0].plot(ts, stf, 'k', label='SCARDEC')
            ax_stf[0].plot(new_ts, wavelet, 'r:', label='Resampled')
            ax_stf[0].set_xlim((ts[0], ts[-1] ) )
            ax_stf[0].set_xlabel('Time (s)')
            ax_stf[0].legend(loc='upper right')
            ###
            df = 1.0/resample_dt/wavelet.size
            fs = np.arange(spec.size)*df
            ax_stf[1].plot(fs, np.abs(spec), label='Amplitude', color='k')
            ax_stf[1].set_xlim(plot_freq_range )
            ax_stf[1].set_xlabel('Frequency (Hz)')
            ax_stf[1].legend(loc='upper right')
            ###
            ax_stf[2].plot(fs, np.rad2deg( np.angle(spec) ), label='Phase', color='k')
            ax_stf[2].set_xlim(plot_freq_range )
            ax_stf[2].set_xlabel('Frequency (Hz)')
            ax_stf[2].legend(loc='upper right')
            #######
            ### deconv
            #######
            ax_deconv[0].plot(new_ts, deconv_inv, 'k', label='Deconv')
            ax_deconv[0].set_xlim((ts[0], ts[-1] ) )
            ax_deconv[0].set_xlabel('Time (s)')
            ax_deconv[0].legend(loc='upper right')
            ###
            ax_deconv[1].plot(fs, np.abs(deconv_inv_spec), label='Amplitude', color='k')
            ax_deconv[1].set_xlim(plot_freq_range )
            ax_deconv[1].set_xlabel('Frequency (Hz)')
            ax_deconv[1].legend(loc='upper right')
            ###
            spike = np.convolve(wavelet, deconv_inv, 'full')
            spike *= (1.0/np.max(spike) )
            spike_ts = np.arange(spike.size) * resample_dt + (1-wavelet.size)*resample_dt
            ax_deconv[2].plot(spike_ts, spike, label='Spike test', color='k')
            ax_deconv[2].set_xlim((-20, 20) )
            ax_deconv[2].set_xlabel('Time (s)')
            ax_deconv[2].legend(loc='upper right')
            ax_deconv[2].grid(True)
            #ax_deconv[2].semilogx(fs, np.rad2deg( np.angle(deconv_inv_spec) ), label='Phase', color='k')
            #ax_deconv[2].set_xlim(right= fs[-1])
            #ax_deconv[2].set_xlabel('Frequency (Hz)')
            #ax_deconv[2].legend(loc='upper right')
            #######
            ### cc
            #######
            ax_cc[0].plot(new_ts, cc_inv, 'k', label='cc')
            ax_cc[0].set_xlim((ts[0], ts[-1] ) )
            ax_cc[0].set_xlabel('Time (s)')
            ax_cc[0].legend(loc='upper right')
            ###
            ax_cc[1].plot(fs, np.abs(nu), label='Amplitude', color='k')
            ax_cc[1].set_xlim(plot_freq_range )
            ax_cc[1].set_xlabel('Frequency (Hz)')
            ax_cc[1].legend(loc='upper right')
            ###
            spike = np.convolve(wavelet, cc_inv, 'full')
            spike *= (1.0/np.max(spike) )
            spike_ts = np.arange(spike.size) * resample_dt + (1-wavelet.size)*resample_dt
            ax_cc[2].plot(spike_ts, spike, label='Spike test', color='k')
            ax_cc[2].set_xlim((-20, 20) )
            ax_cc[2].set_xlabel('Time (s)')
            ax_cc[2].legend(loc='upper right')
            ax_cc[2].grid(True)
            #ax_cc[2].semilogx(fs, np.rad2deg( np.angle(nu) ), label='Phase', color='k')
            #ax_cc[2].set_xlim(right= fs[-1])
            #ax_cc[2].set_xlabel('Frequency (Hz)')
            #ax_cc[2].legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(fig_name, bbox_inches = 'tight', pad_inches = 0.1 )
            plt.close()
        ###### apply the deconv to seismogram
        for it_sacfnm in sacfnms:
            ###
            out_sacfnm = out_dir + '/' + it_sacfnm.split('/')[-1] + '.deconv'
            print(it_sacfnm, out_sacfnm)
            tr = c_rd_sac(it_sacfnm)
            tr.rmean(); tr.detrend(); tr.taper(0.005)
            tr1 = np.convolve(tr.dat, deconv_inv, 'same')
            tr.dat=tr1
            tr.write(out_sacfnm)
            ###
            out_sacfnm = out_dir + '/' + it_sacfnm.split('/')[-1] + '.cc'
            print(it_sacfnm, out_sacfnm)
            tr = c_rd_sac(it_sacfnm)
            tr.rmean(); tr.detrend(); tr.taper(0.005)
            tr1 = np.convolve(tr.dat, cc_inv, 'same')
            tr.dat=tr1
            tr.write(out_sacfnm)
    ##################
    mpi_log_fid.close()



if __name__ == "__main__":
    fnm_wildcard = ''
    out_root_dir = 'junk'
    log_prefnm = 'mpi_log'
    plot_stf_flag = False
    water_level_ratio = 0.005
    ######################
    HMSG = """Search for and remove source time function given many seismogram sac files.

    %s  -I "in*/*.sac" -O out_root_dir --log mpi_log --water_level_ratio 0.005 --plot_stf

    -I  : filename wildcard for event-tree style sac files.
          Usually, those files are downloaded using obspyDMT.
    -O  : output directory.
    --log  : log filename prefix.
    --water_level_ratio:
    --plot_stf:

E.g.,
    %s  -I "in*/*.sac" -O out_root_dir --log mpi_log --water_level_ratio 0.005 --plot_stf
    """
    if len(sys.argv) <= 1:
        print(HMSG % (sys.argv[0], sys.argv[0]), flush=True)
        sys.exit(0)
    ######################
    ######################
    options, remainder = getopt.getopt(sys.argv[1:], 'I:O:V', ['log=', 'plot_stf', 'water_level_ratio='])
    for opt, arg in options:
        if opt in ('-I'):
            fnm_wildcard = arg
        elif opt in ('-O'):
            out_root_dir = arg
        elif opt in ('--log'):
            log_prefnm = arg
        elif opt in ('--plot_stf'):
            plot_stf_flag = True
        elif opt in ('--water_level_ratio'):
            water_level_ratio = float(arg)
    ######
    main(fnm_wildcard, out_root_dir, log_prefnm,
         plot_stf_flag=plot_stf_flag, plot_freq_range=(0, 1),
         resample_t_start=-0.5, water_level_ratio= water_level_ratio)



