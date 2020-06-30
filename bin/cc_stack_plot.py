#!/usr/bin/env python3
"""
Plot the hdf5 file outputed by `cc_stack.py`.
"""
import sys
import getopt
import numpy as np
import h5py
import sacpy.sac as sac
import sacpy.processing as processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal

class cc_stack_plot:
    def __init__(self, fnm, t_range=[-3000, 3000], sym=False, half_taper_length_sec= 10.0, single_norm= True, freq_range=(0.02, 0.0666) ):
        self.mat, self.dt, self.dist, self.stacked_count, self.t_range = self.__obtain_cc_time__(fnm, t_range, sym, half_taper_length_sec, single_norm, freq_range)
    def __obtain_cc_time__(self, fnm, t_range=[-3000, 3000], sym=False, half_taper_length_sec= 0.0, single_norm= True, freq_range=(0.02, 0.0666) ):
        fid = h5py.File(fnm, 'r')
        dt  = fid['cc_stacked_time'].attrs['dt']
        mat = fid['cc_stacked_time'][:]
        dist = fid['distance_bin'][:]
        stacked_count = fid['cc_stacked_count']

        nrow, ncol = mat.shape

        # taper
        wnd = signal.tukey(ncol, half_taper_length_sec/(ncol*dt) )
        for idx in range(nrow):
            mat[idx,:] = signal.detrend(mat[idx,:])
            mat[idx,:] *= wnd
            mat[idx,:] = signal.detrend(mat[idx,:])
        
        # roll
        roll_size = int( (ncol-1)/2 )
        print('Raw mat shape (%d, %d). Roll size %d' % (nrow, ncol, roll_size) )
        for idx in range(nrow):
            mat[idx,:] = np.roll( mat[idx,:],  roll_size )
        #t0 = dt*roll_size
        
        # cut both ends
        cut_length_sec = 10
        cut_size = int(cut_length_sec/dt)
        mat = mat[:,cut_size:-cut_size] 
        nrow, ncol = mat.shape

        ## bandpass filter
        sampling_rate = 1.0/dt
        wnd = signal.tukey(ncol, 500.0/(ncol*dt) )
        for idx in range(nrow):
            mat[idx,:] = signal.detrend(mat[idx,:])
            mat[idx,:] *= wnd
            mat[idx,:] = signal.detrend(mat[idx,:])
            mat[idx,:] *= wnd
            mat[idx,:] = processing.filter(mat[idx,:], sampling_rate, 'bandpass', freq_range, 2, 2)
        
        ## sym
        if (sym):
            mat += mat[:,::-1]

        ##
        old_t0 = -((ncol-1) // 2 )*dt
        ## cut
        self.t_range = t_range
        idx0 = int( round((t_range[0]-old_t0)/dt) )
        idx1 = int( round((t_range[1]-old_t0)/dt) ) + 1
        new_t0 = idx0*dt + old_t0
        new_t1 = idx1*dt + old_t0
        mat = mat[:,idx0:idx1]

        ## norm
        if single_norm:
            for idx in range(nrow):
                max_amp = np.max( mat[idx,:] )
                #print(max_amp)
                if max_amp != 0.0:
                    mat[idx,:] *= (1.0/max_amp)
        ###
        return mat, dt, dist, stacked_count, (new_t0, new_t1)

    def plot(self, fnm_img):
        fig, (ax0, ax) = plt.subplots(2, 1, figsize=(6, 13), gridspec_kw={'height_ratios': [2, 11] } )
        
        max_cc = 0.8
        t0, t1 = self.t_range[0], self.t_range[1]
        dist_step = self.dist[1] - self.dist[0]
        d0, d1 = self.dist[0]-dist_step*0.5, self.dist[-1]+dist_step*0.5
        ext = [d0, d1, t0, t1]
        #print(ext)
        ax.imshow(np.transpose(self.mat[:,::10]), vmin=-max_cc, vmax=max_cc, cmap='gray_r', interpolation='bessel',  extent=ext, origin='lower', aspect='auto' )
        ax.set_ylim(self.t_range)
        ax.set_ylabel('Time (second)')
        ax.set_xlim([d0, d1])
        ax.set_xlabel('Inter-receiver distance (Degree)')
        ##
        ax0.plot(self.dist, self.stacked_count, 'k', linewidth=2 )
        ax0.set_xlim()
        ax0.set_xlim([d0, d1])
        ax0.set_xlabel('Inter-receiver distance (Degree)')
        ax0.set_ylabel('Number of pairs')
        ##
        plt.savefig(fnm_img, bbox_inches = 'tight', pad_inches = 0.2)    

    def toSac(self, fnm_prefix):
        nrow, ncol = self.mat.shape
        for irow in range(nrow):
            fnm = '%s_%03d_%.2f.sac' % (fnm_prefix, irow, self.dist[irow] )
            sac.wrt_sac_2(fnm, self.mat[irow, :], self.dt, self.t_range[0] )

if __name__ == "__main__":
    HMSG = '%s -I fnm.h5 -P img.png -T t1/t2 -S prefix_fnm' % (sys.argv[0] )
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    ###
    h5_fnm = ''
    img_fnm = ''
    sac_prefnm = ''
    t1, t2 = 0, 3600
    sym = False
    ###
    options, remainder = getopt.getopt(sys.argv[1:], 'I:P:T:S:M' )
    for opt, arg in options:
        if opt in ('-I'):
            h5_fnm = arg
        elif opt in ('-P'):
            img_fnm = arg
        elif opt in ('-T'):
            cut_t1, cut_t2 = arg.split('/')
            t1 = float(cut_t1)
            t2 = float(cut_t2)
        elif opt in ('-S'):
            sac_prefnm = arg
        elif opt in ('-M'):
            sym = True
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(0)
    ###
    app = cc_stack_plot(h5_fnm, (t1, t2), sym=sym )
    if (img_fnm != ''):
        app.plot(img_fnm)
    if (sac_prefnm != ''):
        app.toSac(sac_prefnm)
    
    

