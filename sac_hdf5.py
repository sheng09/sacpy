#!/usr/bin/env python3
"""
Transfer sac files into formatted HDF5 file.
"""
import time
import h5py
import sacpy.sac as sac
import sacpy.geomath as geomath
import numpy as np
from time import gmtime, strftime
import pyfftw
import sys
import getpass
from glob import glob

def make_h5_from_sac(sac_fnm_template, h5_fnm, cut_marker=None, cut_range=None, h5_grp_name='raw_sac', user_message=''):
    """
    Generate an HDF5 file from many sac files.
    
    Return an object of `alignedSac2Hdf5`.
    """
    app = alignedSac2Hdf5(getpass.getuser(), user_message)
    app.fromSac(sorted( glob(sac_fnm_template) ), h5_fnm, cut_marker= cut_marker, cut_range=cut_range, h5_grp_name=h5_grp_name )
    return app

class alignedSac2Hdf5:
    """
    Make many aligned sac files into a single hdf5 file.
    The aligned sac files should have same:
        1) npts
        2) delta
    Users should be responsible for time used in different sac files.
    """
    def __init__(self, username, user_message= ''):
        self.username = username
        self.user_message = user_message
        self.__idx_spectra = 0
    ### Functions to generate h5 file from different sources.
    def fromSac(self, sac_fnm_lst, h5_fnm, cut_marker=None, cut_range=None, h5_grp_name = 'raw_sac'):
        """
        sac_fnm_lst: a list of sac filenames.
        h5_fnm: the filename for generated hdf5 file.
        """
        self.sac_fnm_lst = sac_fnm_lst
        self.nsac = len(sac_fnm_lst)
        self.h5 = h5py.File(h5_fnm, 'w' )
        self.__make_hdf5__rawsac__(cut_marker, cut_range, h5_grp_name = h5_grp_name)
    def fromH5(self, h5_fnm, username, user_message= '', open_mode= 'r'):
        if open_mode == 'w':
            print('Err: illegal open mode `w` that would destroy data', file=sys.stderr, flush=True)
            sys.exit(0)
        self.h5 = h5py.File(h5_fnm, open_mode)
    def __make_hdf5__rawsac__(self, cut_marker=None, cut_range=None, h5_grp_name = 'raw_sac'):
        raw = self.h5.create_group(h5_grp_name)
        raw.attrs['user'] = self.username
        raw.attrs['timestamp'] = strftime("%Y-%m-%d %H:%M:%S", gmtime() )
        raw.attrs['message'] = self.user_message
        ###
        #  filenames
        ###
        raw.create_dataset('filename', data=np.array(self.sac_fnm_lst, dtype=h5py.string_dtype() ) ) # variable string
        ###
        #  hdrs
        ###
        hdrs = [sac.rd_sachdr(fnm) for fnm in self.sac_fnm_lst]
        g = raw.create_group('hdr')
        for key in sac.sachdr.f_keys:
            #rint(key)
            g.create_dataset(key, data=np.array([h[key] for h in hdrs], dtype=np.float32) )
        for key in sac.sachdr.i_keys:
            #print(key)
            g.create_dataset(key, data=np.array([h[key] for h in hdrs], dtype=np.int32) )
        for key in sac.sachdr.s_keys:
            #print(key)
            g.create_dataset(key, data=np.array([h[key] for h in hdrs], dtype='S8') ) # fixed 8 length string
        ###
        #  if cut is True, try to read one sac to get correct npts, b, and e
        ###
        npts = hdrs[0]['npts']
        if cut_marker !=None and cut_range!=None:
            t1, t2 = cut_range
            tr = sac.rd_sac_2(self.sac_fnm_lst[0], cut_marker, t1, t2)
            npts = tr['npts']
            #b = tr['b']
            #e = tr['e']
            ###
            #g['npts'][:] = npts
            #g['b'   ][:] = b
            #g['e'   ][:] = e
        ###
        #  data
        ###
        dset = raw.create_dataset('data', (self.nsac, npts), dtype=np.float32 )
        dset.attrs['info'] = 'Aligned sac traces with same npts and delta'
        dset.attrs['dt'] = hdrs[0]['delta']
        if cut_range!=None and cut_marker!=None:
            t1, t2 = cut_range
            for idx, fnm in enumerate(self.sac_fnm_lst):
                tmp = sac.rd_sac_2(fnm, cut_marker, t1, t2)
                dset[idx,:tmp['npts']] = tmp['dat']
                g['npts'][idx] = tmp['npts']
                g['b'][idx] = tmp['b']
                g['e'][idx] = tmp['e']
                #check nan
                if True in np.isnan(dset[idx,:]):
                    dset[idx,:] = 0.0
        else:
            for idx, fnm in enumerate(self.sac_fnm_lst):
                tmp = sac.rd_sac(fnm)
                dset[idx,:tmp['npts']] = tmp['dat']
                #check nan
                if True in np.isnan(dset[idx,:]):
                    dset[idx,:] = 0.0
        ###
    ### Functions to process h5 file and return values.
    def get_raw_sac(self, cut_marker=None, cut_range=None, h5_grp_name = 'raw_sac'):
        """
        Useless functions

        Get a np.ndarray matrix that store the raw sac time-series given cutting parameters.
        """
        grp = self.h5[h5_grp_name]
        if cut_marker != None and cut_range != None:
            t1, t2 = cut_range
            dt = grp['data'].attrs['dt']
            t_ref = grp['hdr/%s' % (cut_marker) ][0]
            idx1 = int( np.round( (t_ref+t1-grp['hdr/b'][0])/dt ) )
            idx2 = int( np.round( (t_ref+t2-grp['hdr/b'][0])/dt ) ) + 1
            return grp['data'][:,idx1:idx2]
        else:
            return grp['data']
    def make_spec(self, nfft_mode='keep', h5_grp_name = 'raw_sac'):
        """
        Useless functions.

        Make a new dataset to store the spectra.
        nfft_mode: 
            1) 'keep': use npts as nfft;
            2) 'cc:    pad zero to npts*2-1, that is used for crosscorrelation;
            3) an integer: abtritray number that is greater than npts
        """
        ###
        self.__idx_spectra = self.__idx_spectra + 1
        ###
        grp = self.h5[h5_grp_name]
        nsac, npts = grp['data'].shape
        nfft = npts
        if nfft_mode == 'keep':
            nfft = npts
        elif nfft_mode == 'cc':
            nfft = 2*npts-1
        else:
            nfft = int(nfft_mode)
        fft_dname = 'spectra_%s' % (nfft_mode)
        dset = self.h5.create_dataset(fft_dname , (nsac, nfft), dtype=np.complex64 )
        ###
        dset.attrs['user'] = self.username
        dset.attrs['timestamp'] = strftime("%Y-%m-%d %H:%M:%S", gmtime() )
        ###
        dset.attrs['nfft_mode'] = nfft_mode
        dset.attrs['nfft'] = nfft
        #print(self.h5['raw_sac/data'].attrs['dt'] )
        dset.attrs['df'] = 1.0/ (grp['data'].attrs['dt'] * nfft)
        ###
        for idx, row in enumerate(grp['data']):
            dset[idx,:] = pyfftw.interfaces.numpy_fft.fft(row, nfft)
        return fft_dname 
    #def make_cc_spec(self, fft_dataset_name):
    #    """
    #    fft_dataset_name: dataset name in h5
    #    """
    #    nsac, nfft = self.h5[fft_dataset_name].shape
    #    ncc = nsac*nsac
    #    cc_dname = '%s_cc' % (fft_dataset_name)
    #    dset = self.h5.create_dataset(cc_dname, (ncc, nfft), dtype=np.complex64 )
    #    cc_idx = 0
    #    spectra = self.h5[fft_dataset_name]
    #    for idx1 in range(nsac):
    #        for idx2 in range(nsac):
    #            dset[cc_idx, :] = spectra[idx1, :] * spectra[idx2, :]
    #            cc_idx = cc_idx + 1
    #    pass

class Sac2ResampleHdf5:
    """
    Useless codes
    Make many sac files into a single hdf5 file.
    The sac time-series will be resampled to a same length.
    Also, the program force same length, so that the overal npts will be the smallest npts after resampling.
    """
    def __init__(self, h5_fnm, sac_fnm_lst, username):
        self.username = username
        self.h5_fnm = h5_fnm
        self.sac_fnm_lst = sac_fnm_lst
        #
        self.h5 = h5py.File(h5_fnm, 'w')
    def build_raw_sac_dataset(self, delta):
        raw = self.h5.create_group('raw_sac')
        raw.attrs['user'] = self.username
        raw.attrs['timestamp'] = strftime("%Y-%m-%d %H:%M:%S", gmtime() )
        raw.attrs['message'] = 'Raw sac data'
        ###
        #  filenames
        ###
        raw.create_dataset('filename', data=np.array(self.sac_fnm_lst, dtype=h5py.string_dtype() ) ) # variable string
        ###
        #  hdrs
        ###
        hdrs = [sac.rd_sachdr(fnm) for fnm in self.sac_fnm_lst]
        g = raw.create_group('hdr')
        for key in sac.sachdr.f_keys:
            #rint(key)
            g.create_dataset(key, data=np.array([h[key] for h in hdrs], dtype=np.float32) )
        for key in sac.sachdr.i_keys:
            #print(key)
            g.create_dataset(key, data=np.array([h[key] for h in hdrs], dtype=np.int32) )
        for key in sac.sachdr.s_keys:
            #print(key)
            g.create_dataset(key, data=np.array([h[key] for h in hdrs], dtype='S8') ) # fixed 8 length string
        ###
        #  time-series 
        ###
        time_range= np.min( [it['npts']*it['delta'] for it in hdrs] )
        new_npts = int( np.round(time_range / delta) )
        nsac = len(hdrs)
        dset = raw.create_dataset('data', (nsac, new_npts), dtype=np.float32 )
        dset.attrs['info'] = 'Time-series with same sampling rate and npts with zero padding.'
        for idx, fnm in enumerate(self.sac_fnm_lst):
            tr = sac.rd_sac(fnm)
            tr.resample(delta)
            dset[idx,:] = tr['dat'][:new_npts]
        #ncol = np.max([it['npts'] for it in hdrs] )
        #dset = raw.create_dataset('data', (nsac, ncol), dtype=np.float32 )
        #dset.attrs['info'] = 'Raw time-seires with different sampling rate and npts with zero padding.'
        #for idx, fnm in enumerate(self.sac_fnm_lst):
        #    dset[idx,:g['npts'][idx]] = sac.rd_sac(fnm)['dat']

class cc_Hdf5:
    """
    Useless codes
    """
    def __init__(self, fnm_alignedSac2Hdf5_lst, fnm_cc_stack, username):
        """
        fnm_alignedSac2Hdf5_lst: list of filenames that correspond to `alignedSac2Hdf5`
        fnm_cc_stack: filename for storing stacked cc.
        """
        self.fnm_alignedSac2Hdf5_lst = fnm_alignedSac2Hdf5_lst
        self.h5 = h5py.File(fnm_cc_stack, 'w')
        self.username = username
        ###
        self.idx_cc_stack_azimuth = 0
    def cc__stack_inter_rcv_distance_all(self, distance_range= [0.0, 180.0], distance_step= 1.0):
        """
        cc and stack with respect to inter-receiver distance
        All cross-correlation functions are used in the stack
        """
        dmin, dmax = distance_range
        dstep = distance_step
        n_cc_stack = int( (dmax-dmin)/dstep )
        ###
        h5_lst = [h5py.File(fnm, 'r') for fnm in self.fnm_alignedSac2Hdf5_lst]
        ###
        junk, nfft = h5_lst[0]['spectra_cc'].shape
        cc_spectra = np.zeros( (n_cc_stack, nfft), dtype=np.complex64 ) ## cc_spectra in the memory
        cc_stack_numer = np.zeros(n_cc_stack, dtype=np.int32)
        ###
        for it in h5_lst:
            nsac = it['spectra_cc'].shape[0]
            stlo_lst = it['raw_sac/hdr/stlo'][:]
            stla_lst = it['raw_sac/hdr/stla'][:]
            for isac1 in range(nsac):
                tmp = np.conj(it['spectra_cc'][isac1, :] )
                stlo1, stla1 = stlo_lst[isac1], stla_lst[isac1]
                for isac2 in range(isac1, nsac):
                    stlo2, stla2 = stlo_lst[isac2], stla_lst[isac2]
                    distance = geomath.haversine(stlo1, stla1, stlo2, stla2)
                    idx_cc = int(np.floor((distance - dmin)/dstep) )
                    cc_spectra[idx_cc, :] += tmp * it['spectra_cc'][isac2, :]
                    cc_stack_numer[idx_cc] += 1
        ###
        g = self.h5.create_group('cc_all')
        g.attrs['info'] = 'cc spectra stacked with respect to inter receiver distance. No selections are applied'
        g.attrs['user'] = self.username
        g.attrs['timestamp'] = strftime("%Y-%m-%d %H:%M:%S", gmtime() )
        dset = g.create_dataset('spectra_cc_stack', data= cc_spectra)
        dset = g.create_dataset('spectra_cc_stack_number', data= cc_stack_numer)
    def cc__stack_inter_rcv_distance_azimuth_diff(self, azimuth_diff_max= 10, reverse=False, distance_range= [0.0, 180.0], distance_step= 1.0):
        """
        cc and stack with respect to inter-receiver distance
        azimuth difference are used to exclude some crosscorrelation functions
        #
        azimuth_diff_max: max azimuth difference
        reverse: False(default) to remove off-great-circle-plane cc
                 True           to remove  in-great-circle-plane cc
        """
        dmin, dmax = distance_range
        dstep = distance_step
        n_cc_stack = int( (dmax-dmin)/dstep )
        ###
        h5_lst = [h5py.File(fnm, 'r') for fnm in self.fnm_alignedSac2Hdf5_lst]
        ###
        junk, nfft = h5_lst[0]['spectra_cc'].shape
        cc_spectra = np.zeros( (n_cc_stack, nfft), dtype=np.complex64 ) ## cc_spectra in the memory
        cc_stack_numer = np.zeros(n_cc_stack, dtype=np.int32)
        ###
        for it in h5_lst:
            nsac = it['spectra_cc'].shape[0]
            stlo_lst = it['raw_sac/hdr/stlo'][:]
            stla_lst = it['raw_sac/hdr/stla'][:]
            evlo = it['raw_sac/hdr/evlo'][0]
            evla = it['raw_sac/hdr/evla'][1]
            for isac1 in range(nsac):
                tmp = np.conj(it['spectra_cc'][isac1, :] )
                stlo1, stla1 = stlo_lst[isac1], stla_lst[isac1]
                az1 = geomath.azimuth(evlo, evla, stlo1, stla1)
                for isac2 in range(isac1, nsac):
                    stlo2, stla2 = stlo_lst[isac2], stla_lst[isac2]
                    az2 = geomath.azimuth(evlo, evla, stlo2, stla2)
                    daz = (az2-az1) % 180
                    if (daz > azimuth_diff_max) != reverse:
                        continue
                    distance = geomath.haversine(stlo1, stla1, stlo2, stla2)
                    idx_cc = int(np.floor((distance - dmin)/dstep) )
                    cc_spectra[idx_cc, :] += tmp * it['spectra_cc'][isac2, :]
                    cc_stack_numer[idx_cc] += 1
        ###
        g = self.h5.create_group('cc_azimuth_%02d' % (self.idx_cc_stack_azimuth) )
        self.idx_cc_stack_azimuth += 1
        ###
        if reverse:
            g.attrs['info'] = 'cc spectra stacked with respect to inter receiver distance. Selected azimuth difference (>%f)' % (azimuth_diff_max)
        else:
            g.attrs['info'] = 'cc spectra stacked with respect to inter receiver distance. Selected azimuth difference (<=%f)' % (azimuth_diff_max)
        g.attrs['user'] = self.username
        g.attrs['timestamp'] = strftime("%Y-%m-%d %H:%M:%S", gmtime() )
        dset = g.create_dataset('spectra_cc_stack', data= cc_spectra)
        dset = g.create_dataset('spectra_cc_stack_number', data= cc_stack_numer)
        ###

if __name__ == "__main__":
    ###
    #
    ###
    sac_template = '/home/catfly/00-LARGE-IMPORTANT-PERMANENT-DATA/AU_dowload/01_resampled_bhz_to_h5/03_workspace_bhz_dt_0.1/2000_008_16_47_20_+0000/*resampled'
    make_h5_from_sac(sac_template, 'junk.h5', h5_grp_name='raw_sac', user_message='test raw_sac')
    make_h5_from_sac(sac_template, 'junk2.h5', h5_grp_name='sheng', user_message='test sheng')
    #import glob
    #fnm_lst = sorted( glob.glob('/home/catfly/workspace/correlation_physics/10_real_data/04_analysis/desample_data/2015_208_21_41_21_+0000/*BHZ*SAC') )
    #app = alignedSac2Hdf5(username= 'Sheng', user_message='')
    #app.fromSac(fnm_lst, 'test.h5')
    #fft_dname = app.make_spec(nfft_mode='cc')
    ###
    #
    ###
    #app = cc_Hdf5(['test.h5'], 'cc.h5', 'Sheng')
    #app.cc__stack_inter_rcv_distance_all()
    #app.cc__stack_inter_rcv_distance_azimuth_diff(azimuth_diff_max=10, reverse=False)
    #app.cc__stack_inter_rcv_distance_azimuth_diff(azimuth_diff_max=10, reverse=True)






