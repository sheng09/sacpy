#!/usr/bin/env python3
import sacpy.sac as sac
import sacpy.geomath as geomath
from glob import glob
import sys
import getopt
from copy import deepcopy
import pickle
import numpy as np
class cc_stack_preproc:
    def __init__(self):
        pass
    def init_from_fnm_template(self, fnm_template, pkl_fnm=None):
        """
        Init from filename template `fnm_template`, and (optional )save into pickle files `pkl_fnm`.

        Two important datasets will be generated to store information of events, receivers, and all sachdrs.
        Those two datasets will be used for following processings. Their structure are:
        # 1. 
        ev_dict(dict):  ev_id(int) --> {    'evlo': (float), 
                                            'evla': (float), 
                                            'evdp': (float), 
                                            'mag' : (float), 
                                             ...
                                            'hdr_dict': (dict) *,
                                        }
        * where hdr_dict(dict): key_rcv(str) --> (sachdr)

        # 2. 
        rcv_dict(dict): key_rcv(str) --> {  'stlo': (float),
                                            'stla': (float),
                                            'rcvid': (int) , 
                                         }
        """
        ev_dir_lst = sorted(glob( '/'.join(fnm_template.split('/')[:-1])  ) )
        remainer_fnm = fnm_template.split('/')[-1]
        ###
        ev_dict  = dict()
        rcv_dict = dict()
        conflict_rcv = set()
        print('Reading from ', fnm_template, '...')
        ### read from files and init
        for idx_ev, ev_dir in enumerate(ev_dir_lst):
            sacfnm_lst = sorted(glob('%s/%s' % (ev_dir, remainer_fnm) ) )
            ### read all sac hdrs for the event
            hdr_lst =  [ sac.rd_sachdr(it) for it in sacfnm_lst ]
            ###
            print('\t', ev_dir, len(sacfnm_lst), len(hdr_lst) )
            ### expand ev_dict
            hdr = sac.rd_sachdr(sacfnm_lst[0])
            ev_dict[idx_ev] = {'ev_dir': ev_dir }
            for key in ['evlo', 'evla', 'evdp', 'mag']:
                ev_dict[idx_ev][key] = hdr[key]
            ev_dict[idx_ev]['hdr_dict'] = dict()
            ### expand the rcv_dict
            for hdr, fnm in zip(hdr_lst, sacfnm_lst):
                key_rcv = fnm.split('/')[-1].replace('.sac', '').replace('.SAC', '').replace('BHZ', '').replace('BHN', '').replace('BHE', '').replace('BH1', '').replace('BH2', '')
                #hdr['key_rcv'] = key_rcv
                stlo, stla = hdr['stlo'], hdr['stla']
                #### 
                if key_rcv in conflict_rcv: # jump over conflict receivers
                    continue
                ####
                if key_rcv in rcv_dict:
                    if abs(rcv_dict[key_rcv]['stlo'] - stlo) > 0.001 or abs(rcv_dict[key_rcv]['stla'] - stla) > 0.001:
                        print('Err: receiver conflict %s. The receiver is removed from the .pkl volume' % (key_rcv), flush=False )
                        rcv_dict.pop(key_rcv, None)
                        conflict_rcv.add(key_rcv)
                else:
                    rcv_dict[key_rcv] = {'stlo': stlo, 'stla': stla}
                    ###
                    hdr['az']  = geomath.azimuth(hdr['evlo'], hdr['evla'], hdr['stlo'], hdr['stla'] )
                    hdr['baz'] = geomath.azimuth(hdr['stlo'], hdr['stla'], hdr['evlo'], hdr['evla'] )
                ###
                if key_rcv not in conflict_rcv:
                    ev_dict[idx_ev]['hdr_dict'][key_rcv] = hdr
            ###
        ### post-reading processing
        for rcv_idx, key_rcv in enumerate( sorted( rcv_dict.keys() ) ):
            rcv_dict[key_rcv]['rcvid'] = rcv_idx
        for key_ev, val in ev_dict.items():
            for key_rcv, hdr in val['hdr_dict'].items():
                hdr['sacid'] = key_ev*100000 + rcv_dict[key_rcv]['rcvid']
        ###
        self.ev_dict = ev_dict
        self.rcv_dict = rcv_dict
        ### save into pkl files
        if pkl_fnm != None:
            with open(pkl_fnm, 'wb')  as fid:
                pickle.dump({'ev_dict': ev_dict, 'rcv_dict': rcv_dict}, fid)
    def init_from_pkl(self,pkl_fnm):
        with open(pkl_fnm, 'rb')  as fid:
            tmp = pickle.load(fid)
            self.ev_dict  = tmp['ev_dict']
            self.rcv_dict = tmp['rcv_dict']
    def run_ev_stack(self, fnm_pair, fnm_in, dist_step=1.0, dist_range=None, daz_range=None, gcd_range=None, tag_info_level=0):
        """
        """
        if True:
            self.__form_rcv_pairs()
            self.__form_ev_rcv_pairs(fnm_pair, dist_step, dist_range, daz_range, gcd_range, tag_info_level)
            self.__output_cc_in(fnm_in)
    def __form_rcv_pairs(self):
        """
        Form all rcv pairs. This will compute the inter-receiver distance, center point of the
        great circle plane formed by two receivers. 
        """
        rcv_pair_dict = dict()
        ###
        rcv_lst = sorted(self.rcv_dict.keys() )
        print('Forming all possible receiver pairs...')
        for idx1, key_rcv1 in enumerate(rcv_lst):
            val1 = self.rcv_dict[key_rcv1]
            stlo1, stla1 = val1['stlo'], val1['stla']
            for key_rcv2 in rcv_lst[idx1:]:
                val2 = self.rcv_dict[key_rcv2]
                stlo2, stla2 = val2['stlo'], val2['stla']
                ####
                key_cc = (key_rcv1, key_rcv2)
                rcv_pair_dict[key_cc] = dict()
                ####
                rcv_pair_dict[key_cc]['inter_dist'] = geomath.haversine(stlo1, stla1, stlo2, stla2)
                if abs(stlo1-stlo2) < 0.001 and abs(stla1-stla2) < 0.001: # two rcvs are at same location
                    rcv_pair_dict[key_cc]['center'] = (None, None)
                else:
                    (lo1, la1), (lo2, la2) = geomath.great_circle_plane_center(stlo1, stla1, stlo2, stla2)
                    if la1 >= 0.0:
                        rcv_pair_dict[key_cc]['center'] = (lo1, la1)
                    else:
                        rcv_pair_dict[key_cc]['center'] = (lo2, la2)
        ###
        self.rcv_pair_dict = rcv_pair_dict
    def __form_ev_rcv_pairs(self, fnm, dist_step=1.0, dist_range=None, daz_range=None, gcd_range=None, tag_info_level=0):
        """
        Form all rcv pairs for all events.
        """
        ###
        print('Form all cross-correlation pairs for all events...')
        ###
        dis_min, dis_max = -9999, 9999
        if dist_range != None:
            dis_min, dis_max = dist_range
        daz_min, daz_max = -9999, 9999
        if daz_range != None:
            daz_min, daz_max = daz_range
        gcd_min, gcd_max = -9999, 9999
        if gcd_range != None:
            gcd_min, gcd_max = gcd_range
        fid = open(fnm, 'w')
        print('#(L0) id1 id2 ccid grpid #(L1) inter-dist daz gcd #(L2) clo cla #(L3) evlo evla stlo1 stla1 stlo2 stla2 ', file=fid )
        ###
        ev_rcv_pair_lst = list()
        for ev_id, val in self.ev_dict.items():
            ###
            print('\t %d/%d %s  %d' %  (ev_id, len(self.ev_dict), val['ev_dir'], len(val['hdr_dict'])*(1+len(val['hdr_dict']))//2 ) )
            ###
            evlo, evla = val['evlo'], val['evla']
            rcv_lst = sorted( val['hdr_dict'].keys() )
            for idx1, rcv1 in enumerate(rcv_lst):
                hdr1 = val['hdr_dict'][rcv1]
                sacid1 = hdr1['sacid']
                stlo1, stla1 = hdr1['stlo'], hdr1['stla']
                for rcv2 in rcv_lst[idx1:]:
                    hdr2 = val['hdr_dict'][rcv2]
                    sacid2 = hdr2['sacid']
                    stlo2, stla2 = hdr2['stlo'], hdr2['stla']
                    ### stack id
                    inter_dist = self.rcv_pair_dict[ (rcv1, rcv2) ]['inter_dist']
                    if inter_dist < dis_min or inter_dist > dis_max:
                        continue
                    ###
                    cc_stack_id = int(np.round(inter_dist/dist_step) ) * int(dist_step*10)
                    ###
                    clo, cla   = self.rcv_pair_dict[ (rcv1, rcv2) ]['center']
                    if clo == None or cla == None:
                        (x1, y1), (x2, y2) = geomath.great_circle_plane_center(stlo1, stla1, evlo, evla )
                        if x1 >= 0.0:
                            clo, cla = x1, y1
                        else:
                            clo, cla = x2, y2
                    ###
                    daz = (hdr1['az'] - hdr2['az']) % 360.0
                    if daz > 180.0:
                        daz = 360.0-daz
                    if daz > 90.0:
                        daz = 180.0-daz
                    ###
                    if daz < daz_min or daz > daz_max:
                        continue
                    ###
                    gcd = geomath.point_distance_to_great_circle_plane(evlo, evla, stlo1, stla1, stlo2, stla2)
                    if gcd < gcd_min or gcd > gcd_max:
                        continue
                    ###
                    ###ev_rcv_pair_lst.append(   (ev_id, hdr1, hdr2, daz, gcd, (rcv1, rcv2) )    )
                    print( '%08d %08d %08d %3d %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f' % 
                            (sacid1, sacid2, cc_stack_id, ev_id, 
                             inter_dist, daz, gcd,
                             clo, cla,
                             evlo, evla, stlo1, stla1, stlo2, stla2 
                            ), 
                            file= fid)
        ###
    def __output_cc_pair(self, fnm, dist_step=1.0, dist_range=None, daz_range=None, gcd_range=None, tag_info_level=0):
        dis_min, dis_max = -9999, 9999
        if dist_range != None:
            dis_min, dis_max = dist_range
        daz_min, daz_max = -9999, 9999
        if daz_range != None:
            daz_min, daz_max = daz_range
        gcd_min, gcd_max = -9999, 9999
        if gcd_range != None:
            gcd_min, gcd_max = gcd_range
        ###
        with open(fnm, 'w') as fid:
            print('#(L0) id1 id2 ccid grpid #(L1) inter-dist daz gcd #(L2) clo cla #(L3) evlo evla stlo1 stla1 stlo2 stla2 ', file=fid )
            for ev_id, hdr1, hdr2, daz, gcd, key_rcv_pair in self.ev_rcv_pair_lst:
                inter_dist = self.rcv_pair_dict[ key_rcv_pair ]['inter_dist']
                clo, cla   = self.rcv_pair_dict[ key_rcv_pair ]['center']
                if clo == None or cla == None:
                    (x1, y1), (x2, y2) = geomath.great_circle_plane_center(hdr1['stlo'], hdr1['stla'], hdr1['evlo'], hdr1['evla'] )
                    if x1 >= 0.0:
                        clo, cla = x1, y1
                    else:
                        clo, cla = x2, y2
                ###
                if inter_dist < dis_min or inter_dist > dis_max:
                    continue
                if daz < daz_min or daz > daz_max:
                    continue
                if gcd < gcd_min or gcd > gcd_max:
                    continue
                ###
                key_rcv1, key_rcv2 = key_rcv_pair
                ###
                cc_stack_id = int(np.round(inter_dist/dist_step) ) * int(dist_step*10)
                if tag_info_level == 0:
                    print( '%08d %08d %08d %3d' % 
                            (hdr1['sacid'], hdr2['sacid'], cc_stack_id, ev_id), 
                            file=fid )
                elif tag_info_level == 1:
                    print( '%08d %08d %08d %3d %6.2f %6.2f %6.2f' % 
                            (hdr1['sacid'], hdr2['sacid'], cc_stack_id, ev_id, 
                             inter_dist, daz, gcd
                            ), 
                            file= fid)
                elif tag_info_level == 2:
                    print( '%08d %08d %08d %3d %6.2f %6.2f %6.2f %6.2f %6.2f' % 
                            (hdr1['sacid'], hdr2['sacid'], cc_stack_id, ev_id, 
                             inter_dist, daz, gcd,
                             clo, cla
                            ), 
                            file= fid)
                else:
                    print( '%08d %08d %08d %3d %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f' % 
                            (hdr1['sacid'], hdr2['sacid'], cc_stack_id, ev_id, 
                             inter_dist, daz, gcd,
                             clo, cla,
                             self.ev_dict[ev_id]['evlo'], self.ev_dict[ev_id]['evla'], hdr1['stlo'], hdr1['stla'], hdr2['stlo'], hdr2['stla'] 
                            ), 
                            file= fid)
                ###  
        ###
    def __output_cc_in(self, fnm):
        """
        Output infnm.
        """
        with open(fnm, 'w') as fid:
            #i = 0
            for key_ev in sorted( self.ev_dict.keys() ):
                #j = 0
                for key_rcv in sorted(self.ev_dict[key_ev]['hdr_dict'].keys() ):
                    hdr = self.ev_dict[key_ev]['hdr_dict'][key_rcv]
                    print('%s %08d' % (hdr['filename'], hdr['sacid']), file=fid)
                    #i = i + 1
                    #j = j + 1
                #print(i, j)
        ###




def main(argv):
    HMSG = """
    Form seismogram pairs with respect to events or receivers.

    %s -P sac/filename/template -M mode -D dist_step [-H]

    -P: filename template for all sac files. 
        All sac files should be grouped in different folders.
        Each folder corresponds to a single event.
        
        E.g.: -P archive/*/processed/II.*.BHZ.sac
        The first wildcard '*' describes different event folders.
        The second describes seismogarms of the same event at
        different receivers.
    
    -M: 'ev' for receiver-to-receiver pairs. 'rcv' for event-to-
        event pairs.

        In other words, '-M ev' produces a pair of seismograms 
        observed at different receivers for a same event.
        '-M rcv' produces a pair of seismograms observed at same
        receiver for two different events.
        The former can be used to form receiver-to-receiver cross-
        correlation stacks, and the latter can be used to form
        event-to-event cross-correlation stacks.
    
        In '-M ev' mode, each line of the output table will be
        'sacid1 sacid2 inter-receiver-dist daz gcd stlo1 stla1 stlo2 stla2'
    
    -D: distance step in degree
    """ % (argv[0])
    ####
    if len(sys.argv) < 2:
        print(HMSG)
        sys.exit(0)
    ####
    fnm_template = None
    mode = 'ev' # 'sta'
    dist_step = 1.0
    ofnm_pair = 'junk_pair.txt'
    ofnm_in   = 'junk_in.txt'
    options, remainder = getopt.getopt(sys.argv[1:], 'P:M:D:O:Hh', ['help'] )
    for opt, arg in options:
        if opt in ('-P'):
            fnm_template = arg
        elif opt in ('-M'):
            mode = arg
        elif opt in ('-D'):
            dist_step = float(arg)
        elif opt in ('-O'):
            ofnm_pair = arg + '_pair.txt'
            ofnm_in   = arg + '_in.txt'
        elif opt in ('-h', '-H', '--help'):
            print(HMSG)
            sys.exit(0)
        else:
            print('invalid options: %s' % (opt) )
            print(HMSG)
            sys.exit(-1)
    ####
    app = cc_stack_preproc()
    if mode == 'ev':
        app.init_from_fnm_template(fnm_template)
        app.run_ev_stack(ofnm_pair, ofnm_in, dist_step, tag_info_level = 3)

if __name__ == "__main__":
    main(sys.argv)