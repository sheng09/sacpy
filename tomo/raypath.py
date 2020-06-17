#!/usr/bin/env python3
import h5py
import obspy.taup as taup
import numpy as np
import sys

model_ak135 = taup.TauPyModel('ak135')


def get_raypaths(vol_list):
    """ 
    `vol_list` is a numpy array of "useless_id evdp evla evlo stla stlo phase_name useless_tag"
    """
    paths = []
    for junk_id, evdp, evla, evlo, stla, stlo, phase, junk_tag in vol_list:
        #print(evdp, evla, evlo, stla, stlo, [phase] )
        arrs = model_ak135.get_ray_paths_geo(evdp, evla, evlo, stla, stlo, [phase])
        idx = np.argmin([it.ray_param for it in arrs])
        arr = arrs[idx]
        lon = arr.path['lon']
        lat = arr.path['lat']
        dep = arr.path['depth']
        time = arr.time
        rp = arr.ray_param
        paths.append( {'lon':lon, 'lat': lat, 'depth': dep, 'time': time, 'ray_param': rp, 'time': time} )
    return paths
def get_raypath_single(vol_it, verbose=None):
    junk_id, evdp, evla, evlo, stla, stlo, phase, junk_tag = vol_it
    if verbose != None:
        print("%d %s ..." % (junk_id, junk_tag) )
    arrs = model_ak135.get_ray_paths_geo(evdp, evla, evlo, stla, stlo, [phase])
    idx = np.argmin([it.ray_param for it in arrs])
    arr = arrs[idx]
    lon = arr.path['lon']
    lat = arr.path['lat']
    dep = arr.path['depth']
    time = arr.time
    rp = arr.ray_param
    return {'lon':lon, 'lat': lat, 'depth': dep, 'time': time, 'ray_param': rp, 'time': time, 'id': junk_id, 'tag': junk_tag, 'phase': phase}

def run_h5(filename, h5_fnm):
    """
    `filename` is a table file that each line is 
    "id evdp evla evlo stla stlo phase_name tag"
    """
    vol_list = np.loadtxt(filename, dtype= {'names':   ('id',    'evdp',    'evla',    'evlo',    'stla',    'stlo',   'phase', 'tag' ),
                                 'formats': ('int32',  'float64', 'float64', 'float64', 'float64', 'float64', object, object) } )
    paths = get_raypaths(vol_list)
    fid = h5py.File(h5_fnm, 'w')
    # `settings` that store the input table file data
    time  = np.array([it['time'] for it in paths], dtype=np.float64 )
    rp    = np.array([it['ray_param'] for it in paths], dtype=np.float64 )

    grp_settings = fid.create_group('settings')
    grp_settings.create_dataset('id',    data=vol_list['id'],   dtype=np.int32)
    grp_settings.create_dataset('evdp',  data=vol_list['evdp'], dtype=np.float64)
    grp_settings.create_dataset('evla',  data=vol_list['evla'], dtype=np.float64)
    grp_settings.create_dataset('evlo',  data=vol_list['evlo'], dtype=np.float64)
    grp_settings.create_dataset('stla',  data=vol_list['stla'], dtype=np.float64)
    grp_settings.create_dataset('stlo',  data=vol_list['stlo'], dtype=np.float64)
    grp_settings.create_dataset('phase', data=vol_list['phase'], dtype=h5py.string_dtype(encoding='ascii') )
    grp_settings.create_dataset('tag',   data=vol_list['tag'],   dtype=h5py.string_dtype(encoding='ascii') )
    grp_settings.create_dataset('time',        data=time, dtype=np.float64 )
    grp_settings.create_dataset('ray_param',   data=rp,   dtype=np.float64 )
    # `raypath`
    grp = fid.create_group('raypath')
    for id, r, t, it in zip(vol_list['id'], rp, time, paths):
        tmp = grp.create_group('%d' % id )
        tmp.attrs['id'] = id
        tmp.attrs['ray_param'] = r
        tmp.attrs['time'] = t
        tmp.attrs['npts'] = it['lon'].size
        tmp.create_dataset('lon', data=it['lon'], dtype=np.float64 )
        tmp.create_dataset('lat', data=it['lat'], dtype=np.float64 )
        tmp.create_dataset('depth', data=it['depth'], dtype=np.float64 )
    fid.close()

def run_h5_less_mem(filename, h5_fnm, verbose=None):
    """
    `filename` is a table file that each line is 
    "id evdp evla evlo stla stlo phase_name tag"
    """
    vol_list = np.loadtxt(filename, dtype= {'names':   ('id',    'evdp',    'evla',    'evlo',    'stla',    'stlo',   'phase', 'tag' ),
                                 'formats': ('int32',  'float64', 'float64', 'float64', 'float64', 'float64', object, object) } )
    fid = h5py.File(h5_fnm, 'w')
    # setting group
    grp_settings = fid.create_group('settings')
    grp_settings.create_dataset('id',    data=vol_list['id'],   dtype=np.int32)
    grp_settings.create_dataset('evdp',  data=vol_list['evdp'], dtype=np.float64)
    grp_settings.create_dataset('evla',  data=vol_list['evla'], dtype=np.float64)
    grp_settings.create_dataset('evlo',  data=vol_list['evlo'], dtype=np.float64)
    grp_settings.create_dataset('stla',  data=vol_list['stla'], dtype=np.float64)
    grp_settings.create_dataset('stlo',  data=vol_list['stlo'], dtype=np.float64)
    grp_settings.create_dataset('phase', data=vol_list['phase'], dtype=h5py.string_dtype(encoding='ascii') )
    grp_settings.create_dataset('tag',   data=vol_list['tag'],   dtype=h5py.string_dtype(encoding='ascii') )
    grp_settings.attrs['size'] = vol_list.size
    # raypath group
    rps = []
    ts  = []
    grp_path = fid.create_group('raypath')
    grp_path.attrs['size'] = vol_list.size
    for it_vol in vol_list:
        it_return = get_raypath_single(it_vol, verbose)
        tmp = grp_path.create_group('%d' % it_return['id'] )
        tmp.attrs['id'] = it_return['id']
        tmp.attrs['ray_param'] = it_return['ray_param']
        tmp.attrs['time'] = it_return['time']
        tmp.attrs['npts'] = it_return['lon'].size
        tmp.attrs['tag']   =  np.string_( it_return['tag'] )
        tmp.attrs['phase'] =  np.string_( it_return['phase'] )
        tmp.create_dataset('lon', data=it_return['lon'], dtype=np.float64 )
        tmp.create_dataset('lat', data=it_return['lat'], dtype=np.float64 )
        tmp.create_dataset('depth', data=it_return['depth'], dtype=np.float64 )
        rps.append(it_return['ray_param'])
        ts.append(it_return['time'])
        del it_return
    #
    grp_settings.create_dataset('time',        data=np.array(ts, dtype=np.float64), dtype=np.float64 )
    grp_settings.create_dataset('ray_param',   data=np.array(rps, dtype=np.float64),   dtype=np.float64 )
    fid.close()


if __name__ == "__main__":
    HMSG = """ Usage %s infnm outfnm [v]

    infnm: the input text file each line of which is `id evdp evla evlo stla stlo phase_name tag`. 
        id is an integer, tag is a string that should not contain whitespace.
    outfnm: output hdf5 that contains raypaths for event-station pairs.
    [v]: turn on verbose
    """
    argv = sys.argv
    if len(argv) < 2:
        print(HMSG % argv[0] )
        sys.exit(0)
    fin = argv[1]
    fout = argv[2]
    verbose = None
    try:
        verbose = argv[3]
    except:
        pass
    run_h5_less_mem(fin, fout, verbose)
    

