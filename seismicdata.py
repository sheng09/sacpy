#!/usr/bin/env python3
from obspy.core.utcdatetime import UTCDateTime
from numba import jit
from obspy.clients.fdsn.client import Client
from obspy import read, read_inventory
from obspy.clients.fdsn.client import Client
from obspy.core.stream import Stream
from obspy.core.util.attribdict import AttribDict
from obspy.core.inventory.inventory import Inventory
from obspy import read as obspy_read
import pickle
from h5py import File as h5_File
from numpy import float32, int32, zeros, array
from numpy import max as np_max
from numpy import min as np_min
import math
from matplotlib import mlab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.colors as colors
import warnings, traceback, sys
import os, os.path

from .geomath import haversine, azimuth, decluster_spherical_pts
from .utils import send_email, get_http_files, wget_http_files, deprecated_run, Timer

#####################################################################################################################
# Using the IRIS's BREQ_FAST service to request and download seismic data
#####################################################################################################################
class BreqFast:
    """
    e.g.,
        >>> from obspy.core.utcdatetime import UTCDateTime
        >>> from obspy.clients.fdsn.client import Client
        >>> from obspy.core.utcdatetime import UTCDateTime
        >>> #
        >>> app = BreqFast(email='who_email', name='name', institute='inst', mail_address='where', phone='555 5555 555', fax='555 5555 555', hypo=None, sender_email='sender@email', sender_passwd='pass', sender_host='what', sender_port=25 )
        >>> #
        >>> # 1. Request continuous time series
        >>> starttime, endtime = UTCDateTime(2021, 8, 21, 0, 0, 0), UTCDateTime(2021, 8, 21, 4, 0, 0)
        >>> app.do_not_send_emails = True # set to False to send emails
        >>> app.continuous_run('cont', starttime, endtime, interval_sec=3600, stations=('IU.PAB', 'II.ALE', 'AU.MCQ'), channels=('BH?', 'SH?'), location_identifiers=('00', '10'), label_prefix='cont', request_dataless=True, request_miniseed=True )
        >>> #
        >>> # 2. Request records for catalog events
        >>> client = Client('USGS')
        >>> starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
        >>> endtime   = UTCDateTime(2021, 3, 1, 0, 0, 0)
        >>> catalog = client.get_events(starttime, endtime, mindepth=10, minmagnitude=6.5) #, includeallorigins=True)
        >>> catalog = catalog[:2]
        >>> app.do_not_send_emails = True # set to False to send emails
        >>> app.earthquake_run('cat.txt', catalog, -10, 100, stations=('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'), label='cat', location_identifiers=('00', '10'), request_dataless=True, request_miniseed=True )
        >>> #
        >>> # 3. Customize run
        >>> app.do_not_send_emails = True # set to False to send emails
        >>> time_windows = [(UTCDateTime(2000,1,1,0), UTCDateTime(2000,1,1,1)), (UTCDateTime(2002,2,3,12), UTCDateTime(2002,2,3,14)) ]
        >>> app.custom_time_run('tw.txt', time_windows, stations=('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'), label='tw', location_identifiers=('00', '10'), request_dataless=True, request_miniseed=True )
        >>> #
        >>> # 4. Download the files from BREQFAST HTTP server once they are ready
        >>> url = 'http://ds.iris.edu/pub/userdata/Sheng_Wang'
        >>> app.breqfast_wget(url, re_template_string='^exam-.*mseed$', output_filename_prefix='junk1034343/d_', overwrite=True)

    """
    def __init__(self, email="you@where.com", name='your_name', institute='your_institute', mail_address='', phone='', fax='', hypo=None,
                       sender_email=None, sender_passwd=None, sender_host=None, sender_port=None,
                       do_not_send_emails=False ):
        """
        e.g.,:
            >>> hypo={  'ot': an datetime object, 
                        'evlo': 0.0, 'evla': 0.0, 'evdp_km': 0.0, 
                        'mag': 0.0, 'magtype': 'Mw' }, 
        """
        self.email = email
        self.name = name
        self.institute = institute
        self.mail_address = mail_address
        self.phone = phone
        self.fax = fax
        self.hypo = hypo
        self.sender_email=sender_email
        self.sender_passwd=sender_passwd
        self.sender_host=sender_host
        self.sender_port=sender_port
        self.do_not_send_emails=do_not_send_emails
    @staticmethod
    def earthquake_time_segments(catalog, pretime_sec=-100, posttime_sec=3600):
        """
        catalog: an object returned from `obspy.clients.fdsn.client.Client.get_events(...)`

        e.g.,
        """
        time_segments = list()
        for event in catalog:
            ot = event.origins[0].time
            t0, t1 = ot+pretime_sec, ot+posttime_sec
            time_segments.append( (t0, t1) )
        return time_segments
    @staticmethod
    def continuous_time_segments(min_time, max_time, interval_sec=7200):
        """
        min_time, max_time: an object of `obspy.core.utcdatetime.UTCDateTime`.
        interval_sec: time interval in seconds for subdividing long continuous time series.
        """
        segments = list()
        t0 = min_time
        while t0<max_time:
            t1 = t0+interval_sec
            if t1 > max_time:
                t1 = max_time
            segments.append( (t0, t1) )
            t0 = t1
        return segments
    @staticmethod
    def table_for_time_time_interval(stations, starttime, endtime, channels=('BH?', 'SH?'), location_identifiers=[]):
        """
        stations: a list of string in the formate of XXXX.YYYY where XXXX is network code and YYYY the station code.
        """
        ch_args = ' '.join(channels)
        ch_args = '%d %s' % (len(channels), ch_args)
        t1str = starttime.strftime('%Y %m %d %H %M %S.%f')[:-3]
        t2str = endtime.strftime('%Y %m %d %H %M %S.%f')[:-3]
        table = list()
        if not location_identifiers:
            location_identifiers = ('', )
        for net, sta in [it.split('.') for it in stations]:
            for loc in location_identifiers:
                table.append( '%-4s %-2s %s %s %s %s' % (sta, net, t1str, t2str, ch_args, loc) )
        return sorted(table)
    def __send_email_and_print(self, filename, tab, request_miniseed, request_dataless ):
        content = '%s\n%s\n' % (self.__hdr(), '\n'.join(tab) )
        if filename:
            with open(filename, 'w') as fid:
                print(content, file=fid, end='')
        subject = self.label
        address_book = {'mseed': 'miniseed@iris.washington.edu', 'dataless': 'DATALESS@iris.washington.edu'}
        if self.sender_email:
            recipients = list()
            if request_miniseed:
                recipients.append( address_book['mseed'] )
            if request_dataless:
                recipients.append( address_book['dataless'] )
            for it in recipients:
                print('Content: %s, Subject/BreqfastLabel: %s, Host: %s, Port: %s, Recipient: %s Sender: %s' % (
                        filename, subject, self.sender_host, self.sender_port, it, self.sender_email) )
                if not self.do_not_send_emails:
                    send_email(content, subject, it, sender=self.sender_email, passwd=self.sender_passwd, host=self.sender_host, port=self.sender_port)
    def __hdr(self):
        """
        """
        hdr_lines=[ ".NAME %s" %  (self.name) ,
                    ".INST %s" %  (self.institute),
                    ".MAIL %s" %  (self.mail_address) ,
                    ".EMAIL %s" % (self.email) ,
                    ".PHONE %s" % (self.phone)  ,
                    ".FAX   %s" % (self.fax)  ,
                    ".MEDIA FTP",
                    ".ALTERNATE MEDIA 1/2 tape\" - 6250",
                    ".ALTERNATE MEDIA EXABYTE",
                    ".LABEL %s" % (self.label) ,
        ]
        if self.hypo:
            hypo = self.hypo
            ot = hypo['ot']
            evla, evlo, evdp_km = hypo['evla'], hypo['evlo'], hypo['evdp_km']
            mag, magtype = hypo['mag'], hypo['magtype']
            otstr = '%04d %02d %02d %02d %02d %05.3f' % (ot.year, ot.month, ot.day, ot.hour, ot.minute, ot.second+ot.microsecond*0.000001 )
            hdr_lines.extend((  ".SOURCE",
                                ".HYPO ~%s~%7.3f~%8.3f~%5.1f~" % (otstr, evla, evlo, evdp_km) ,
                                ".MAGNITUDE ~%.1f~%s~" % (mag, magtype) ,
                            ))
        hdr_lines.extend((  ".QUALITY B",
                            ".END", 
                        ))
        hdr = '\n'.join(hdr_lines)
        return hdr
    def continuous_run( self, filename_prefix, min_time, max_time, interval_sec,
                        stations, channels=('BH?', 'SH?'), location_identifiers=[], label_prefix=None,
                        request_miniseed=True, request_dataless=False):
        """
        Generate BREQFAST request files by providing a time span, and send emails to breqfast@IRIS.
        For each time segment, a BREQFAST request file will be generated and a request will be made.

        filename_prefix: the filename prefix to store the BREQFAST request file.
        min_time, max_time: the time span to download continuous time series.
        interval_sec: the time interval in seconds for separating the continuous time series into many segments.
        stations: a list of station names in the format of XXXX.YYYY in which XXXX represent
                  network code and YYYY station code.
        channels: a list of channels that allow wildcards of `?` and `*`. (e.g., channels=('BH?', 'SH?') ).
        location_identifiers: a list of location identifiers. (e.g., location_identifiers=('00', '10') ).
                              in default, all location identifiers will be considered.
        label_prefix: a prefix string for label used in the BREQFAST request files.
                      If set to None, then the filename will be used for the label.
        request_miniseed: True or False to send email to request miniseed data.
        request_dataless: True or False to send email to request dataless metadata.
        """
        tseg = BreqFast.continuous_time_segments(min_time, max_time, interval_sec)
        print('Will make %d BREQFAST(+DATALESS) requests' % (len(tseg)) )
        for t0, t1 in tseg:
            tab = BreqFast.table_for_time_time_interval(stations, t0, t1, channels=channels, location_identifiers=location_identifiers)
            tab = sorted(tab)
            filename = '%s%s.txt' % (filename_prefix, t0.__str__() )
            if label_prefix:
                self.label = '%s_%s' % (label_prefix, t0.__str__() )
            else:
                self.label = filename.split('/')[-1].replace('.txt', '')
            self.__send_email_and_print(filename, tab, request_miniseed, request_dataless)
        self.label=None
    def earthquake_run( self, filename, catalog, pretime_sec, posttime_sec,
                        stations, channels=('BH?', 'SH?'), location_identifiers=[], label=None,
                        request_miniseed=True, request_dataless=False):
        """
        Generate BREQFAST request file by providing an event catalog, and send email to breqfast@IRIS.

        filename: the filename to store the BREQFAST request file.
        catalog: an object returned from `obspy.clients.fdsn.client.Client.get_events(...)`
        pretime_sec, posttime_sec: the time window in respect to the origin time of an event.
        stations: a list of station names in the format of XXXX.YYYY in which XXXX represent
                  network code and YYYY station code.
        channels: a list of channels that allow wildcards of `?` and `*`. (e.g., channels=('BH?', 'SH?') ).
        location_identifiers: a list of location identifiers. (e.g., location_identifiers=('00', '10') ).
                              in default, all location identifiers will be considered.
        label: a string for label used in the BREQFAST request files.
               If set to None, then the filename will be used for the label.
        request_miniseed: True or False to send email to request miniseed data.
        request_dataless: True or False to send email to request dataless metadata.
        """
        t_segs = BreqFast.earthquake_time_segments(catalog, pretime_sec, posttime_sec)
        tab = list()
        for t0, t1 in t_segs:
            tab.extend( BreqFast.table_for_time_time_interval(stations, t0, t1, channels=channels, location_identifiers=location_identifiers) )
        tab = sorted(tab)
        if label:
            self.label = label
        else:
            self.label = filename.split('/')[-1].replace('.txt', '')
        print('Will make 1 BREQFAST(+DATALESS) requests' )
        self.__send_email_and_print(filename, tab, request_miniseed, request_dataless)
        self.label=None
    def custom_time_run(self, filename, time_windows,
                        stations, channels=('BH?', 'SH?'), location_identifiers=[], label=None,
                        request_miniseed=True, request_dataless=False):
        """
        Generate BREQFAST request files by providing a list of time windows, and send a single email to breqfast@IRIS.

        filename: the filename to store the BREQFAST request file.
        time_windows: a list of time windows. e.g., time_windows = [ (UTCDateTime(2000,1,1,0), UTCDateTime(2000,1,1,1)), (UTCDateTime(2002,2,3,12), UTCDateTime(2002,2,3,14)) ]
        stations: a list of station names in the format of XXXX.YYYY in which XXXX represent
                  network code and YYYY station code.
        channels: a list of channels that allow wildcards of `?` and `*`. (e.g., channels=('BH?', 'SH?') ).
        location_identifiers: a list of location identifiers. (e.g., location_identifiers=('00', '10') ).
                              in default, all location identifiers will be considered.
        label: a string for label used in the BREQFAST request files.
               If set to None, then the filename will be used for the label.
        request_miniseed: True or False to send email to request miniseed data.
        request_dataless: True or False to send email to request dataless metadata.
        """
        tab = list()
        for t0, t1 in time_windows:
            tab.extend( BreqFast.table_for_time_time_interval(stations, t0, t1, channels=channels, location_identifiers=location_identifiers) )
        tab = sorted(tab)
        if label:
            self.label = label
        else:
            self.label = filename.split('/')[-1].replace('.txt', '')
        print('Will make 1 BREQFAST(+DATALESS) requests' )
        self.__send_email_and_print(filename, tab, request_miniseed, request_dataless)
        self.label=None
    @classmethod
    def breqfast_wget(cls, url, re_template_string, output_filename_prefix, overwrite=True):
        """
        Use wget to download files from the breqfast `http://ds.iris.edu/pub/userdata/user_name`.

        url: your BREQFAST http url.
        re_template_string: the template string to select files.
                            e.g., `re_template_string=r'^head.*txt$'` will match any filenames
                            starting with `head` and end with `txt`. `.*` means zero or any number
                            of any characters in the middle.
        output_filename_prefix: filename prefix to save the files.
        overwrite: True or False to overwrite the existing files.
        """
        urls = get_http_files(url, re_template_string=re_template_string )
        print(url, re_template_string, urls)
        wget_http_files(urls, filename_prefix=output_filename_prefix, overwrite=overwrite )
    @classmethod
    def test_run(cls):
        from obspy.core.utcdatetime import UTCDateTime
        from obspy.clients.fdsn.client import Client
        from obspy.core.utcdatetime import UTCDateTime
        #
        app = BreqFast(email='who_email', name='name', institute='inst', mail_address='where', phone='555 5555 555', fax='555 5555 555', hypo=None, sender_email='sender@email', sender_passwd='pass', sender_host='what', sender_port=25 )
        #
        # 1 Request continuous time series
        starttime, endtime = UTCDateTime(2021, 8, 21, 0, 0, 0), UTCDateTime(2021, 8, 21, 4, 0, 0)
        app.do_not_send_emails = True # set to False to send emails
        app.continuous_run('cont', starttime, endtime, interval_sec=3600,
                            stations=('IU.PAB', 'II.ALE', 'AU.MCQ'), channels=('BH?', 'SH?'), location_identifiers=('00', '10'),
                            label_prefix='cont', request_dataless=True, request_miniseed=True )
        #
        # 2 Request records for catalog events
        client = Client('USGS')
        starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
        endtime   = UTCDateTime(2021, 3, 1, 0, 0, 0)
        catalog = client.get_events(starttime, endtime, mindepth=10, minmagnitude=6.5) #, includeallorigins=True)
        catalog = catalog[:2]
        app.do_not_send_emails = True # set to False to send emails
        app.earthquake_run('cat.txt', catalog, -10, 100, stations=('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'),
                            label='cat', location_identifiers=('00', '10'), request_dataless=True, request_miniseed=True )
        #
        # 3 Customize run
        app.do_not_send_emails = True # set to False to send emails
        time_windows = [(UTCDateTime(2000,1,1,0), UTCDateTime(2000,1,1,1)),
                        (UTCDateTime(2002,2,3,12), UTCDateTime(2002,2,3,14)) ]
        app.custom_time_run('tw.txt', time_windows, stations=('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'),
                            label='tw', location_identifiers=('00', '10'), request_dataless=True, request_miniseed=True )
        #
        # 4. Download the files from BREQFAST HTTP server once they are ready
        url = 'http://ds.iris.edu/pub/userdata/Sheng_Wang'
        app.breqfast_wget(url, re_template_string='^exam-.*mseed$', output_filename_prefix='junk1034343/d_', overwrite=True)

#####################################################################################################################
# !!! This seems useless now because we no longer hanleing SAC data format.
#####################################################################################################################
class Waveforms(Stream):
    """
    """
    def __init__(self, traces=None):
        """
        traces: a list of `Trace`, or an object of `Stream` or `Waveforms`.
        """
        if traces:
            self.traces = Stream(traces)
        else:
            self.traces = Stream()
    def read(self, filenames):
        """
        Read from a list of files.
        Can be of `mseed`, `sac` formats.

        filenames: a list of mseed_filenames. The sequence is not changed if the input is sorted.
        """
        for it in filenames:
            self.traces.extend( read(it) )
    def get_inventory(self, client, level='response', filename=None, format='STATIONXML', **kwargs):
        """
        Return inventory/stati metadata for all the traces within `self.traces`.

        client: an object of `obspy.clients.fdsn.client.Client` for getting station metadata.
        level:  can be 'network', 'station', 'channel', or 'response'.
        filename: if not `None`, then write the invertory to a file.
        format:   the format to write.
        kwargs:   other arguments of `obspy.clients.fdsn.client.Client.get_stations(...)`

        Return: the object of `obspy.core.inventory.inventory.Inventory` for all the traces within `self.traces`.
        #
        #(station_name_list, inventory)
        #        station_name_list:  a sorted list of station names XXXX.YYYY.ZZ where XXXX is network code,
        #                            YYYY is station code, ZZ the location identifier.
                
        """
        vol_dict = dict()
        starttime, endtime = UTCDateTime('9999-01-01T00:00:00.000000'), UTCDateTime('1000-01-01T00:00:00.000000')
        for tr in self.traces:
            t0, t1 = tr.stats.starttime, tr.stats.endtime
            if t0<starttime:
                starttime = t0
            if t1>endtime:
                endtime = t1
            #
            net, sta, loc, chn = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
            if net not in vol_dict:
                vol_dict[net] = { 'sta': set(), 'loc': set(), 'chn': set() }
            net_dict = vol_dict[net]
            net_dict['sta'].add(sta)
            net_dict['loc'].add(loc)
            net_dict['chn'].add(chn)
        #
        starttime, endtime = starttime-100, endtime+100 # safe values
        #
        inv = Inventory()
        for net, net_dict in vol_dict.items():
            sta_str = ','.join(net_dict['sta'])
            loc_str = ','.join(net_dict['loc'])
            chn_str = ','.join(net_dict['chn'])
            tmp = client.get_stations(starttime=starttime, endtime=endtime, network=net, sta=sta_str, loc=loc_str, channel=chn_str, level=level, **kwargs)
            inv.extend(tmp) 
        #
        if filename:
            inv.write(filename, format=format)
        return inv
    def get_time_coverage(self):
        """
        Return a list of [tstart, tend] for the coverage of all the traces within.
        """
        segs = [(tr.stats.starttime, tr.stats.endtime) for tr in self]
        return union_time_segments(segs)
    def select_time(self, min_time, max_time, method='tight'):
        """
        Select a portion of traces within `self.traces` using the time range (`min_time`, `max_time`),
        Return the a new object of `Waveforms` that contain the selected traces. The traces
        will not be trimmed.
        If you want to select and also trim traces, please consider using `Waveforms.trim(...)`
        which in fact is `Stream.trim(...)`.

        min_time, max_time: objects of `UTCDateTime`.
        method:  a string can be `tight` or`loose`.
                `tight`: we only select the `trace` that `trace.stats.starttime <= min_time < max_time <= trace.stats.endtime`
                `loose`: we select the `trace` if the [`min_time`, `max_time`] and [`trace.stats.starttime`, `trace.stats.endtime`] intersect.
        """
        if method == 'tight':
            return Waveforms( [tr for tr in self.traces if (tr.stats.starttime<=min_time and tr.stats.endtime>=max_time) ] )
        elif method =='loose':
            return Waveforms( [tr for tr in self.traces if (tr.stats.starttime<=max_time and min_time<=tr.stats.endtime) ] )
    def group_network(self):
        """
        Group all the traces within the `self.traces` with respect to network code `XXXX`
        where `XXXX` is the network code.

        Return a dictionary. The keys of the dictionary are in the format of `XXXX`. Each
        value of the dictionary is a list of objects of `Waveforms`. Each element
        of the list contains traces having the same network code `XXXX` while their
        station codes, location identifier, and channels codes can be different.
        """
        nets = [tr.stats.network for tr in self.traces]
        vol = { it:Waveforms() for it in set(nets) }
        for net, tr in zip(nets, self.traces):
            vol[net].append(tr)
        return vol
    def group_station(self):
        """
        Group all the traces within the `self.traces` with respect to station name `XXXX.YYYY.ZZZZ`
        where `XXXX` is the network code, and `YYYY` the station code, and `ZZZZ` the location identifier.

        Return a dictionary. The keys of the dictionary are in the format of `XXXX.YYYY.ZZZZ`. Each
        value of the dictionary is a list of objects of `Waveforms`. Each element of the list
        contains traces having the same station name `XXXX.YYYY.ZZZZ` while their channels can be
        different.
        """
        stations = ['%s.%s.%s' % (tr.stats.network, tr.stats.station, tr.stats.location)  for tr in self.traces]
        stations = [it[:-1] if it[-1]=='.' else it for it in stations]
        vol = { it:Waveforms() for it in set(stations) }
        for st, tr in zip(stations, self.traces):
            vol[st].append(tr)
        return vol
    def group_location(self):
        """
        Group all the traces within the `self.traces` with respect to location identifier
        `ZZZZ` where the `ZZZZ` is the location identifier.

        Return a dictionary. The keys of the dictionary are in the format of `ZZZZ`. Each
        value of the dictionary is a list of objects of `Waveforms`. Each element
        of the list contains traces having the same location identifier `ZZZZ` while
        their network codes, station codes, and channel codes can be different.
        """
        locs = [tr.stats.location for tr in self.traces]
        vol = { it:Waveforms() for it in set(locs) }
        for loc, tr in zip(locs, self.traces):
            vol[loc].append(tr)
        return vol
    def group_channel(self):
        """
        Group all the traces within the `self.traces` with respect to channel name `CCC`
        where the `CCC` is the channel code.

        Return a dictionary. The keys of the dictionary are in the format of `CCC`. Each
        value of the dictionary is a list of objects of `Waveforms`. Each element
        of the list contains traces having the same channel name `CCC` while their
        network codes, station codes, and location identifier can be different.
        """
        channels = [tr.stats.channel for tr in self.traces]
        vol = { it:Waveforms() for it in set(channels) }
        for ch, tr in zip(channels, self.traces):
            vol[ch].append(tr)
        return vol
    def group_station_channel(self):
        """
        Group all the traces within the `self.traces` with respect to station-channel name
        `XXXX.YYYY.ZZZZ.CCC` where `XXXX` is the network code, and `YYYY` the station code,
        and `ZZZZ` the location identifier, and `CCC` the channel code.

        Return a dictionary. The keys of the dictionary are in the format of `XXXX.YYYY.ZZZZ.CCC`.
        Each value of the dictionary is a list of objects of `Waveforms`. Each element
        of the list contains traces having the same station-channel name `XXXX.YYYY.ZZZZ.CCC`.
        """
        station_chns = ['%s.%s.%s.%s' % (tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel)  for tr in self.traces]
        vol = { it:Waveforms() for it in set(station_chns) }
        for st, tr in zip(station_chns, self.traces):
            vol[st].append(tr)
        return vol
    def update_stats_sac(self, lcalda=False):
        """
        Update the `trace.stats.sac` for each element of `self.traces`
        """
        for tr in self.traces:
            if 'sac' not in tr.stats:
                tr.stats.sac = AttribDict()
            sac_dict = tr.stats.sac
            delta, npts = tr.stats.delta, tr.stats.npts
            sac_dict.delta = delta
            sac_dict.npts  = npts
            if 'b' in sac_dict:
                sac_dict.e = (npts-1)*delta+sac_dict.b
            #
            if lcalda:
                if 'evlo' in sac_dict and  'evla' in sac_dict   and   'stlo' in sac_dict  and   'stla' in sac_dict:
                    evlo, evla, stlo, stla = sac_dict.evlo, sac_dict.evla, sac_dict.stlo, sac_dict.stla
                    az    = azimuth(evlo, evla, stlo, stla)
                    baz   = azimuth(stlo, stla, evlo, evla)
                    gcarc = haversine(stlo, stla, evlo, evla)
                    sac_dict.az    = az
                    sac_dict.baz   = baz
                    sac_dict.gcarc = gcarc
                    sac_dict.lcalda = 1
                    tr.stats.back_azimuth = baz
                    tr.stats.azimuth = az

        pass
    def set_stats_sac_reference_time(self, reference_time):
        """
        Adjust the reference time used for sac header.
        The `trace.stats.sac` will be revised for each `trace` within `self.traces`.
        Also, the revision will affect the writing of sac file.

        reference_time: an object of UTCDateTime.
        """
        for tr in self.traces:
            if not hasattr(tr.stats, 'sac_reference_time'):
                tr.stats.sac_reference_time = tr.stats.starttime
            #############################################################
            if not hasattr(tr.stats, 'sac'):
                tr.stats.sac = AttribDict()
            sac_dict = tr.stats.sac
            sac_dict['nzyear'] = reference_time.year
            sac_dict['nzjday'] = reference_time.julday
            sac_dict['nzhour'] = reference_time.hour
            sac_dict['nzmin']  = reference_time.minute
            sac_dict['nzsec']  = reference_time.second
            sac_dict['nzmsec'] = int(reference_time.microsecond*0.001)
            sac_dict['b'] = tr.stats.starttime - reference_time
            #############################################################
            time_shift_sec = tr.stats.sac_reference_time - reference_time
            for k in ('t0', 't1', 't2', 't3', 't4',  't5', 't6', 't7', 't8', 't9', 'a', 'f', 'o'):
                if hasattr(sac_dict, k):
                    if sac_dict[k] != -12345.0:
                        sac_dict[k] = sac_dict[k] + time_shift_sec
    def set_stats_sac_station(self, inventory, verbose=False):
        """
        Set metadata of stations from `inventory` to each element of `self.traces`.
        The `trace.stats.sac` will be revised for each `trace` within `self.traces`.
        Also, the revision will affect the writing of sac file.

        inventory: an object of `obspy.core.inventory.inventory.Inventory`.
                   If the metadata of some stations are missing in the provided `inventory`,
                   that station and the related trace will be jumped and a warning will
                   be printed if `verbose=True`
        verbose:   whether to print warning for missing station metadata.

        """
        for tr in self.traces:
            knetwk, kstnm, kcmpnm = tr.stats.network, tr.stats.station, tr.stats.channel
            id = '%s.%s.%s.%s' % (knetwk, kstnm, tr.stats.location, kcmpnm)
            try:
                metadata = inventory.get_channel_metadata(id, tr.stats.starttime) # may fail here.
                #
                if not hasattr(tr.stats, 'sac'):
                    tr.stats.sac = AttribDict()
                sac_dict = tr.stats.sac
                #
                sac_dict['knetwk'] = knetwk
                sac_dict['kstnm']  = kstnm
                sac_dict['kcmpnm'] = kcmpnm
                #
                if 'dip' in metadata:
                    sac_dict['cmpinc'] = 90+metadata['dip']
                for meta_key, sac_key in zip(   ('latitude', 'longitude', 'elevation', 'local_depth', 'azimuth' ),
                                                ('stla',     'stlo',      'stel',      'stdp',        'cmpaz'   )   ):
                    if meta_key in metadata:
                        sac_dict[sac_key] = metadata[meta_key]
            except Exception:
                if verbose:
                    print('Warning: metadata cannot be found from the provided inventory for %s' % id, file=sys.stderr )
    def set_stats_sac_event(self, evlo, evla, evdp_km, mag, origin_time):
        """
        Fill metadata of events to each element of `self.traces`.
        The `trace.stats.sac` will be revised for each `trace` within `self.traces`.
        Also, the revision will affect the writing of sac file.

        origin_time: an object of UTCDateTime.
        """
        self.set_stats_sac_reference_time(origin_time) #
        for tr in self.traces:
            sac_dict = tr.stats.sac
            sac_dict['evlo'] = evlo
            sac_dict['evla'] = evla
            sac_dict['evdp'] = evdp_km
            sac_dict['mag'] = mag
            #
            sac_dict['o'] = 0.0
    def plot_wv(self, ax, reftime=None, baselines=None, normalize=False, scale=None, color='k', fill_between=None, **kwargs):
        """
        Plot the waveforms into an existed `ax`. This function will not change the data.

        reftime: an single `UTCDateTime` object or a list of `UTCDateTime` object.
                 The `reftime` will be used to compute the relative time of time axies.
                 In default, set `reftime=None` to use the `starttime` for each trace.
        baselines: a list of floating values for the baselines to plot each of the traces.
                   In default, set `baselines=None` to automatically adjust the waveforms to avoid overlapping.
        normalize: `False`(default) or a float value to normalize in respect with the maximal positive amplitude.
        scale: a single or a list of floating values to scale the waveforms.
               In default, set `scale=None` to disable this option.
        fill_between: 'positive' or 'negative' to fill the postive or negative amplitudes, respectively.
               In default, set `scale=None` to disable this option.
        """
        ##############
        st = self
        if normalize:
            st = self.copy()
            #st.detrend()
            for tr in st:
                v = tr.data.max()
                if v>0.0:
                    tr.data *= (normalize/tr.data.max() )
        if type(scale)!=type(None):
            try:
                n = len(scale)
                if n!=len(st):
                    errmsg = "Err: number of scale does not match the number of traces. %d!=%d " % (n, len(st))
                    raise RuntimeError(errmsg)
                else:
                    for tr, s in zip(st, scale):
                        tr.data *= s
            except Exception:
                for tr in st:
                    tr.data *= scale
        ##############
        if type(reftime)!=type(None):
            try:
                n = len(reftime)
                if n!=len(st):
                    errmsg = "Err: number of reftime does not match the number of traces. %d!=%d " % (n, len(st))
                    raise RuntimeError(errmsg)
            except Exception:
                reftime = [reftime for it in st]
        else:
            reftime = [it.stats.starttime for it in st]
        ###############
        if type(baselines)!=type(None):
            try:
                n = len(baselines)
                if n!=len(st):
                    errmsg = "Err: number of baselines does not match the number of traces. %d!=%d " % (n, len(st))
                    raise RuntimeError(errmsg)
            except Exception:
                errmsg = "Err: set a list of numbers for baselines"
                raise RuntimeError(errmsg)
        else:
            baselines = list()
            _base = 0.0
            for tr in st:
                _base = _base - tr.data.min()
                baselines.append(_base)
                _base = _base + tr.data.max()
        baselines = array(baselines)
        ###############
        for tr, tref, _base in zip(st, reftime, baselines):
            ys = tr.data + _base
            ts = tr.times(type='relative', reftime=tref)
            ax.plot(ts, ys, color=color, **kwargs)
            if fill_between == 'positive':
                ax.fill_between(ts, ys, _base, where=ys>=_base, interpolate=True, color=color, alpha=0.3)
            elif fill_between == 'negative':
                ax.fill_between(ts, ys, _base, where=ys<=_base, interpolate=True, color=color, alpha=0.3)

        return baselines
    def plot_wavefield(self, ax, normalize=False, scale=None, cmap='gray', interpolation=None, vmin=None, vmax=None, extent=None):
        """
        Plot the wavefield into an existed `ax`. This function will not change the data.

        normalize: `False`(default) or a float value to normalize in respect with the maximal positive amplitude.
        scale: a single or a list of floating values to scale the waveforms.
               In default, set `scale=None` to disable this option.
        """
        ##############
        st = self
        if normalize:
            st = self.copy()
            st.detrend()
            for tr in st:
                tr.data *= (normalize/tr.data.max() )
        if type(scale)!=type(None):
            try:
                n = len(scale)
                if n!=len(st):
                    errmsg = "Err: number of scale does not match the number of traces. %d!=%d " % (n, len(st))
                    raise RuntimeError(errmsg)
                else:
                    for tr, s in zip(st, scale):
                        tr.data *= s
            except Exception:
                for tr in st:
                    tr.data *= scale
        ##############
        ncol = np_max([tr.data.size for tr in st])
        nrow = len(st)
        mat = zeros((nrow, ncol), dtype=float32)
        for idx, tr in enumerate(st):
            mat[idx,:][:tr.data.size] = tr.data
        ##############
        if type(extent) == type(None):
            extent = array((0.0, st[0].stats.delta*(ncol-1), 0.0, nrow) )
        ax.imshow(mat, origin="lower", cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, aspect="auto", extent=extent)
        return extent
    def to_hdf5(self, h5_filename, info=''):
        """
        Convert the time-series and their metadata to hdf5 format file.
        """
        self.update_stats_sac(lcalda=True)
        data = [it.data for it in self.traces]
        # trace.stats keys()
        obspy_trace_starttime = [it.stats.starttime.__str__() for it in self.traces]
        obspy_trace_endtime   = [it.stats.endtime.__str__() for it in self.traces]
        #delta     = [it.stats.delta    for it in self.traces]
        #npts      = [it.stats.npts     for it in self.traces]
        obspy_trace_network   = [it.stats.network  for it in self.traces]
        obspy_trace_station   = [it.stats.station  for it in self.traces]
        obspy_trace_location  = [it.stats.location for it in self.traces]
        obspy_trace_channel   = [it.stats.channel  for it in self.traces]
        #
        filename = ['%s.%s.%s.%s' % it for it in zip(obspy_trace_network, obspy_trace_station, obspy_trace_location, obspy_trace_channel) ]
        # trace.stats.sac keys
        sachdr = dict()
        if hasattr(self.traces[0].stats, 'sac'):
            stats_sac_keys = set(self.traces[0].stats.sac.keys() )
            for k in stats_sac_keys:
                sachdr[k] = [it.stats.sac[k] for it in self.traces]
        Converter.to_hdf5(h5_filename, data, LL=obspy_trace_location, filename=filename,
                            obspy_trace_starttime=obspy_trace_starttime, obspy_trace_endtime=obspy_trace_endtime, 
                            obspy_trace_network=obspy_trace_network, obspy_trace_station=obspy_trace_station,
                            obspy_trace_location=obspy_trace_location, obspy_trace_channel=obspy_trace_channel,
                            **sachdr )
    def test_run(self):
        import os.path, os
        from glob import glob
        ### catalog
        pkl_fnm = 'junk_catalog.pkl'
        if not os.path.exists(pkl_fnm):
            client = Client('USGS')
            starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
            endtime   = UTCDateTime(2021, 9, 1, 0, 0, 0)
            catalog = client.get_events(starttime, endtime, mindepth=25, minmagnitude=6.5) #, includeallorigins=True)
            with open(pkl_fnm, 'wb') as fid:
                pickle.dump(catalog, fid)
        else:
            with open(pkl_fnm, 'rb') as fid:
                catalog = pickle.load(fid)
        print(catalog, '\n', len(catalog) )
        pre_time, tail_time = -10, 100
        ### download via breqfast
        # 
        ### read
        app = Waveforms()
        filenames = sorted(glob('exam-events-data-new1.549089/*mseed') )
        app.read(filenames)
        ### get inventory
        client_iris = Client('IRIS')
        inv_fnm = 'junk_test_Waveforms.xml'
        if not os.path.exists(inv_fnm):
            inv = app.get_inventory(client_iris, level='response', filename=inv_fnm)
        else:
            inv = read_inventory(inv_fnm)
        print(inv)
        ### attach response from the inv
        app.attach_response(inv)
        ###
        app.set_stats_sac_station(inv)
        ### group into event folders
        for event in catalog:
            ori = event.origins[0]
            ot, evlo, evla, evdp = ori.time, ori.longitude, ori.latitude, ori.depth
            mag = event.magnitudes[0].mag
            t0, t1 = ot+pre_time, ot+tail_time
            app2 = app.select_time(t0, t1, 'loose')
            app2.set_stats_sac_event(evlo, evla, evdp, mag, ot)
            #
            if app2:
                event_folder = '%s' % (ot.__str__() )
                #print(event_folder)
                if not os.path.exists(event_folder):
                    os.makedirs(event_folder)
                if False: # write to sac
                    for tr in app2:
                        fnm = '%s/%s.%s.%s.%s.sac' % (event_folder, tr.stats.network, tr.stats.station,  tr.stats.location, tr.stats.channel)
                        tr.write(fnm, format='SAC')
                if False: # write to mseed
                    app2.write('%s.mseed' % event_folder)
                if True: # write to h5
                    app2.to_hdf5('%s.h5' % event_folder)

#####################################################################################################################
# !!! This seems useless now because we no longer hanleing SAC data format.
#####################################################################################################################
class Converter:
    sachdr_float32_keys={   'delta',     'depmin',    'depmax',    'scale',     'odelta',
                            'b',         'e',         'o',         'a',         'internal1',
                            't0',        't1',        't2',        't3',        't4',
                            't5',        't6',        't7',        't8',        't9',
                            'f',         'resp0',     'resp1',     'resp2',     'resp3',
                            'resp4',     'resp5',     'resp6',     'resp7',     'resp8',
                            'resp9',     'stla',      'stlo',      'stel',      'stdp',
                            'evla',      'evlo',      'evel',      'evdp',      'mag',
                            'user0',     'user1',     'user2',     'user3',     'user4',
                            'user5',     'user6',     'user7',     'user8',     'user9',
                            'dist',      'az',        'baz',       'gcarc',     'internal2',
                            'internal3', 'depmen',    'cmpaz',     'cmpinc',    'unused2',
                            'unused3',   'unused4',   'unused5',   'unused6',   'unused7',
                            'unused8',   'unused9',   'unused10',  'unused11',  'unused12' }
    sachdr_int32_keys={ 'nzyear',    'nzjday',    'nzhour',    'nzmin',     'nzsec',
                        'nzmsec',    'nvhdr',     'internal5', 'internal6', 'npts',
                        'internal7', 'internal8', 'unused13',  'unused14',  'unused15',
                        'iftype',    'idep',      'iztype',    'unused16',  'iinst',
                        'istreg',    'ievreg',    'ievtyp',    'iqual',     'isynth',
                        'unused17',  'unused18',  'unused19',  'unused20',  'unused21',
                        'unused22',  'unused23',  'unused24',  'unused25',  'unused26',
                        'leven',     'lpspol',    'lovrok',    'lcalda',    'unused27'  }
    sachdr_S8_keys={'kstnm',
                    'khole',     'ko',        'ka',
                    'kt0',       'kt1',       'kt2',
                    'kt3',       'kt4',       'kt5',
                    'kt6',       'kt7',       'kt8',
                    'kt9',       'kf',        'kuser0',
                    'kuser1',    'kuser2',    'kcmpnm',
                    'knetwk',    'kdatrd',    'kinst'  }
    sachdr_S16_keys={'kevnm', }
    def __init__(self):
        pass
    @classmethod
    def to_hdf5(cls, h5_filename, data, info=None, ignore_data=False, **metadata):
        """
        Write out data into an hdf5 file.

        h5_filename:  the output hdf5 filename.
        data:         a list of time series, or a matrix that each row is a time series.
        info:         a string of information to describe the output file.
        ignore_data:  set to `True` to ignore writing the data to the file. (default: False)
        **metadata:   a list of whatever metadata for describing the time series.

        e.g., 
            >>> data = [ [1, 2, 3], [6, 7], [9, 10, 11, 12] ] # 3 traces
            >>> info = 'exam1_dataset'
            >>> delta = [1, 1, 2]
            >>> b = [0, 0, 3]
            >>> kstnm = ['s1', 's2', 's3']
            >>> Converter.to_hdf5('junk.h5', data, delta=delta, b=b, kstnm=kstnm)
        """
        nfile = len(data)
        fid = h5_File(h5_filename, 'w')
        # write attrs
        fid.attrs['nfile'] = nfile
        if info:
            fid.attrs['info'] = info
        #####################################################################################################################
        def check_length(key, vector):
            if len(vector) != nfile:
                print('Warning: missmatched numbers of metadata(%s) and data (%d!=%d)' % (key, len(vector), nfile), file=sys.stderr)
        #####################################################################################################################
        # write two / groups of strings
        for key in ('filename', 'LL'):
            if key in metadata:
                v = metadata[key]
                check_length(key, v)
                fid.create_dataset(key, data=[it.encode('ascii') for it in v] )
        #####################################################################################################################
        # write / hdr group
        grp_hdr = fid.create_group('hdr')
        for k, v in metadata.items():
            check_length(key, v)
            #if k in ('filename', 'LL'):
            #    continue
            if k in cls.sachdr_S16_keys: # sac hdr S16
                grp_hdr.create_dataset(k, data=[it.encode('ascii') for it in v], dtype='S16'  )
            elif k in cls.sachdr_S8_keys: # S8
                grp_hdr.create_dataset(k, data=[it.encode('ascii') for it in v], dtype='S8'  )
            elif k in cls.sachdr_int32_keys: # sac hdr int things
                grp_hdr.create_dataset(k, data=v, dtype=int32)
            elif k in cls.sachdr_float32_keys: # sac hdr float things
                grp_hdr.create_dataset(k, data=v, dtype=float32)
            elif type(v[0] ) == str: # strings
                grp_hdr.create_dataset(k, data=[it.encode('ascii') for it in v] )
            else: # numbers
                grp_hdr.create_dataset(k, data=v)
        # write the default values for other sac headers
        sachdr_key_sets  = (cls.sachdr_float32_keys, cls.sachdr_int32_keys, cls.sachdr_S8_keys, cls.sachdr_S16_keys)
        sachdr_dtypes = (float32, int32, 'S8', 'S16')
        sachdr_defaults = (-12345.5, -12345, '-12345'.encode('ascii'), '-12345'.encode('ascii') )
        for key_set, dtype, default_value in zip(sachdr_key_sets, sachdr_dtypes, sachdr_defaults):
            keys =[it for it in key_set if it not in metadata]
            for k in keys:
                grp_hdr.create_dataset(k, data=[default_value,]*nfile, dtype=dtype )
        #####################################################################################################################
        # write /data
        if not ignore_data:
            ncol = np_max([len(row) for row in data])
            mat = zeros((nfile, ncol), dtype=float32)
            for irow, row in enumerate(data):
                try:
                    sz = len(row)
                    mat[irow, :sz] = row
                except:
                    mat[irow] = 0.0
                    print('Warning: error data in row(%d). Set all to zeros' % (irow), file=sys.stderr)
            fid.create_dataset('dat', data=mat, dtype=float32)
        #####################################################################################################################
        fid.close()


#####################################################################################################################
# For getting adaptive colormap, and plotting spectrogram considering adaptive colormap
#####################################################################################################################
def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b
@deprecated_run(message='This function will be deprecated since 26/08/2024. Use `spectrogram_ax` instead.')
def old_spectrogram_ax(axes, data, tstart, samp_rate, per_lap=0.9, wlen=None, log=False,
                cax=None, cmap_hist_ax=None,
                outfile=None, fmt=None, dbscale=False,
                mult=8.0, cmap='plasma', zorder=None, title=None,
                show=True, clip=None):
    """
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to a
        window length matching 128 samples.
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type clip: [float, float]
    :param clip: absolute values to adjust colormap to clip at lower and/or upper end.
    """
    import matplotlib.pyplot as plt
    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(data)

    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    time += tstart

    if len(time) < 2:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, {nlap} samples window '
               f'overlap, sampling rate {samp_rate} Hz)')
        raise ValueError(msg)

    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    vmin, vmax = specgram.min(), specgram.max()
    if clip:
        _vmin, _vmax = clip
        if _vmin>vmax or _vmax<vmin or _vmin>=_vmax:
            print('Warning: invalid clip range:', clip)
            print('Change to data value range:', (vmin, vmax) )
        else:
            vmin, vmax = _vmin, _vmax
    norm, bin_edges, density, (vmin, vmax) = get_adaptive_colormap_norm(time, freq, specgram, logx=False, logy=log, v_range=(vmin, vmax) )
    cmap = get_adaptive_colormap(cmap, norm)
    ax = axes

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
              if v is not None}

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        im = ax.pcolormesh(time, freq, specgram, vmin=vmin, vmax=vmax, **kwargs)
    else:
        # this method is much much faster!
        specgram = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        im = ax.imshow(specgram, interpolation="nearest", extent=extent, vmin=vmin, vmax=vmax, **kwargs)


    print(time[0], time[-1])

    if cax:
        plt.colorbar(im, cax=cax)
    if cmap_hist_ax:
        #norm_func, bin_edges, density, v_range = adaptive_cmap_things
        bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:] )
        height = bin_edges[1]-bin_edges[0]
        clrs = [cmap( (it-vmin)/(vmax-vmin) ) for it in bin_centers]
        cmap_hist_ax.barh(bin_centers, density, height=height, color=clrs)
        cmap_hist_ax.set_ylim((vmin, vmax) )

    # set correct way of axis, whitespace before and after with window
    # length
    ax.axis('tight')
    #ax.set_xlim(0+tstart, end+tstart)
    ax.grid(False)
    ####
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    return im, (norm, bin_edges, density, (vmin, vmax))
@deprecated_run(message='This function will be deprecated since 26/08/2024. Use `get_adaptive_cmap` instead.')
def old_get_adaptive_colormap(cmap, norm):
    """
    #Return new `colormap` and `norm` with evenly distributed colors.
    """
    cmap = plt.cm.get_cmap(cmap)
    vs = norm.inverse(np.linspace(0.0, 1.0, 500) )
    dv = np.abs(np.diff(vs)).min()
    print('vs:', vs[0], vs[-1], vs.min(), vs.max(), dv )
    n = int((vs[-1]-vs[0])/dv)+1
    vs = np.linspace(vs[0],vs[-1], n+2)
    #print(vs)
    #print(n)
    clrs = cmap(norm(vs))
    newcmap = ListedColormap(clrs)
    return newcmap
@deprecated_run(message='This function will be deprecated since 26/08/2024.')
def old_get_adaptive_colormap_norm(xs, ys, mat, logx=False, logy=True, v_range=None):
    """
    Get an object of `matplotlib.colors.FuncNorm(...)` that
    can be used for `imshow(...)`, `pcolormesh(...)`, etc.

    xs, ys, mat: the objects that will be used to plot.
    logx: the x axis is in log scale.
    logy: the y axis is in log scale.
    vmin, vmax: `None` in default, and min and max values in the `mat` will be used.

    Return: norm_func, bin_edges, density, v_range
        norm_func: the targeted `matplotlib.colors.FuncNorm(...)`.
        bin_edges, density: the histogram of the pixal values for plotting the `mat`.
        v_range
    """
    try:
        vmin, vmax = v_range
    except Exception:
        vmin, vmax = mat.min(), mat.max()
    bin_edges = np.linspace(vmin, vmax, 200)
    ######################
    if logx or logy:
        weight = np.ones(mat.shape)
        for axis, log_flag, tmp in zip('xy', (logx, logy), (xs, ys) ):
            if not log_flag:
                continue
            if tmp[0] <= 1e-7: # if this is zero
                tmp = tmp[:]
                tmp[0] += 0.5*(tmp[-1]-tmp[-2])
            scale = np.zeros(tmp.size)
            logs = np.log(tmp)*(1.0/np.log(10))
            lmin, lmax = np.floor(logs.min()), np.ceil(logs.max())+1
            count, junk = np.histogram(logs, np.arange(lmin, lmax))
            istart = 0
            for c in count:
                if c>0:
                    iend = istart+c
                    scale[istart:iend] = 1.0/c
                    istart = iend
            scale *= (np.ceil(logs)-logs)
            if axis == 'y':
                for irow, scale in enumerate(scale):
                    weight[irow] *= scale
            elif axis == 'x':
                junk = np.transpose(weight)
                for irow, scale in enumerate(scale):
                    junk[irow] *= scale
                weight = np.transpose(junk)
        density, junk = np.histogram(mat, bins=bin_edges, density=True, weights=weight )
    else:
        density, junk = np.histogram(mat, bins=bin_edges, density=True )
    density *= (1.0/density.sum() )
    cum_density = np.cumsum(density)

    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    xs = np.concatenate( [[bin_centers[0]], bin_centers,  [bin_centers[-1]]] )
    ys = np.concatenate( [[cum_density[0]], cum_density, [cum_density[-1]]] )
    smooth = 0
    if smooth:
        ys = np.correlate(ys, np.ones(smooth), mode='same' ) *(1.0/smooth)
    def _forward(x):
        return np.interp(x, xs, ys)
    def _inverse(y):
        return np.interp(y, ys, xs)
    vmin = _inverse(0.0)
    vmax = _inverse(1.0)
    norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax )
    return norm, bin_edges, density, (vmin, vmax)

def spectrogram_ax(ax, data, tstart, samp_rate, per_lap=0.9, wlen=None, mult=8.0, 
                log=False, dbscale=False, clip=None, xlim=None, ylim=None,
                cmap='plasma', zorder=None, enable_adaptive_cmap=True,
                cax=None, cmap_hist_ax=None):
    """
    Computes and plots spectrogram of the input data.

    ax:        an object of `axes` for where to plot.
    data:      Input 1D data.
    tstart:    the start time of the first data point.
    samp_rate: Samplerate in Hz
    per_lap:   Percentage of overlap of sliding window, ranging from 0
               to 1. High overlaps take a long time to compute.
    wlen:      Window length for fft in seconds. If this parameter is too small,
               the calculation will take forever. If None, it defaults to a
               window length matching 128 samples.
    mult:      Pad zeros to length mult * wlen. This will make the
               spectrogram smoother.
    log:       Logarithmic frequency axis if `True`, linear frequency axis if `False`.
    dbscale:   If `True` 10 * log10 of color values is taken,
               if `False` the SQRT is taken.
    clip:      absolute values to adjust colormap to clip at lower and upper end.
    xlim, ylim:to adjust the xlim and ylim of the `ax`.
    cmap:      an object of `matplotlib.colors.Colormap` or a string that
               specify a colormap instance.
    zorder:    zorder of the plot.
    enable_adaptive_cmap: True or False to enable adaptive colormap adjustment.
    cax:       an object of `axes` for where to plot colorbar.
               `None` in default to disable this plotting.
    cmap_hist_ax: an object of `axes` for where to plot histogram when
                  adaptive colorbar is enabled.
                  `None` in default to disable this plotting.
    """
    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(data)

    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    # Note: while the `mlab.specgram` can plots db (10log10(values)), it returns the raw values.
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft, pad_to=mult, noverlap=nlap)

    time += tstart

    if len(time) < 2:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, {nlap} samples window '
               f'overlap, sampling rate {samp_rate} Hz)')
        raise ValueError(msg)

    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    vmin, vmax = specgram.min(), specgram.max()
    if clip:
        try:
            _vmin, _vmax = clip
            if _vmin>vmax or _vmax<vmin or _vmin>=_vmax:
                print('Warning: invalid clip range:', clip)
                print('Change to data value range:', (vmin, vmax) )
            else:
                vmin, vmax = _vmin, _vmax
        except Exception:
            msg = 'Wrong clip: {}'.formate(clip)
            raise ValueError(msg)
    # run adaptive cmap
    if enable_adaptive_cmap:
        tmp = get_adaptive_cmap(specgram, xs=time, ys=freq, 
                                logx=False, logy=log, xlim=xlim, ylim=ylim, 
                                vmin=vmin, vmax=vmax, cmap=cmap, ax_hist=cmap_hist_ax )
        cmap, cmap_norm, (bin_edges, density) = tmp
    else:
        vmax_minus_vmin = vmax-vmin
        inverted_vmax_minus_vmin = 1.0/vmax_minus_vmin
        def _forward(x):
            return (x-vmin)*inverted_vmax_minus_vmin
        def _inverse(y):
            return y*vmax_minus_vmin+vmin
        cmap_norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax )
        bin_edges, density = None, None
    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (    ('cmap', cmap),
                                    ('zorder', zorder),
                                    ('norm', cmap_norm) )
                                if v is not None }

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        im = ax.pcolormesh(time, freq, specgram, **kwargs)
    else:
        # this method is much much faster!
        # specgram = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        im = ax.imshow(specgram, interpolation="nearest", extent=extent, **kwargs, origin='lower')

    if cax:
        plt.colorbar(im, cax=cax)

    # set correct way of axis, whitespace before and after with window length
    ax.axis('tight')
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0+tstart, end+tstart)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(False)
    ####
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    return im, (cmap, cmap_norm, (bin_edges, density) )

def get_adaptive_cmap(data, xs=None, ys=None, 
                        logx=False, logy=False, 
                        xlim=None, ylim=None,
                        vmin=None, vmax=None, cmap='Spectral_r',
                        ax_hist=None):
    """
    Form an object of `matplotlib.colors.Colormap` and an objec of `matplotlib.colors.Norlize` 
    that can be used when calling contour-like plotting functions, such as `imshow(... cmap=..., norm=...)`
    based on the statistics of the input `data`.

    data: array-like input data. We will use the flattened array. The input data will not be modified.
    xs, ys: x and y coordinates along the row and colume directions of the `data` if it is a 2D matrix.
            The `xs` and `ys` are only useful when `logx` or `logy` are `True` or `xlim` and `ylim` are set.
            In default `xs` and `ys` are None.
            Please note: `data.shape = (xs.size, ys.size)`.
            o--+---+---+---+-------o
            | d00 d01 d02 d03 ...  +   y0
            |                      |
            | d10 d11 d12 d13 ...  +   y1
            |                      |
            | d20 d21 d22 d23 ...  +   y2
            | ...                  |   ...
            |                      |
            o--+---+---+---+-------o
               x0  x1  x2  x3 ...
    logx, logy: True or False (default) to declare it will be log10 style for the X or Y axis.
    xlim, ylim: the (xmin, xmax) and (ymin, ymax) that will be used for plotting.
    vmin, vmax: min and max values that will be used when calling contour-like plotting functions.
    cmap: the input basic colormap to adjust. This input cmap will not be modified.
          Can be an object of `matplotlib.colors.Colormap` or a string (e.g., `Spectral_r`)
    ax_hist: where to plot the histogram statistics of the input `data` with colors from the adjusted
             cmap. `None` in default to disable the plotting. 

    Return: cmap, norm, (bin_edges, density)
            the `(bin_edges, density)` describe the histogram statistics of the input `data`.
    """
    ####
    def check_lim(lim, values, search=True):
        lim_min, lim_max = lim
        val_min, val_max = np.min(values), np.max(values)
        if lim_min>=lim_max or lim_min>val_max or lim_max<val_min:
            msg = 'Wrong range setting ({}, {}) vs. data ragnge ({}, {})'.format(lim_min, lim_max, val_min, val_max)
            raise Exception(msg)
        #print('set', lim, 'data', (_min, _max))
        if lim_min < val_min:
            lim_min = val_min
        if lim_max > val_max:
            lim_max = val_max
        if search:
            imin = np.searchsorted(values, lim_min)
            imax = np.searchsorted(values, lim_max, side='right')
            return lim_min, lim_max, imin, imax
        else:
            return lim_min, lim_max
    #####
    if xlim and type(xs)!=type(None):
        xmin, xmax, icol0, icol1 = check_lim(xlim, xs)
        data = data[:, icol0:icol1]
        xs = xs[icol0:icol1]
    if ylim and type(ys)!=type(None):
        ymin, ymax, irow0, irow1 = check_lim(ylim, ys)
        data = data[irow0:irow1, :]
        ys = ys[irow0:irow1]
    #####
    if type(vmin) == type(None):
        vmin = np.min(data)
    if type(vmax) == type(None):
        vmax = np.max(data)
    check_lim( (vmin, vmax), data, False ) # after here, vmin and vmax should not be modified
    #####
    # now start to statisticize the values within the `data`
    def old_compute_log_scale(tmp, safe_value): # the tmp should be an evenly increasing array
        if tmp[0]<=1.0e-7: # if this is zero
            tmp = tmp.copy()
            tmp[0] = safe_value
        scale = np.zeros(tmp.size)
        logs = np.log(tmp)*(1.0/np.log(10))
        lmin, lmax = np.floor(logs.min()), np.ceil(logs.max())+1
        count, junk = np.histogram(logs, np.arange(lmin, lmax))
        istart = 0
        for c in count:
            if c>0:
                iend = istart+c
                scale[istart:iend] = 1.0/c
                istart = iend
        scale *= (np.ceil(logs)-logs)
        return scale
    def compute_log_scale(tmp, safe_value): # the tmp should be an evenly increasing array
        scale = np.ones(tmp.size)
        #scale *= 1/tmp
        step = tmp[1]-tmp[0]
        half_step = 0.5*step
        tmp = np.concatenate( (tmp, (tmp[-1]+step, )) )
        tmp -= half_step
        if tmp[0] < 1.0e-6: # if zeros
            tmp[0] += 1.0e-6
        log_edges = np.log(tmp)*(1.0/np.log(10))
        width = np.diff(log_edges)
        width *= (1.0/width.max() )
        #width = np.ones(width.size)
        scale *= width
        return scale
        #
    nbins = 200
    bin_edges = np.linspace(vmin, vmax, nbins+1)
    bin_width = bin_edges[1]-bin_edges[0]
    bin_centers = bin_edges[:-1]+bin_width*0.5
    if logx or logy:
        weight = np.ones(data.shape)
        for axis, log_flag, tmp in zip('xy', (logx, logy), (xs, ys) ):
            if not log_flag:
                continue
            scale = compute_log_scale(tmp, 0.5*(tmp[-1]-tmp[-2]))
            if axis == 'x':
                weight *= scale
            elif axis == 'y':
                weight = weight.transpose()
                weight *= scale
                weight = weight.transpose()
        weight *= (1.0/np.sum(weight) )
        density, junk = np.histogram(data, bins=bin_edges, density=True, weights=weight )
    else:
        density, junk = np.histogram(data, bins=bin_edges, density=True)
    density *= (1.0/density.sum() )  # this will not be modified in later operations
    cum_density = np.cumsum(density) # this can be modified in later operations
    ####
    # now get the first version norm function
    smooth = 0
    if smooth:
        cum_density = np.correlate(cum_density, np.ones(smooth), mode='same' ) *(1.0/smooth)
    def _forward(x):
        return np.interp(x, bin_centers, cum_density)
    def _inverse(y):
        return np.interp(y, cum_density, bin_centers)
    first_norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax )
    ####
    # now adjust the cmap and get the second version norm function
    cmap = plt.cm.get_cmap(cmap)
    nl = 100
    levels = first_norm.inverse( np.linspace(0.0, 1.0, nl) )
    dls = np.abs( np.diff(levels) )
    dlmin = np.extract(dls>0, dls).min()
    levels = np.linspace(vmin, vmax, int((vmax-vmin)/dlmin)+1 )
    clrs = cmap( first_norm(levels) )
    newcmap = ListedColormap(clrs)
    vmax_minus_vmin = vmax-vmin
    inverted_vmax_minus_vmin = 1.0/vmax_minus_vmin
    def _forward(x):
        return (x-vmin)*inverted_vmax_minus_vmin
    def _inverse(y):
        return y*vmax_minus_vmin+vmin
    second_norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax )
    ####
    # now plot the histogram with colors
    if ax_hist:
        bar_clrs = [newcmap( second_norm(it) ) for it in bin_centers]
        ax_hist.barh(bin_centers, density, height=bin_width, color=bar_clrs)
        ax_hist.set_ylim((vmin, vmax) )
    ####
    return newcmap, second_norm, (bin_edges, density)

#####################################################################################################################
# Historial methods for processing time windows/segments
#####################################################################################################################
def union_time_segments(segments):
    """
    Union a list of time segments.

    segments: a list of (tstart, tend)
    """
    if len(segments) < 1:
        return list()
    ######
    segs = sorted(segments)
    result = [segs[0] ]
    for t2, t3 in segs:
        t0, t1 = result[-1]
        if t1<t2:
            result.append( (t2, t3) )
        else:
            tend = t1 if t1>t3 else t3
            result[-1] = (t0, tend)
    return result
def join_two_time_segments(s1, s2):
    """
    Return a list of time segments derived from the intersection of two lists of time segments.
    Each of the time segment within the returned should be coverged by both the s1 and s2.

    s1: a list of (tstart, tend).
    s2: a list if (tstart, tend).
    """
    def __intersect(t0, t1, t2, t3): # check if (t0, t1) and (t2, t3) intersect
        if t1<t2 or t3<t0:
            return False
        return True
    def __join(t0, t1, t2, t3): # intersect the (t0, t1) and (t2, t3) if they intersect
        tstart = t0 if t0>t2 else t2
        tend = t1 if t1<t3 else t3
        return (tstart, tend)
    seg1 = union_time_segments(s1)
    seg2 = union_time_segments(s2)
    result = list()
    for (t0, t1) in seg1:
        for t2, t3 in (seg2):
            if __intersect(t0, t1, t2, t3):
                result.append( __join(t0, t1, t2, t3) )
    return union_time_segments(result)
def join_time_segments(list_of_segments):
    if len(list_of_segments) < 2:
        return list_of_segments[0]
    result = join_two_time_segments( list_of_segments[0], list_of_segments[1] )
    for it in list_of_segments[2:]:
        result = join_two_time_segments(result, it)
    return result
def plot_time_segments(semgents, ax, y=None, **kwargs):
    """
    Plot time segments.

    segments: a list of (tstart, tend)
    ax: where to plot
    y: None to set step-like time segments.
       a number (1,2,3,4,...) to specify the y coordinate to plot the time segments.
    """
    for idx,(t0, t1) in enumerate(semgents):
        v = y if type(y)!=type(None) else idx
        ax.plot((t0, t1), (v, v), **kwargs)

#####################################################################################################################
# Newer Processing time windows
#####################################################################################################################
class TimeWindow:
    """
    """
    __time_zero = UTCDateTime("1970-01-01T00:00:00")
    def __init__(self, starttime, endtime):
        """
        starttime, endtime: the start and end time of the time window. In objects of `obspy.UTCDateTime`.
        """
        if starttime >= endtime:
            #warnings.warn('An empty time window is created due to invalid: %s---%s' % (str(starttime), str(endtime)) )
            self.starttime = TimeWindow.__time_zero
            self.endtime   = TimeWindow.__time_zero
            return
        self.starttime = starttime
        self.endtime = endtime
    def duration(self):
        return self.endtime - self.starttime # in seconds
    def intersect(self, another_time_window):
        """
        Return the intersection of the current time window and another time window.
        """
        t0, t1 = self.starttime, self.endtime
        t2, t3 = another_time_window.starttime, another_time_window.endtime
        #                t0------------t1
        #      t2----t3                                   x
        #      t2----------======t3                       t0-t3
        #      t2----------============-------t3          t0-t1
        #                   t2===t3                       t2-t3
        #                   t2=========-------t3          t2-t1
        #                                t2---t3          x
        t0 = max(t0, t2)
        t1 = min(t1, t3)
        if t0 >= t1:
            t0 = TimeWindow.__time_zero
            t1 = TimeWindow.__time_zero
        return TimeWindow(t0, t1)
    def union(self, another_time_window):
        """
        Return the union of the current time window and another time window if them intersect!
        #
        Empty time window does not intersect with any other time windows.
        """
        t0, t1 = self.starttime, self.endtime
        t2, t3 = another_time_window.starttime, another_time_window.endtime
        #                t0------------t1
        #      t2----t3                                   x
        #      t2================t3====                   t0-t3
        #      t2=============================t3          t0-t1
        #                  ==t2===t3                      t2-t3
        #                  ==t2===============t3          t2-t1
        #                                t2---t3          x
        if t3<t0 or t1<t2:
            return TimeWindow( TimeWindow.__time_zero, TimeWindow.__time_zero )
        return TimeWindow( min(t0, t2), max(t1, t3) )
    def plot(self, ax, y=0, **kwargs):
        """
        Plot the time window on the ax.
        """
        ax.plot( (self.starttime.datetime, self.endtime.datetime), (y, y), **kwargs)
    def __eq__(self, other):
        if isinstance(other, TimeWindow):
            return (self.starttime == other.starttime) and (self.endtime == other.endtime)
        raise ValueError('Invalid comparison between a TimeWindow and %s' % str(type(other)) )
    def __ne__(self, other):
        return not self.__eq__(other)
    def __lt__(self, other):
        if isinstance(other, TimeWindow):
            return (self.starttime, self.endtime) < (other.starttime, other.endtime)
        raise ValueError('Invalid comparison between a TimeWindow and %s' % str(type(other)) )
    def __le__(self, other):
        if isinstance(other, TimeWindow):
            return (self.starttime, self.endtime) <= (other.starttime, other.endtime)
        raise ValueError('Invalid comparison between a TimeWindow and %s' % str(type(other)) )
    def __gt__(self, other):
        if isinstance(other, TimeWindow):
            return (self.starttime, self.endtime) > (other.starttime, other.endtime)
        raise ValueError('Invalid comparison between a TimeWindow and %s' % str(type(other)) )
    def __ge__(self, other):
        if isinstance(other, TimeWindow):
            return (self.starttime, self.endtime) >= (other.starttime, other.endtime)
        raise ValueError('Invalid comparison between a TimeWindow and %s' % str(type(other)) )
    def __add__(self, seconds):
        return TimeWindow(self.starttime+seconds, self.endtime+seconds)
    def __sub__(self, seconds):
        return TimeWindow(self.starttime-seconds, self.endtime-seconds)
    def __iadd__(self, seconds):
        self.starttime += seconds
        self.endtime   += seconds
        return self
    def __isub__(self, seconds):
        self.starttime -= seconds
        self.endtime   -= seconds
        return self
    def __str__(self) -> str:
        return '%s---%s' % (str(self.starttime), str(self.endtime) )
    def __repr__(self) -> str:
        return self.__str__()
    def __iter__(self):
        # This method returns an iterator
        yield self.starttime
        yield self.endtime
class TimeWindows(list):
    """
    A list of TimeWindow objects.
    """
    def __init__(self, time_windows=None):
        """
        This init function takes care of sorting and merging if any time windows overlap.
        """
        self.clear()
        if time_windows:
            time_windows = [(it.starttime, it.endtime) for it in time_windows]
            time_windows = sorted(time_windows)
            fixed = [time_windows[0]]
            for (t2, t3) in time_windows[1:]:
                t0, t1 = fixed[-1]
                #   t0--------------t1
                #   t2---------t3                    t0 = t2 < t3 < t1   x
                #       t2-----t3                    t0 < t2 < t3 < t1   x
                #   t2--------------t3               t0 = t2 < t1 = t3   x
                #       t2----------t3               t0 < t2 < t3 = t1   x
                #   t2-----------------------t3      t0 = t2 < t1 < t3   merge  --> (t0, t3)
                #       t2-------------------t3      t0 < t2 < t1 < t3   merge  --> (t0, t3)
                #                   t2-------t3      t0 < t2 = t1 < t3   merge  --> (t0, t3)
                #                        t2--t3      t0 < t1 < t2 < t3   append --> (t0, t1), (t2, t3)
                if t0 < t1 < t2 < t3:
                    fixed.append( (t2, t3) )
                elif t0 <= t2 <= t1 < t3:
                    fixed[-1] = (t0, t3)
            self.extend( [TimeWindow(t0, t1) for (t0, t1) in fixed] )
    def __str__(self):
        return ', '.join( [str(it) for it in self] )
    def intersect(self, other_wnds):
        """
        Return a list of time windows are the intersection of two list of time windows.
        That means the returned time windows are the time windows that are common in both ws1 and ws2.
        """
        result = list()
        for w1 in self:
            result.extend( [w1.intersect(w2) for w2 in other_wnds] )
        result = [it for it in result if it.duration()>0 ]
        return TimeWindows(result) # this will take care of sorting and merging if necessary
    def plot(self, ax, y=0, **kwargs):
        """
        Plot the time windows on the ax.
        """
        for idx, wnd in enumerate(self):
            wnd.plot(ax, y=y, **kwargs)

#####################################################################################################################
# Stream and Trace Processing considering the same channels or stations or events
#####################################################################################################################
class ChannelStream(Stream): #A stream for all Traces at the same channel
    """
    A stream for all Traces at the same channel
    """
    def __init__(self, traces=None):
        """
        """
        st = traces
        self.clear()
        if st:
            if len( set([tr.get_id() for tr in st]) ) != 1:
                raise ValueError('The input stream contains traces from different stations/locations/channels! %s' % (str(st) ) )
            self.extend(st)
    def get_id(self):
        return self[0].get_id()
    def get_station(self):
        """
        Return net.sta.loc
        """
        stats = self[0].stats
        return '.'.join( (stats.network, stats.station, stats.location) )
    def trim_ws(self, ws):
        """
        trim using a list of time windows
        """
        ws = TimeWindows(ws) # ws will be updated for sorting and merging regarding overlapping time windows
        tmp = [self.copy().trim(wnd.starttime, wnd.endtime) for wnd in ws] # could be the bottleneck!
        #tmp = [self.slice(t0, t1).copy() for (t0, t1) in ws] #?
        self.clear()
        for it in tmp:
            self.extend(it)
        self.sort(keys=['starttime', 'endtime'] )
    def remove_short_traces(self, min_length_sec, min_gap_sec=0, min_single_length_sec=0, verbose_print_func=None): # this will modify the content of self
        """
        Remove traces with length less than min_length_sec and min_gap_sec.
        min_length_sec:        the minimum length in seconds for the total duration of all the traces.
        min_gap_sec:           the minimum gap between two near traces in seconds.
        min_single_length_sec: the minimum length in seconds for a single trace.
        """
        ####
        n1 = len(self) # verbose stuff
        msg1 = 'Before removing short traces: %s' % self.__str__(extended=True) if verbose_print_func else ''
        ####
        min_length_sec = min_length_sec if min_length_sec > 0 else 0
        min_gap_sec = min_gap_sec if min_gap_sec > 0 else 0
        min_single_length_sec = min_single_length_sec if min_single_length_sec > 0 else 0
        ####
        if min_single_length_sec > 0:
            self[:] = [tr for tr in self if ( (tr.stats.endtime-tr.stats.starttime)>=min_single_length_sec) ] # remove short individual traces
        ####
        if len(self) > 0:
            self.sort( keys=['starttime', 'endtime'] )
            ws = TimeWindows( [TimeWindow(tr.stats.starttime, tr.stats.endtime) for tr in self] )
            dur= [it.duration() for it in ws]
            idx_ref = np.argmax(dur)
            idx_start, idx_end = idx_ref, idx_ref+1
            ####
            t2, t3 = ws[idx_ref]
            for idx in range(idx_ref-1, -1, -1):
                t0, t1 = ws[idx]
                if (t2-t1)>min_gap_sec or t0>=t1: # t0 < t1 < t2 < t3 & (t2-t1)>min_gap_sec
                    break
                idx_start = idx
                t2, t3 = t0, t1
            ####
            t0, t1 = ws[idx_ref]
            for idx in range(idx_ref+1, len(ws)):
                t2, t3 = ws[idx]
                if (t2-t1)>min_gap_sec or t2>=t3: # t0 < t1 < t2 < t3 & (t2-t1)>min_gap_sec
                    break
                idx_end = idx+1
                t0, t1 = t2, t3
            ####
            junk  = TimeWindows( ws[idx_start:idx_end] )
            total_length = np.sum([float(t1-t0) for (t0, t1) in junk] )
            if total_length < min_length_sec:
                self.clear()
            elif (idx_end-idx_start) != len(self):
                tmp = self[idx_start:idx_end]
                self[:] = tmp
        ####
        n2 = len(self) # verbose stuff
        if n1 != n2 and verbose_print_func:
            msg2 = 'After removing short traces: %s' % tmp.__str__(extended=True)
            verbose_print_func( msg1 )
            verbose_print_func( msg2 )
    def get_time_windows(self):
        """
        Return a list of time windows as the object of TimeWindows.
        Note, overlapping and sorting are taken care of.
        """
        ws = [TimeWindow(tr.stats.starttime, tr.stats.endtime) for tr in self if tr.stats.starttime<tr.stats.endtime ] # we don't need window with length of zero
        return TimeWindows(ws) #  this will take care of sorting and merging if necessary
    @staticmethod
    def intersect_time_windows_lst(lst_of_chst, time_summary=None):
        if len(lst_of_chst) > 1:
            if time_summary == None:
                # obtain the intersection time windows
                ws = lst_of_chst[0].get_time_windows()
                for chst in lst_of_chst[1:]:
                    ws2 = chst.get_time_windows()
                    ws = ws.intersect(ws2)
                # now trim for each channel stream
                for chst in lst_of_chst:
                    if ws != chst.get_time_windows():
                        chst.trim_ws(ws)
            else:
                # obtain the intersection time windows
                with Timer(tag='get_intersection_wnd', verbose=False, summary=time_summary):
                    ws = lst_of_chst[0].get_time_windows()
                    for chst in lst_of_chst[1:]:
                        ws2 = chst.get_time_windows()
                        ws = ws.intersect(ws2)
                # now trim for each channel stream
                with Timer(tag='trim_intersection_wnd', verbose=False, summary=time_summary):
                    for chst in lst_of_chst:
                        if ws != chst.get_time_windows():
                            chst.trim_ws(ws)
class StationStreams(dict):  #A dict for all ChanneStream at the same channel (channel_code-->ChannelStream)
    """
    A dict for grouping all traces w.r.t. different channels. The traces should be at the same station and location.
    The dict key is the channel name (e.g, BHZ, BH1,...), and the value is an object of ChannelStream.
    """
    def __init__(self, st=None):
        """
        st: an object of obspy.Stream, or a list of obspy.Trace objects.
        """
        self.clear()
        if st:
            if len( set( [(tr.stats.network,tr.stats.station, tr.stats.location)  for tr in st] ) ) != 1:
                raise ValueError('The input stream contains traces from different stations/location! %s' % (str(st) ) )
            keys = [tr.stats.channel for tr in st]
            tmp  = {k:ChannelStream() for k in set(keys) }
            for k, v in zip(keys, st):
                tmp[k].append(v)
            for v in tmp.values():
                v.sort(keys=['starttime', 'endtime'] )
            self.update(tmp)
    def fix_segment_times(self, min_length_sec, min_gap_sec=0, min_single_length_sec=0, verbose_print_func=None, time_summary=None):
        """
        Fix the time windows at each channel, so that all channels have the same time windows.
        And remove traces with length less than min_length_sec and min_gap_sec.
        """
        msg1, msg2 = '', ''
        if verbose_print_func:
            msg1 = self.to_stream().__str__(extended=True)
        ####
        tmp = list( self.values() )
        ChannelStream.intersect_time_windows_lst( tmp, time_summary=time_summary ) # this will modify content within tmp
        #####
        if time_summary:
            with Timer(tag='remove_short_empty', verbose=False, summary=time_summary):
                for it in tmp:
                    it.remove_short_traces(min_length_sec, min_gap_sec, min_single_length_sec=min_single_length_sec) # this will modify content within tmp
                self.__remove_empty_stream__()
        else:
            for it in tmp:
                it.remove_short_traces(min_length_sec, min_gap_sec, min_single_length_sec=min_single_length_sec) # this will modify content within tmp
            self.__remove_empty_stream__()
        ####
        if verbose_print_func:
            msg2 = self.to_stream().__str__(extended=True)
        if verbose_print_func and msg1 != msg2: # verbose
            verbose_print_func("StationStreams Before fixing time segments", msg1 )
            verbose_print_func("StationStreams After fixing time segments", msg2 )
    def trim(self, starttime, endtime, pad=False, fill_value=None):
        """
        Trim the time windows for all the channel streams.
        """
        for v in self.values():
            v.trim(starttime, endtime, pad=pad, fill_value=fill_value)
        self.__remove_empty_stream__()
    def merge(self, fill_value=None):
        """
        Merge the time windows for all the channel streams
        """
        for v in self.values():
            v.merge(fill_value=fill_value)
        self.__remove_empty_stream__()
    def to_stream(self, deepcopy=False):
        """
        Convert an object of Stream for all the traces. The object is return.
        NOTE! Shallow copy is used here.
        """
        st = Stream()
        for v in self.values():
            st.extend(v)
        if deepcopy:
            st = st.copy()
        return st
    def is_zne_z12(self):
        """
        Check if the channel streams are for three components ZNE, or Z12.
        """
        chs = set( [it[-1] for it in self.keys()] )
        return (len(self) == 3 and (chs == set('ZNE') or chs == set('Z12') ) )
    def is_ne_12(self):
        """
        Check if the channel stream are for two components NE or 12.
        """
        chs = set( [it[-1] for it in self.keys()] )
        return (len(self) == 2 and (chs == set('NE') or chs == set('12') ) )
    def __remove_empty_stream__(self):
        """
        Pop the key for the empty stream
        """
        ks = [k for k, v in self.items() if len(v)<=0]
        for k in ks:
            self.pop(k)
    def __str__(self):
        s = '\n'.join( ['%s: %s' % (k, v.__str__().replace('\n', '\n    ') ) for k, v in self.items()] )
        return s
class EventRecords(dict):    #A dict for all StationStreams for the same event (net.sta.loc-->StationStreams)
    """
    A dict for grouping all traces w.r.t. different stations.
    The dict key is the station name, and the value is an object of StRecords.
    The station name is in the format of `net.sta.loc`.
    """
    def __init__(self, st=None):
        """
        st: an object of obspy.Stream, or a list of obspy.Trace objects.
        """
        self.__func_init(st)
    def from_file(self, mseed_fnm):
        """
        mseed_fnm: the filename of the mseed file. Wildcard is supported.
        """
        self.__func_init( obspy_read(mseed_fnm) )
    def __func_init(self, st=None):
        self.clear()
        if st:
            keys = ['.'.join( (tr.stats.network,tr.stats.station, tr.stats.location) ) for tr in st]
            tmp  = {k:Stream() for k in set(keys)}
            for k, tr in zip(keys, st):
                tmp[k].append(tr)
            for k, v in tmp.items():
                self[k] = StationStreams(v)
    def fix_segment_times(self, min_length_sec, min_gap_sec=0, min_single_length_sec=0, verbose_print_func=None, time_summary=None):
        """
        Fix the time windows at each station, so that all channels at each station have the same time windows.
        That means,  at one station, the time windows at each channel are the intersection of all channels.
        """
        for v in self.values():
            v.fix_segment_times(min_length_sec, min_gap_sec, min_single_length_sec=min_single_length_sec,
                                verbose_print_func=verbose_print_func,
                                time_summary=time_summary)
        self.__remove_empty_station()
    def trim(self, starttime, endtime, pad=False, fill_value=None):
        """
        Trim all traces w.r.t. the time windows
        """
        for v in self.values():
            v.trim(starttime, endtime, pad=pad, fill_value=fill_value)
        self.__remove_empty_station()
    def merge(self, fill_value=None):
        """
        Merge all traces at each channel of each station
        """
        for v in self.values():
            v.merge(fill_value=fill_value)
        self.__remove_empty_station()
    def to_stream(self, deepcopy=False):
        """
        Convert to an object of Stream for all the traces. The object is returned.
        NOTE! Shallow copy is used in default.
        """
        st = Stream()
        for v in self.values():
            st.extend( v.to_stream() )
        if deepcopy:
            st = st.copy()
        return st
    def __remove_empty_station(self):
        """
        Pop the key for the empty station
        """
        for v in self.values():
            v.__remove_empty_stream__()
        ####
        ks = [k for k, v in self.items() if len(v)<=0]
        for k in ks:
            self.pop(k)
    def __str__(self):
        keys = sorted(self.keys())
        s = ''
        for k in keys:
            s = s + '%s\n%s\n' % (k, self[k].__str__() )
        return s
# Organize and align traces for the same event, and convert to hdf5
def event_mseed2h5(input, h5_fnm, inventory,
                   sampling_interval, freq_band, starttime, endtime,
                   time_summary,
                   channels='ZNE', evlo=None, evla=None,
                   min_length_sec=0, min_gap_sec=0, min_single_length_sec=0, verbose_print_func=None):
    """
    Pre-process mseed file, and convert to h5 file.
    #
    The generate h5 file will has datasets:
        /dat: the data matrix
        /hdr: the metadata group
        /hdr/az
        /hdr/baz
        /hdr/dist
        /hdr/evla
        /hdr/evlo
        /hdr/stdp
        /hdr/stla
        /hdr/stlo
        /hdr/stnm
    and metadata
        fid.attrs['nsta']
        fid.attrs['nch']
        fid.attrs['nt']
        fid.attrs['channels']
        fid.attrs['sampling_interval']
        fid.attrs['freq_band']
        fid.attrs['starttime']
        fid.attrs['endtime']
    #
    Will:
    (0) trim, rmean, taper, bandpass, and resample.
        `starttime, endtime`
        `sampling_interval, freq_band`
    (1) fix the time windows, so that all channels at the same station have the same time windows;
    (2) (optional) remove short traces, gap;
        `min_length_sec, min_gap_sec`
    (3) merge traces at the same channel to a single trace
    (4) pad zeros to make sure all the traces cover starttime to endtime
    (5) rotate the traces to specified channels and select the specific channels.
        `channels` could be `ZNE, NE, Z, N, E`.
    """
    if type(input) == str:
        st = obspy_read(input)
    elif type(input) == Stream:
        st = input
    else:
        raise ValueError('The input should be either a string or an object of Stream! %s' % str(input) )
    #### (0) trim, rmean, taper, bandpass, and resample.
    with Timer(tag='trim_rtr_bp_resample', verbose=False, summary=time_summary):
        starttime = UTCDateTime(starttime)
        endtime   = UTCDateTime(endtime)
        st = Stream(traces=[tr for tr in st if ( len(tr.data)>=3 and (tr.stats.endtime-tr.stats.starttime)>=min_single_length_sec ) ]  )
        st.trim(starttime, endtime)
        st = Stream(traces=[tr for tr in st if ( len(tr.data)>=3 and (tr.stats.endtime-tr.stats.starttime)>=min_single_length_sec ) ]  )
        st.detrend()
        #st.taper(max_percentage=0.005, max_length=100.0)
        st.filter('bandpass', freqmin=freq_band[0], freqmax=freq_band[1], corners=4, zerophase=True)
        st.interpolate(1.0/sampling_interval)
    #
    #### (1) fix the time windows, so that all channels at the same station have the same time windows;
    #### (2) (optional) remove short traces, gap;
    vol = EventRecords(st)
    vol.fix_segment_times(min_length_sec=min_length_sec, min_gap_sec=min_gap_sec, min_single_length_sec=min_single_length_sec,
                          verbose_print_func=None,
                          time_summary=time_summary)
    verbose_print_func('Finish fixing time segments for all stations')
    #verbose_print_func(vol)
    #
    #### (3) merge traces at the same channel to a single trace;
    #### (4) pad zeros to make sure all the traces cover starttime to endtime
    with Timer(tag='merge_pad_zeros', verbose=False, summary=time_summary):
        vol.merge(fill_value=0.0)
        vol.trim(starttime, endtime, pad=True, fill_value=0.0)
        verbose_print_func('Finish merge and trim')
    #
    #### (5) rotate the traces to specified channels and select the specific channels.
    ####     `channels` could be `ZNE, NE, Z, N, E`.
    with Timer(tag='rotate_ZNE', verbose=False, summary=time_summary):
        lst_st = list()
        lst_strec = [vol[k] for k in sorted(vol.keys()) ]
        if channels in ('ZNE', 'NE', 'N', 'E'): # rotate to ZNE is necessary
            tmp = [it.to_stream() for it in lst_strec if it.is_zne_z12()]
            for it in tmp:
                msg = it.__str__(extended=True)
                try:
                    it.rotate( '->ZNE', inventory=inventory)
                    lst_st.append(it)
                except Exception as err: # rotation could fail due to the lack of inventory
                    print('<<<<<<<<<<<<<<<<<<', file=sys.stderr)
                    print('Failed to rotate to ZNE for %s' % msg, file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    print(err, file=sys.stderr)
                    print('>>>>>>>>>>>>>>>>>>', file=sys.stderr)
    #### search for stlo, stla, stel, stdp for each station
    with Timer(tag='get_station_coords', verbose=False, summary=time_summary):
        ids = [it[0].get_id() for it in lst_st]
        stnms = ['.'.join( it.split('.')[:-1] ) for it in ids ]
        st_coords = [inventory.get_coordinates(it[0].get_id(), it[0].stats.starttime) for it in lst_st]
        stlas = np.array([it['latitude']    for it in st_coords], dtype=np.float64)
        stlos = np.array([it['longitude']   for it in st_coords], dtype=np.float64)
        stels = np.array([it['elevation']   for it in st_coords], dtype=np.float64)
        stdps = np.array([it['local_depth'] for it in st_coords], dtype=np.float64)
        dists, azs, bazs = None, None, None
        if (evlo is not None) and (evla is not None):
            dists = np.array([haversine(evlo, evla, stlo, stla) for stlo, stla in zip(stlos, stlas)], dtype=np.float64 )
            azs   = np.array([azimuth(evlo, evla, stlo, stla)   for stlo, stla in zip(stlos, stlas)], dtype=np.float64 )
            bazs  = np.array([azimuth(stlo, stla, evlo, evla)   for stlo, stla in zip(stlos, stlas)], dtype=np.float64 )
    #### convert to a matrix
    with Timer(tag='convert_to_mat', verbose=False, summary=time_summary):
        nch = len(channels)
        nt = int( np.ceil((endtime-starttime)/sampling_interval)+1 )
        nrow = len(lst_st) * nch
        mat = np.zeros( (nrow, nt), dtype=np.float32 )
        for ista, st in enumerate(lst_st):
            for ich, ch in enumerate(channels):
                tr = st.select(component=ch)[0]
                data = tr.data
                sz = min(data.size, nt)
                mat[ista*nch+ich][:sz] = data[:sz]
    #### save to h5
    with Timer(tag='write_h5', verbose=False, summary=time_summary):
        fid = h5_File(h5_fnm, 'w')
        fid.attrs['nsta'] = len(lst_st)
        fid.attrs['nch'] = nch
        fid.attrs['nt'] = nt
        fid.attrs['channels'] = channels
        fid.attrs['sampling_interval'] = sampling_interval
        fid.attrs['freq_band'] = freq_band
        fid.attrs['starttime'] = str(starttime)
        fid.attrs['endtime'] = str(endtime)
        #
        fid.create_dataset('dat',  data=mat,   dtype=np.float32)
        #
        grp = fid.create_group('hdr')
        grp.create_dataset('stlo', data=stlos, dtype=np.float64)
        grp.create_dataset('stla', data=stlas, dtype=np.float64)
        grp.create_dataset('stel', data=stels, dtype=np.float64)
        grp.create_dataset('stdp', data=stdps, dtype=np.float64)
        #
        #byte_list = [s.encode("utf-8") for s in stnms]
        grp.create_dataset('stnm', data=stnms)
        if (evlo is not None) and (evla is not None):
            evlos = np.zeros(len(lst_st), dtype=np.float64) + evlo
            evlas = np.zeros(len(lst_st), dtype=np.float64) + evla
            grp.create_dataset('evlo', data=evlos, dtype=np.float64)
            grp.create_dataset('evla', data=evlas, dtype=np.float64)
            #
            grp.create_dataset('dist', data=dists, dtype=np.float64)
            grp.create_dataset('az',   data=azs,   dtype=np.float64)
            grp.create_dataset('baz',  data=bazs,  dtype=np.float64)
        fid.close()
    return len(lst_st), mat.shape[0]

#####################################################################################################################
# For processing station metadata based on obspy.Inventory or obspy.Station objects
#####################################################################################################################
def download_obspy_inventory(pkl_fnm, starttime, endtime, client_nms=None,
                             network="*", station="*", channel="BH?,HH?", level='channel',
                             log_fnm='log.txt', **kwargs):
    """
    Download all available inventory from all FDSN clients that come with ObsPy
    Will return an object
    client_name -> year-> inventory
    #
    clients: a list of client names seperated by comma. `None` in default to download from all clients.
    """
    if os.path.exists(pkl_fnm):
        raise ValueError('The file exists! %s' % pkl_fnm)
    ######################################################################
    if client_nms:
        client_nms = client_nms.split(',')
    else:
        from obspy.clients.fdsn.header import URL_MAPPINGS
        client_nms = sorted( URL_MAPPINGS.keys() )
    ######################################################################
    # we download year by year because of the download limitation of the server
    starttime = UTCDateTime(starttime)
    endtime   = UTCDateTime(endtime)
    y0 = starttime.year + 1
    y1 = endtime.year + 1
    ts = [starttime]
    ts.extend( [UTCDateTime(y, 1, 1) for y in range(y0, y1, 1) ] )
    ts.append( endtime )
    ######################################################################
    vol_dict = dict()
    log_fid = open(log_fnm, 'w')
    y0 = UTCDateTime(starttime).year
    y1 = UTCDateTime(endtime).year + 1
    for client_name in client_nms:
        try:
            client = Client(client_name)
            vol_dict[client_name] = dict()
            for t0, t1 in zip(ts[:-1], ts[1:]):
                time_range_str = '%s---%s' % (str(t0), str(t1) )
                try:
                    inventory = client.get_stations(starttime=t0, endtime=t1,
                                                    network=network, station=station, channel=channel, level=level, **kwargs)
                    vol_dict[client_name][t0.year] = inventory
                    print('Succeed', client_name, time_range_str, len(inventory), file=log_fid, flush=True)
                except Exception as e:
                    print('Failed', client_name, time_range_str, file=log_fid, flush=True)
            if len(vol_dict[client_name]) == 0:
                print('Nothing obtained', client_name, time_range_str, file=log_fid, flush=True)
                vol_dict.pop(client_name)
        except Exception as e:
            print('Failed', client_name, file=log_fid, flush=True)
    ######################################################################
    with open(pkl_fnm, 'wb') as f:
        pickle.dump(vol_dict, f)
    log_fid.close()
def select_inv_given_channels(inv, lst_channels_string=['Z12', 'ZNE']):
    """
    Return a new inventory to make sure each station within the inventory has all the specific channels
    """
    ##################################################################################################
    # e.g., has_all_channels(a_station, 'ZNE')
    def has_all_channels(station, channels_string):
        channels_string = channels_string.upper()
        cs = set( [it.code[-1].upper() for it in station.channels] )
        not_included = [it for it in channels_string if (it not in cs)]
        if len(not_included) > 0:
            return False
        return True
    def has_all_channels2(station, lst_channels_string):
        flag = False
        for channels_string in lst_channels_string:
            if has_all_channels(station, channels_string):
                flag = True
                break
        return flag
    ##################################################################################################
    new_inv = inv.copy()
    for net, new_net in zip(inv.networks, new_inv.networks):
        new_net.stations = [it for it in net if has_all_channels2(it, lst_channels_string) ]
    return new_inv
def flatten_inv_to_stations(inventory, client_name=None):
    """
    Flatten the inventory to a list of station information.
    For each station, the net_code and client_name is added to the station information.
    """
    stas = list()
    for net in inventory.networks:
        net_code = net.code
        stations = net.stations
        for sta in stations:
            sta.net_code = net_code
            sta.client_name = client_name
        stas.extend( stations )
    return stas
def decluster_stations(lst_stations, approximate_lo_dif=2, approximate_la_dif=2):
    """
    Decluster a list of stations.
    lst_stations: a list of obspy.Station objects.
    approximate_lo_dif, approximate_la_dif: the approximate longitude and latitude grid step to decluster the stations.
    """
    stlos = np.array([it.longitude for it in lst_stations] )
    stlas = np.array([it.latitude  for it in lst_stations] )
    idx_clusters = decluster_spherical_pts(stlos, stlas, approximate_lo_dif, approximate_la_dif)
    ####
    clusters = dict()
    for key, idxs in idx_clusters.items():
        idxs = sorted(idxs)
        clusters[key] = [lst_stations[idx] for idx in idxs]
    return clusters
def sort_stations(lst_stations, preferred_client_names=None, preferred_nets='II,IU,AU', removed_duplicated_station=True):
    """
    Sort a list of stations given the preferred client names and networks.
    lst_stations: a list of obspy.Station objects, and each of them should have the attributes of `client_name` and `net_code`.
    preferred_client_names: a list of client names seperated by comma. `None` in default.
    preferred_nets: a list of network codes seperated by comma. `II,IU,AU` in default.
    removed_duplicated_station: a boolean, if True, then remove the duplicated station objects which have the same net and station code
    """
    def func(stations, prefered_single_client=None, preferred_single_net=None):
        """
        Return a list of indexes of stations that match the prefered_single_client and preferred_single_net.
        """
        preferred_idxs = list(range(len(stations)) )
        if prefered_single_client:
            preferred_idxs = [idx for idx in preferred_idxs if stations[idx].client_name == prefered_single_client]
        if preferred_single_net:
            preferred_idxs = [idx for idx in preferred_idxs if stations[idx].net_code == preferred_single_net]
        return preferred_idxs
    ####
    preferred_client_names = preferred_client_names.split(',') if preferred_client_names else [None, ]
    preferred_nets = preferred_nets.split(',') if preferred_nets else [None, ]
    ####
    preferred_idxs = list()
    for single_client in preferred_client_names:
        for single_net in preferred_nets:
            preferred_idxs.extend( func(lst_stations, single_client, single_net) )
    ####
    preferred_stations = [lst_stations[idx] for idx in preferred_idxs]
    other_stations = [it for idx, it in enumerate(lst_stations) if idx not in preferred_idxs]
    sorted_stations = preferred_stations
    sorted_stations.extend(other_stations)
    ####
    if removed_duplicated_station:
        tmp = list()
        net_sta_set = set()
        for sta in sorted_stations:
            v = sta.net_code, sta.code
            if v not in net_sta_set:
                net_sta_set.add(v)
                tmp.append(sta)
        sorted_stations = tmp
    return sorted_stations

if __name__ == '__main__':
    if False:
        download_obspy_inventory('junk1.pkl', starttime='2000-01-01', endtime='2004-01-01',
                             network="*", station="A*", channel="LH?,BH?,HH?", level='station',
                             log_fnm='log1.txt')
        download_obspy_inventory('junk2.pkl', starttime='2000-01-01', endtime='2004-01-01',
                             network="*", station="A*", channel="LH?,BH?,HH?", level='channel',
                             log_fnm='log2.txt')
    if False:
        wnd1 = TimeWindow(UTCDateTime("2021-01-01T00:00:00"), UTCDateTime("2021-01-01T01:00:00") )
        t0, t1 = wnd1
        print(t0, t1)
        print(wnd1)
    if False:
        wnd1 = TimeWindow(UTCDateTime("2021-01-01T00:00:00"), UTCDateTime("2021-01-01T01:00:00") )
        print(wnd1)
        wnd1 += 100
        print(wnd1)
        wnd1 -= 100
        print(wnd1)
        wnd2 = wnd1 + 3600
        wnd3 = wnd1 - 3600
        print(wnd1)
        print(wnd2)
        print(wnd3)
        print(wnd1 <= wnd1 )
        print(wnd1 <= wnd2 )
        print(wnd1 <= wnd3 )
        ###
        wnds1 = TimeWindows([wnd1, wnd2, wnd2+5000])
        wnds2 = TimeWindows([wnd3, wnd2, wnd2+6000] ) #, wnd3+7200])
        wnds3 = wnds1.intersect(wnds2)
        print(wnds1)
        print(wnds2)
        print(wnds3)
        ax = plt.subplot(111)
        wnds1.plot(ax, y=0, color='r', linewidth=15, alpha=0.6)
        wnds2.plot(ax, y=1, color='b', linewidth=15, alpha=0.6)
        wnds3.plot(ax, y=2, color='g', linewidth=15, alpha=0.6)
        plt.show()
    if False:
        BreqFast.test_run()
    if False:
        starttime, endtime = UTCDateTime(2021, 8, 20, 0, 0, 0), UTCDateTime(2021, 8, 22, 0, 0, 0)
        #
    if False:
        import os.path
        from obspy.core.utcdatetime import UTCDateTime
        from obspy.clients.fdsn.client import Client
        pkl_fnm = 'junk_catalog.pkl'
        if not os.path.exists(pkl_fnm):
            client = Client('USGS')
            starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
            endtime   = UTCDateTime(2021, 9, 1, 0, 0, 0)
            catalog = client.get_events(starttime, endtime, mindepth=25, minmagnitude=6.5) #, includeallorigins=True)
            with open(pkl_fnm, 'wb') as fid:
                pickle.dump(catalog, fid)
        else:
            with open(pkl_fnm, 'rb') as fid:
                catalog = pickle.load(fid)
            print(catalog, '\n', len(catalog) )
        pre_time, tail_time = -10, 100
        #
        #app = BreqFast(email="seisdata_sss_www@163.com", label='exam-events-data-new1', 
        #               name='Sheng_Wang', institute='ANU', mail_address='142 Mills Road ACT Australia 2601', phone='555 5555 555', fax='555 5555 555', hypo=None,
        #               sender_email='seisdata_sss_www@163.com', sender_passwd='WSNXGQUBFUWLSSBK', sender_host='smtp.163.com', sender_port=25 )
        #app.earthquake_run('202101-202106.txt', catalog, pre_time, tail_time, ('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'), location_identifiers=('00', '10'), request_dataless=False, request_miniseed=True)

        app = Waveforms()
        app.test_run()
    
        #
    pass

