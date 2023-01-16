#!/usr/bin/env python3
from obspy.core.utcdatetime import UTCDateTime
from numba import jit
from obspy.clients.fdsn.client import Client
from obspy import read, read_inventory
from obspy.clients.fdsn.client import Client
from obspy.core.stream import Stream
from obspy.core.util.attribdict import AttribDict
from obspy.core.inventory.inventory import Inventory
import pickle
from sacpy.utils import send_email, get_http_files, wget_http_files
from h5py import File as h5_File
from numpy import float32, int32, zeros
from numpy import max as np_max
from numpy import min as np_min
from sacpy.geomath import haversine, azimuth

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
        >>> # 2。 Request records for catalog events
        >>> client = Client('USGS')
        >>> starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
        >>> endtime   = UTCDateTime(2021, 3, 1, 0, 0, 0)
        >>> catalog = client.get_events(starttime, endtime, mindepth=10, minmagnitude=6.5) #, includeallorigins=True)
        >>> catalog = catalog[:2]
        >>> app.do_not_send_emails = True # set to False to send emails
        >>> app.earthquake_run('cat.txt', catalog, -10, 100, stations=('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'), label='cat', location_identifiers=('00', '10'), request_dataless=True, request_miniseed=True )
        >>> #
        >>> # 3。 Customize run
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
        Group all the traces within the `self.traces` with respect to network code XXXX
        where XXXX is the network code.

        Return a dictionary. The keys of the dictionary are in the format of XXXX. Each
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
        Group all the traces within the `self.traces` with respect to station name XXXX.YYYY.ZZZZ
        where XXXX is the network code, and YYYY the station code, and ZZZZ the location identifier.

        Return a dictionary. The keys of the dictionary are in the format of XXXX.YYYY.ZZZZ. Each
        value of the dictionary is a list of objects of `Waveforms`. Each element of the list
        contains traces having the same station name `XXXX.YYYY.ZZZZ` while their channels can be
        different.
        """
        stations = ['%s.%s.%s' % (tr.stats.network, tr.stats.station, tr.stats.location)  for tr in self.traces]
        vol = { it:Waveforms() for it in set(stations) }
        for st, tr in zip(stations, self.traces):
            vol[st].append(tr)
        return vol
    def group_location(self):
        """
        Group all the traces within the `self.traces` with respect to location identifier
        ZZZZ where the ZZZZ is the location identifier.

        Return a dictionary. The keys of the dictionary are in the format of ZZZZ. Each
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
        Group all the traces within the `self.traces` with respect to channel name CCCC
        where the CCCC is the station code.

        Return a dictionary. The keys of the dictionary are in the format of CCCC. Each
        value of the dictionary is a list of objects of `Waveforms`. Each element
        of the list contains traces having the same channel name `CCCC` while their
        network codes, station codes, and location identifier can be different.
        """
        channels = [tr.stats.channel for tr in self.traces]
        vol = { it:Waveforms() for it in set(channels) }
        for ch, tr in zip(channels, self.traces):
            vol[ch].append(tr)
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

if __name__ == '__main__':
    if True:
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

