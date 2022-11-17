#!/usr/bin/env python3
from obspy.core.utcdatetime import UTCDateTime
from numba import jit
from obspy.clients.fdsn.client import Client

from sacpy.utils import send_email

class BreqFast:
    """
    e.g.,
        >>> from obspy.core.utcdatetime import UTCDateTime
        >>> from obspy.clients.fdsn.client import Client
        >>> from obspy.core.utcdatetime import UTCDateTime
        >>> #
        >>> starttime, endtime = UTCDateTime(2021, 8, 21, 0, 0, 0), UTCDateTime(2021, 8, 21, 4, 0, 0)
        >>> #
        >>> app = BreqFast(email='email' label='exam-events-data-new1', name='name', institute='inst', mail_address='where', phone='555 5555 555', fax='555 5555 555', hypo=None, sender_email='email', sender_passwd='pass', sender_host='what', sender_port=25 )
        >>> app.continuous_run('bfc.txt', starttime, endtime, 3600, ('IU.PAB', 'II.ALE', 'AU.MCQ'), location_identifiers=('00', '10') ) )
        >>> app.continuous_run('bfc.txt', starttime, endtime, 3600, ('IU.PAB', 'II.ALE', 'AU.MCQ'), location_identifiers=('00', '10') )
        >>> #
        >>> client = Client('USGS')
        >>> starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
        >>> endtime   = UTCDateTime(2021, 3, 1, 0, 0, 0)
        >>> catalog = client.get_events(starttime, endtime, mindepth=10, minmagnitude=6.5) #, includeallorigins=True)
        >>> catalog = catalog[:2]
        #
        >>> app = BreqFast(email='email' label='exam-events-data-new1', name='name', institute='inst', mail_address='where', phone='555 5555 555', fax='555 5555 555', hypo=None, sender_email='email', sender_passwd='pass', sender_host='what', sender_port=25 )
        >>> app.continuous_run('bfc.txt', starttime, endtime, 3600, ('IU.PAB', 'II.ALE', 'AU.MCQ'), location_identifiers=('00', '10') ) )
        >>> app.earthquake_run('bfe.txt', catalog, -10, 100, ('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'), location_identifiers=('00', '10'), request_dataless=False, request_miniseed=True)

    """
    def __init__(self, email="you@where.com", label='your-label', 
                       name='your_name', institute='your_institute', mail_address='', phone='', fax='', hypo=None,
                       sender_email=None, sender_passwd=None, sender_host=None, sender_port=None ):
        """
        e.g.,:
            >>> hypo={  'ot': an datetime object, 
                        'evlo': 0.0, 'evla': 0.0, 'evdp_km': 0.0, 
                        'mag': 0.0, 'magtype': 'Mw' }, 
        """
        self.email = email
        self.label = label
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
    def continuous_run( self, filename, min_time, max_time, interval_sec, stations, channels=('BH?', 'SH?'), location_identifiers=[], 
                        request_miniseed=True, request_dataless=False):
        """
        e.g., 
            >>> from obspy.core.utcdatetime import UTCDateTime
            >>> starttime, endtime = UTCDateTime(2021, 8, 21, 0, 0, 0), UTCDateTime(2021, 8, 21, 4, 0, 0)
            >>> #
            >>> app = BreqFast(email='email' label='exam-events-data-new1', 
            >>>                name='name', institute='inst', mail_address='where', phone='555 5555 555', fax='555 5555 555', hypo=None,
            >>>                sender_email='email', sender_passwd='pass', sender_host='what', sender_port=25 )
            >>> app.continuous_run('bfc.txt', starttime, endtime, 3600, ('IU.PAB', 'II.ALE', 'AU.MCQ'), location_identifiers=('00', '10') )
            
        """
        tseg = BreqFast.continuous_time_segments(min_time, max_time, interval_sec)
        tab = list()
        for t0, t1 in tseg:
            tab.extend( BreqFast.table_for_time_time_interval(stations, t0, t1, channels=channels, location_identifiers=location_identifiers) )
        tab = sorted(tab)
        return self.__send_email_and_print(filename, tab, request_miniseed, request_dataless)
    def earthquake_run( self, filename, catalog, pretime_sec, posttime_sec, stations, channels=('BH?', 'SH?'), location_identifiers=[], 
                        request_miniseed=True, request_dataless=False):
        """
        catalog: an object returned from `obspy.clients.fdsn.client.Client.get_events(...)`

        e.g.,

            >>> from obspy.core.utcdatetime import UTCDateTime
            >>> from obspy.clients.fdsn.client import Client
            >>> client = Client('USGS')
            >>> starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
            >>> endtime   = UTCDateTime(2021, 3, 1, 0, 0, 0)
            >>> catalog = client.get_events(starttime, endtime, mindepth=10, minmagnitude=6.5) #, includeallorigins=True)
            >>> catalog = catalog[:2]
            >>> #
            >>> app = BreqFast(email='email' label='exam-events-data-new1', 
            >>>                name='name', institute='inst', mail_address='where', phone='555 5555 555', fax='555 5555 555', hypo=None,
            >>>                sender_email='email', sender_passwd='pass', sender_host='what', sender_port=25 )
            >>> app.earthquake_run('bfe.txt', catalog, -10, 100, ('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'), location_identifiers=('00', '10'), request_dataless=False,    >>> request_miniseed=True)
        """
        t_segs = BreqFast.earthquake_time_segments(catalog, pretime_sec, posttime_sec)
        tab = list()
        for t0, t1 in t_segs:
            tab.extend( BreqFast.table_for_time_time_interval(stations, t0, t1, channels=channels, location_identifiers=location_identifiers) )
        tab = sorted(tab)
        return self.__send_email_and_print(filename, tab, request_miniseed, request_dataless)

if __name__ == '__main__':
    if True:
        starttime, endtime = UTCDateTime(2021, 8, 20, 0, 0, 0), UTCDateTime(2021, 8, 22, 0, 0, 0)
        #
        app = BreqFast(email="seisdata_sss_www@163.com", label='exam5-continuous-data-new', 
                       name='Sheng_Wang', institute='ANU', mail_address='142 Mills Road ACT Australia 2601', phone='555 5555 555', fax='555 5555 555', hypo=None,
                       sender_email='seisdata_sss_www@163.com', sender_passwd='WSNXGQUBFUWLSSBK', sender_host='smtp.163.com', sender_port=25 )
        app.continuous_run('bfc.txt', starttime, endtime, 3600*100, ('IU.PAB', 'II.ALE', 'AU.MCQ'), channels=('BH?',) )
    if False:
        from obspy.core.utcdatetime import UTCDateTime
        from obspy.clients.fdsn.client import Client
        client = Client('USGS')
        starttime = UTCDateTime(2021, 1, 1, 0, 0, 0)
        endtime   = UTCDateTime(2021, 3, 1, 0, 0, 0)
        catalog = client.get_events(starttime, endtime, mindepth=10, minmagnitude=6.5) #, includeallorigins=True)
        catalog = catalog[:2]
        #
        app = BreqFast(email="seisdata_sss_www@163.com", label='exam-events-data-new1', 
                       name='Sheng_Wang', institute='ANU', mail_address='142 Mills Road ACT Australia 2601', phone='555 5555 555', fax='555 5555 555', hypo=None,
                       sender_email='seisdata_sss_www@163.com', sender_passwd='WSNXGQUBFUWLSSBK', sender_host='smtp.163.com', sender_port=25 )
        app.earthquake_run('bfe.txt', catalog, -10, 100, ('IU.PAB', 'II.ALE', 'AU.WRKA'), channels=('BH?','SH?'), location_identifiers=('00', '10'), request_dataless=False, request_miniseed=True)

    pass

