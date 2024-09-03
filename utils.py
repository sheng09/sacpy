#!/usr/bin/env python3

from time import perf_counter as compute_time
from functools import wraps
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

from numpy.random import randint
import time
from smtplib import SMTP, SMTP_SSL
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
import requests
import re
import wget
import rangeparser, types
import copy

def deprecated_run(message='will be deprecated soon!'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Do something with arg1 and arg2 before calling the function
            print('%s() %s' % (func.__name__, message ))
            # Call the original function
            result = func(*args, **kwargs)
            # Do something with the result or after calling the function
            return result
        return wrapper
    return decorator
class TimeSummary(OrderedDict):
    """
    This is used for as an argument for `Timer`, which is for
    summarizing the time consumption of selected operations.
    Also, it contains methods for plotting the time consumption.
    Check the Timer for usage.
    """
    def __init__(self, accumulative=False):
        """
        accumulative: True or False. If True, the time consumption from the same tag will be summed together.
        """
        self.accumulative = 1 if accumulative else 0
    def id(self):
        return len(self)+1
    def push(self, tag, time_ms, color):
        funcs = [self.__noncum_push, self.__cum_push]
        return funcs[self.accumulative](tag, time_ms, color)
    def __cum_push(self, tag, time_ms, color):
        """
        tag: a string
        time_ms: time consumption in miliseconds
        color:   string of color hex value. (default '#999999')
        """
        if tag not in self:
            self[tag] = { 'tag': tag, 'color': color, 'time_ms': 0.0}
        self[tag]['time_ms'] += time_ms
    def __noncum_push(self, tag, time_ms, color):
        """
        tag: a string
        time_ms: time consumption in miliseconds
        color:   string of color hex value. (default '#999999')
        """
        self[ self.id() ] = { 'tag': tag, 'color': color, 'time_ms': time_ms}
    def total_t(self):
        """
        Return the total time consumption in miliseconds.
        """
        total_t = 0.0
        for vol in self.values():
            total_t = total_t + vol['time_ms']
        return total_t
    def plot_rough(self, file=sys.stdout, prefix_str=''):
        """
        Plot a rough histogram of time consumptions using pure text printing.
        #
        file: where to print. (default:sys.stdout)
        prefix_each_line: the prefix string for each line. (default: '')
        """
        total_t = self.total_t()
        plot_lines = ['%s+--------------------+' % (prefix_str) ]
        for id, vol in self.items():
            t_percentage = vol['time_ms']/total_t * 100.0 if total_t>0.0 else 0.0
            t_bin = '*'*int(t_percentage/5)
            #line = '%s%-10s :%-10s %5.2f%%|%-25s|\n' % (prefix_str, vol['tag'][:10], TimeSummary.pretty_time(vol['time_ms']), t_percentage, t_bin)
            line = '%s|%-20s|%5.2f%% %-8s %s' % (prefix_str, t_bin, t_percentage, TimeSummary.pretty_time(vol['time_ms']), vol['tag'])
            plot_lines.append(line)
        plot_lines.append( '%s+--------------------+' % (prefix_str) )
        plot_lines.append( '%sTotal: %s' % (prefix_str, TimeSummary.pretty_time(total_t)) )
        #plot_lines.append('%s%-10s :%s' % (prefix_str, 'Total', TimeSummary.pretty_time(total_t)) )
        histogram = '\n'.join(plot_lines)
        print( histogram, file=file )
    def plot(self, figname=None, show=False, plot_percentage=True):
        """
        Plot histogram of time consumptions, and pie plot (optional).
        #
        figname: where to save the figure. (default: None)
        show:    True or False to show the figure. (default: False)
        plot_percentage: True or False to plot the pie plot (percentage of time consumption). (default: True)
        """
        if plot_percentage:
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 8) )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 8) )
        labels = [vol['tag'] for id, vol in self.items()]
        ts = np.array( [vol['time_ms'] for id, vol in self.items()] )
        colors = np.array( [vol['color'] for id, vol in self.items()] )
        total_t = np.sum(ts)
        percentages = ts/total_t*100.0
        ys = percentages if plot_percentage else ts
        locs = -1*np.arange(ts.size, dtype=np.int64)
        if plot_percentage:
            texts = ['%s\n%.2f%%' % (TimeSummary.pretty_time(t), perc) for t, perc in zip(ts, percentages)]
        else:
            texts = ['%s' % TimeSummary.pretty_time(t) for t, perc in zip(ts, percentages)]
        for loc, text, y, clr in zip(locs, texts, ys, colors):
            ax.barh(loc, y, color=clr)
            ax.text(y*1.01, loc, text, horizontalalignment='left', verticalalignment='center' )
        ax.set_yticks(locs)
        ax.set_yticklabels(labels)
        ax.set_xlim((0, ys.max()*1.25 ))
        ax.set_xlabel('Time Consumption (ms)')
        ax.set_title('Time Consumption Summary')

        if plot_percentage:
            wedges, texts, autotexts =  ax2.pie(percentages, colors=colors, autopct=lambda pct: '%.2f%%' % pct)
            ax2.set_title('Total: %s' % TimeSummary.pretty_time(total_t) )
            ax.legend(wedges, labels, ncol=2, loc=(1.1, 0.0) )
        if figname:
            plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0, dpi=300, transparent=True)
        if show:
            plt.show()
        plt.close()
    def __add__(self, other):
        """
        Merge two TimeSummary objects. The time consumptions with the same key or tag (if accumulative is on) will be summed together, and the others will be kept.
        Return a new TimeSummary object. The returned object will use the accumulative and color setting of the first object.
        """
        if not isinstance(other, TimeSummary):
            raise ValueError('Invalid add between an object of TimeSummary and %s' % str(type(other)) )
        result = TimeSummary(accumulative=self.accumulative)
        ###### copy and preserve the order of self
        for id, vol in self.items():
            result[id] = copy.deepcopy(vol)
        ######
        result += other
        return result
    def __iadd__(self, other):
        """

        """
        if not isinstance(other, TimeSummary):
            raise ValueError('Invalid add(+=) an object of %s to object of TimeSummary' % str(type(other)) )
        for id, vol in other.items():
            if id in self:
                self[id]['time_ms'] += vol['time_ms']
            else:
                self[id] =  copy.deepcopy(vol)
        return self
    def __str__(self):
        result  = """################ Marked Time Consumption Summary ################\n"""
        for id, vol in self.items():
            tmp1 = '#%s\n' % (id)
            tmp2 = ''.join( ['#    %-7s: %s\n' % (key, value) for key, value in vol.items()] )
            result += '%s%s' % (tmp1, tmp2)
        result += """#################################################################"""
        return result
    @staticmethod
    def pretty_time(t_ms):
        """
        Convert time in miliseconds to a pretty string in miliseconds, seconds, minutes, hours, or days when appopriate.
        """
        if t_ms < 100:
            return '%.1fms' % t_ms # ms
        elif t_ms < 60000:
            return '%.1fs' % (t_ms/1000.0) # sec
        elif t_ms < 3600000:
            return '%.1fm' % (t_ms/60000.0) # minute
        elif t_ms < 86400000:
            return '%.1fh' % (t_ms/3600000.0) # hour
        elif t_ms < 31536000000:
            return '%.1fd' % (t_ms/86400000.0) # days
        else:
            return '~%.1fy' % (t_ms/31536000000.0) # rough year
class Timer:
    """
    This is for marking the time consumption of selected operations.
    #
    Example:
        >>> # Initialize a TimeSummary object
        >>> time_summary = TimeSummary()
        >>> #
        >>> # Use the Timer to mark the time consumption
        >>> with Timer(tag='test1', summary=time_summary):
        >>>     a = 1
        >>>     b = 2
        >>>     c = a + b
        >>> #
        >>> # Use the Timer as a decorator
        >>> @Timer(summary=time_summary)
        >>> def sub(a, b):
        >>>     return a-b
        >>> sub(1, 2)
        >>> #
        >>> print(time_summary) # Print the summary
        >>> time_summary.plot_rough() # Plot a rough histogram in terminal
        >>> time_summary.plot(show=False, figname='time_summary.png') # Plot time consumption summary and save it to a file
    """
    color_index = 0
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    def __init__(self, tag=None, color=None, verbose=True, file=sys.stdout, summary=None):
        """
        tag:
        color:
        verbose:
        file:
        summary: an object of OrderedDict
        """
        if summary != None:
            if summary.accumulative:
                if tag in summary:
                    color = summary[tag]['color']
        if color == None:
            color = Timer.colors[Timer.color_index]
            Timer.color_index = (Timer.color_index+1) % len(Timer.colors)
        self.tag = tag
        self.color = color
        self.verbose = verbose
        self.file = file
        self.summary = summary
    def __enter__(self):
        self.__do_enter()
    def __exit__(self, *exc_args): #def __exit__(self, exc_type, exc_value, traceback):
        self.__do_exit()
    def __do_enter(self):
        self.starttime = compute_time()
    def __do_exit(self):
        t_ms = (compute_time()-self.starttime)*1000.0
        if self.verbose:
            line = '%s: %f ms' % (self.tag, t_ms)
            print(line, file=self.file)
        if self.summary != None:
            try:
                self.summary.push(tag=self.tag, time_ms=t_ms, color=self.color)
            except Exception as err:
                print('Err: Timer(...summary=...). The `summary` must be an object of `TimeSummary`', self.summary, file=sys.stderr)
                raise
    def __call__(self, func):
        if not self.tag:
            self.tag = 'call %s(...)' % func.__name__
        @wraps(func)
        def __wrapper(*args, **kwargs):
            self.__do_enter()
            result =  func(*args, **kwargs)
            self.__do_exit()
            return result
        return __wrapper

class CacheRun:
    """
    This is used for caching runtime data in order to accelerating multiple calls.
    For example, if calling a functions requires a heavy computation of a dataset and the dataset
    can be used for next calling of a function, then we can cache the dataset in the first calling
    and use the cached dataset in the next calling. This will avoid multiple heavy computations of
    the same dataset.
    In seismology, a example could be the computations of travel-time curves for seismic phases, which
    can be cached for the next calling of the same phase.
    #
    Example:
        >>> from obspy.taup import TauPyModel
        >>> import numpy as np
        >>> #
        >>> # Define a function that will be cached
        >>> @CacheRun('local_travel_times.h5', clear=True)
        >>> def get_travel_time_curves(model_name, phase_name, evdp_km):
        >>>     # the `get_travel_time_curves(...)` after decoration will have two additional methods:
        >>>     #     `get_travel_time_curves.load_from_cache(key_name)`
        >>>     #     and
        >>>     #     `get_travel_time_curves.dump_to_cache(key_name, data_dict, attrs_dict)`
        >>>     #
        >>>     # We first try to load the data from cache. It will return None if the data is not in cache.
        >>>     key_name = '%s_%s_%d' % (model_name, phase_name, int(evdp_km*1000) )
        >>>     tmp = get_travel_time_curves.load_from_cache(key_name)
        >>>     if tmp != None:
        >>>         xs = tmp['xs']
        >>>         ts = tmp['ts']
        >>>         ps = tmp['ps']
        >>>         return xs, ts, ps
        >>>     else: # Else, we compute the data and cache it.
        >>>         mod = TauPyModel(model_name)
        >>>         distances = np.arange(0, 180, 2)
        >>>         xs, ts, ps = [], [], []
        >>>         for x in distances:
        >>>             arrs = mod.get_travel_times(source_depth_in_km=evdp_km, distance_in_degree=x, phase_list=[phase_name] )
        >>>             xs.extend( [x for it in arrs] )
        >>>             ts.extend( [it.time for it in arrs] )
        >>>             ps.extend( [it.ray_param for it in arrs] )
        >>>         xs = np.array(xs)
        >>>         ts = np.array(ts)
        >>>         ps = np.array(ps)
        >>>         idxs = np.argsort(ps)
        >>>         xs = xs[idxs]
        >>>         ts = ts[idxs]
        >>>         ps = ps[idxs]
        >>>         # Cache the data, with a unique key_name and the data_dict and attrs_dict(optional, in default is None).
        >>>         get_travel_time_curves.dump_to_cache(key_name, data_dict={'xs': xs, 'ts': ts, 'ps': ps}, attrs_dict=None )
        >>> #
        >>> # Test the time function for the first and second calling.
        >>> with Timer(tag='1st run', verbose=True, summary=None):
        >>>     get_travel_time_curves('ak135', 'P', 100)
        >>> with Timer(tag='2nd run', verbose=True, summary=None):
        >>>     get_travel_time_curves('ak135', 'P', 100)
    #
    Please note. This CacheRun as a decorator with be constructed for once no matter how
    many times the function that is decoratored will be called.
    """
    def __init__(self, h5_filename=None, clear=False):
        """
        h5_filename: Use an additional hdf5 file in the disk to save files.
                     The memory cache will still be used.
        clear: clear the h5 file.
        """
        try:
            if h5_filename and clear:
                os.remove(h5_filename)
        except Exception:
            pass
        #print('enter CacheRun._init__(...)')
        self.h5cache = None
        if h5_filename:
            try:
                mpi_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
                h5_filename = '%s_mpi_%4d' % (h5_filename, mpi_rank)
            except:
                pass
            self.h5cache = h5py.File(h5_filename, 'a')
        self.cache   = dict()
        #######################################################
        def __load(group_name):
            try:
                cache = self.cache
                if group_name in cache:
                    return cache[group_name]
                h5cache = self.h5cache
                if h5cache:
                    if group_name in h5cache:
                        cache[group_name] = {_k: _v for _k, _v in h5cache[group_name].items()}
                        h5cache[group_name]
                        cache[group_name]['attrs'] = {_k: _v for _k, _v in h5cache[group_name].attrs.items() }
                        return cache[group_name]
            except Exception:
                pass
            return None
        self.load_from_cache = __load
        def __dump(group_name, data_dict, attrs_dict=None):
            h5cache = self.h5cache
            if h5cache:
                if group_name in h5cache:
                    del h5cache[group_name]
                grp = h5cache.create_group(group_name)
                for _k, _v in data_dict.items():
                    if _k != 'attrs':
                        grp.create_dataset(_k, data=_v)
                if attrs_dict:
                    for _k, _v in attrs_dict.items():
                        grp.attrs[_k] = _v
            cache = self.cache
            cache[group_name] = data_dict
            cache[group_name]['attrs'] = attrs_dict
        self.dump_to_cache = __dump
    def __enter__(self):
        return self
        pass
    def __exit__(self, *exc_args):
        self.h5cache.close()
        pass
    def __call__(self, func):
        #func.h5cache = self.h5cache
        #func.cache = self.cache
        ########################################################
        func.load_from_cache = self.load_from_cache
        ########################################################
        func.dump_to_cache = self.dump_to_cache
        ########################################################
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return __wrapper

class Logger:
    """
    Logger that supports MPI.
    """
    prompt_dict = {-1: '>>> ', 0: '',
                   1: ' '*4,   2: ' '*8,   3: ' '*12,  4: ' '*16,
                   5: ' '*20,  6: ' '*24,  7: ' '*28,  8: ' '*32,
                   9: ' '*36, 10: ' '*40, 11: ' '*44, 12: ' '*48, }
    def __init__(self, log_dir='./', mpi_comm=None, rank_only='all'):
        """
        log_dir: where to save the log files.
        mpi_comm: MPI communicator when MPI is used (default: None)
        rank_only: which rank to print the log. Only useful when `mpi_comm` is not None.
                   `None` to a print at all ranks.
                   `0,1,2,5,7` to print at specific ranks

        """
        self.log_fid       = types.new_class("DummyObject", (), {})
        self.log_fid.close = lambda *args, **kwargs: None
        self.log_print     = lambda *args, **kwargs: None
        get_log_fnm = lambda direc, rank: '%s/log_%04d.txt' % (direc,rank)
        if mpi_comm:
            if mpi_comm.Get_rank() == 0:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            mpi_comm.Barrier()
            ####
            if rank_only == 'all':
                self.log_fid   = open(get_log_fnm(log_dir, mpi_comm.Get_rank() ), 'w')
                self.log_print = print
            else:
                p = rangeparser.RangeParser()
                ranks = set( list( p.parse(rank_only) ) )
                if mpi_comm.Get_rank() in ranks:
                    self.log_fid = open(get_log_fnm(log_dir, mpi_comm.Get_rank() ), 'w')
                    self.log_print = print
        else:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.log_fid = open(get_log_fnm(log_dir, 0), 'w')
            self.log_print = print
    def __call__(self, level=0, *args, **kwargs):
        prompt = Logger.prompt_dict[level] if level in Logger.prompt_dict else ''
        self.log_print(prompt, end='', file=self.log_fid, flush=False)
        self.log_print(*args, **kwargs, file=self.log_fid)
    def close(self):
        self.log_fid.close()

def mpi_makedirs(mpi_comm, wd):
    """
    Make a directory if it does not exist.
    """
    mpi_comm.Barrier()
    if mpi_comm.Get_rank() == 0:
        if not os.path.exists(wd):
            os.makedirs(wd)
    mpi_comm.Barrier()
def get_folder(filename, makedir=True, mpi_comm=None):
    """
    Get the folder for hosting a filename, and make the folder if it does not exist and `makedir=True`.

    filename:
    makedir:  True or False
    mpi_comm: MPI communicator, which is only useful in mpi run and `makedir=True`.
    """
    if makedir and mpi_comm:
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank == 0:
            get_folder(filename=filename, makedir=makedir, mpi_comm=None)
        mpi_comm.Barrier()
    ########################################################
    folder = '/'.join(filename.split('/')[:-1])
    if folder and makedir and (not os.path.exists(folder)):
        os.makedirs(folder)
    return folder
def send_email(content, subject, recipient, sender, passwd, host="smtp.163.com", port=465, use_ssl=True):
    """
    Send an email.

    content, subject: string, the content and subject of the email. 
    recipient: email address of the recipient.
    sender:    email address of the sender.
    passwd:    email passwd of the sender.
    host:      host of the sender server.
    port:      port of the sender server.

    use_ssl:   True or False to use SMTP or SMTP_SSL.
    """
    msg = MIMEText(content)
    msg['From'] = sender
    msg['To']   = recipient
    msg['Subject'] = subject
    if use_ssl:
        with SMTP_SSL(host=host, port=port) as smtp_ssl:
            smtp_ssl.login(sender, passwd)
            smtp_ssl.sendmail(sender, [recipient], msg.as_string() )
    else:
        with SMTP(host=host, port=port) as smtp:
            smtp.set_debuglevel(0)
            smtp.ehlo()
            smtp.starttls()
            smtp.login(sender, passwd)
            smtp.sendmail(sender, [recipient], msg.as_string() )
    time.sleep(randint(3, 9) )
def get_http_files(url, re_template_string):
    """
    Return a list of file urls inside a http page by matching a template string.

    url: the url of the http page.
    re_template_string: the template string to select files.
                        e.g., `re_template_string=r'^head.*txt$'` will match any filenames
                        starting with `head` and end with `txt`. `.*` means zero or any number
                        of any characters in the middle.
    """
    template = re.compile(re_template_string)
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    fnms = [node.get('href') for node in soup.find_all('a') ]
    return ['%s/%s' % (url, it)  for it in fnms if template.match(it)]
def wget_http_files(urls, filename_prefix='testtest/s_', overwrite=True, content_disposition=False):
    """
    Use wget to download files from a list of urls.

    urls:
    filename_prefix: filename prefix to save the files.
    overwrite: True or False to overwrite the existing files.
    content_disposition: True or False(default) to use the content disposition to save the files (comparable to wget --content-disposition).
    """
    get_folder(filename_prefix, True)
    for url in urls:
        ofnm = '%s%s'% (filename_prefix, url.split('/')[-1])
        if content_disposition:
            tmp = get_filename_from_url_content_disposition(url)
            if tmp is not None:
                ofnm = '%s%s'% (filename_prefix, tmp)
        print('wget %s ...' % (url)  )
        download_flag = True
        if os.path.exists(ofnm):
            if overwrite:
                print('Will overwrite the file %s' % (ofnm) )
                os.remove(ofnm)
            else:
                print('Jump over the file %s' % (ofnm) )
                download_flag = False
        if download_flag:
            with Timer(tag='Downloaded', color='red', summary=None):
                real_ofnm = wget.download(url, ofnm)
            print('\nSaved to %s\nDone!' % (real_ofnm) )
    pass
def get_filename_from_url_content_disposition(url):
    """
    Get filename from an url for content-disposition
    """
    print(url)
    response = requests.head(url, allow_redirects=True)
    cd = response.headers.get('content-disposition')
    if not cd:
        return None
    fname = cd.split('filename=')[-1]
    if fname[0] == '"' or fname[0] == "'":
        fname = fname[1:-1]
    return fname

if __name__ == '__main__':
    if False:
        t_ms = [10**x+0.1 for x in range(0, 12)]
        for t in t_ms:
            print('%f -> %s' % (t, TimeSummary.pretty_time(t)))
    if False:
        from obspy.taup import TauPyModel
        import numpy as np
        @CacheRun('junkjunkjunk.h5', clear=True)
        def get_travel_time_curves(model_name, phase_name, evdp_km):
            key_name = '%s_%s_%d' % (model_name, phase_name, int(evdp_km*1000) )
            tmp = get_travel_time_curves.load_from_cache(key_name)
            if tmp != None:
                xs = tmp['xs']
                ts = tmp['ts']
                ps = tmp['ps']
                return xs, ts, ps
            else:
                mod = TauPyModel(model_name)
                distances = np.arange(0, 180, 0.5)
                xs, ts, ps = [], [], []
                for x in distances:
                    arrs = mod.get_travel_times(source_depth_in_km=evdp_km, distance_in_degree=x, phase_list=[phase_name] )
                    xs.extend( [x for it in arrs] )
                    ts.extend( [it.time for it in arrs] )
                    ps.extend( [it.ray_param for it in arrs] )
                xs = np.array(xs)
                ts = np.array(ts)
                ps = np.array(ps)
                idxs = np.argsort(ps)
                xs = xs[idxs]
                ts = ts[idxs]
                ps = ps[idxs]
                get_travel_time_curves.dump_to_cache(key_name, {'xs': xs, 'ts': ts, 'ps': ps})
        with Timer(tag='1st run', verbose=True, summary=None):
            get_travel_time_curves('ak135', 'P', 100)
        with Timer(tag='2nd run', verbose=True, summary=None):
            get_travel_time_curves('ak135', 'P', 100)
    if False:
        @deprecated_run(message='This will be deprecated soon since ??????!')
        def somefunc(a, b):
            return a+b
        c = somefunc(1, 3)
        print(c)
    if False:
        urls = get_http_files('http://ds.iris.edu/pub/userdata/Sheng_Wang/', '^exam-.*')
        for it in urls:
            print(it)
        wget_http_files(urls, overwrite=True)
    if False:
        content = 'Test content %d' % randint(0, 99999999)
        subject = 'Test Subject %d' % randint(0, 99999999)
    if False:
        time_summary  = TimeSummary(accumulative=True)
        time_summary2 = TimeSummary(accumulative=True)
        with Timer(tag='part1', summary=time_summary):
            a = 1
            b = 2
            c = a + b

        with Timer(tag='part1', summary=time_summary):
            a = 1
            b = 2
            c = a + b

        with Timer(tag='part1', summary=time_summary2):
            a = 1
            b = 2
            c = a + b
        from numba import jit
        @Timer(summary=time_summary)
        #@jit(nopython=True, nogil=True)
        def sub(a, b):
            return a-b
        sub(1, 2)
        sub(2, 3)
        sub(4, 5)
        sub(1, 2)
        sub(2, 3)
        sub(4, 5)

        #print(time_summary)
        time_summary.plot_rough(prefix_str= '    ')
        #time_summary.plot(show=False, figname='time_summary.png')

        #print('\n\n\n')
        #print(time_summary)
        #print(time_summary2)
        #print(time_summary+time_summary2)
    if False:
        with CacheRun('junk.h5', clear=True) as cache:
            print( cache.load_from_cache('key1') )
            cache.dump_to_cache( 'key2', {'xs': [1,2,3,4,5], 'ys': 'sdfdfadfd'}, {'maker': 'her', 'date': 'first day'} )
            print( cache.load_from_cache('key2') )
            print( cache.load_from_cache('key1') )
        @CacheRun(None, clear=True)
        def add(x, y):
            print( add.load_from_cache('key1') )
            add.dump_to_cache( 'key2', {'xs': x, 'ys': y}, {'maker': 'her', 'date': 'first day'} )
            print( add.load_from_cache('key2') )
            print( add.load_from_cache('key1') )
        add([0, 0, 0], [9, 8, 7])
    if True:
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD.Dup()
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        time_summary = TimeSummary(accumulative=True)
        if mpi_rank != 0:
            mpi_comm.send(time_summary, dest=0, tag=mpi_rank)
        else:
            for i in range(1, mpi_size):
                it = mpi_comm.recv(source=i, tag=i)