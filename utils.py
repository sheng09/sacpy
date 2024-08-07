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

class TimeSummary(OrderedDict):
    def __init__(self):
        pass
    def id(self):
        return len(self)+1
    def push(self, message=None, time_ms=0.0, color='#999999'):
        """
        message: a string
        time_ms: time consumption in miliseconds
        color:   string of color hex value. (default '#999999')
        """
        self[self.id()] = { 'message': message, 'color': color, 'time_ms': time_ms}
    def plot_rough(self, file=sys.stdout):
        """
        Plot a rough histogram of time consumptions using pure text printing.

        file: where to print. (default:sys.stdout)
        """
        total_t = 0.0
        for id, vol in self.items():
            total_t = total_t + vol['time_ms']
        plot_lines = ['\n%-10s :%-10.2f\n' % ('Total(ms)', total_t), ]
        for id, vol in self.items():
            t_percentage = vol['time_ms']/total_t * 100.0
            t_bin = '*'*int(t_percentage/5)
            line = '%-10s :%-10.2f %5.2f%%|%-25s \n' % (vol['message'][:10], vol['time_ms'], t_percentage, t_bin)
            plot_lines.append(line)
        histogram = ''.join(plot_lines)
        print( histogram, file=file )
    def plot(self, figname=None, show=True, plot_percentage=True):
        """
        """
        if plot_percentage:
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 8) )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 8) )
        labels = [vol['message'] for id, vol in self.items()]
        ts = np.array( [vol['time_ms'] for id, vol in self.items()] )
        colors = np.array( [vol['color'] for id, vol in self.items()] )
        total_t = np.sum(ts)
        percentages = ts/total_t*100.0
        ys = percentages if plot_percentage else ts
        locs = np.arange(ts.size, dtype=np.int)
        if plot_percentage:
            texts = ['%.1ems\n%.2f%%' % (t, perc) for t, perc in zip(ts, percentages)]
        else:
            texts = ['%.1ems' % t for t, perc in zip(ts, percentages)]
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
            ax.legend(wedges, labels, ncol=2, loc=(1.1, 0.0) )
        if figname:
            plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0, dpi=300, transparent=True)
        if show:
            plt.show()
        plt.close()
    def __str__(self):
        result  = """################ Marked Time Consumption Summary ################\n"""
        for id, vol in self.items():
            tmp1 = '#%04d call\n' % (id)
            tmp2 = ''.join( ['#    %-7s: %s\n' % (key, value) for key, value in vol.items()] )
            result += '%s%s' % (tmp1, tmp2)
        result += """#################################################################"""
        return result
class Timer:
    def __init__(self, message=None, color='red', verbose=True, file=sys.stdout, summary=None):
        """
        message:
        color:
        verbose:
        file:
        summary: an object of OrderedDict
        """
        self.message = message
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
            line = '%s: %f ms' % (self.message, t_ms)
            print(line, file=self.file)
        if self.summary:
            try:
                self.summary.push(message=self.message, time_ms=t_ms, color=self.color)
            except Exception as err:
                print('Err: Timer(...summary=...). The `summary` must be an object of `TimeSummary`', self.summary, file=sys.stderr)
                raise
    def __call__(self, func):
        if not self.message:
            self.message = 'call %s(...)' % func.__name__
        @wraps(func)
        def __wrapper(*args, **kwargs):
            self.__do_enter()
            result =  func(*args, **kwargs)
            self.__do_exit()
            return result
        return __wrapper

class CacheRun:
    """
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
            with Timer(message='Downloaded', color='red', summary=None):
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
    if True:
        urls = get_http_files('http://ds.iris.edu/pub/userdata/Sheng_Wang/', '^exam-.*')
        for it in urls:
            print(it)
        wget_http_files(urls, overwrite=True)
    if False:
        content = 'Test content %d' % randint(0, 99999999)
        subject = 'Test Subject %d' % randint(0, 99999999)
    if False:
        time_summary = TimeSummary()
        with Timer(message='test1', color='red', summary=None):
            a = 1
            b = 2
            c = a + b

        @Timer(color='blue', summary=time_summary)
        def sub(a, b):
            return a-b
        sub(1, 2)

        print(time_summary)
        time_summary.plot_rough()
        time_summary.plot(show=True)
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
