#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import string
###
#  Cross-term related data
###
cc_info = dict()
for line in open('06_workspace/cc.txt', 'r'):
    if line[0] =='#':
        continue
    ccid, id1, id2, tags = line.strip().split()
    ccid, id1, id2 = int(ccid), int(id1), int(id2)
    ### d1(5.76)_d2(49.08)_interd(43.59)_az1(210.24)_az2(227.16)_daz(-16.91)_cc(I13PKIKS-I14)
    tmp = tags.split('_')
    d1     = float(tmp[0].replace('d1(', '').replace(')', '') )
    d2     = float(tmp[1].replace('d2(', '').replace(')', '') )
    interd = float(tmp[2].replace('interd(', '').replace(')', '') )
    az1    = float(tmp[3].replace('az1(', '').replace(')', '') )
    az2    = float(tmp[4].replace('az2(', '').replace(')', '') )
    daz    = float(tmp[5].replace('daz(', '').replace(')', '') )
    name  = tmp[6].replace('cc(', '').replace(')', '') 
    n_rever= int(name[-2:]) # number of reverberations
    #print(ccid, interd, name, n_rever)
    ###

    cc_info[ccid] = {   'id1': id1, 'id2': id2, 'd1': d1, 'd2': d2,
                        'interd': interd, 'az1': az1, 'az2': az2, 'daz': daz, 
                        'name': name, 'n_rever': n_rever  }

###
#  Read theoretical times
###
tmp = np.loadtxt('07_workspace/cc_time.txt', comments='#', dtype={'names':  ('index', 't1d_taup', 't1d',  't3d',  'dt'),
                                                                  'formats':('i4',    'f4',       'f4',   'f4',   'f4') } )

tmp2 = np.loadtxt('08_workspace/cScP.txt')

###
#  Plot inter-receiver distance v.s. correlation-time
#       ...                     v.s. correlation-time shift
###
xs = [cc_info[it]['interd'] for it in tmp['index'] ]
t0 = tmp['t1d']
t  = tmp['t3d']
dt = tmp['dt']


x_c = tmp2[:,0]
t_c = tmp2[:,2]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5) )
ax1.plot(xs, t0, 'k.', alpha= 0.9, label ='Single event')
ax1.plot(x_c, t_c, 'r-', alpha= 0.9, label ='Optimal events')
ax1.set_title('Correlation time in 1-D model')
#ax1.plot(xs, t,  'C0.', alpha= 0.3, markeredgewidth= 0, label = 'Plume model')
ax2.plot(xs, t,  'C0.', alpha= 0.6, label = 'Single event')
ax2.plot(x_c, t_c, 'r-', alpha= 0.9, label ='Optimal events')
ax2.set_title('Correlation time in 3-D plume model')
ax3.plot(xs, dt, 'C0.', alpha= 0.6 )
ax3.set_title('Correlation-time difference')
ax4.hist(dt, np.arange(-25, 10, 1.0), orientation='horizontal', align='mid', edgecolor='black', linewidth=1 )
ax4.set_title('Histogram of correlation-time difference')
ax4.grid(True, linestyle= ':')
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('Inter-receiver distance ($\degree$)')
    ax.set_xlim([0, 20.0] )
for ax in [ax1, ax2]:
    ax.set_ylim([150, 230] )
    ax.set_ylabel('cS-cP correlation time (s)')
    ax.legend(loc='lower right')
for ax in [ax3, ax4]:
    ax.set_ylim([-22, 10])
    ax.set_ylabel('cS-cP correlation-time difference (s)')
ax4.set_xlabel('Numbers of cross-terms')

for n, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20, weight='bold')

plt.tight_layout()
plt.savefig('08_workspace/cScP_inter-dist.png')
plt.close()

###
#  Plot number of reverberations v.s. correlation-time
###
xs = [cc_info[it]['n_rever'] for it in tmp['index'] ]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5) )
ax1.plot(xs, t0, 'k.', alpha= 0.35 )
ax1.set_title('Correlation time in 1-D model')
ax2.plot(xs, t,  'C0.', alpha= 0.6 )
ax2.set_title('Correlation time in 3-D plume model')
ax3.plot(xs, dt, 'C0.', alpha= 0.6 )
ax3.set_title('Correlation-time difference')
ax4.hist(dt, np.arange(-25, 10, 1.0), orientation='horizontal', align='mid', edgecolor='black', linewidth=1 )
ax4.set_title('Histogram of correlation-time difference')
ax4.grid(True, linestyle= ':')
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('Cross-terms')
    ax.set_xticks(np.arange(6, 17, 2) )
    ax.set_xticklabels(['I5PKIKS-I6', 'I7PKIKS-I8', 'I9PKIKS-I10', 'I11PKIKS-I12', 'I13PKIKS-I14', 'I15PKIKS-I16'], rotation= 20)
for ax in [ax1, ax2]:
    ax.set_ylim([185, 225] )
    ax.set_ylabel('cS-cP correlation time (s)')
for ax in [ax3, ax4]:
    ax.set_ylim([-22, 10])
    ax.set_ylabel('cS-cP correlation-time difference (s)')
ax4.set_xlabel('Numbers of cross-terms')

for n, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20, weight='bold')

plt.tight_layout()
plt.savefig('08_workspace/cScP_crossterms.png')
plt.close()

###
#
###
t0 = tmp['t1d']
t  = tmp['t3d']
dt = tmp['dt']
xs = [cc_info[it]['interd'] for it in tmp['index'] ]
ys = [cc_info[it]['n_rever'] for it in tmp['index'] ]

for it_rever, cross_term in zip(np.arange(6, 17, 2), ['I5PKIKS-I6', 'I7PKIKS-I8', 'I9PKIKS-I10', 'I11PKIKS-I12', 'I13PKIKS-I14', 'I15PKIKS-I16']):
    idx_lst = [idx for idx, n_rever in enumerate(ys) if n_rever == it_rever]
    new_xs = [xs[idx] for idx in idx_lst]
    new_t0 = [t0[idx] for idx in idx_lst]
    new_t  = [t[idx]  for idx in idx_lst]
    new_dt = [dt[idx] for idx in idx_lst]



    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5) )
    ax1.plot(new_xs, new_t0, 'k.' , alpha= 0.35, label =cross_term )
    ax1.set_title('Correlation time in 1-D model')
    ax2.plot(new_xs, new_t,  'C0.', alpha= 0.6,  label =cross_term )
    ax2.set_title('Correlation time in 3-D plume model')
    ax3.plot(new_xs, new_dt, 'C0.', alpha= 0.6,  label =cross_term )
    ax3.set_title('Correlation-time difference')
    ax4.hist(new_dt, np.arange(-25, 10, 1.0), orientation='horizontal', align='mid', edgecolor='black', linewidth=1 )
    ax4.set_title('Histogram of correlation-time difference')
    ax4.grid(True, linestyle= ':')
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Inter-receiver distance ($\degree$)')
        ax.set_xlim([0, 20.0] )
    for ax in [ax1, ax2]:
        ax.set_ylim([185, 230] )
        ax.set_ylabel('cS-cP correlation time (s)')
        ax.legend(loc='upper left', prop={'size': 13} )
    ax3.legend(loc='upper left', prop={'size': 13} )
    for ax in [ax3, ax4]:
        ax.set_ylim([-22, 10])
        ax.set_ylabel('cS-cP correlation-time difference (s)')
    ax4.set_xlabel('Numbers of cross-terms')

    for n, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=20, weight='bold')

    plt.tight_layout()
    plt.savefig('08_workspace/cScP_%s.png' % (cross_term) )
    plt.close()