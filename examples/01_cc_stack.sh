#!/usr/bin/env bash
mpirun -np 2 cc_stack_v2.py -I dataset/waveforms_h5/*h5 -T -5/10800/32400 -D 0.1 --input_format h5 -O junk_cc/o --out_format hdf5 --pre_detrend --pre_taper 0.005 --w_temporal 128.0/0.02/0.06667 --w_spec 0.02 --stack_dist 0/180/1 --post_fold --post_taper 0.005 --post_filter bandpass/0.01/0.06667 --post_norm --post_cut 0/5000 --log junk_cc/log --acc 0.01
