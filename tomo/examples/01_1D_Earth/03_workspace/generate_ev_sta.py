#!/usr/bin/env python3
import numpy as np

evdp, evla, evlo = 50, 60, 150
fid = open("ev-sta.txt", "w")
fid.write("#id evdp evla evlo stla stlo phase_name tag\n")
id = 0
for stla in np.arange(-30, -10, 2):
    for stlo in np.arange(120, 140, 2):
        for nI in range(5, 17, 2):
            phase1 = "PKIKP" * nI + "PKIKS"
            phase2 = "PKIKP" * nI + "PKIKP"
            tag1 = "I%02dPKIKS" % nI
            tag2 = "I%02d" % (nI+1)

            fid.write("%d %f %f %f %f %f %s %s\n"% (id, evdp, evla, evlo, stla, stlo, phase1, tag1) )
            id = id + 1

            fid.write("%d %f %f %f %f %f %s %s\n"% (id, evdp, evla, evlo, stla, stlo, phase2, tag2) )
            id = id + 1
fid.close()
