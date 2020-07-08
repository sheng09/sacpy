#!/usr/bin/env bash

> cScP.txt
for it in $(seq 0 20)
do
    echo ${it}
    /home/catfly/Program_Sheng/sacpy/bin/cc_feature_time.py -F PcS-PcP -D ${it} -E -50/0 >> cScP.txt
done
