#!/usr/bin/env bash

###
#  generate Earth model
###

/home/catfly/Program_Sheng/sacpy/tomo/cxx/fw_generate_mod3d.x --dlon=2.5 --dlat=2.5 \
    --model_o=01_workspace/earth1d.mod.txt --model_o_mod=p \
    --grd_o=01_workspace/earth1d.grd.txt --grd_o_mod=a


/home/catfly/Program_Sheng/sacpy/tomo/cxx/fw_generate_mod3d.x --dlon=2.5 --dlat=2.5 \
    --model_o=01_workspace/plume.mod.txt --model_o_mod=p \
    --grd_o=01_workspace/plume.grd.txt --grd_o_mod=d \
    --plume=30.0/0.0/5.0/660/2850/0.99/0.95
