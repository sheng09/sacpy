Cross-correlation Tomography
============================

[TOC]

# 1. Basic Steps

Cross-correlation tomography includes forward modelling and inversion. The former computes tomographic sensitivity kernel given a background/initial/prior model (either 1-D or 3-D). The latter use the sensitivity kernel to invert velocity structure given time observations/measurements of cross-correlation features. 

# 2. Forward Modelling

The modelling consists of successive steps: 
## 2.1 Parameterization of the Earth structure 

`fw_generate_mod3d.x`

This step creates many 3-D grid points to represent the Earth structure. The grid points are regularly distributed within the longitude-latitude-depth coordinates to accelerated computations. Portions of points can be joined together latter in inversion, and hence don't worry about singularity in parameterization. 

```bash
$ # 01_generate_mod.sh
$ # Create the grid for 1-D model.
$ fw_generate_mod3d.x --dlon=2.5 --dlat=2.5 \
    --model_o=01_workspace/earth1d.mod.txt --model_o_mod=p \
    --grd_o=01_workspace/earth1d.grd.txt --grd_o_mod=p
$ 
$ # Create the grid via adding a plume to 1-D model
$ # This 3-D model will used in section 2.4
$ # The plume centered at (30.0E, 0.0N), with radius of 5.0 degree.
$ # The depth range is 660~2850km. There are -0.01 Vp and -0.05 Vsanomalies.
$ fw_generate_mod3d.x --dlon=2.5 --dlat=2.5 \
    --model_o=01_workspace/plume.mod.txt --model_o_mod=p \
    --grd_o=01_workspace/plume.grd.txt --grd_o_mod=p \
    --plume=30.0/0.0/5.0/660/2850/0.99/0.95
```

There are two outputs: `--model_o=mod.txt` and `--grd_o=grd.txt`. The former has special format, and it will be used in *section 2.3*. The latter that depends on the former is purely a table file. Each line of the table corresponds to a grid point. The line format is `inedx lon lat depth x y z vp vs rho  dvp(1+e) dvs(1+e)`, where the index will be used latter in storing the sensitivities. The grid points can be used in plotting.

<table><tr>
<td> <img src="01_workspace/pics/anim005.png"/> </td>
<td> <img src="01_workspace/pics_plume/anim335.png"/> </td>
</tr></table>

 <img src="" style="zoom:45%;" />



Besides, the model can be parameterized manually although it is tiring. The model file has the format of:

```
+-------------------------+
|ndep nlat nlon           |
|dep1 dep2 dep3 ... depN  |
|lat1 lat2 lat3 ... latM  |
|lon1 lon2 lon3 ... lonL  |
|dvp1 dvp2 dvp3 ... dvpX  |
|dvs1 dvs2 dvs3 ... dvsX  |
+-------------------------+
```
where `X=N*M*L`, and please avoid depth discontinuities, such as `20`, `35`, `210`, `410`, `660`, `2891.5`, `5153.5`km in ak135. Instead, use two depths around both sides of a discontinuity. For example, `19.999` and `20.001` for the `2`0 discontinuity, and `2891.499` and `2891.501` for the `2891.5` discontinuity, etc.

Besides, please note, `dvp` and `dvs` here is `1.0+perturbation`. For example, use `1.05` for `5 percent` fast velocity anomaly, and `0.95` for `5 percent` low velocity anomaly.




## 2.2 Modelling of ray-paths

`raypath.py`

This step forward model ray-paths given many body-waves  and event-receiver geometries. We use `taup` method and `obspy.taup` module based on 1D model.

```bash
$ # Generate event-receiver pairs and seismic phases
$ cd 03_workspace/
$ ./generate_ev_sta.py
$ cd ..
$
$ # 03_generate_raypath.sh
$ # Compute ray-paths
$ raypath.py 03_workspace/ev-sta.txt  03_workspace/raypath.h5 verbose
$
$ # Visualize ray-paths
$ python3 ./05_plot_ray.py
```

Please note, the input file `03_workspace/ev-sta.txt` is a table. Each line describes a body wave with an important ID/index. A pair of ID/index will be used in *section 2.4* to describe a cross-term.
```bash
#id/index evdp evla evlo stla stlo phase_name tag
0 50.000000 20.000000 45.000000 -15.000000 0.000000 PKIKPPKIKPPKIKPPKIKPPKIKPPKIKS I05PKIKS
1 50.000000 20.000000 ... ... 
```

And, the output `03_workspace/raypath.h5` stores ray-paths, and it can be viewed by the software `HDFView`.

## 2.3 Computation of body-wave sensitivity

`fw_traveltime_sensitivity.x`.

This step uses the parameterized model, represented by the 3-D grid points, and the computed ray-paths to compute sensitivity of body waves.

```bash
$ # 04_generate_sensitivity.sh
$ fw_traveltime_sensitivity.x --model3d=p/01_workspace/earth1d.mod.txt \
    --raypath=03_workspace/raypath.h5 --verbose \
    --output_sensitivity=04_workspace/sens.h5 > 04.log
```

The output `--output_sensitivity=04_workspace/sens.h5` describes the sensitivity of each body wave. We use `index-value` to store the sensitivities instead of the whole sensitivity matrix since the matrix is highly sparse. The `index` is the grid point index as described in *section 2.1*.

Besides, check `04.log` for running information, such as correctness or relative errors of the computations.


## 2.4 Computation of cross-term sensitivity

`fw_crossterm_sensitivity.x`

We compute cross-term sensitivities via taking the difference between body-wave sensitivities. 

```bash
$ # Generate random cross-terms
$ cd 06_workspace
$ cd ..
$
$ # 06_cross_term_sensitivity.sh
$ # compute the sensitivity
$ fw_crossterm_sensitivity.x --cc=06_workspace/cc.txt --sens=04_workspace/sens.h5 \
$     -O=06_workspace/cc_sens.h5 -V
```
where the input `--cc=06_workspace/cc.txt` declares cross-terms between body waves. A cross-term has a cc-ID and two body-wave ID. The body-wave  ID/index is described in *section 2.2*.


## 2.5 Modelling of correlation-time given a 3-D model

Until now, we have obtained cross-term sensitivities, and the parameterized model. Based on those, we can  do either modelling of correlation time given velocity anomalies, or inversion of structure given correlation time data-set. This subsection focus on the former.

```bash
$ fw_crossterm_time.py  01_workspace/plume.grd.txt  \
    06_workspace/cc_sens.h5  07_workspace/cc_time.txt
```



## 2.6 Analysis of cS-cP given a single event

<img src="05_workspace/raypaths.png" style="zoom:50%;" />

Body wave ray-paths (colored line) from a single event (red dot) to receivers (green dots). Body waves include `I6, I8,...,I16` and `I5PKIKS, I7PKIKS,....,I15PKIKS`.

![](08_workspace/cScP_inter-dist.png)

Correlation time versus inter-receiver distance.



### (1) Comparison between cross-terms

![](08_workspace/cScP_crossterms.png)
Correlation time versus different cross-terms.

### (2) Different cross-terms

![](08_workspace/cScP_I5PKIKS-I6.png)
Correlation time versus inter-receiver distance for the cross-term `I5PKIKS-I6`.

![](08_workspace/cScP_I7PKIKS-I8.png)
Correlation time versus inter-receiver distance for the cross-term `I7PKIKS-I8`.

![](08_workspace/cScP_I9PKIKS-I10.png)
Correlation time versus inter-receiver distance for the cross-term `I9PKIKS-I10`.

![](08_workspace/cScP_I11PKIKS-I12.png)
Correlation time versus inter-receiver distance for the cross-term `I11PKIKS-I12`.

![](08_workspace/cScP_I13PKIKS-I14.png)
Correlation time versus inter-receiver distance for the cross-term `I15PKIKS-I14`.

![](08_workspace/cScP_I15PKIKS-I16.png)
Correlation time versus inter-receiver distance for the cross-term `I17PKIKS-I16`.

# 3. Inversion

# Generate, Modify, and Output a 3-D model

To generate and output 3-D forward model. 
`fw_generate_mod3d.x --dlon=2.5 --dlat=2.5 --model_i=infnm.txt --model_i_mod=[p|b]
    --model_o=outfnm.txt --model_o_mod=[p|b] --grd_o=outfnm.txt --grd_o_mod=a 
    [--plume=lon/lat/radius/d0/d1/dvp/dvs]  [--plume=lon/lat/radius/d0/d1/dvp/dvs] 
    [--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs] [--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs]
    --smooth=5.0 `

The 3-D model is generated 1) by setting a background model, and then 2) by adding some elements.

    - 1) The background model can be
        a. a built-in 1-D ak-135 model. 
           Use `--dlon=X.X --dlat=X.X`, and the discretization in depth dimension
           is the same as the setting in ak135.
        b. an input 3-D model.
           Use `--model_i=filename --model_i_mod=[p|b]` where `p` represent plain-text 
           format file and `b` binary format. The model file has the format of:
                +-------------------------+
                |ndep nlat nlon           |
                |dep1 dep2 dep3 ... depN  |
                |lat1 lat2 lat3 ... latM  |
                |lon1 lon2 lon3 ... lonL  |
                |dvp1 dvp2 dvp3 ... dvpX  |
                |dvs1 dvs2 dvs3 ... dvsX  |
                +-------------------------+
           If binary format is used, then the datatype should be `int`(int32) and `double`(float64).
           
           Please avoid depth discontinuities, such as 20, 35, 210, 410, 660, 2891.5, 5153.5km in ak135.
           Instead, use two depths around both sides of a discontinuity. For example, 19.999 and 20.001
           for the 20 discontinuity, and 2891.499 and 2891.501 for the 2891.5 discontinuity, etc.
           
           Please note, `dvp` and `dvs` here is 1.0+perturbation. For example, use `1.05` for 5 percent
           fast velocity anomaly, and `0.95` for 5 percent low velocity anomaly.
    
    Elements can be added to the background model by:
        a. cylinder plume.
           Use mutiple `--plume=lon/lat/radius/d0/d1/dvp/dvs` to add plumes.
        b. cube.
           Use mutiple `--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs` to add cubes.
           Please note, `dvp` and `dvs` here is 1.0+perturbation. For example, use `1.05` for 5 percent
           fast velocity anomaly, and `0.95` for 5 percent low velocity anomaly.

Args details:

    --dlon=        :
    --dlat=        :
    --model_i=     : filename for input 3-D model.
    --model_i_mod= : `p` for plain-text format.
                     `b` for binary format.
    --grd_o=       : filename to output 3-D model grid points.
    --grd_o_mod=   : `a` to output all points.
                     `p` to output points with P velocity perturbation being not zero.
                     `s` ...                   S ...
                     `d` ...                   P and S ...
    --model_o=     : filename to output 3-D model.
    --model_o_mod= : `p` for plain-text format.
                     `b` for binary format.
    --plume=lon/lat/radius/d0/d1/dvp/dvs: add a plume.
    --cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs : add a cube.
    --smooth=radius: to smooth the 3-Dmodel in longitude-latitude dimension given a radius in degree.
