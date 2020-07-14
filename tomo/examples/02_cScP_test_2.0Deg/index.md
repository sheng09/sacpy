Tests of Event-Receiver Geometry and Cross-terms
====
[TOC]



# 1. Introduction and Motivations

Here we investigate sensitivity variations of body waves and their cross-terms. The variations are related to 1) inter-receiver distance, 2) event locations, and 3) body-wave types. The investigation will help us to select appropriate geometry and cross-term settings in realistic tomography practices. 

In realistic cases, it is impossible to have an event on optimal location (same ray-parameter location) given a body-wave cross-term. Such mislocation would deny the perfect overlap between two body-wave ray-paths, and hence their cross-term is sensitivity to a global-scale volume. That  makes regional tomography hard. Besides, some body-waves cannot be separated from each other however they have different ray-paths, such as PKIKS and SKIKP. That makes the sensitivities inaccurate. 

However, being inaccurate does not equal to being totally wrong. The inaccuracy can be decreased via proper selections of events, receivers and cross-terms. Through the selection, we can avoid the subsets characterisized by significant variations of sensitivities. Those subsets would significantly bias the tomographic sensitivities if they are included. Besides, the sensitivity inaccuracy can be too small and negligible with specific parameterization of Earth structure. The inaccuracy due to ray-path variations can be smaller than the resolution capability. For example, the ray-path variations smaller than 1 km in latteral direction can be negligible if the Earth structure is parameterized as many 3-D grid points with 2 degree separations in lateral direction.

In the following, we quantitatively investigate the variations/inaccuracy of tomographic sensitivities due to event-receiver geometry and cross-term types. We only focus on cS-cP cross-correlation features, and we look forward to quantitative guidance in practicing realistic cross-correlation tomography.



# 2. Settings in Different Levels

## 2.0 Pre-settings
Here we test an parameterization of 2.0 degree in lateral (longitude and latitude) directions. 

We set a plume model. The plume locates at `30.0E 0.0N`, and is at depth of `660~2850km`, and has the radius of `10 degree`, and presents S wave velocity reductions of `2` per cent. Figure 2.1 shows planform of the plume.

```bash
$ ./01_generate_mod.sh
```



![Fig2.1](02_workspace/geo.png)

*Fig 2.1 Settings of plume, events (blue crosses), receivers (green triangles). The plume is at depth of `660~2850` km. Receivers are grouped into three regional networks.*



![](01_workspace/plume.gif)

*Fig 2.2 Setting of plume, parameterized as 3-D grid points.*

## 2.1 Level 1: Receivers and events

Here we set three networks, as shown in Figure 2.1. The first (middle) network is inside the plume area, and the other two (left and right) are outside plume area. Each network has an aperture of `10` degree of diameter, and consists of `10` receivers with random distributions. That setting mimic the realistic.  We set `10` receivers to ease computation, although there are usually more than `10` receivers in the realistic that definitely bring better tomographic resolution.

In each network, we form receiver pairs between all receivers. We do not allow receiver pairs across different networks. Therefore, we have 50 receiver pairs in each network, that include 10 auto-correlation and 40 cross-correlation between different receivers.

```bash
$ ./02a_prepare_ev-rcv.py 
$ # check the generated `02_workspace/event.txt` and `02_workspace/rcv.txt`
```

## 2.2 Level 2: Body-wave cross-terms

cS-cP feature's formation is dominated by cross-terms between body waves reverberated from the Earth's free surface and the Core-Mantle-Boundary (CMB) (Wang & Tkalcic 2020b). Here, we focus those cross-terms that are `I_{n-1}PKIKS-I_{n}` of which the body waves have traveltime in `1-9` h.  

We generate cross-terms with `02b_prepare_cross-terms.py`. The cross-terms are stored at `02_workspace/cross-terms.txt`.

## 2.3 Level 3: Event locations

Generally speaking, cross-correlation tomography prefer events close to receivers or the antipodal locus of the receivers. Ray-paths from those events concentrate on a volume of relatively small lateral dimension, although the ray-paths bounce from the Earth's free surface or the CMB for many time. If an event is far away from the receivers (e.g., 60 degree), then the ray-paths would sample the whole mantle volume between the event and the receivers, and hence any structure anomalies within the volume would result in cross-terms time shifts. 

Besides, events close to receivers present nearly zero ray parameter. The ray-paths would be nearly vertically. That setting allows good lateral resolution in the tomography when having many receivers and events. However, nearly vertical ray-paths decrease the vertical resolution.

We quantitatively test the critical distance between an event and receivers. First, we compute slowness for different body waves (Figs 2.4-2.7). Second, we compute the slowness and ray-paths for `P` and `S`  legs from the CMB to the Earth's free surface. As shown in Figure 2.3, for nearly vertical `P` or `S` legs (the lateral distance smaller than `5` degree), the slowness should be smaller than `1.0` sec/degree. Being more restrict, we can use the slowness smaller than `0.5` sec/degree. The computation results show that events with distance smaller than `20` degree or greater than `160` degree always present slowness smaller than `0.5` sec/degree. Therefore, we limit the epi-central distance smaller than `20` degree or greater than `160` degree to select events and receivers. 

Fig 2.1 shows the events (blue crosses) based on the selection criteria.

```bash
$ ./02b_prepare_cross-terms.py
$ ./02c_test_PcP.py 
```


<img src="02_workspace/PcP_ScP_slowness.png" style="zoom: 67%;" />

*Fig 2.3 (A) P (black) and S (red) ray-path legs from the CMB to the Earth's free surface for different distances. (B) Slowness of P (black) and S (red) waves with respect to the distance. The blue dash line indicate the distance of `5` degree.*

<img src="02_workspace/cross-term_1-3hr.png" style="zoom: 50%;" />

*Fig 2.4 Body waves in 1-3 h.*

<img src="02_workspace/cross-term_3-5hr.png" style="zoom: 50%;" />

*Fig 2.5 Body waves in 3-5 h.*

<img src="02_workspace/cross-term_5-7hr.png" style="zoom: 50%;" />

*Fig 2.6 Body waves in 5-7 h.*

<img src="02_workspace/cross-term_7-9hr.png" style="zoom: 50%;" />

*Fig 2.7 Body waves in 7-9 h.* 




## 2.4 Issues

Although we select receivers and events, the ray-paths are still sensitivity to structure of volume larger than the plume shown in Figures 2.1-2.2. Any structure anomaly traverserd by ray-paths results in time shift. Besides, the sensitivity value (ray-path segment length) are not comparable between different depth, because the grid separation decrease with depth. Therefore, comparing the sensitivity value in differen depths hardly works in determining where is the sensitive zone. 

Instead, it is more reasonable and direct to test time variations when anomalies are set in different locations, such as in different depth or in different lateral locations.

Nevertheless, it is still necessary to compute the ray-path ans the sensitivities. Those are prerequisites to the computation of synthetic time variations. Therefore, we still use the criteria in this section to set events, receivers and cross-terms.


