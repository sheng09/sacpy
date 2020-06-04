#include "global_tomo.hh"
#include <cstdio>

//#include <Python.h> 
//#include <numpy/arrayobject.h>
namespace ak135
{
    int nlayer = 127;
    double d0[] = {0.  ,   20.  ,   35.  ,   77.5 ,  120.  ,  165.  ,  210.  ,
        260.  ,  310.  ,  360.  ,  410.  ,  460.  ,  510.  ,  560.  ,
        610.  ,  660.  ,  710.  ,  760.  ,  809.5 ,  859.  ,  908.5 ,
        958.  , 1007.5 , 1057.  , 1106.5 , 1156.  , 1205.5 , 1255.  ,
       1304.5 , 1354.  , 1403.5 , 1453.  , 1502.5 , 1552.  , 1601.5 ,
       1651.  , 1700.5 , 1750.  , 1799.5 , 1849.  , 1898.5 , 1948.  ,
       1997.5 , 2047.  , 2096.5 , 2146.  , 2195.5 , 2245.  , 2294.5 ,
       2344.  , 2393.5 , 2443.  , 2492.5 , 2542.  , 2591.5 , 2640.  ,
       2690.  , 2740.  , 2789.67, 2839.33, 2891.5 , 2939.33, 2989.66,
       3039.99, 3090.32, 3140.66, 3190.99, 3241.32, 3291.65, 3341.98,
       3392.31, 3442.64, 3492.97, 3543.3 , 3593.64, 3643.97, 3694.3 ,
       3744.63, 3794.96, 3845.29, 3895.62, 3945.95, 3996.28, 4046.62,
       4096.95, 4147.28, 4197.61, 4247.94, 4298.27, 4348.6 , 4398.93,
       4449.26, 4499.6 , 4549.93, 4600.26, 4650.59, 4700.92, 4801.58,
       4851.91, 4902.24, 4952.58, 5002.91, 5053.24, 5103.57, 5153.5 ,
       5204.61, 5255.32, 5306.04, 5356.75, 5407.46, 5458.17, 5508.89,
       5559.6 , 5610.31, 5661.02, 5711.74, 5813.16, 5863.87, 5914.59,
       5965.3 , 6016.01, 6066.72, 6117.44, 6168.15, 6218.86, 6269.57,
       6320.29};
    double d1[] = {20.  ,   35.  ,   77.5 ,  120.  ,  165.  ,  210.  ,  260.  ,
        310.  ,  360.  ,  410.  ,  460.  ,  510.  ,  560.  ,  610.  ,
        660.  ,  710.  ,  760.  ,  809.5 ,  859.  ,  908.5 ,  958.  ,
       1007.5 , 1057.  , 1106.5 , 1156.  , 1205.5 , 1255.  , 1304.5 ,
       1354.  , 1403.5 , 1453.  , 1502.5 , 1552.  , 1601.5 , 1651.  ,
       1700.5 , 1750.  , 1799.5 , 1849.  , 1898.5 , 1948.  , 1997.5 ,
       2047.  , 2096.5 , 2146.  , 2195.5 , 2245.  , 2294.5 , 2344.  ,
       2393.5 , 2443.  , 2492.5 , 2542.  , 2591.5 , 2640.  , 2690.  ,
       2740.  , 2789.67, 2839.33, 2891.5 , 2939.33, 2989.66, 3039.99,
       3090.32, 3140.66, 3190.99, 3241.32, 3291.65, 3341.98, 3392.31,
       3442.64, 3492.97, 3543.3 , 3593.64, 3643.97, 3694.3 , 3744.63,
       3794.96, 3845.29, 3895.62, 3945.95, 3996.28, 4046.62, 4096.95,
       4147.28, 4197.61, 4247.94, 4298.27, 4348.6 , 4398.93, 4449.26,
       4499.6 , 4549.93, 4600.26, 4650.59, 4700.92, 4801.58, 4851.91,
       4902.24, 4952.58, 5002.91, 5053.24, 5103.57, 5153.5 , 5204.61,
       5255.32, 5306.04, 5356.75, 5407.46, 5458.17, 5508.89, 5559.6 ,
       5610.31, 5661.02, 5711.74, 5813.16, 5863.87, 5914.59, 5965.3 ,
       6016.01, 6066.72, 6117.44, 6168.15, 6218.86, 6269.57, 6320.29,
       6371.  };
    double p0[] = {5.8   ,  6.5   ,  8.04  ,  8.045 ,  8.05  ,  8.175 ,  8.3   ,
        8.4825,  8.665 ,  8.8475,  9.36  ,  9.528 ,  9.696 ,  9.864 ,
       10.032 , 10.79  , 10.9229, 11.0558, 11.1353, 11.2221, 11.3068,
       11.3896, 11.4705, 11.5495, 11.6269, 11.7026, 11.7766, 11.8491,
       11.92  , 11.9895, 12.0577, 12.1245, 12.1912, 12.255 , 12.3185,
       12.3819, 12.4426, 12.5031, 12.5631, 12.6221, 12.6804, 12.7382,
       12.7956, 12.8526, 12.9096, 12.9668, 13.0222, 13.0783, 13.1336,
       13.1894, 13.2465, 13.3018, 13.3585, 13.4156, 13.4741, 13.5312,
       13.59  , 13.6494, 13.653 , 13.6566,  8.    ,  8.0382,  8.1283,
        8.2213,  8.3122,  8.4001,  8.4861,  8.5692,  8.6496,  8.7283,
        8.8036,  8.8761,  8.9461,  9.0138,  9.0792,  9.1426,  9.2042,
        9.2634,  9.3205,  9.376 ,  9.4297,  9.4814,  9.5306,  9.5777,
        9.6232,  9.6673,  9.71  ,  9.7513,  9.7914,  9.8304,  9.8682,
        9.9051,  9.941 ,  9.9761, 10.0103, 10.0439, 10.0768, 10.1415,
       10.1739, 10.2049, 10.2329, 10.2565, 10.2745, 10.2854, 11.0427,
       11.0585, 11.0718, 11.085 , 11.0983, 11.1166, 11.1316, 11.1457,
       11.159 , 11.1715, 11.1832, 11.1941, 11.2134, 11.2219, 11.2295,
       11.2364, 11.2424, 11.2477, 11.2521, 11.2557, 11.2586, 11.2606,
       11.2618};
    double p1[] = {5.8   ,  6.5   ,  8.045 ,  8.05  ,  8.175 ,  8.3   ,  8.4825,
        8.665 ,  8.8475,  9.03  ,  9.528 ,  9.696 ,  9.864 , 10.032 ,
       10.2   , 10.9229, 11.0558, 11.1353, 11.2221, 11.3068, 11.3896,
       11.4705, 11.5495, 11.6269, 11.7026, 11.7766, 11.8491, 11.92  ,
       11.9895, 12.0577, 12.1245, 12.1912, 12.255 , 12.3185, 12.3819,
       12.4426, 12.5031, 12.5631, 12.6221, 12.6804, 12.7382, 12.7956,
       12.8526, 12.9096, 12.9668, 13.0222, 13.0783, 13.1336, 13.1894,
       13.2465, 13.3018, 13.3585, 13.4156, 13.4741, 13.5312, 13.59  ,
       13.6494, 13.653 , 13.6566, 13.6602,  8.0382,  8.1283,  8.2213,
        8.3122,  8.4001,  8.4861,  8.5692,  8.6496,  8.7283,  8.8036,
        8.8761,  8.9461,  9.0138,  9.0792,  9.1426,  9.2042,  9.2634,
        9.3205,  9.376 ,  9.4297,  9.4814,  9.5306,  9.5777,  9.6232,
        9.6673,  9.71  ,  9.7513,  9.7914,  9.8304,  9.8682,  9.9051,
        9.941 ,  9.9761, 10.0103, 10.0439, 10.0768, 10.1415, 10.1739,
       10.2049, 10.2329, 10.2565, 10.2745, 10.2854, 10.289 , 11.0585,
       11.0718, 11.085 , 11.0983, 11.1166, 11.1316, 11.1457, 11.159 ,
       11.1715, 11.1832, 11.1941, 11.2134, 11.2219, 11.2295, 11.2364,
       11.2424, 11.2477, 11.2521, 11.2557, 11.2586, 11.2606, 11.2618,
       11.2622};
    double s0[] = {3.46  , 3.85  , 4.48  , 4.49  , 4.5   , 4.509 , 4.523 , 4.609 ,
       4.696 , 4.783 , 5.08  , 5.186 , 5.292 , 5.398 , 5.504 , 5.96  ,
       6.0897, 6.2095, 6.2426, 6.2798, 6.316 , 6.3512, 6.3854, 6.4187,
       6.451 , 6.4828, 6.5138, 6.5439, 6.5727, 6.6008, 6.6285, 6.6555,
       6.6815, 6.7073, 6.7326, 6.7573, 6.7815, 6.8052, 6.8286, 6.8515,
       6.8742, 6.8972, 6.9194, 6.9418, 6.9627, 6.9855, 7.0063, 7.0281,
       7.05  , 7.072 , 7.0931, 7.1144, 7.1369, 7.1586, 7.1807, 7.2031,
       7.2258, 7.249 , 7.2597, 7.2704, 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       3.5043, 3.5187, 3.5314, 3.5435, 3.5551, 3.5661, 3.5765, 3.5864,
       3.5957, 3.6044, 3.6126, 3.6202, 3.6337, 3.6396, 3.645 , 3.6498,
       3.654 , 3.6577, 3.6608, 3.6633, 3.6653, 3.6667, 3.6675};
    double s1[] = {3.46  , 3.85  , 4.49  , 4.5   , 4.509 , 4.518 , 4.609 , 4.696 ,
       4.783 , 4.87  , 5.186 , 5.292 , 5.398 , 5.504 , 5.61  , 6.0897,
       6.2095, 6.2426, 6.2798, 6.316 , 6.3512, 6.3854, 6.4187, 6.451 ,
       6.4828, 6.5138, 6.5439, 6.5727, 6.6008, 6.6285, 6.6555, 6.6815,
       6.7073, 6.7326, 6.7573, 6.7815, 6.8052, 6.8286, 6.8515, 6.8742,
       6.8972, 6.9194, 6.9418, 6.9627, 6.9855, 7.0063, 7.0281, 7.05  ,
       7.072 , 7.0931, 7.1144, 7.1369, 7.1586, 7.1807, 7.2031, 7.2258,
       7.249 , 7.2597, 7.2704, 7.2811, 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       3.5187, 3.5314, 3.5435, 3.5551, 3.5661, 3.5765, 3.5864, 3.5957,
       3.6044, 3.6126, 3.6202, 3.6337, 3.6396, 3.645 , 3.6498, 3.654 ,
       3.6577, 3.6608, 3.6633, 3.6653, 3.6667, 3.6675, 3.6678};
    double rho0[] = {2.72  ,  2.92  ,  3.3198,  3.3455,  3.3713,  3.3985,  3.4258,
        3.4561,  3.4864,  3.5167,  3.7557,  3.8175,  3.8793,  3.941 ,
        4.0028,  4.3714,  4.401 ,  4.4305,  4.4596,  4.4885,  4.5173,
        4.5459,  4.5744,  4.6028,  4.631 ,  4.6591,  4.687 ,  4.7148,
        4.7424,  4.7699,  4.7973,  4.8245,  4.8515,  4.8785,  4.9052,
        4.9319,  4.9584,  4.9847,  5.0109,  5.037 ,  5.0629,  5.0887,
        5.1143,  5.1398,  5.1652,  5.1904,  5.2154,  5.2403,  5.2651,
        5.2898,  5.3142,  5.3386,  5.3628,  5.3869,  5.4108,  5.4345,
        5.4582,  5.4817,  5.5051,  5.5284,  9.9145,  9.9942, 10.0722,
       10.1485, 10.2233, 10.2964, 10.3679, 10.4378, 10.5062, 10.5731,
       10.6385, 10.7023, 10.7647, 10.8257, 10.8852, 10.9434, 11.0001,
       11.0555, 11.1095, 11.1623, 11.2137, 11.2639, 11.3127, 11.3604,
       11.4069, 11.4521, 11.4962, 11.5391, 11.5809, 11.6216, 11.6612,
       11.6998, 11.7373, 11.7737, 11.8092, 11.8437, 11.8772, 11.9414,
       11.9722, 12.0001, 12.0311, 12.0593, 12.0867, 12.1133, 12.7037,
       12.7289, 12.753 , 12.776 , 12.798 , 12.8188, 12.8387, 12.8574,
       12.8751, 12.8917, 12.9072, 12.9217, 12.9474, 12.9586, 12.9688,
       12.9779, 12.9859, 12.9929, 12.9988, 13.0036, 13.0074, 13.01  ,
       13.0117};
    double rho1[] = {2.72  ,  2.92  ,  3.3455,  3.3713,  3.3985,  3.4258,  3.4561,
        3.4864,  3.5167,  3.547 ,  3.8175,  3.8793,  3.941 ,  4.0028,
        4.0646,  4.401 ,  4.4305,  4.4596,  4.4885,  4.5173,  4.5459,
        4.5744,  4.6028,  4.631 ,  4.6591,  4.687 ,  4.7148,  4.7424,
        4.7699,  4.7973,  4.8245,  4.8515,  4.8785,  4.9052,  4.9319,
        4.9584,  4.9847,  5.0109,  5.037 ,  5.0629,  5.0887,  5.1143,
        5.1398,  5.1652,  5.1904,  5.2154,  5.2403,  5.2651,  5.2898,
        5.3142,  5.3386,  5.3628,  5.3869,  5.4108,  5.4345,  5.4582,
        5.4817,  5.5051,  5.5284,  5.5515,  9.9942, 10.0722, 10.1485,
       10.2233, 10.2964, 10.3679, 10.4378, 10.5062, 10.5731, 10.6385,
       10.7023, 10.7647, 10.8257, 10.8852, 10.9434, 11.0001, 11.0555,
       11.1095, 11.1623, 11.2137, 11.2639, 11.3127, 11.3604, 11.4069,
       11.4521, 11.4962, 11.5391, 11.5809, 11.6216, 11.6612, 11.6998,
       11.7373, 11.7737, 11.8092, 11.8437, 11.8772, 11.9414, 11.9722,
       12.0001, 12.0311, 12.0593, 12.0867, 12.1133, 12.1391, 12.7289,
       12.753 , 12.776 , 12.798 , 12.8188, 12.8387, 12.8574, 12.8751,
       12.8917, 12.9072, 12.9217, 12.9474, 12.9586, 12.9688, 12.9779,
       12.9859, 12.9929, 12.9988, 13.0036, 13.0074, 13.01  , 13.0117,
       13.0122};
    earthmod1d model(6371.0, nlayer, d0, d1, p0, p1, s0, s1, rho0, rho1);
}; // namespace ak135

/*
    earthmod1d
*/
int earthmod1d::init(   double earth_radius, int nlayer, 
                        const double d0[], const double d1[],
                        const double p0[], const double p1[],
                        const double s0[], const double s1[],
                        const double r0[], const double r1[] )
{
    d_earth_radius = earth_radius;
    d_layers.resize(nlayer);
    for(int idx=0; idx<nlayer; ++idx) {
        d_layers[idx].init( d0[idx], d1[idx], p0[idx], p1[idx], 
                            s0[idx], s1[idx], r0[idx], r1[idx] );
    }
    find_velocity_jump();
    /////// not necessary
    //d_profile_depth.resize(nlayer+1);
    //d_profile_vp.resize(nlayer+1);
    //d_profile_vs.resize(nlayer+1);
    //d_profile_rho.resize(nlayer+1);
    //for(int idx=0; idx<nlayer; ++idx) {
    //    d_profile_depth[idx] = d_layers[idx].d_top_depth;
    //    d_profile_vp[idx]    = d_layers[idx].d_top_vp;
    //    d_profile_vs[idx]    = d_layers[idx].d_top_vs;
    //    d_profile_rho[idx]   = d_layers[idx].d_top_rho;
    //}
    //d_profile_depth[nlayer] = d_layers[nlayer-1].d_bot_depth;
    //d_profile_vp[nlayer]  = d_layers[nlayer-1].d_top_vp;
    //d_profile_vs[nlayer]  = d_layers[nlayer-1].d_top_vs;
    //d_profile_rho[nlayer] = d_layers[nlayer-1].d_top_rho;
    ///
    return 0;
}
int earthmod1d::find_velocity_jump() {
    int n_interface = d_layers.size() - 1;
    std::vector<int> tmp(n_interface);
    int n_jump = 0;
    for(int idx=0; idx<n_interface; ++idx) {
        if (       d_layers[idx].is_continuous(d_layers[idx+1], 'a' ) ) {
            //pass
            //printf("Y %d, %d\n", idx, idx+1);
        }
        else {
            tmp[n_jump] = idx;
            ++n_jump;
        }
    }
    d_layer_jump.resize(n_jump);
    for(int idx=0; idx<n_jump; ++idx) {
        d_layer_jump[idx].first = tmp[idx];
        d_layer_jump[idx].second = d_layers[tmp[idx]].d_bot_depth;
        printf("NOTE earthmod1d: a jump between two layers: (%d, %d, %f, %f)\n", 
            tmp[idx], tmp[idx]+1, d_layers[tmp[idx]].d_bot_depth, d_layers[tmp[idx]+1].d_top_depth );
    }
    if(!tmp.empty() ) tmp.clear();
    return 0;
}
int earthmod1d::search_layer(double depth) {
    /* 
        search for the index of layer that contain the depth.
        */
    // check if the depth is exactly on the discontinuity
    int i_above, i_below;
    if ( is_depth_on_discontinuity(depth, &i_above, &i_below) ) {
        printf("Warning, earthmod1d::search_layer: a given depth on discontinuity (%f)\n", depth);
    }
    //
    int i0 = 0, i2= d_layers.size()-1;
    int i1 = 0;
    // depth outside of the earth
    if (depth <= 0.0) { return 0; }
    if (depth >= d_earth_radius) { return i2; }
    // depth within the earth
    while(true)
    {
        i1 = (i0+i2)/2;
        char v = d_layers[i1].inside_layer(depth);
        //printf("idx %f %d %d %d, %c\n", depth, i0, i1, i2, v);
        //printf("%f %d %d %d %c\n", depth, i0, i1, i2, v);
        if (v=='i')
        {
            break;
        }
        else if (v=='a') {
            i2 = i1-1;
        }
        else {
            i0 = i1+1;
        }
    }
    return i1;
}
double earthmod1d::evaulate_from_depth(double depth, char type, bool return_slowness) {
    //printf("%f ", depth);
    int idx = search_layer(depth);
    //printf("%f %d\n", depth, idx);
    double v = (return_slowness) ? d_layers[idx].interpolate_slowness(depth, type) : d_layers[idx].interpolate_velocity(depth, type);
    return v;
}
int earthmod1d::output_profile(const char *filename) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "#depth_top(km) vp_top(km/s) vs_top(km/s) rho_top(g/cm^3) depth_bot(km) vp_bot(km/s) vs_bot(km/s) rho_bot(g/cm^3) \n");
    for (std::vector<layer>::iterator it = d_layers.begin(); it != d_layers.end(); ++it ) {
        fprintf(fp, "%.2f %.4f %.4f %.4f    %.2f %.4f %.4f %.4f\n", it->d_top_depth, it->d_top_vp, it->d_top_vs, it->d_top_rho, it->d_bot_depth, it->d_bot_vp, it->d_bot_vs, it->d_bot_rho );
    }
    fclose(fp);
    return 0;
}
int earthmod1d::adjust_raypath(std::list<double> & lon, std::list<double> & lat, std::list<double> & depth ) {
    ////////////////////////////////////////////////////////////////////////// 
    // adjust the raypath to avoid line segments across the discontinuity   //
    //////////////////////////////////////////////////////////////////////////
    std::list<double>::iterator itlon = lon.begin();
    std::list<double>::iterator itlat = lat.begin();
    std::list<double>::iterator itdep = depth.begin();
    double dis_depth;
    for( ; itdep != depth.end();  ) {
        ++itdep;
        if (itdep == depth.end() ) {
            break;
        }
        double d1 = *(itdep);
        double d0 = *(--itdep);
        if ( is_depth_cross_discontinuity(d0, d1, &dis_depth) ) {
            ++itdep;
            double lon0 = *(itlon), lon1 = *(++itlon);
            double lat0 = *(itlat), lat1 = *(++itlat);
            double new_lon = (dis_depth-d0)/(d1-d0)* lon1 + (d1-dis_depth)/(d1-d0)* lon0; 
            double new_lat = (dis_depth-d0)/(d1-d0)* lat1 + (d1-dis_depth)/(d1-d0)* lat0; 
            depth.insert(itdep, dis_depth);
            lon.insert(itlon, new_lon);
            lat.insert(itlat, new_lat);

            --itlon; --itlat; --itdep;
        }
        else
        {
            ++itdep, ++itlon, ++itlat;
        }
    }
    /////////////////////////////////////////////////////////////// 
    // adjust the raypath to avoid points on the discontinuity   //
    ///////////////////////////////////////////////////////////////
    
    static double derr = 1.0e-6; // 1.0e-6 is very good
    int junk1, junk2;
    itlon = lon.begin();
    itlat = lat.begin();
    itdep = depth.begin();

    double x0, x1, x2;
    for( ; itdep != depth.end() ; ++itdep, ++itlon, ++itlat ) {
        if ( is_depth_on_discontinuity(*itdep, &junk1, &junk2) ) {
            x0 = *(--itdep);
            x1 = *(++itdep);
            x2 = *(++itdep);
            --itdep;
            if (x0 < x1 && x2 <= x1) 
            {
                ///
                //   * x0  * x2 
                //    \   /
                //     \ /
                //      * x1 (itdep)
                ///
                *itdep -= derr;
            } 
            else if (x0 < x1 && x2 > x1) 
            { 
                ///
                //   * x0 
                //    \   
                //     * x1 (itdep)
                //      \
                //       * x2 
                ///
                depth.insert(itdep, *itdep);
                --itdep;
                *itdep -= derr;
                ++itdep;
                *itdep += derr;
                lon.insert(itlon, *itlon);
                lat.insert(itlat, *itlat);

            } 
            else if ( x0 > x1 && x2 >= x1) 
            {
                ///
                //      * x1 (itdep)
                //     / \
                //    /   \
                //   * x0  * x2 
                ///
                *itdep += derr;
            } 
            else if ( x0 > x1 && x2 < x1) 
            {
                ///
                //        * x2 
                //       /
                //      * x1 (itdep)
                //     / 
                //    * x0
                ///
                depth.insert(itdep, *itdep);
                --itdep;
                *itdep += derr;
                ++itdep;
                *itdep -= derr;
                lon.insert(itlon, *itlon);
                lat.insert(itlat, *itlat);
            }
        }
    }
    
    return 0;
}
/* 
    earthmod3d
*/
int earthmod3d::init(earthmod1d * mod1d, double dlon, double dlat, const double depth[], int ndep) {
    int nlon = 360.0/dlon;
    int nlat = 180.0/dlat;
    std::vector<double> lons, lats;
    lons.resize(nlon); 
    lats.resize(nlat);
    for(int idx=0; idx<nlon; ++idx) { lons[idx] = idx*dlon; }
    for(int idx=0; idx<nlat; ++idx) { lats[idx] = idx*dlat-90.0; }
    init(mod1d, lons.data(), nlon, lats.data(), nlat, depth, ndep);
    lons.clear();
    lats.clear();
    return 0;
}
// lons should be values in the range of [0, 360) degree
// lats should be values in the range of [-90, 90] degree
// depts should be values in the range of [0, 6371] km
int earthmod3d::init(earthmod1d * mod1d, const double lons[], int nlon,  const double lats[], int nlat, const double depth[], int ndep )
{
    d_mod1d = mod1d;
    // init grid lines
    d_nlon = nlon;
    d_nlat = nlat;
    d_ndep = ndep;
    d_nlonlat = nlon*nlat;
    d_lon.resize(nlon);
    d_lat.resize(nlat);
    d_depth.resize(ndep);
    d_lon.assign(lons, lons+nlon);
    for(int idx=0; idx<d_nlon; ++idx) {
        d_lon[idx] = ROUND_DEG360( d_lon[idx] );
    }
    d_lat.assign(lats, lats+nlat);
    d_depth.assign(depth, depth+ndep);
    //
    d_npts = ndep*nlat*nlon;
    d_vp.resize(d_npts) ; d_vp.assign( d_npts, 0.0);
    d_vs.resize(d_npts) ; d_vs.assign( d_npts, 0.0);
    d_rho.resize(d_npts); d_rho.assign(d_npts, 0.0);
    d_dvp.resize(d_npts); d_dvp.assign(d_npts, 1.0);
    d_dvs.resize(d_npts); d_dvs.assign(d_npts, 1.0);
    d_slowness_p.resize(d_npts);  d_slowness_p.assign( d_npts, 0.0);
    d_slowness_s.resize(d_npts);  d_slowness_s.assign( d_npts, 0.0);
    d_dslowness_p.resize(d_npts); d_dslowness_p.assign(d_npts, 1.0);
    d_dslowness_s.resize(d_npts); d_dslowness_s.assign(d_npts, 1.0);
    d_pts.resize(d_npts);
    //
    long ipt = 0;
    double theta, phi, r, x, y, z;
    double vp, vs, rho;
    double sp, ss;
    for (int idep=0; idep<ndep; ++idep) {
        //d_mod1d->evaulate_from_depth(d_depth[idep], &vp, &vs, &rho );
        vp = d_mod1d->evaulate_from_depth(d_depth[idep], 'p', false);
        vs = d_mod1d->evaulate_from_depth(d_depth[idep], 's', false);
        sp = d_mod1d->evaulate_from_depth(d_depth[idep], 'p', true);
        ss = d_mod1d->evaulate_from_depth(d_depth[idep], 's', true);
        rho = d_mod1d->evaulate_from_depth(d_depth[idep], 'r', false);
        //printf("mod3d: depth %lf vp: %lf\n", d_depth[idep], vp);
        for (int ilat=0; ilat<nlat; ++ilat) {
            for (int ilon=0; ilon<nlon; ++ilon) {
                //printf("%d %d %d %ld\n", idep, ilat, ilon, ipt);
                // cooridnates
                geo2sph(d_lon[ilon], d_lat[ilat], d_depth[idep], &theta, &phi, &r);
                sph2xyz(theta, phi, r, &x, &y, &z);
                d_pts[ipt].init(d_lon[ilon], d_lat[ilat], d_depth[idep], theta, phi, r, x, y, z );
                // velocities and density
                d_vp[ipt] = vp; 
                d_vs[ipt] = vs;
                d_rho[ipt] = rho;
                d_slowness_p[ipt] = sp;
                d_slowness_s[ipt] = ss;
                //
                ++ipt;
            }
        }
    }
}
// Init 3-D Earth model from file  that can be 'p' for plain text, 'b' for binary file.
// The Format is:
//      +-------------------------+
//      |ndep nlat nlon           |
//      |dep1 dep2 dep3 ... depN  |
//      |lat1 lat2 lat3 ... latM  |
//      |lon1 lon2 lon3 ... lonL  |
//      |dvp1 dvp2 dvp3 ... dvpX  |
//      |dvs1 dvs2 dvs3 ... dvsX  |
//      +-------------------------+
int earthmod3d::init(earthmod1d * mod1d, const char *fnm, char mode) {
    std::vector<double> deps, lats, lons, dvp, dvs;
    int junk;
    if (mode == 'p')
    {
        FILE *fid = fopen(fnm, "r");
        // part 1
        int ndep, nlat, nlon;
        junk = fscanf(fid, "%d %d %d", &ndep, &nlat, &nlon);
        deps.resize(ndep); lats.resize(nlat); lons.resize(nlon);
        // part 2
        for(int idx=0; idx<ndep; ++idx) {
            junk = fscanf(fid, "%lf", &(deps[idx]) );
        }
        for(int idx=0; idx<nlat; ++idx) {
            junk = fscanf(fid, "%lf", &(lats[idx]) );
        }
        for(int idx=0; idx<nlon; ++idx) {
            junk = fscanf(fid, "%lf", &(lons[idx]) );
        }
        // part 3
        int npts = ndep*nlat*nlon;
        dvp.resize(npts); dvp.assign(npts, 0.0);
        dvs.resize(npts); dvs.assign(npts, 0.0);
        for(int idx=0; idx<npts; ++idx) {
            junk = fscanf(fid, "%lf", &(dvp[idx]) );
        }
        for(int idx=0; idx<npts; ++idx) {
            junk = fscanf(fid, "%lf", &(dvs[idx]) );
        }
        //
        fclose(fid);
        //
        init(mod1d, lons.data(), nlon, lats.data(), nlat, deps.data(), ndep );
        set_mod3d(dvp.data(), dvs.data() );
    }
    else if (mode == 'b')
    {
        FILE *fid = fopen(fnm, "rb");
        // part 1
        int ndep, nlat, nlon;
        junk = fread(&ndep, sizeof(int), 1, fid);
        junk = fread(&nlat, sizeof(int), 1, fid);
        junk = fread(&nlon, sizeof(int), 1, fid);
        deps.resize(ndep); lats.resize(nlat); lons.resize(nlon);
        // part 2
        junk = fread(deps.data(), sizeof(double), ndep, fid);
        junk = fread(lats.data(), sizeof(double), nlat, fid);
        junk = fread(lons.data(), sizeof(double), nlon, fid);
        // part 3
        int npts = ndep*nlat*nlon;
        dvp.resize(npts); dvp.assign(npts, 1.0);
        dvs.resize(npts); dvs.assign(npts, 1.0);
        junk = fread(dvp.data(), sizeof(double), npts, fid);
        junk = fread(dvs.data(), sizeof(double), npts, fid);
        //
        fclose(fid);
        //
        init(mod1d, lons.data(), nlon, lats.data(), nlat, deps.data(), ndep );
        set_mod3d(dvp.data(), dvs.data() );
    }
    return 0;
}
int earthmod3d::output_model(const char * fnm, char mode) {
    int junk;
    if (mode == 'p')
    {
        FILE *fid = fopen(fnm, "w");
        // part 1
        int ndep = d_depth.size();
        int nlat =  d_lat.size();
        int nlon =  d_lon.size();
        junk = fprintf(fid, "%d %d %d\n", ndep, nlat, nlon );
        // part 2
        for(int idx=0; idx<d_depth.size(); ++idx) {
            junk = fprintf(fid, "%.12lf ", d_depth[idx] );
        }
        fprintf(fid, "\n");
        for(int idx=0; idx<d_lat.size(); ++idx) {
            junk = fprintf(fid, "%.4lf ", d_lat[idx] );
        }
        fprintf(fid, "\n");
        for(int idx=0; idx<d_lon.size(); ++idx) {
            junk = fprintf(fid, "%.4lf ", d_lon[idx] );
        }
        fprintf(fid, "\n");
        // part 3
        int ipt =0;
        for(int idep=0; idep<d_depth.size(); ++idep) {
            for(int ilat=0;ilat<d_lat.size(); ++ilat) {
                for(int ilon=0;ilon<d_lon.size(); ++ilon) {
                    junk = fprintf(fid, "%lf ", d_dvp[ipt]);
                    ++ipt;
                }
                fprintf(fid, "\n");
            }
        }
        ipt =0;
        for(int idep=0; idep<d_depth.size(); ++idep) {
            for(int ilat=0;ilat<d_lat.size(); ++ilat) {
                for(int ilon=0;ilon<d_lon.size(); ++ilon) {
                    junk = fprintf(fid, "%lf ", d_dvs[ipt]);
                    ++ipt;
                }
                fprintf(fid, "\n");
            }
        }
        //
        fclose(fid);
    }
    else if (mode == 'b')
    {
        FILE *fid = fopen(fnm, "wb");
        // part 1
        int ndep = d_depth.size();
        int nlat =  d_lat.size();
        int nlon =  d_lon.size();
        junk = fwrite(&ndep, sizeof(int), 1, fid);
        junk = fwrite(&nlat, sizeof(int), 1, fid);
        junk = fwrite(&nlon, sizeof(int), 1, fid);
        // part 2
        junk = fwrite(d_depth.data(), sizeof(double), ndep, fid);
        junk = fwrite(d_lat.data(),   sizeof(double), nlat, fid);
        junk = fwrite(d_lon.data(),   sizeof(double), nlon, fid);
        // part 3
        junk = fread(d_dvp.data(), sizeof(double), d_dvp.size(), fid);
        junk = fread(d_dvs.data(), sizeof(double), d_dvs.size(), fid);
        //
        fclose(fid);
        //
    }
    return 0;
}
// type : 'a' to output all points
//        'p' to output points with Vp perturbation != 0.0
//        's' to output points ...  Vs ..           != 0.0
//        'd' to output points with Vp and Vs ...   != 0.0
int earthmod3d::output_grd_pts(const char * filename, char type ) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "#lon lat depth  theta phi r  x y z vp vs rho  dvp(1+e) dvs(1+e)\n" );
    if (type == 'a') 
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].d_lon, d_pts[idx].d_lat, d_pts[idx].d_depth,
                    d_pts[idx].d_theta, d_pts[idx].d_phi, d_pts[idx].d_r,
                    d_pts[idx].d_x, d_pts[idx].d_y, d_pts[idx].d_z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_dvp[idx], d_dvs[idx] );
        }
    }
    else if (type == 'p' || type == 'P')
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            if ( ISEQUAL(d_dvp[idx], 1.0) ) 
                continue;
            fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].d_lon, d_pts[idx].d_lat, d_pts[idx].d_depth,
                    d_pts[idx].d_theta, d_pts[idx].d_phi, d_pts[idx].d_r,
                    d_pts[idx].d_x, d_pts[idx].d_y, d_pts[idx].d_z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_dvp[idx], d_dvs[idx] );
        }
    }
    else if (type == 's' || type == 'S')
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            if ( ISEQUAL(d_dvs[idx], 1.0) ) 
                continue;
            fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].d_lon, d_pts[idx].d_lat, d_pts[idx].d_depth,
                    d_pts[idx].d_theta, d_pts[idx].d_phi, d_pts[idx].d_r,
                    d_pts[idx].d_x, d_pts[idx].d_y, d_pts[idx].d_z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_dvp[idx], d_dvs[idx] );
        }
    }
    else if (type == 'd' || type == 'D')
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            if ( ISEQUAL(d_dvp[idx], 1.0) &&  ISEQUAL(d_dvs[idx], 1.0) ) 
                continue;
            fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].d_lon, d_pts[idx].d_lat, d_pts[idx].d_depth,
                    d_pts[idx].d_theta, d_pts[idx].d_phi, d_pts[idx].d_r,
                    d_pts[idx].d_x, d_pts[idx].d_y, d_pts[idx].d_z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_dvp[idx], d_dvs[idx] );
        }
    }
    fclose(fp);
    return 0;
}

/*
    raypath3d
*/
int raypath3d::init(earthmod3d *mod3d, double evdp, double evlo, double evla, double stlo, double stla , char * phase) {
    /*
    d_mod3d = mod3d;
    printf("start\n");
    //
    
    PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *pArgs;
    Py_Initialize();
    pModule = PyImport_ImportModule("test");
    pDict = PyModule_GetDict(pModule);
    pFunc = PyDict_GetItemString(pDict, "run" );
    if (PyCallable_Check(pFunc)) 
    {
        //PyObject_CallObject(pFunc, NULL);
        pArgs = PyTuple_New(6);
        pValue = PyFloat_FromDouble( evdp );
        PyTuple_SetItem(pArgs, 0, pValue); 
        pValue = PyFloat_FromDouble( evla );
        PyTuple_SetItem(pArgs, 1, pValue); 
        pValue = PyFloat_FromDouble( evlo );
        PyTuple_SetItem(pArgs, 2, pValue); 
        pValue = PyFloat_FromDouble( stla );
        PyTuple_SetItem(pArgs, 3, pValue); 
        pValue = PyFloat_FromDouble( stlo );
        PyTuple_SetItem(pArgs, 4, pValue); 
        pValue = PyUnicode_FromString( phase );
        PyTuple_SetItem(pArgs, 5, pValue); 
        pValue = PyObject_CallObject(pFunc, pArgs);
    } else 
    {
        PyErr_Print();
    }
    */
    //pModule_taup = PyImport_ImportModule("obspy.taup"); 
    //pFunc = PyObject_GetAttrString(pModule_taup, "TauPyModel");    
    //pValue_mod_ak135 = PyDict_GetItemString(pFunc, "ak135");
    //if (PyCallable_Check(pValue_mod_ak135) )
    //{
    //    pInstance = PyObject_CallObject(pValue_mod_ak135, NULL); 
    //    //pFunc = PyObject_GetAttrString(pInstance, "get_ray_paths_geo");
    //    pArr = PyObject_CallMethod(pInstance, "get_ray_paths_geo", "(fffff)", evdp, evla, evlo, stla, stlo);
    //    //PyObject_Call(pFunc, "f", evdp, "f", evla, "f", evlo, "f", stla, "f", stlo );
    //    //pArr = PyObject_CallMethod(, 
    //    //    evdp, evla, evlo, stla, stlo);
    //}
    //
    //
    //
    //Py_Finalize();
    //
    //printf("end\n");
    return 0;
}
int raypath3d::init(earthmod3d *mod3d, const int npts, const double lons[], const double lats[], const double depth[]) {
    d_mod3d = mod3d;
    d_npts = npts;
    d_lon_list.assign(  lons,  lons+npts);
    d_lat_list.assign(  lats,  lats+npts);
    d_depth_list.assign(depth, depth+npts);
    adjust_avoid_discontinuity();

    list2vec();
    geo2xyz();

    d_sensitivity.resize( d_mod3d->npts() ); 
    d_sensitivity.assign(d_sensitivity.size(), 0.0);
    return 0;
}
int raypath3d::init(earthmod3d *mod3d, const std::list<double> lon, const std::list<double> lat, const std::list<double> depth ) {
    d_mod3d = mod3d;
    d_npts = lon.size();
    d_lon_list.assign(  lon.begin(),   lon.end()   );
    d_lat_list.assign(  lat.begin(),   lat.end()   );
    d_depth_list.assign(depth.begin(), depth.end() );

    adjust_avoid_discontinuity();

    list2vec();
    geo2xyz();

    d_sensitivity.resize( d_mod3d->npts() ); 
    d_sensitivity.assign(d_sensitivity.size(), 0.0);
    return 0;
}
int raypath3d::init(earthmod3d *mod3d, const char *filename) {
#define MAXNPTS 4194304
    //
    FILE *fid = fopen(filename, "rb");
    int npts;
    int junk;
    static double lons[MAXNPTS];
    static double lats[MAXNPTS];
    static double depth[MAXNPTS];
    junk = fread(&npts, sizeof(int), 1, fid);
    if (junk != 1) { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
    junk = fread(lons,  sizeof(double), npts, fid );
    if (junk != npts) { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
    junk = fread(lats,  sizeof(double), npts, fid );
    if (junk != npts) { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
    junk = fread(depth, sizeof(double), npts, fid );
    if (junk != npts) { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
    fclose(fid);
    //
    init(mod3d, npts, lons, lats, depth);
    return 0;
#undef MAXNPTS
}
int raypath3d::init(earthmod3d *mod3d, const int id, const char * filename, const char * tag) {
    d_id = id;
    d_raypath_filename = filename;
    d_tag = tag;
    init(mod3d, filename);
    return 0;
}
int raypath3d::list2vec() {
    d_lon.resize(d_lon_list.size() ); d_lon.assign(    d_lon_list.begin(),      d_lon_list.end()   );
    d_lat.resize(d_lat_list.size() ); d_lat.assign(    d_lat_list.begin(),      d_lat_list.end()   );
    d_depth.resize(d_depth_list.size() ); d_depth.assign(  d_depth_list.begin(),    d_depth_list.end() );
    return 0;
}
int raypath3d::geo2xyz() {
    d_theta.resize(d_npts);
    d_phi.resize(d_npts);  
    d_r.resize(d_npts);    
    d_x.resize(d_npts);    
    d_y.resize(d_npts);    
    d_z.resize(d_npts);    
    //
    for(int idx=0; idx<d_npts; ++idx) {
        d_mod3d->geo2sph(d_lon[idx], d_lat[idx], d_depth[idx], &(d_theta[idx]), &(d_phi[idx]), &(d_r[idx]) );
        d_mod3d->sph2xyz(d_theta[idx], d_phi[idx], d_r[idx], &(d_x[idx]), &(d_y[idx]), &(d_z[idx]) );
    }
    //
    return 0;
}
int raypath3d::xyz2geo() {
    d_lon.resize(d_npts);
    d_lat.resize(d_npts);
    d_depth.resize(d_npts);
    d_theta.resize(d_npts);
    d_phi.resize(d_npts);  
    d_r.resize(d_npts);  
    for(int idx=0; idx<d_npts; ++idx) {
        d_mod3d->xyz2sph(d_x[idx], d_y[idx], d_z[idx], &(d_theta[idx]), &(d_phi[idx]), &(d_r[idx]) );
        d_mod3d->sph2geo(d_theta[idx], d_phi[idx], d_r[idx], &(d_lon[idx]), &(d_lat[idx]), &(d_depth[idx]) );
    }
    return 0;
}
int raypath3d::adjust_avoid_discontinuity() {
    // xyz and sph coordinates are not updated
    earthmod1d *mod1d = d_mod3d->mod1d();
    mod1d->adjust_raypath(d_lon_list, d_lat_list, d_depth_list);
    d_npts = d_lon_list.size();
    return 0;
}
int raypath3d::old_adjust_avoid_discontinuity() {
    /*
    earthmod1d *mod1d = d_mod3d->mod1d();
    int junk1, junk2;
    for (int idx=0; idx<d_npts; ++idx) {
        if( mod1d->is_depth_on_discontinuity(d_depth[idx], &junk1, &junk2) ) {
            printf("Ray-path point on discontinuity %f \n", d_depth[idx] );
            d_depth[idx] += 0.001;
        }
    }
    */
    earthmod1d *mod1d = d_mod3d->mod1d();
    std::vector<int> pt_idx(256); // 256 seems enough here
    int sz = 0;
    //
    pt_idx[sz] = -1; ++sz;
    //
    int junk1, junk2;
    for (int idx=0; idx<d_npts; ++idx) {
        if( mod1d->is_depth_on_discontinuity(d_depth[idx], &junk1, &junk2) ) {
            printf("Ray-path point on discontinuity %f \n", d_depth[idx] );
            pt_idx[sz] = idx;
            ++sz;
        }
    }
    //
    pt_idx[sz] = d_npts; ++sz;
    //
    int new_npts = d_npts + sz - 2;
    std::vector<double> v_new_lon(new_npts), v_new_lat(new_npts), v_new_depth(new_npts);
    double *old_lon = d_lon.data();
    double *old_lat = d_lat.data();
    double *old_dep = d_depth.data();
    double *new_lon = v_new_lon.data();
    double *new_lat = v_new_lat.data();
    double *new_dep = v_new_depth.data();

    int idx_start_old, idx_end_old, sublen, idx_seg;
    int idx_start_new=0, idx_end_new;
    for (idx_seg=0; idx_seg<sz-1; ++idx_seg) {
        // find the segment in old npts
        idx_start_old = pt_idx[idx_seg]+1; // [start, end) and hence length = end-start
        idx_end_old   = pt_idx[idx_seg+1];
        sublen  = idx_end_old - idx_start_old;
        // copy segements
        std::memcpy(new_dep+idx_start_new, old_dep+idx_start_old, sublen*sizeof(double) );
        std::memcpy(new_lat+idx_start_new, old_lat+idx_start_old, sublen*sizeof(double) );
        std::memcpy(new_lon+idx_start_new, old_lon+idx_start_old, sublen*sizeof(double) );
        // two points
        if ( idx_seg != sz-2) {
            idx_end_new = idx_start_new + sublen;
            new_lon[idx_end_new]   = old_lon[idx_end_old];
            new_lon[idx_end_new+1] = old_lon[idx_end_old];
            new_lat[idx_end_new]   = old_lat[idx_end_old];
            new_lat[idx_end_new+1] = old_lat[idx_end_old];
            if ( old_dep[idx_end_old] > old_dep[idx_end_old-1] ) {
                new_dep[idx_end_new]   = old_dep[idx_end_old]-1.0e-6; // 1.0e-6 is very good given double type
                new_dep[idx_end_new+1] = old_dep[idx_end_old]+1.0e-6;
            } else {
                new_dep[idx_end_new]   = old_dep[idx_end_old]+1.0e-6;
                new_dep[idx_end_new+1] = old_dep[idx_end_old]-1.0e-6;
            }
            idx_end_new += 2;
            idx_start_new = idx_end_new;
        }
    }
    //
    init(d_mod3d, new_npts, new_lon, new_lat, new_dep);
    return 0;
}
int raypath3d::inter_denser_raypath(double dl) {
    std::list<double> new_lon, new_lat, new_dep;
    for(int idx=1; idx<d_npts; ++idx) {
        double x0 = d_x[idx-1], y0 = d_y[idx-1], z0 = d_z[idx-1];
        double x1 = d_x[idx],   y1 = d_y[idx],   z1 = d_z[idx];
        double l = raypath3d::length(x0, y0, z0, x1, y1, z1);
        if (l <= dl) {
            new_lon.push_back(d_lon[idx-1]   );
            new_lat.push_back(d_lat[idx-1]   );
            new_dep.push_back(d_depth[idx-1] );
            continue;
        }
        //////
        int n = (int)(l/dl+1);
        double deltax = (x1-x0)/n, deltay = (y1-y0)/n, deltaz = (z1-z0)/n;
        for(int ipt=0; ipt<n; ++ipt) {
            double x = (x0 + deltax*ipt);
            double y = (y0 + deltay*ipt);
            double z = (z0 + deltaz*ipt);
            double lon, lat, dep;
            d_mod3d->xyz2geo(x, y, z, &lon, &lat, &dep);
            new_lon.push_back(lon);
            new_lat.push_back(lat);
            new_dep.push_back(dep);
        }
    }
    new_lon.push_back(d_lon[d_npts-1]   );
    new_lat.push_back(d_lat[d_npts-1]   );
    new_dep.push_back(d_depth[d_npts-1] );
    init(d_mod3d, new_lon, new_lat, new_dep);
    if (!new_lon.empty() ) { new_lon.clear(); }
    if (!new_lat.empty() ) { new_lat.clear(); }
    if (!new_dep.empty() ) { new_dep.clear(); }
    return 0;
}
int raypath3d::double_denser_raypath() {
    std::vector<double> old_x, old_y, old_z;
    std::vector<double> new_x, new_y, new_z;
    old_x.assign(d_x.begin(), d_x.end() );
    old_y.assign(d_y.begin(), d_y.end() );
    old_z.assign(d_z.begin(), d_z.end() );
    int old_npts = d_npts;
    //
    d_npts = old_npts*2-1;
    new_x.resize(d_npts);
    new_y.resize(d_npts);
    new_z.resize(d_npts);
    for(int idx=0; idx<old_npts; ++idx) {
        new_x[idx*2] = old_x[idx];
        new_y[idx*2] = old_y[idx];
        new_z[idx*2] = old_z[idx];
    }
    for(int idx=1; idx<d_npts; idx+=2) {
        new_x[idx] = 0.5*(new_x[idx-1]+new_x[idx+1] );
        new_y[idx] = 0.5*(new_y[idx-1]+new_y[idx+1] );
        new_z[idx] = 0.5*(new_z[idx-1]+new_z[idx+1] );
    }
    //
    for(int idx=0; idx<d_npts; ++idx) {
        double x= new_x[idx], y= new_y[idx], z= new_z[idx];
        double theta, phi, r;
        d_mod3d->xyz2sph(x, y, z, &theta, &phi, &r);
        d_mod3d->sph2geo(theta, phi, r, &(new_x[idx]), &(new_y[idx]), &(new_z[idx]) );
    }
    init(d_mod3d, d_npts, new_x.data(), new_y.data(), new_z.data() );
    //
    old_x.clear();
    old_y.clear();
    old_z.clear();
    new_x.clear();
    new_y.clear();
    new_z.clear();
    return 0;
}
int raypath3d::grd_sensitivity() {
    d_sensitivity.assign(d_sensitivity.size(), 0.0);
    int nseg = d_npts - 1;
    for(int idx = 0; idx<nseg; ++idx) {
        // length
        int pt1=idx, pt2=idx+1;
        double segment_length = raypath3d::length( d_x[pt1], d_y[pt1], d_z[pt1], d_x[pt2], d_y[pt2], d_z[pt2] );
        //printf("\n>>> segment length: %lf\n", segment_length);
        double half_segment_length = 0.5*segment_length;
        // interpolation for the middle point
        if (false)
        {
            double x_mid = 0.5*(d_x[pt1]+d_x[pt2]);
            double y_mid = 0.5*(d_y[pt1]+d_y[pt2]);
            double z_mid = 0.5*(d_z[pt1]+d_z[pt2]);
            double theta, phi, r;
            double lon, lat, depth;
            d_mod3d->xyz2sph(x_mid, y_mid, z_mid, &theta, &phi, &r);
            d_mod3d->sph2geo(theta, phi, r, &lon, &lat, &depth);
            //double lon=0.5*(d_lon[pt1]+d_lon[pt2]); 
            //double lat=0.5*(d_lat[pt1]+d_lat[pt2]);
            //double depth=0.5*(d_depth[pt1]+d_depth[pt2]);
            int ilon, ilat, idep;
            d_mod3d->search_grd(lon, lat, depth, &ilon, &ilat, &idep);
            int i0, i1, i2, i3, i4, i5, i6, i7;
            i0 = d_mod3d->point_index(ilon,   ilat,   idep  );
            i1 = d_mod3d->point_index(ilon+1, ilat,   idep  );
            i2 = d_mod3d->point_index(ilon,   ilat+1, idep  );
            i3 = d_mod3d->point_index(ilon+1, ilat+1, idep  );
            i4 = d_mod3d->point_index(ilon,   ilat,   idep+1);
            i5 = d_mod3d->point_index(ilon+1, ilat,   idep+1);
            i6 = d_mod3d->point_index(ilon,   ilat+1, idep+1);
            i7 = d_mod3d->point_index(ilon+1, ilat+1, idep+1);
            //double c0, c1, c2, c3, c4, c5, c6, c7;
            //// s   = c0*s[i0] + c1*s[i1] + ...
            //// t   = 0.5*l*(c0*s[i0] + c1*s[i1] + ...)
            //// t   = d[i0] * s[i0] + ...
            //// d[i0] = 0.5*l*c[i0]
            //d_mod3d->interpolate3d(i0, i1, i2, i3, i4, i5, i6, i7, d_lon[pt1], d_lat[pt1], d_depth[pt1],
            //                        &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7);

            double d0, d1, d2, d3, d4, d5, d6, d7;
            d_mod3d->interpolate_grd3d(ilon, ilat, idep, d_lon[pt1], d_lat[pt1], d_depth[pt1],
                                    &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);
            //printf("%f %f %f %f    %f %f %f %f\n", c0, c1, c2, c3, c4, c5, c6, c7);
            //printf("%f %f %f %f    %f %f %f %f\n", d0, d1, d2, d3, d4, d5, d6, d7);
            //printf("%f %f %f %f    %f %f %f %f\n", c0-d0, c1-d1, c2-d2, c3-d3, c4-d4, c5-d5, c6-d6, c7-d7);
            d_sensitivity[i0] += (d0*segment_length)/d_mod3d->vp(i0);
            d_sensitivity[i1] += (d1*segment_length)/d_mod3d->vp(i1);
            d_sensitivity[i2] += (d2*segment_length)/d_mod3d->vp(i2);
            d_sensitivity[i3] += (d3*segment_length)/d_mod3d->vp(i3);
            d_sensitivity[i4] += (d4*segment_length)/d_mod3d->vp(i4);
            d_sensitivity[i5] += (d5*segment_length)/d_mod3d->vp(i5);
            d_sensitivity[i6] += (d6*segment_length)/d_mod3d->vp(i6);
            d_sensitivity[i7] += (d7*segment_length)/d_mod3d->vp(i7);
            //printf("sens %lf %lf %lf  %d %d %d\n", lon, lat, depth, ilon, ilat, idep);
            //printf("sens i0: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i0, d_sensitivity[i0], (d0*segment_length)/d_mod3d->vp(i0), d0,  d_mod3d->vp(i0), d_mod3d->depth(i0)  );
            //printf("sens i1: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i1, d_sensitivity[i1], (d1*segment_length)/d_mod3d->vp(i1), d1,  d_mod3d->vp(i1), d_mod3d->depth(i1)  );
            //printf("sens i2: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i2, d_sensitivity[i2], (d2*segment_length)/d_mod3d->vp(i2), d2,  d_mod3d->vp(i2), d_mod3d->depth(i2)  );
            //printf("sens i3: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i3, d_sensitivity[i3], (d3*segment_length)/d_mod3d->vp(i3), d3,  d_mod3d->vp(i3), d_mod3d->depth(i3)  );
            //printf("sens i4: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i4, d_sensitivity[i4], (d4*segment_length)/d_mod3d->vp(i4), d4,  d_mod3d->vp(i4), d_mod3d->depth(i4)  );
            //printf("sens i5: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i5, d_sensitivity[i5], (d5*segment_length)/d_mod3d->vp(i5), d5,  d_mod3d->vp(i5), d_mod3d->depth(i5)  );
            //printf("sens i6: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i6, d_sensitivity[i6], (d6*segment_length)/d_mod3d->vp(i6), d6,  d_mod3d->vp(i6), d_mod3d->depth(i6)  );
            //printf("sens i7: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i7, d_sensitivity[i7], (d7*segment_length)/d_mod3d->vp(i7), d7,  d_mod3d->vp(i7), d_mod3d->depth(i7)  );
            //printf("%f\n", segment_length);
            //printf("%d %d %d @@ %f %f %f %f %f %f %f %f \n", ilon, ilat, idep, d0, d1, d2, d3, d4, d5, d6, d7);
        }
        // interpolation for the end point pt1
        if (true)
        {
            int ilon, ilat, idep;
            d_mod3d->search_grd(d_lon[pt1], d_lat[pt1], d_depth[pt1], &ilon, &ilat, &idep);
            int i0, i1, i2, i3, i4, i5, i6, i7;
            i0 = d_mod3d->point_index(ilon,   ilat,   idep  );
            i1 = d_mod3d->point_index(ilon+1, ilat,   idep  );
            i2 = d_mod3d->point_index(ilon,   ilat+1, idep  );
            i3 = d_mod3d->point_index(ilon+1, ilat+1, idep  );
            i4 = d_mod3d->point_index(ilon,   ilat,   idep+1);
            i5 = d_mod3d->point_index(ilon+1, ilat,   idep+1);
            i6 = d_mod3d->point_index(ilon,   ilat+1, idep+1);
            i7 = d_mod3d->point_index(ilon+1, ilat+1, idep+1);
            //double c0, c1, c2, c3, c4, c5, c6, c7;
            //// s   = c0*s[i0] + c1*s[i1] + ...
            //// t   = 0.5*l*(c0*s[i0] + c1*s[i1] + ...)
            //// t   = d[i0] * s[i0] + ...
            //// d[i0] = 0.5*l*c[i0]
            //d_mod3d->interpolate3d(i0, i1, i2, i3, i4, i5, i6, i7, d_lon[pt1], d_lat[pt1], d_depth[pt1],
            //                        &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7);

            double d0, d1, d2, d3, d4, d5, d6, d7;
            d_mod3d->interpolate_grd3d(ilon, ilat, idep, d_lon[pt1], d_lat[pt1], d_depth[pt1],
                                    &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);
            //printf("%f %f %f %f    %f %f %f %f\n", c0, c1, c2, c3, c4, c5, c6, c7);
            //printf("%f %f %f %f    %f %f %f %f\n", d0, d1, d2, d3, d4, d5, d6, d7);
            //printf("%f %f %f %f    %f %f %f %f\n", c0-d0, c1-d1, c2-d2, c3-d3, c4-d4, c5-d5, c6-d6, c7-d7);
            d_sensitivity[i0] += (d0*half_segment_length)*d_mod3d->slowness_p(i0);
            d_sensitivity[i1] += (d1*half_segment_length)*d_mod3d->slowness_p(i1);
            d_sensitivity[i2] += (d2*half_segment_length)*d_mod3d->slowness_p(i2);
            d_sensitivity[i3] += (d3*half_segment_length)*d_mod3d->slowness_p(i3);
            d_sensitivity[i4] += (d4*half_segment_length)*d_mod3d->slowness_p(i4);
            d_sensitivity[i5] += (d5*half_segment_length)*d_mod3d->slowness_p(i5);
            d_sensitivity[i6] += (d6*half_segment_length)*d_mod3d->slowness_p(i6);
            d_sensitivity[i7] += (d7*half_segment_length)*d_mod3d->slowness_p(i7);
            //printf("%d %d %d @@ %f %f %f %f %f %f %f %f \n", ilon, ilat, idep, d0, d1, d2, d3, d4, d5, d6, d7);
            //printf("sens %lf %lf %lf \n", d_lon[pt1], d_lat[pt1], d_depth[pt1] );
            //printf("sens i0: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i0, d_sensitivity[i0], (d0*segment_length)/d_mod3d->vp(i0), d0,  d_mod3d->vp(i0), d_mod3d->depth(i0)  );
            //printf("sens i1: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i1, d_sensitivity[i1], (d1*segment_length)/d_mod3d->vp(i1), d1,  d_mod3d->vp(i1), d_mod3d->depth(i1)  );
            //printf("sens i2: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i2, d_sensitivity[i2], (d2*segment_length)/d_mod3d->vp(i2), d2,  d_mod3d->vp(i2), d_mod3d->depth(i2)  );
            //printf("sens i3: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i3, d_sensitivity[i3], (d3*segment_length)/d_mod3d->vp(i3), d3,  d_mod3d->vp(i3), d_mod3d->depth(i3)  );
            //printf("sens i4: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i4, d_sensitivity[i4], (d4*segment_length)/d_mod3d->vp(i4), d4,  d_mod3d->vp(i4), d_mod3d->depth(i4)  );
            //printf("sens i5: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i5, d_sensitivity[i5], (d5*segment_length)/d_mod3d->vp(i5), d5,  d_mod3d->vp(i5), d_mod3d->depth(i5)  );
            //printf("sens i6: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i6, d_sensitivity[i6], (d6*segment_length)/d_mod3d->vp(i6), d6,  d_mod3d->vp(i6), d_mod3d->depth(i6)  );
            //printf("sens i7: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i7, d_sensitivity[i7], (d7*segment_length)/d_mod3d->vp(i7), d7,  d_mod3d->vp(i7), d_mod3d->depth(i7)  );
            //printf("%f\n", segment_length);
            //printf("%d %d %d @@ %f %f %f %f %f %f %f %f \n", ilon, ilat, idep, d0, d1, d2, d3, d4, d5, d6, d7);
        
        }
        // interpolation for the end point pt2
        if (true)
        {
            int ilon, ilat, idep;
            d_mod3d->search_grd(d_lon[pt2], d_lat[pt2], d_depth[pt2], &ilon, &ilat, &idep);
            int i0, i1, i2, i3, i4, i5, i6, i7;
            i0 = d_mod3d->point_index(ilon,   ilat,   idep  );
            i1 = d_mod3d->point_index(ilon+1, ilat,   idep  );
            i2 = d_mod3d->point_index(ilon,   ilat+1, idep  );
            i3 = d_mod3d->point_index(ilon+1, ilat+1, idep  );
            i4 = d_mod3d->point_index(ilon,   ilat,   idep+1);
            i5 = d_mod3d->point_index(ilon+1, ilat,   idep+1);
            i6 = d_mod3d->point_index(ilon,   ilat+1, idep+1);
            i7 = d_mod3d->point_index(ilon+1, ilat+1, idep+1);
            
            double d0, d1, d2, d3, d4, d5, d6, d7;
            d_mod3d->interpolate_grd3d(ilon, ilat, idep, d_lon[pt1], d_lat[pt1], d_depth[pt1],
                                    &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);
            d_sensitivity[i0] += (d0*half_segment_length)*d_mod3d->slowness_p(i0);
            d_sensitivity[i1] += (d1*half_segment_length)*d_mod3d->slowness_p(i1);
            d_sensitivity[i2] += (d2*half_segment_length)*d_mod3d->slowness_p(i2);
            d_sensitivity[i3] += (d3*half_segment_length)*d_mod3d->slowness_p(i3);
            d_sensitivity[i4] += (d4*half_segment_length)*d_mod3d->slowness_p(i4);
            d_sensitivity[i5] += (d5*half_segment_length)*d_mod3d->slowness_p(i5);
            d_sensitivity[i6] += (d6*half_segment_length)*d_mod3d->slowness_p(i6);
            d_sensitivity[i7] += (d7*half_segment_length)*d_mod3d->slowness_p(i7);
            //
            //printf("sens %lf %lf %lf  %d %d %d\n", lon, lat, depth, ilon, ilat, idep);
            //printf("sens %lf %lf %lf \n", d_lon[pt2], d_lat[pt2], d_depth[pt2] );
            //printf("sens i0: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i0, d_sensitivity[i0], (d0*segment_length)/d_mod3d->vp(i0), d0,  d_mod3d->vp(i0), d_mod3d->depth(i0)  );
            //printf("sens i1: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i1, d_sensitivity[i1], (d1*segment_length)/d_mod3d->vp(i1), d1,  d_mod3d->vp(i1), d_mod3d->depth(i1)  );
            //printf("sens i2: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i2, d_sensitivity[i2], (d2*segment_length)/d_mod3d->vp(i2), d2,  d_mod3d->vp(i2), d_mod3d->depth(i2)  );
            //printf("sens i3: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i3, d_sensitivity[i3], (d3*segment_length)/d_mod3d->vp(i3), d3,  d_mod3d->vp(i3), d_mod3d->depth(i3)  );
            //printf("sens i4: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i4, d_sensitivity[i4], (d4*segment_length)/d_mod3d->vp(i4), d4,  d_mod3d->vp(i4), d_mod3d->depth(i4)  );
            //printf("sens i5: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i5, d_sensitivity[i5], (d5*segment_length)/d_mod3d->vp(i5), d5,  d_mod3d->vp(i5), d_mod3d->depth(i5)  );
            //printf("sens i6: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i6, d_sensitivity[i6], (d6*segment_length)/d_mod3d->vp(i6), d6,  d_mod3d->vp(i6), d_mod3d->depth(i6)  );
            //printf("sens i7: %d %lf %lf = %lf * l / %lf (dep: %lf)\n", i7, d_sensitivity[i7], (d7*segment_length)/d_mod3d->vp(i7), d7,  d_mod3d->vp(i7), d_mod3d->depth(i7)  );
            //printf("%f\n", segment_length);
            //printf("%d %d %d @@ %f %f %f %f %f %f %f %f \n", ilon, ilat, idep, d0, d1, d2, d3, d4, d5, d6, d7);
        }
    }
    ///
    d_traveltime_1d = 0.0;
    for(int idx=0; idx<d_mod3d->npts(); ++idx) {
        if ( fabs(d_sensitivity[idx]) < 1.0e-6 ) continue;
        //double tmp = d_sensitivity[idx] / d_mod3d->vp(idx);
        double tmp = d_sensitivity[idx]; // * d_mod3d->slowness_p(idx);
        //printf("dt: %d %f %f %f \n", idx, d_sensitivity[idx], d_mod3d->slowness_p(idx),  tmp );
        d_traveltime_1d += tmp;
    }
    //printf("%f\n", d_traveltime_1d);
    return 0;
}
int raypath3d::output(const char * filename) {
    FILE *fid = fopen(filename, "w");
    fprintf(fid, "#index lon lat depth  theta phi r  x y \n");
    for(int idx=0; idx< d_npts; ++idx) {
        fprintf(fid, "%d %f %f %f  %f %f %f  %f %f %f\n", idx, d_lon[idx], d_lat[idx], d_depth[idx], d_theta[idx], d_phi[idx], d_r[idx], d_x[idx], d_y[idx], d_z[idx] );
    }
    fclose(fid);
}


/*
int main(int argc, char *argv[])
{
    //ak135::model.output_profile("tmp.txt")
    double err = 1.0e-9; // 1.0e-9 is very good given double type
    double deps[140] = {0. ,  5, 10, 15,
        20.0-err, 20.0+err , 25, 30,  35.-err  , 35.+err,
        77.5 ,  120.  ,  165.  ,  210.-err  , 210+err,
        260.  ,  310.  ,  360.  ,  410.-err , 410+err,  460.  ,  510.  ,  560.  ,
        610.  ,  660.-err, 660.+err  ,  710.  ,  760.  ,  809.5 ,  859.  ,  908.5 ,
        958.  , 1007.5 , 1057.  , 1106.5 , 1156.  , 1205.5 , 1255.  ,
       1304.5 , 1354.  , 1403.5 , 1453.  , 1502.5 , 1552.  , 1601.5 ,
       1651.  , 1700.5 , 1750.  , 1799.5 , 1849.  , 1898.5 , 1948.  ,
       1997.5 , 2047.  , 2096.5 , 2146.  , 2195.5 , 2245.  , 2294.5 ,
       2344.  , 2393.5 , 2443.  , 2492.5 , 2542.  , 2591.5 , 2640.  ,
       2690.  , 2740.  , 2789.67, 2839.33, 2891.5-err, 2891.5+err , 2939.33, 2989.66,
       3039.99, 3090.32, 3140.66, 3190.99, 3241.32, 3291.65, 3341.98,
       3392.31, 3442.64, 3492.97, 3543.3 , 3593.64, 3643.97, 3694.3 ,
       3744.63, 3794.96, 3845.29, 3895.62, 3945.95, 3996.28, 4046.62,
       4096.95, 4147.28, 4197.61, 4247.94, 4298.27, 4348.6 , 4398.93,
       4449.26, 4499.6 , 4549.93, 4600.26, 4650.59, 4700.92, 4801.58,
       4851.91, 4902.24, 4952.58, 5002.91, 5053.24, 5103.57, 5153.5-err, 5153.5+err,
       5204.61, 5255.32, 5306.04, 5356.75, 5407.46, 5458.17, 5508.89,
       5559.6 , 5610.31, 5661.02, 5711.74, 5813.16, 5863.87, 5914.59,
       5965.3 , 6016.01, 6066.72, 6117.44, 6168.15, 6218.86, 6269.57,
       6320.29, 6371 };
    //
    //int size = 7;
    //double xlon[7] = {0, 0,   0,  0, 0,  0, 0};
    //double xlat[7] = {0, 0,   0,  0, 0,  0, 0};
    //double xdep[7] = {0, 35, 20, 35, 0, 20, 0};
    //std::list<double> llon(xlon, xlon+size), llat(xlat, xlat+size), ldep(xdep, xdep+size);
    //ak135::model.adjust_raypath(llon, llat, ldep);
    //std::list<double>::iterator itlon = llon.begin();
    //std::list<double>::iterator itlat = llat.begin();
    //std::list<double>::iterator itdep = ldep.begin();
    //for( ; itlon != llon.end(); ++itlon, ++itlat, ++itdep) {
    //    printf("%f %f %f\n", *itlon, *itlat, *itdep );
    //}
    //return 0;
    //double d = 5.0;
    //double vp = ak135::model.evaulate_from_depth(d, 'p', false);
    //printf("%lf %lf\n", d, vp);
    //return  0;
    //
    earthmod3d mod3d(&ak135::model, 5., 5., deps, 140);
    //mod3d.output_grd_pts("tmp.grd.txt");
    ////
    int npts = 1505;
    double lons[npts], lats[npts], depth[npts];
    FILE *fp1 = fopen("PcP_lon.bin", "rb");
    fread(lons, sizeof(double), npts, fp1);
    fclose(fp1);
    FILE *fp2 = fopen("PcP_lat.bin", "rb");
    fread(lats, sizeof(double), npts, fp2);
    fclose(fp2);
    FILE *fp3 = fopen("PcP_dep.bin", "rb");
    fread(depth, sizeof(double), npts, fp3);
    fclose(fp3);
    raypath3d I2(&mod3d, npts, lons, lats, depth);
    I2.inter_denser_raypath(1000.0);
    //I2.double_denser_raypath();
    //I2.double_denser_raypath();
    //I2.denser_raypath();
    //I2.denser_raypath();
//
    //I2.output("I2.txt");
    I2.grd_sensitivity();
    
    //
    //raypath3d I2(&mod3d, 0.0, 0.0, 0.0, 0.0, 0.0, "PKIKPPKIKP" );
    return 0;
}
*/