#include "earthmodel.hh"
#include <cstdio>

/* earthmod1d
*/
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
    earthmod1d model(6371.0, nlayer, d0, d1, p0, p1, s0, s1, rho0, rho1, "built-in ak135", 0);
    
    double err = 1.0e-9;
    int    dep_grd_size = 135;
    double dep_grd[135] = {0. ,  
                 20.0-err, 20.0+err ,   35.-err  , 35.+err,
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
}; // namespace ak135

int earthmod1d::find_velocity_jump(int verbose) 
{
    for(int idx=1; idx<d_layers.size(); ++idx)
    {
        if ( !(d_layers[idx-1].is_continuous(d_layers[idx], 'a' ) ) )
        {
            d_layer_jump.push_back( std::pair<int, double>(idx-1, d_layers[idx].d_top_depth ) );
            //printf("Warning earthmod1d: a jump between two layers: (%d, %d, %f, %f)\n", 
            //    idx-1, idx, d_layers[idx-1].d_bot_depth, d_layers[idx].d_top_depth );
        }
    }
    //
    if ( d_layer_jump.size() > 0 && verbose != 0) 
    {
        fprintf(stderr, ">>> Warning. There are discontinuities in the 1-D model (`earthmod1d`) %s \n", d_info.c_str() );
    }
    for(auto it= d_layer_jump.begin(); it!= d_layer_jump.end(); ++it)
    {
        int i0 = it->first; 
        double depth = it->second;
        if (verbose)
        {
            fprintf(stderr, "    %d, %d, %lf\n", i0, i0+1, depth);
        }
    }
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

int earthmod1d::output_layers(const char *filename) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "#depth_top(km) vp_top(km/s) vs_top(km/s) rho_top(g/cm^3) depth_bot(km) vp_bot(km/s) vs_bot(km/s) rho_bot(g/cm^3) \n");
    for (std::vector<layer>::iterator it = d_layers.begin(); it != d_layers.end(); ++it ) {
        fprintf(fp, "%.2f %.4f %.4f %.4f    %.2f %.4f %.4f %.4f\n", it->d_top_depth, it->d_top_vp, it->d_top_vs, it->d_top_rho, it->d_bot_depth, it->d_bot_vp, it->d_bot_vs, it->d_bot_rho );
    }
    fclose(fp);
    return 0;
}



/* earthmod3d
*/
// lons should be values in the range of [0, 360) degree
//      lats should be values in the range of [-90, 90] degree
//      depts should be values in the range of [0, 6371] km
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
    // init grid points
    d_npts = ndep*nlat*nlon;
    d_pts.resize(d_npts);

    d_vp.resize(d_npts) ; d_vp.assign( d_npts, 0.0);
    d_vs.resize(d_npts) ; d_vs.assign( d_npts, 0.0);
    d_rho.resize(d_npts); d_rho.assign(d_npts, 0.0);
    d_vp_pert.resize(d_npts); d_vp_pert.assign(d_npts, 1.0);
    d_vs_pert.resize(d_npts); d_vs_pert.assign(d_npts, 1.0);

    d_slowness_p.resize(d_npts);  d_slowness_p.assign( d_npts, 0.0);
    d_slowness_s.resize(d_npts);  d_slowness_s.assign( d_npts, 0.0);
    d_slowness_p_pert.resize(d_npts); d_slowness_p_pert.assign(d_npts, 1.0);
    d_slowness_s_pert.resize(d_npts); d_slowness_s_pert.assign(d_npts, 1.0);

    //
    int ipt = 0;
    double R0 = d_mod1d->radius();
    for (int idep=0; idep<ndep; ++idep)
    {
        double vp = d_mod1d->evaulate_from_depth( d_depth[idep], 'p', false);
        double vs = d_mod1d->evaulate_from_depth( d_depth[idep], 's', false);
        double rho = d_mod1d->evaulate_from_depth(d_depth[idep], 'r', false);
        double sp = ( ISEQUAL(vp, 0.0) ) ? 0.0 : 1.0/vp;
        double ss = ( ISEQUAL(vs, 0.0) ) ? 0.0 : 1.0/vs;
        for (int ilat=0; ilat<nlat; ++ilat)
        {
            for (int ilon=0; ilon<nlon; ++ilon)
            {
                d_pts[ipt].init(d_lon[ilon], d_lat[ilat], d_depth[idep], R0 ); 
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
    return 0;
}
// 
int earthmod3d::init(earthmod1d * mod1d, double dlon, double dlat, const double depth[], int ndep) 
{
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
    if (mode == 'p')
    {
        int junk;
        std::vector<double> deps, lats, lons;
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
        init(mod1d, lons.data(), nlon, lats.data(), nlat, deps.data(), ndep );
        deps.clear();
        lats.clear();
        lons.clear();
        // part 4
        for(int idx=0; idx<d_npts; ++idx) {
            junk = fscanf(fid, "%lf", &(d_vp_pert[idx]) );
        }
        for(int idx=0; idx<d_npts; ++idx) {
            junk = fscanf(fid, "%lf", &(d_vs_pert[idx]) );
        }
        //
        fclose(fid);
        //
        set_mod3d(d_vp_pert.data(), d_vs_pert.data() );
    }
    else if (mode == 'b')
    {
        int junk;
        std::vector<double> deps, lats, lons;
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
        init(mod1d, lons.data(), nlon, lats.data(), nlat, deps.data(), ndep );
        deps.clear();
        lats.clear();
        lons.clear();
        // part 4
        junk = fread(d_vp_pert.data(), sizeof(double), d_npts, fid);
        junk = fread(d_vs_pert.data(), sizeof(double), d_npts, fid);
        //
        fclose(fid);
        //
        set_mod3d(d_vp_pert.data(), d_vs_pert.data() );
    }
    return 0;
}
// Output the model setting files, that is in the same format for `earthmod3d::init`
// So that the output file of this method can be used to init another `earthmod3d`
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
                    junk = fprintf(fid, "%lf ", d_vp_pert[ipt]);
                    ++ipt;
                }
                fprintf(fid, "\n");
            }
        }
        ipt =0;
        for(int idep=0; idep<d_depth.size(); ++idep) {
            for(int ilat=0;ilat<d_lat.size(); ++ilat) {
                for(int ilon=0;ilon<d_lon.size(); ++ilon) {
                    junk = fprintf(fid, "%lf ", d_vs_pert[ipt]);
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
        junk = fread(d_vp_pert.data(), sizeof(double), d_vp_pert.size(), fid);
        junk = fread(d_vs_pert.data(), sizeof(double), d_vs_pert.size(), fid);
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
    fprintf(fp, "#lon lat depth x y z   vp vs rho  dvp(1+e) dvs(1+e)\n" );
    if (type == 'a') 
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            fprintf(fp, "%d %.4f %.4f %.12f %.4f %.4f %.4f   %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].lon, d_pts[idx].lat, d_pts[idx].depth,
                    d_pts[idx].x, d_pts[idx].y, d_pts[idx].z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_vp_pert[idx], d_vs_pert[idx] );
        }
    }
    else if (type == 'p' || type == 'P')
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            if ( ISEQUAL(d_vp_pert[idx], 1.0) ) 
                continue;
            fprintf(fp, "%d %.4f %.4f %.12f %.4f %.4f %.4f   %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].lon, d_pts[idx].lat, d_pts[idx].depth,
                    d_pts[idx].x, d_pts[idx].y, d_pts[idx].z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_vp_pert[idx], d_vs_pert[idx] );
        }
    }
    else if (type == 's' || type == 'S')
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            if ( ISEQUAL(d_vs_pert[idx], 1.0) ) 
                continue;
            fprintf(fp, "%d %.4f %.4f %.12f %.4f %.4f %.4f   %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].lon, d_pts[idx].lat, d_pts[idx].depth,
                    d_pts[idx].x, d_pts[idx].y, d_pts[idx].z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_vp_pert[idx], d_vs_pert[idx] );
        }
    }
    else if (type == 'd' || type == 'D')
    {
        for(int idx = 0; idx<d_npts; ++idx)
        {
            if ( ISEQUAL(d_vp_pert[idx], 1.0) &&  ISEQUAL(d_vs_pert[idx], 1.0) ) 
                continue;
            fprintf(fp, "%d %.4f %.4f %.12f %.4f %.4f %.4f   %f %f %f %f %f\n", 
                    idx,
                    d_pts[idx].lon, d_pts[idx].lat, d_pts[idx].depth,
                    d_pts[idx].x, d_pts[idx].y, d_pts[idx].z,
                    d_vp[idx], d_vs[idx], d_rho[idx],
                    d_vp_pert[idx], d_vs_pert[idx] );
        }
    }
    fclose(fp);
    return 0;
}





// There are mutiple format of an earthmod3d INPUT model setting files. 
// A file is composed of 1~2 parts. 
// 1. The 1st part declares longitude and latitude settings. There are two types:
//
//    1). TYPE 1 (Plain-text or Binary)
//       +----------------+
//       | 1 1 dlon dlat  |
//       +----------------+
//    2). TYPE 2 (Plain-text or Binary)
//       +---------------------+
//       | 1 2                 |
//       | nlon nlat           |
//       | lon1 lon2 ... lonN  |
//       | lat1 lat2 ... latM  |
//       +---------------------+
//
// 2. The 2nd part declares depth setting. There are two types:
//
//    1). TYPE 1 (Plain-text or Binary). This would use built-in 1D model depth setting.
//       +-------+
//       | 2 1   |
//       +-------+
//
//    2). TYPE 2 (Plain-text or Binary)
//       +---------------------+
//       | 2 2                 |
//       | ndep                |
//       | dep1 dep2 ... depL  |
//       +---------------------+
//
int earthmod3d::read_part1(FILE *fid, char mod, 
        int *nlon, double **lons, int *nlat, double **lats)
{
#define MAXGRD 4096    
    static double static_lons[MAXGRD];
    static double static_lats[MAXGRD];
    if (mod == 'p')
    {
        int part_id, type_id;
        int junk = fscanf(fid, "%d %d", &part_id, &type_id);
        if (part_id != 1 || junk != 2)
        {
            fprintf(stderr, "Err. wrong part one ID (%d), or missing of type_id\n.", part_id);
            exit(-1);
        }

        if (type_id == 1)
        {
            //    1). TYPE 1 (Plain-text or Binary)
            //       +----------------+
            //       | 1 1 dlon dlat  |
            //       +----------------+
            double dlon, dlat;
            junk = fscanf(fid, "%lf %lf", &dlon, &dlat);
            if (junk != 2)
            {
                fprintf(stderr, "Err in reading dlon and dlat\n." );
                exit(-1);
            }
            *nlon = (int) (round(360.0/dlon) );
            *nlat = (int) (round(180.0/dlat) );
            for(int idx=0; idx<*nlon; ++idx) { static_lons[idx] = idx*dlon; }
            for(int idx=0; idx<*nlat; ++idx) { static_lats[idx] = idx*dlat-90.0; }
            *lons = static_lons;
            *lats = static_lats;
        }
        else if (type_id == 2)
        {
            //    2). TYPE 2 (Plain-text or Binary)
            //       +---------------------+
            //       | 1 2                 |
            //       | nlon nlat           |
            //       | lon1 lon2 ... lonN  |
            //       | lat1 lat2 ... latM  |
            //       +---------------------+
            int junk = fscanf(fid, "%d %d", nlon, nlat);
            for(int idx=0; idx<*nlon; ++idx) { junk = fscanf(fid, "%lf", &(static_lons[idx]) ); }
            for(int idx=0; idx<*nlat; ++idx) { junk = fscanf(fid, "%lf", &(static_lats[idx]) ); }
            *lons = static_lons;
            *lats = static_lats;
        }
        else 
        {
            fprintf(stderr, "Err. wrong type_id (%d) in part one\n", type_id);
            exit(-1);
        }
    }
    else if (mod == 'b')
    {
        int part_id, type_id;
        int junk;
        junk = fread(&part_id, sizeof(int), 1, fid);
        junk = fread(&type_id, sizeof(int), 1, fid);
        if (part_id != 1 || junk != 2)
        {
            fprintf(stderr, "Err. wrong part one ID (%d), or missing of type_id\n.", part_id);
            exit(-1);
        }

        if (type_id == 1)
        {
            //    1). TYPE 1 (Plain-text or Binary)
            //       +----------------+
            //       | 1 1 dlon dlat  |
            //       +----------------+
            double dlon, dlat;
            junk = fread(&dlon, sizeof(double), 1, fid);
            junk = fread(&dlat, sizeof(double), 1, fid);
            *nlon = (int) (round(360.0/dlon) );
            *nlat = (int) (round(180.0/dlat) );
            for(int idx=0; idx<*nlon; ++idx) { static_lons[idx] = idx*dlon; }
            for(int idx=0; idx<*nlat; ++idx) { static_lats[idx] = idx*dlat-90.0; }
            *lons = static_lons;
            *lats = static_lats;
        }
        else if (type_id == 2)
        {
            //    2). TYPE 2 (Plain-text or Binary)
            //       +---------------------+
            //       | 1 2                 |
            //       | nlon nlat           |
            //       | lon1 lon2 ... lonN  |
            //       | lat1 lat2 ... latM  |
            //       +---------------------+
            junk = fread(nlon, sizeof(int), 1, fid);
            junk = fread(nlat, sizeof(int), 1, fid);
            junk = fread(static_lons, sizeof(double), *nlon, fid);
            junk = fread(static_lats, sizeof(double), *nlat, fid);
            *lons = static_lons;
            *lats = static_lats;
        }
        else 
        {
            fprintf(stderr, "Err. wrong type_id (%d) in part one\n", type_id);
            exit(-1);
        }
    }
    return 0;
#undef MAXGRD
}
int earthmod3d::read_part2(FILE *fid, char mod, int *ndep, double **deps)
{
#define MAXGRD 4096
    static double static_deps[MAXGRD];
    if (mod == 'p')
    {
        int part_id, type_id;
        int junk = fscanf(fid, "%d %d", &part_id, &type_id);
        if (part_id != 2 || junk != 2)
        {
            fprintf(stderr, "Err. wrong part two ID (%d), or missing of type_id\n.", part_id);
            exit(-1);
        }

        if (type_id == 1)
        {
            // +-------+
            // | 2 1   |
            // +-------+
            *ndep  = ak135::dep_grd_size;
            *deps  = ak135::dep_grd;
        }
        else if (type_id == 2)
        {
            // +---------------------+
            // | 2 2                 |
            // | ndep                |
            // | dep1 dep2 ... depL  |
            // +---------------------+
            junk = fscanf(fid, "%d", ndep);
            for(int idx=0; idx<*ndep; ++idx) { junk = fscanf(fid, "%lf", &(static_deps[idx]) ); }
            *deps = static_deps;
        }
        else
        {
            fprintf(stderr, "Err. wrong typeid (%d) in part two\n", type_id);
            exit(-1);
        }
    }
    else if (mod == 'b')
    {
        int part_id, type_id;
        int junk;
        junk = fread(&part_id, sizeof(int), 1, fid);
        junk = fread(&type_id, sizeof(int), 1, fid);
        if (part_id != 2 || junk != 2)
        {
            fprintf(stderr, "Err. wrong part two ID (%d), or missing of type_id\n.", part_id);
            exit(-1);
        }

        if (type_id == 1)
        {
            // +-------+
            // | 2 1   |
            // +-------+
            *ndep  = ak135::dep_grd_size;
            *deps  = ak135::dep_grd;
        }
        else if (type_id == 2)
        {
            // +---------------------+
            // | 2 2                 |
            // | ndep                |
            // | dep1 dep2 ... depL  |
            // +---------------------+
            junk = fread(ndep, sizeof(int), 1, fid);
            junk = fread(static_deps, sizeof(double), *ndep, fid);
            *deps = static_deps;
        }
        else
        {
            fprintf(stderr, "Err. wrong typeid (%d) in part two\n", type_id);
            exit(-1);
        }
    }
#undef MAXGRD
}
