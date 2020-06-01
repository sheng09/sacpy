#include "global_tomo.hh"
#include "getopt/getopt.hpp"


//int add_plume(earthmod3d * mod, const double dvp, const double dvs,
//    const double lon, const double lat, const double radius, 
//    const double d0, const double d1, const double smooth_degree= 5.0) 
//{
//    mod->set_mod3d_cylinder(d0, d1, lon, lat, radius, dvp, dvs);
//    mod->set_mod3d_smooth_sphere(smooth_degree);
//    return 0;
//}

static char HMSG[] = "%s --dlon=2.5 --dlat=2.5 -O=outfnm.txt \
--plume=lon/lat/radius/d0/d1/dvp/dvs  [--plume=lon/lat/radius/d0/d1/dvp/dvs] \
--smooth=5.0\
\n";


bool obtain_single_plume(char av[], double *lon, double *lat, double *radius, double *d0, double *d1, double *dvp, double *dvs ) 
{
    if (strncmp(av, "--plume=", 8 ) == 0 )
    {
        int n = sscanf(av+8, "%lf/%lf/%lf/%lf/%lf/%lf/%lf", lon, lat, radius, d0, d1, dvp, dvs);
        if (n != 7) 
        {
            fprintf(stderr, "Err: wrong plume settings (%s)\n", av );
            exit(-1);
        }
        return true;
    }
    return false;
}

int main(int argc, char *argv[])
{
    if (argc<3) {
        fprintf(stderr, HMSG, argv[0]);
        exit(0);
    }
    double dlon = getarg( 1.0, "--dlon");
    double dlat = getarg( 1.0, "--dlat");
    std::string outfnm = getarg("ak135mod.txt", "-O", "--output");
    fprintf(stderr, ">>> dlon: %lf, dlat: %lf\n", dlon, dlat);
    //
    double err = 1.0e-9; // 1.0e-9 is very good given double type
    double deps[135] = {0. ,  
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
    earthmod3d mod3d(&ak135::model, dlon, dlat, deps, 135);
    //
    int nplume = 0;
    double lon, lat, radius, d0, d1, dvp, dvs;
    for(int iav=1; iav<argc; ++iav) 
    {
        if (obtain_single_plume(argv[iav], &lon, &lat, &radius, &d0, &d1, &dvp, &dvs) ) {
            mod3d.set_mod3d_cylinder(d0, d1, lon, lat, radius, dvp, dvs);
            ++nplume;
            fprintf(stdout, ">>> Adding plume [%d]. %s\n", nplume, argv[iav] );
        }
    }
    //
    double smooth_grid_deg = getarg( 0.0, "--smooth");
    mod3d.set_mod3d_smooth_sphere(smooth_grid_deg);
    //
    mod3d.output_grd_pts(outfnm.c_str() );
    //
    return 0;
}
