#include "global_tomo.hh"
#include "getopt/getopt.hpp"

static char HMSG[] = "%s --dlon=2.5 --dlat=2.5 --model_i=infnm.txt --model_i_mod=[p|b]\n\
--model_o=outfnm.txt --model_o_mod=[p|b] --grd_o=outfnm.txt --grd_o_mod=a \n\
[--plume=lon/lat/radius/d0/d1/dvp/dvs]  [--plume=lon/lat/radius/d0/d1/dvp/dvs] \n\
[--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs] [--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs]\n\
--smooth=5.0\n\
\n\
To generate and output 3-D forward model. \
The 3-D model can be generated with a background model, then be modified by adding different elements.\n\
\n\
The background model can be\n\
1. a built-in 1-D ak-135 model. \n\
   Use `--dlon=XXX --dlat=XXX`, and the discretization in depth dimension is the same as the setting in ak135.\n\
2. an input 3-D model.\n\
   Use `--model_i=XXX --model_i_mod=[p|b]` where `p` represent plain-text format file and `b` binary format.\n\
   The model file has the format of:\n\
        +-------------------------+\n\
        |ndep nlat nlon           |\n\
        |dep1 dep2 dep3 ... depN  |\n\
        |lat1 lat2 lat3 ... latM  |\n\
        |lon1 lon2 lon3 ... lonL  |\n\
        |dvp1 dvp2 dvp3 ... dvpX  |\n\
        |dvs1 dvs2 dvs3 ... dvsX  |\n\
        +-------------------------+\n\
   If binary format is used, then the datatype should be `int`(int32) and `double`(float64).\n\
   Please note, `dvp` and `dvs` here is 1.0+perturbation. For example, use `1.05` for 5 percent fast velocity\n\
   anomaly, and `0.95` for 5 percent low velocity anomaly.\n\
\n\
Elements can be added to the background model by:\n\
1. cylinder plume.\n\
   Use mutiple `--plume=lon/lat/radius/d0/d1/dvp/dvs` to add plumes.\n\
2. cube.\n\
   Use mutiple `--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs` to add cubes.\n\
   Please note, `dvp` and `dvs` here is 1.0+perturbation. For example, use `1.05` for 5 percent fast velocity\n\
   anomaly, and `0.95` for 5 percent low velocity anomaly.\n\
\n\
Args:\n\
--dlon=        :\n\
--dlat=        :\n\
--model_i=     : filename for input 3-D model.\n\
--model_i_mod= : `p` for plain-text format.\n\
                 `b` for binary format.\n\
--grd_o=       : filename to output 3-D model grid points.\n\
--grd_o_mod=   : `a` to output all points.\n\
                 `p` to output points with P velocity perturbation being not zero.\n\
                 `s` ...                   S ...\n\
                 `d` ...                   P and S ...\n\
--model_o=     : filename to output 3-D model.\n\
--model_o_mod= : `p` for plain-text format.\n\
                 `b` for binary format.\n\
--plume=lon/lat/radius/d0/d1/dvp/dvs: add a plume.\n\
--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs : add a cube.\n\
--smooth=radius: to smooth the 3-Dmodel in longitude-latitude dimension given a radius in degree.\n\
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
bool obtain_single_cube(char av[], double *lon0, double *lon1, double *lat0, double *lat1, double *d0, double *d1, double *dvp, double *dvs) 
{
    if (strncmp(av, "--cube=", 7 ) == 0 )
    {
        int n = sscanf(av+7, "%lf/%lf/%lf/%lf/%lf/%lf/%lf/%lf", lon0, lon1, lat0, lat1, d0, d1, dvp, dvs);
        if (n != 8) 
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
    /*
     parameter to background model.
    */
    double dlon = getarg( -10.0, "--dlon");
    double dlat = getarg( -10.0, "--dlat");
    std::string fnm_model_in = getarg("",  "--model_i");
    char fnm_model_in_mod    = getarg('p', "--model_i_mod");
    if ( (dlon>0.0 && dlat>0.0) && !fnm_model_in.empty() ) {
        fprintf(stderr, "Err, conflict through setting two background models.\n");
        exit(-1);
    } else if ( (dlon<0.0 && dlat<0.0) && fnm_model_in.empty() ) {
        fprintf(stderr, "Err, one background model must be set.\n");
        exit(-1);
    }
    earthmod3d * mod3d;
    if (dlon>0.0 && dlat>0.0) {
        double err = 1.0e-9; // 1.0e-9 is very good given double type
        static double deps[135] = {0. ,  
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
        fprintf(stderr, ">>> Init 3-D model from internal ak135. dlon: %lf, dlat: %lf\n", dlon, dlat);
        mod3d = new earthmod3d(&ak135::model, dlon, dlat, deps, 135);
    }
    else if (!fnm_model_in.empty() ) {
        fprintf(stderr, ">>> Init 3-D model from external file (%s)\n", fnm_model_in.c_str());
        mod3d = new earthmod3d(&ak135::model, fnm_model_in.c_str(), fnm_model_in_mod);
    }
    /*
        add plume
    */
    int nplume = 0;
    double lon, lat, radius, d0, d1, dvp, dvs;
    for(int iav=1; iav<argc; ++iav) 
    {
        if (obtain_single_plume(argv[iav], &lon, &lat, &radius, &d0, &d1, &dvp, &dvs) ) {
            mod3d->set_mod3d_cylinder(d0, d1, lon, lat, radius, dvp-1.0, dvs);
            ++nplume;
            fprintf(stdout, ">>> Adding plume [%d]. %s\n", nplume, argv[iav] );
        }
    }
    /*
        add cube
    */
    double lon0, lon1, lat0, lat1;
    int ncube = 0;
    for(int iav=1; iav<argc; ++iav) 
    {
        if (obtain_single_cube(argv[iav], &lon0, &lon1, &lat0, &lat1, &d0, &d1, &dvp, &dvs) ) {
            mod3d->set_mod3d_cube(d0, d1, lon0, lon1, lat0, lat1, dvp-1.0, dvs-1.0);
            ++ncube;
            fprintf(stdout, ">>> Adding cube [%d]. %s\n", ncube, argv[iav] );
        }
    }
    /*
        smooth
    */
    double smooth_grid_deg = getarg( 0.0, "--smooth");
    if (smooth_grid_deg > 0.0)
    {
        fprintf(stdout, ">>> Smoothing... %lf\n", smooth_grid_deg );
        mod3d->set_mod3d_smooth_lonlat(smooth_grid_deg);
    }
    /*
        output
    */
    std::string outfnm = getarg("", "-O", "--grd_o");
    char output_mod = getarg('d', "--output_mod", "--grd_o_mod");
    if (!outfnm.empty())
    {
        fprintf(stdout, ">>> Output 3-D model grid points... (%s)\n", outfnm.c_str() );
        mod3d->output_grd_pts(outfnm.c_str(), output_mod);
    }
    outfnm = getarg("", "--model_o");
    output_mod = getarg('p', "--model_o_mod");
    if (!outfnm.empty())
    {
        fprintf(stdout, ">>> Output 3-D model file... (%s)\n", outfnm.c_str() );
        mod3d->output_model(outfnm.c_str(), output_mod);
    }
    //
    delete mod3d;
    //
    return 0;
}
