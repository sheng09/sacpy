#include "earthmodel.hh"
#include "getopt/getopt.hpp"

static char HMSG[] = "\
To generate and output 3-D forward model. \
\n\
%s --dlon=2.5 --dlat=2.5 --model_i=infnm.txt --model_i_mod=[p|b]\n\
    --model_o=outfnm.txt --model_o_mod=[p|b] --grd_o=outfnm.txt --grd_o_mod=a \n\
    [--plume=lon/lat/radius/d0/d1/dvp/dvs]  [--plume=lon/lat/radius/d0/d1/dvp/dvs] \n\
    [--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs] [--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs]\n\
    --smooth=5.0\n\
\n\
The 3-D model is generated 1) by setting a background model, and then 2) by adding some elements.\n\
\n\
    1) The background model can be\n\
        a. a built-in 1-D ak-135 model. \n\
           Use `--dlon=X.X --dlat=X.X`, and the discretization in depth dimension\n\
           is the same as the setting in ak135.\n\
        b. an input 3-D model.\n\
           Use `--model_i=filename --model_i_mod=[p|b]` where `p` represent plain-text \n\
           format file and `b` binary format. The model file has the format of:\n\
                +-------------------------+\n\
                |ndep nlat nlon           |\n\
                |dep1 dep2 dep3 ... depN  |\n\
                |lat1 lat2 lat3 ... latM  |\n\
                |lon1 lon2 lon3 ... lonL  |\n\
                |dvp1 dvp2 dvp3 ... dvpX  |\n\
                |dvs1 dvs2 dvs3 ... dvsX  |\n\
                +-------------------------+\n\
           If binary format is used, then the datatype should be `int`(int32) and `double`(float64).\n\
           \n\
           Please avoid depth discontinuities, such as 20, 35, 210, 410, 660, 2891.5, 5153.5km in ak135.\n\
           Instead, use two depths around both sides of a discontinuity. For example, 19.999 and 20.001\n\
           for the 20 discontinuity, and 2891.499 and 2891.501 for the 2891.5 discontinuity, etc.\n\
           \n\
           Please note, `dvp` and `dvs` here is 1.0+perturbation. For example, use `1.05` for 5 percent\n\
           fast velocity anomaly, and `0.95` for 5 percent low velocity anomaly.\n\
\n\
    2) Elements can be added to the background model by:\n\
        a. cylinder plume.\n\
           Use mutiple `--plume=lon/lat/radius/d0/d1/dvp/dvs` to add plumes.\n\
        b. cube.\n\
           Use mutiple `--cube=lon0/lon1/lat0/lat1/d0/d1/dvp/dvs` to add cubes.\n\
           Please note, `dvp` and `dvs` here is 1.0+perturbation. For example, use `1.05` for 5 percent\n\
           fast velocity anomaly, and `0.95` for 5 percent low velocity anomaly.\n\
\n\
Args details:\n\
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
        int ndep = ak135::dep_grd_size;
        double *dep = ak135::dep_grd;
        fprintf(stderr, ">>> Init 3-D model from internal ak135. dlon: %lf, dlat: %lf\n", dlon, dlat);
        mod3d = new earthmod3d(&ak135::model, dlon, dlat, dep, ndep);
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
