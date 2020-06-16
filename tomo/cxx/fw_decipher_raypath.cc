#include "raypath.hh"
#include "getopt/getopt.hpp"
#include "hdf5_hl.h"


static char HMSG[] = "\
Decipher raypath sgements with respect to seismic phases.\n\
\n\
%s --raypath=fnm --output=fnm [--verbose]\n\
\n\
";

int main(int argc, char const *argv[])
{
    static char verbose_msg[MAXMSG_LEN];
    if (argc == 1)
    {
        fprintf(stdout, HMSG, argv[0]);
        return 0;
    }
    // verbose
    bool verbose = getarg(false, "--verbose", "-v");
    // input HDF5
    std::string ray_fnm = getarg("", "--raypath");
    if (ray_fnm.empty() )
    {
        fprintf(stderr, "Err. please provide raypath file.\n");
        exit(-1);
    }
    int nray;
    hid_t fid = H5Fopen(ray_fnm.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t  grp_raypath = H5Gopen2(fid, "raypath", H5P_DEFAULT);
    H5LTget_attribute_int(grp_raypath, ".", "size", &nray);
    // output HDF5
    std::string output_raypath_segments = getarg("", "--output");
    if (output_raypath_segments.empty() )
    {
        fprintf(stderr, "Err. please provide output_raypath_segments file.\n");
        exit(-1);
    }
    hid_t out_fid  = H5Fcreate(output_raypath_segments.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t grp_segments = H5Gcreate2(out_fid, "raypath_segments", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //hid_t grp_sens = H5Gcreate2(out_fid, "sensitivity_zip", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //H5LTset_attribute_int(grp_sens, ".", "size", &nray, 1);
    //H5LTset_attribute_string(grp_sens, ".", "info", "The sensitivity is based on 1D earth model");
    // Loop over all raypath
    raypath3d   ray;
    if (verbose)
    {
        verbose_print(0, "Start running for each raypath...\n");
    }
    for (int idx=0; idx<nray; ++idx)
    {
        static char sub_grp_name[4096];
        std::memset(sub_grp_name, 0, 4096);
        ssize_t junk = H5Lget_name_by_idx (grp_raypath, ".", H5_INDEX_NAME, H5_ITER_INC, idx, sub_grp_name, 4096, H5P_DEFAULT);
        
        int npts;
        int id;
        double rp, time;
        static char c_phase[4096], c_tag[4096];
        std::memset(c_phase, 0, 4096);
        std::memset(c_tag, 0, 4096);

        H5LTget_attribute_int(grp_raypath,    sub_grp_name, "id", &id);
        H5LTget_attribute_int(grp_raypath,    sub_grp_name, "npts", &npts);
        H5LTget_attribute_double(grp_raypath, sub_grp_name, "ray_param", &rp);
        H5LTget_attribute_double(grp_raypath, sub_grp_name, "time", &time);
        H5LTget_attribute_string(grp_raypath, sub_grp_name, "phase", c_phase);
        H5LTget_attribute_string(grp_raypath, sub_grp_name, "tag", c_tag);
        std::string phase(c_phase);
        std::string tag(c_tag);

        if (verbose)
        {
            std::memset(verbose_msg, 0, MAXMSG_LEN);
            sprintf(verbose_msg, "[%d/%d] running... raypath id(%d) tag(%s) npts(%d)\n", idx+1, nray, id, c_tag, npts );
            verbose_print(1, verbose_msg);
        }

        std::vector<double> lons(npts), lats(npts), deps(npts);
        std::string loc(sub_grp_name);
        std::string loclon = loc+"/lon";
        std::string loclat = loc+"/lat";  
        std::string locdep = loc+"/depth";
        H5LTread_dataset_double(grp_raypath,  loclon.c_str(), lons.data() );
        H5LTread_dataset_double(grp_raypath,  loclat.c_str(), lats.data() );
        H5LTread_dataset_double(grp_raypath,  locdep.c_str(), deps.data() );

        // check ray-path
        // int test_idx = 900;
        // fprintf(stdout, "    %d: %.12lf %.12lf %.12lf\n", test_idx, lons[test_idx], lats[test_idx], deps[test_idx] );
        ray.init(c_phase, npts, lons.data(), lats.data(), deps.data(),time, rp, c_tag, 2);

        //
        int n_seg = ray.size();
        hid_t single_path = H5Gcreate2(grp_segments, sub_grp_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5LTset_attribute_int(   single_path, ".", "size", &n_seg, 1);
        H5LTset_attribute_int(   single_path, ".", "id", &id, 1);
        H5LTset_attribute_double(single_path, ".", "ray_param", &rp, 1);
        H5LTset_attribute_double(single_path, ".", "time", &time, 1);
        H5LTset_attribute_string(single_path, ".", "phase", c_phase);
        H5LTset_attribute_string(single_path, ".", "tag", c_tag);
        //
        int iseg=0;
        for(auto seg=ray.begin(); seg!=ray.end(); ++seg, ++iseg )
        {
            static char name[5];
            std::memset(name, 0, 5);
            sprintf(name, "%03d", iseg);
            hid_t single_seg = H5Gcreate2(single_path, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5LTset_attribute_int(   single_seg, ".", "idx", &iseg, 1);
            char type[1];
            type[0] = seg->type();
            H5LTset_attribute_string(  single_seg, ".", "type", type);
            int npts = seg->size();
            H5LTset_attribute_int(  single_seg, ".", "npts", &npts, 1);

            hsize_t dim[1];
            dim[0] = seg->size();
            std::vector<double> lon, lat, dep;
            seg->tovec(lon, lat, dep);

            H5LTmake_dataset_double(single_seg, "lon",   1, dim, lon.data() );
            H5LTmake_dataset_double(single_seg, "lat",   1, dim, lat.data() );
            H5LTmake_dataset_double(single_seg, "depth", 1, dim, dep.data() );

            H5Gclose(single_seg);
        }


        H5Gclose(single_path);
    }
    H5Gclose(grp_raypath);
    H5Fclose(fid);
    H5Gclose(grp_segments);
    H5Fclose(out_fid);
    //
    verbose_print(0, "Safely exit\n");
    return 0;
}
