#ifndef TOMO_EARTHMODEL______
#define TOMO_EARTHMODEL______

#include "raypath.hh"
#include <vector>
#include <string>
#include <list>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <algorithm> 
#include <string>
#include <cstring>
#include <unordered_map>

class layer {
public:
    layer() {}
    layer(double d0,   double d1, double p0,   double p1, double s0,   double s1, double rho0, double rho1)
    {
        init(d0, d1, p0, p1, s0, s1, rho0, rho1);
    }
    ~layer() {}
    int init(double d0,   double d1, double p0,   double p1, double s0,   double s1, double rho0, double rho1)
    {
        d_top_depth = d0;
        d_bot_depth = d1;
        d_top_vp = p0;
        d_bot_vp = p1;
        d_top_slowness_p = (p0 > 0.0) ? (1.0/p0) : 0.0 ;
        d_bot_slowness_p = (p1 > 0.0) ? (1.0/p1) : 0.0 ;
        d_top_vs = s0;
        d_bot_vs = s1;
        d_top_slowness_s = (s0 > 0.0) ? (1.0/s0) : 0.0;
        d_bot_slowness_s = (s1 > 0.0) ? (1.0/s1) : 0.0;
        d_top_rho = rho0;
        d_bot_rho = rho1;
        //
        double tmp = d1-d0;
        kp   =  (d1!=d0) ? (p1-p0)/tmp     : 0.0 ;
        ks   =  (d1!=d0) ? (s1-s0)/tmp     : 0.0 ;
        krho =  (d1!=d0) ? (rho1-rho0)/tmp : 0.0 ;
        k_slowness_p = (d1!=d0) ? (d_bot_slowness_p-d_top_slowness_p)/tmp : 0.0;
        k_slowness_s = (d1!=d0) ? (d_bot_slowness_s-d_top_slowness_s)/tmp : 0.0;
        //
        double r0 = 6371.0-d0; 
        double r1 = 6371.0-d1;
        Bp = log(p0/p1) / log(r0/r1);
        Ap = p0/pow(r0, Bp);
        Bs = (ISEQUAL(s0, 0.0)) ?  0.0: log(s0/s1) / log(r0/r1) ;
        As = (ISEQUAL(s0, 0.0)) ?  0.0: s0/pow(r0, Bs);
        return 0;
    }
    char inside_layer(double depth)
    {
        /* check if a depth is inside or outside the layer.
            return 'i' inside this layer that includes the point exactly on the up or lower boundaries.
                   'b' below this layer 
                   'a' above this layer
        */
       if (depth < d_top_depth)
       {
           return 'a';
       }
       else if (depth > d_bot_depth)
       {
           return 'b';
       }
       else 
       {
           return 'i';
       }
    }
    double interpolate_velocity(double depth, char type )
    {
        // return interpolated value using Bullen's law
        // Type: 'p', 's', or 'r/d'
        double v = 0.0;
        double dx = depth - d_top_depth;
        if (type == 'p')
        {
            // v = d_top_vp+dx*kp;
            v = Ap * pow(6371.0-depth, Bp);
        }
        else if (type == 's')
        {
            //v = d_top_vs+dx*ks;
            v = As * pow(6371.0-depth, Bs);
        }
        else if (type == 'r' || type == 'd')
        {
            v = d_top_rho+dx*krho;
        }
        return v;
    }
    double interpolate_slowness(double depth, char type ) {
        double v = interpolate_velocity(depth, type);
        v = ISEQUAL(v, 0.0) ? 0.0 : 1.0/v;
        //double v=0.0;
        //double dx = depth - d_top_depth;
        //double k  = (type == 'p' ) ? k_slowness_p : ( (type=='s') ? k_slowness_s : krho );
        //double x0 = (type == 'p' ) ? d_top_slowness_p : ( (type=='s') ? d_top_slowness_s : d_top_rho );
        //v = k*dx+x0;
        return v;
    }
    /* check if `this` layer and `layer2` are continuous, or this is a jump (e.g., depth jump or velocity jump).
        `type` can be `p` for P velocity,  `s` for S velocity, `r` or `d` for density, and `a` for all of those. 
    */
    bool  is_continuous(layer & layer2, char type ) {
        if (type == 'a') {
            return is_continuous(layer2, 'p') && is_continuous(layer2, 's') && is_continuous(layer2, 'r');
        }
        if (ISEQUAL(d_top_depth, layer2.d_bot_depth) ) {
            if (type == 'p' && ISEQUAL(d_top_vp, layer2.d_bot_vp)  ) 
            { 
                //double x = d_top_vp, y = layer2.d_bot_vp;
                //printf("%f %f %f %f\n", d_top_vp, layer2.d_bot_vp, fabs((x)-(y)), fabsf((x)+(y))*1.0e-20);
                return true; }
            else if (type == 's' && ISEQUAL(d_top_vs, layer2.d_bot_vs )) 
            { return true; }
            else if ( (type == 'd' || type == 'r') && ISEQUAL(d_top_rho, layer2.d_bot_rho) ) 
            { return true; }
            else 
            { return false; }
        } 
        else {
            return layer2.is_continuous(*this, type);
        }
        return false;
    }
public: 
    double d_top_depth, d_bot_depth;
    double d_top_vp, d_bot_vp, d_top_slowness_p, d_bot_slowness_p;
    double d_top_vs, d_bot_vs, d_top_slowness_s, d_bot_slowness_s;
    double d_top_rho, d_bot_rho;
    // linear velocity and slowness
    double kp, ks, krho;
    double k_slowness_p, k_slowness_s;
    // bullen law for velocity: v(r) = Ar^B
    double Ap, Bp, As, Bs;
};



class earthmod1d {
public:
    earthmod1d() {}
    earthmod1d( double earth_radius, double nlayer, 
                const double d0[], const double d1[],
                const double p0[], const double p1[],
                const double s0[], const double s1[],
                const double r0[], const double r1[], 
                const char * info, int verbose= 0 )
    {
        set_info(info);
        init(earth_radius, nlayer, d0, d1, p0, p1, s0, s1, r0, r1, verbose);
    }
    ~earthmod1d() {
        if (!d_layers.empty() ) d_layers.clear();
        if (!d_layer_jump.empty() ) d_layer_jump.clear();
    }
    /* Set the 1-D layered model
    */
    int init(   double earth_radius, int nlayer, 
                const double d0[], const double d1[],
                const double p0[], const double p1[],
                const double s0[], const double s1[],
                const double r0[], const double r1[], int verbose= 0 ) 
    {
        d_earth_radius = earth_radius;
        d_layers.resize(nlayer);
        for(int idx=0; idx<nlayer; ++idx) {
            d_layers[idx].init( d0[idx], d1[idx], p0[idx], p1[idx], 
                                s0[idx], s1[idx], r0[idx], r1[idx] );
        }
        find_velocity_jump(verbose);
        return 0;
    }
    inline int set_info(const char *info) { d_info = info; return 0;  }
    //
    int search_layer(double depth);
    //int search_layer(const double depth1, const double depth2);
    double evaulate_from_depth(double depth, char type, bool return_slowness);
    double radius() { return d_earth_radius; }
    int  output_layers(const char *filename); // output layers into plain-text
    bool is_depth_on_discontinuity(double depth, int *ptr_above, int *ptr_below) {
        for(std::vector<std::pair<int, double> >::iterator it = d_layer_jump.begin(); it != d_layer_jump.end(); ++it) {
            int i_above = it->first;
            double dis_depth = it->second;
            if ( ISEQUAL(depth, dis_depth ) ) {
                *ptr_above  = i_above;
                *ptr_below  = i_above+1;
                return true;
            }
        }
        return false;
    }
    bool is_depth_cross_discontinuity(double d0, double d1, double * d_dis) {
        // check if a line by two points across any discontinuity
        // return the minimal depth if there are more than 1 across
        for(std::vector<std::pair<int, double> >::iterator it = d_layer_jump.begin(); it != d_layer_jump.end(); ++it) {
            double dis_depth = it->second;
            if ((d0-dis_depth) * (d1-dis_depth) < 0 ) {
                *d_dis = dis_depth;
                return true;
            }
        }
        return false;
    }
    //
    std::vector<std::pair<int, double> > & discontinuous_layers() { return d_layer_jump; }
private:
    int find_velocity_jump(int verbose=0);
private:
    /* Model
    */
    std::string d_info; // a string to briefly describe this model
    double d_earth_radius;
    std::vector<layer> d_layers;
    /* Discontinuities are:
        -------------------------------
         layer i
        ---------depthd-iscontinuity---
         layer i+1
        -------------------------------

        <i, depth> is stored in `d_layer_jump` 
    */
    std::vector<std::pair<int, double> >  d_layer_jump;
};



class earthmod3d {
public:
    earthmod3d() {};
    earthmod3d(earthmod1d * mod1d, const double lons[], int nlon,  const double lats[], int nlat, const double depth[], int ndep)  {
        init(mod1d, lons, nlon, lats, nlat, depth, ndep);
    }
    earthmod3d(earthmod1d * mod1d, double dlon, double dlat, const double depth[], int ndep)  {
        init(mod1d, dlon, dlat, depth, ndep);
    }
    earthmod3d(earthmod1d * mod1d, const char *filename, char mode) {
        init(mod1d, filename, mode);
    }
    ~earthmod3d() {
        if(!d_lon.empty() ) d_lon.clear();
        if(!d_lat.empty() ) d_lat.clear();
        if(!d_depth.empty() ) d_depth.clear();
        if(!d_vp.empty() ) d_vp.clear();
        if(!d_vs.empty() ) d_vs.clear();
        if(!d_rho.empty() ) d_rho.clear();

        if(!d_vp_pert.empty() ) d_vp_pert.clear();
        if(!d_vs_pert.empty() ) d_vs_pert.clear();
        if(!d_pts.empty() ) d_pts.clear();
        if(!d_slowness_p.empty()  ) d_slowness_p.clear();
        if(!d_slowness_s.empty()  ) d_slowness_s.clear();

        if(!d_slowness_p_pert.empty() ) d_slowness_p_pert.clear();
        if(!d_slowness_s_pert.empty() ) d_slowness_s_pert.clear();
        d_mod1d = NULL;
    }
    /* Init 3-D grid
    */
    int init(earthmod1d * mod1d, const double lons[], int nlon,  const double lats[], int nlat, const double depth[], int ndep );
    int init(earthmod1d * mod1d, double dlon, double dlat, const double depth[], int ndep) ;
    int init(earthmod1d * mod1d, const char *fnm, char mode);
private:
    int read_part1(FILE *fid, char mod, int *nlon, double **lons, int *nlat, double **lats);
    int read_part2(FILE *fid, char mod, int *ndep, double **deps);
public:
    /* Access 3-D grid 
    */
    const earthmod1d * mod1d() const { return d_mod1d; }
    int   output_model(const char * filename, char mode); 
    int   output_grd_pts(const char * filename, char type); 
    long   npts() const { return d_npts; }
    double radius() const { return d_mod1d->radius(); }
    earthmod1d * mod1d() { return d_mod1d; }

    // double vp(int pt_idx) { return d_vp[pt_idx]; }
    // double vs(int pt_idx) { return d_vs[pt_idx]; }
    // double slowness_p(int pt_idx) { return d_slowness_p[pt_idx]; }
    // double slowness_s(int pt_idx) { return d_slowness_s[pt_idx]; }
    std::vector<double> & vp_vector() { return d_vp; }
    std::vector<double> & vs_vector() { return d_vs; }
    std::vector<double> & slowness_p_vector() { return d_slowness_p; }
    std::vector<double> & slowness_s_vector() { return d_slowness_s; }
    std::vector<double> & slowness_p_pert() { return d_slowness_p_pert; }
    std::vector<double> & slowness_s_pert() { return d_slowness_s_pert; }

    // const  pt3d &  point(int pt_idx) const { return d_pts[pt_idx]; }
    // double lon(int pt_idx)   const { return d_pts[pt_idx].lon;    }
    // double lat(int pt_idx)   const { return d_pts[pt_idx].lat;    }
    // double depth(int pt_idx) const { return d_pts[pt_idx].depth;  }
    
    // search for the location given a 3-D point
    inline int round_lon_index(int ilon)  { return ilon % d_lon.size(); }
    inline int round_lat_index(int ilat) {
        if (ilat<0) 
        {
            return 0;
        } 
        else if (ilat >= d_lat.size() ) 
        {
            return d_lat.size()-1;
        }
        else
        {
            return ilat;
        }
        //return (ilat >= d_lat.size() ) ? (d_lat.size()-1) : ilat;
    }
    inline int round_depth_index(int idep) {
        if (idep <0) 
        {
            return 0;
        }
        else if (idep >= d_depth.size() )
        {
            return d_depth.size()-1;
        }
        else
        {
            return idep;
        }
        //return (idep >= d_depth.size() ) ? (d_depth.size()-1) : idep;
    }
    inline int point_index(int ilon, int ilat, int idep) {
        return round_depth_index(idep) * d_nlonlat + round_lat_index(ilat)*d_nlon + round_lon_index(ilon);
    }
    inline int search_grd_lon(double lon) {
        std::vector<double>::iterator ptr = std::upper_bound(d_lon.begin(), d_lon.end(), lon );
        int ilon = ptr-d_lon.begin()-1; // d_lon[ilon] <= lon < d_lon[ilon+1]
        return round_lon_index(ilon);
    }
    inline int search_grd_lat(double lat) {
        std::vector<double>::iterator ptr = std::upper_bound(d_lat.begin(), d_lat.end(), lat );
        int ilat = ptr-d_lat.begin()-1; // d_lat[ilat] <= lat < d_lat[ilat+1]
        return round_lat_index(ilat);
    }
    inline int search_grd_dep(double dep) {
        std::vector<double>::iterator ptr = std::upper_bound(d_depth.begin(), d_depth.end(), dep );
        int idep = ptr-d_depth.begin()-1; // d_dep[idep] <= dep < d_dep[idep+1]
        return round_depth_index(idep);
    }
    inline int search_grd(double lon, double lat, double dep, int *ptr_ilon, int *ptr_ilat, int *ptr_idep) {
        // search for the point index in the lon-lat-depth grid given a point.
        int ilon, ilat, idep;
        *ptr_ilon = search_grd_lon(lon);
        *ptr_ilat = search_grd_lat(lat);
        *ptr_idep = search_grd_dep(dep);
        return 0;
    }
    inline int interpolate3D_coef(pt3d &p, 
            double *c0, double *c1, double *c2, double *c3, 
            double *c4, double *c5, double *c6, double *c7,
            int *i0, int *i1, int *i2, int *i3,
            int *i4, int *i5, int *i6, int *i7  )
    {
        // A cube in lon-lat-dep is described by two points `p0` and `p7`.
        // Here, we look forward to linear interpolation given another point `p`, so
        // that V(p) = c0* V(p0) + c1* V(p1) + ... + c7* V(p7)
        //
        //                   ^ X (lat)
        //                  /        
        //              p1 o-----------o p3
        //                /|          /|
        //               / |         / |
        //              /  |        /  |
        //         p0  /   |    p2 /   |
        //            o-----------o---------->  Y (lon)
        //            | p5 o------|----o p7
        //            |   /       |   /  
        //            |  /        |  /
        //            | /         | /
        //            |/          |/
        //            o-----------o
        //         p4 |         p6
        //            |
        //            V 
        //             Z (depth)
        //
        //
        int ilon0 = search_grd_lon(p.lon);
        int ilat0 = search_grd_lat(p.lat);
        int idep0 = search_grd_dep(p.depth);
        int ilon7 = round_lon_index(ilon0+1);
        int ilat7 = round_lat_index(ilat0+1);
        int idep7 =  round_depth_index(idep0+1);
        double x0, y0, z0, x7, y7, z7;
        x0 = d_lat[ilat0]; y0 = d_lon[ilon0]; z0 = d_depth[idep0];
        x7 = d_lat[ilat7]; y7 = d_lon[ilon7]; z7 = d_depth[idep7];
        if (ilon7 == 0) y7 += 360.0; // for the points cross the meridian at 0/360 degree
        double xp, yp, zp;
        xp = p.lat; yp = p.lon; zp = p.depth;
        interpolate3D_cube(x0, y0, z0, x7, y7, z7, xp, yp, zp, c0, c1, c2, c3, c4, c5, c6, c7);
        //
        *i0 = this->point_index(ilon0,   ilat0,   idep0);
        *i1 = this->point_index(ilon0,   ilat0+1, idep0);
        *i2 = this->point_index(ilon0+1, ilat0,   idep0);
        *i3 = this->point_index(ilon0+1, ilat0+1, idep0);
        *i4 = this->point_index(ilon0,   ilat0,   idep0+1);
        *i5 = this->point_index(ilon0,   ilat0+1, idep0+1);
        *i6 = this->point_index(ilon0+1, ilat0,   idep0+1);
        *i7 = this->point_index(ilon0+1, ilat0+1, idep0+1);
        return 0;
    }

public:
    // To set 3D velocity anomaly
    // the input lon0, lon1 can be ouside of [0, 360) for blocks across the meridian at 0/360 longitude degree.
    int set_mod3d_cube(double d0,   double d1, double lon0, double lon1, double lat0, double lat1, double dvp,  double dvs ) 
    {
        double v0 = 1.0+dvp;
        double v1 = 1.0+dvs;
        double s0 = (1.0/v0);
        double s1 = (1.0/v1);
        //
        lon0 = ROUND_DEG360(lon0);
        lon1 = ROUND_DEG360(lon1);
        int ilon0 = round_lon_index( search_grd_lon(lon0)+1 ); 
        int ilon1 = search_grd_lon(lon1); // 
        int ilat0 = round_lat_index( search_grd_lat(lat0)+1 );
        int ilat1 = search_grd_lat(lat1); // ilat0 <= it <= ilat
        int idep0 = round_depth_index( search_grd_dep(d0)+1 );
        int idep1 = search_grd_dep(d1);   // idep0 <= it <= idep1
        // 
        if ( (ilat1 < ilat0) || (idep1 < idep0) ) 
        {
            return 0;
        }
        if (ilon1 >= ilon0)
        {
            for (int idep=idep0; idep<=idep1; ++idep)
            {
                for (int ilat=ilat0; ilat<=ilat1; ++ilat)
                {
                    for (int ilon=ilon0; ilon<=ilon1; ++ilon)
                    {
                        int idx = point_index(ilon, ilat, idep);
                        d_vp_pert[idx] = v0;
                        d_vs_pert[idx] = v1;
                        d_slowness_p_pert[idx] = s0;
                        d_slowness_s_pert[idx] = s1;
                    }
                }
            }
        }
        else 
        {
            for (int idep=idep0; idep<=idep1; ++idep)
            {
                for (int ilat=ilat0; ilat<=ilat1; ++ilat)
                {
                    for (int ilon=0; ilon<=ilon1; ++ilon)
                    {
                        int idx = point_index(ilon, ilat, idep);
                        d_vp_pert[idx] = v0;
                        d_vs_pert[idx] = v1;
                        d_slowness_p_pert[idx] = s0;
                        d_slowness_s_pert[idx] = s1;
                    }
                    for (int ilon=ilon0; ilon<d_lon.size(); ++ilon) 
                    {
                        int idx = point_index(ilon, ilat, idep);
                        d_vp_pert[idx] = v0;
                        d_vs_pert[idx] = v1;
                        d_slowness_p_pert[idx] = s0;
                        d_slowness_s_pert[idx] = s1;
                    }
                }
            }
        }
        //
        return 0;
    }
    int set_mod3d_cylinder(double d0, double d1, double lon, double lat, double radius, double dvp,   double dvs ) 
    {
        double v0 = 1.0+dvp;
        double v1 = 1.0+dvs;
        double s0 = (1.0/v0);
        double s1 = (1.0/v1);
        for(int idx=0; idx<npts(); ++idx) {
            pt3d & pt = d_pts[idx];
            if (great_circle_distance(lon, lat, pt.lon, pt.lat) <= radius )
            {
                if (pt.depth >= d0 && pt.depth <= d1) {
                    //printf("%f %f %f %f %d\n",  lon, lat, pt.d_lon, pt.d_lat, idx);
                    d_vp_pert[idx] = v0;
                    d_vs_pert[idx] = v1;
                    d_slowness_p_pert[idx] = s0;
                    d_slowness_s_pert[idx] = s1; 
                }
            }
        }
        return 0;
    }
    int set_mod3d(const double * dvp, const double *dvs) {
        d_vp_pert.assign(dvp, dvp+d_vp_pert.size() );
        d_vs_pert.assign(dvs, dvs+d_vs_pert.size() );
        for(int idx=0; idx< d_vp_pert.size(); ++idx) {
            d_slowness_p_pert[idx] = 1.0/d_vp_pert[idx];
            d_slowness_s_pert[idx] = 1.0/d_vs_pert[idx];
        }
        return 0;
    }
    int set_mod3d_smooth_lonlat(double d_lonlat ) {
        int step = (int)( round(d_lonlat*0.5/(d_lon[1]-d_lon[0]) ) );
        if (step <= 0) {
            return 0;
        }
        /////////////////////////////////////////////////////////////////////
        std::vector<double> new_dvp(d_npts, 0.0);
        std::vector<double> new_dvs(d_npts, 0.0);
        double alpha = 1.0/(2.0*step+1.0)/(2.0*step+1.0);
        for (int idep=0; idep<d_depth.size(); ++idep) 
        {
            for (int ilat=0; ilat<d_lat.size(); ++ilat) 
            {
                for (int ilon=0; ilon<d_lon.size(); ++ilon)
                {
                    int ipt = point_index(ilon, ilat, idep);
                    // internal summing
                    for (int loop_lat=-step; loop_lat<=step; ++loop_lat) 
                    {
                        for (int loop_lon=-step; loop_lon<=step; ++loop_lon)
                        {
                            int it_lon = round_lon_index(ilon+loop_lon);
                            int it_lat = round_lat_index(ilat+loop_lat);
                            int it_ipt = point_index(it_lon, it_lat, idep);
                            new_dvp[ipt] += d_vp_pert[it_ipt];
                            new_dvs[ipt] += d_vs_pert[it_ipt];
                        }
                    }
                    new_dvp[ipt] *= alpha;
                    new_dvs[ipt] *= alpha;
                }
            }
        }
        d_vp_pert.assign(new_dvp.begin(), new_dvp.end() );
        d_vs_pert.assign(new_dvs.begin(), new_dvs.end() );
        for(int idx=0; idx<d_npts; ++idx) {
            d_slowness_p_pert[idx] = 1.0/d_vp_pert[idx];
            d_slowness_s_pert[idx] = 1.0/d_vs_pert[idx];
        }
        return 0;
    }
public:
private:
    /* data */
    earthmod1d * d_mod1d;
    // grd lines
    int d_nlon, d_nlat, d_ndep, d_nlonlat;
    std::vector<double> d_lon;   // should be values in [0, 360) degree
    std::vector<double> d_lat;   // should be values in [-90, 90] degree
    std::vector<double> d_depth; // should be values in [0, 6371.0] km
    // grd points
    long d_npts;
    std::vector<pt3d>   d_pts;
    //
    std::vector<double> d_vp; // actual 3D vp = d_vp * d_dvp
    std::vector<double> d_vs;
    std::vector<double> d_rho;
    std::vector<double> d_vp_pert; // 1.0 represents no anomally
    std::vector<double> d_vs_pert; // the same
    //
    std::vector<double> d_slowness_p; // actual 3D d_slowness_p = d_slowness_p * d_dslowness_p
    std::vector<double> d_slowness_s;
    std::vector<double> d_slowness_p_pert; // 1.0 represents no anomally
    std::vector<double> d_slowness_s_pert;
    //
};

namespace ak135
{
    extern earthmod1d model;
    extern int    dep_grd_size;
    extern double dep_grd[];
}; // namespace ak135


#endif