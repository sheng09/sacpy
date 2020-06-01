#ifndef GLOBAL_TOMO_H
#define GLOBAL_TOMO_H

#include <vector>
#include <list>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm> 
#include <string>
#include <cstring>
#include <unordered_map>

#define DEG2RAD(x) ( (x)/180.0*3.1415926535)
#define RAD2DEG(x) ( (x)*180.0/3.1415926535)
//#define DEG360(x)  ( (x) < 0.0 ? ((x)+360) : (x) )
#define ROUND_DEG360(x) ( fmod((x), 360.0) )       // return angle in [0, 360) degree
#define ROUND_DEG180(x) ( fmod((x), 360.0)-180.0 ) // return angle in [-180, 180) degree
//#define ISEQUAL(x, y) (fabsf((x)-(y)) < fabsf((x)+(y))*1.0e-20+1.0e-20 ? 1 : 0 ) // check if x and y equals or not
template<typename T>
bool ISEQUAL(T x, T y) {
    // the least accuray is 10^(-6) for double and 10^(-14) for double
    int sz = sizeof(T);
    char *c1 = (char*)(&x), *c2 = (char*)(&y);
    for(int idx=0; idx<sz; ++idx) {
        if (c1[idx] ^ c2[idx]) return false;
    }
    return true;
}


class layer {
public:
    layer() {};
    layer(  const double d0,   const double d1, 
            const double p0,   const double p1, 
            const double s0,   const double s1, 
            const double rho0, const double rho1)
    {
        init(d0, d1, p0, p1, s0, s1, rho0, rho1);
    }
    ~layer() {}
    int init(   const double d0,   const double d1, 
                const double p0,   const double p1, 
                const double s0,   const double s1, 
                const double rho0, const double rho1)
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
    }
    char inside_layer(const double depth)
    {
        /* check if a depth is inside or outside the layer 
            return 'i' inside this layer
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
    double interpolate_velocity(const double depth, const char type )
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
    double interpolate_slowness(const double depth, const char type ) {
        double v = interpolate_velocity(depth, type);
        v = ISEQUAL(v, 0.0) ? 0.0 : 1.0/v;
        //double v=0.0;
        //double dx = depth - d_top_depth;
        //double k  = (type == 'p' ) ? k_slowness_p : ( (type=='s') ? k_slowness_s : krho );
        //double x0 = (type == 'p' ) ? d_top_slowness_p : ( (type=='s') ? d_top_slowness_s : d_top_rho );
        //v = k*dx+x0;
        return v;
    }
    // check if `this` layer and `layer2` are continuous, or this is a jump (e.g., depth jump or velocity jump).
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
    earthmod1d( double earth_radius, const double nlayer, 
                const double d0[], const double d1[],
                const double p0[], const double p1[],
                const double s0[], const double s1[],
                const double r0[], const double r1[] )
    {
        init(earth_radius, nlayer, d0, d1, p0, p1, s0, s1, r0, r1);
    }
    ~earthmod1d() {
        if (!d_layers.empty() ) d_layers.clear();
        if (!d_layer_jump.empty() ) d_layer_jump.clear();
    }
    int init(   double earth_radius, const int nlayer, 
                const double d0[], const double d1[],
                const double p0[], const double p1[],
                const double s0[], const double s1[],
                const double r0[], const double r1[] );
    int search_layer(const double depth);
    //int search_layer(const double depth1, const double depth2);
    double evaulate_from_depth(const double depth, char type, bool return_slowness);
    double radius() { return d_earth_radius; }
    int output_profile(const char *filename);
    bool is_depth_on_discontinuity(const double depth, int *ptr_above, int *ptr_below) {
        for(std::vector<std::pair<int, float> >::iterator it = d_layer_jump.begin(); it != d_layer_jump.end(); ++it) {
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
    bool is_depth_cross_discontinuity(const double d0, const double d1, double * d_dis) {
        // check if a line by two points across any discontinuity
        // return the minimal depth if there are more than 1 across
        for(std::vector<std::pair<int, float> >::iterator it = d_layer_jump.begin(); it != d_layer_jump.end(); ++it) {
            double dis_depth = it->second;
            if ((d0-dis_depth) * (d1-dis_depth) < 0 ) {
                *d_dis = dis_depth;
                return true;
            }
        }
        return false;
    }
    int adjust_raypath(std::list<double> & lon, std::list<double> & lat, std::list<double> & depth );
    // static method
    static double great_cirle_distance(const double lon1, const double lat1, const double lon2, const double lat2) {
        double dlamda = DEG2RAD( lon1-lon2 );
        double p1 = DEG2RAD(lat1);
        double p2 = DEG2RAD(lat2);
        double v = acos(sin(p1)*sin(p2)+cos(p1)*cos(p2)*cos(dlamda) );
        return RAD2DEG(v) ;
    }
private:
    int find_velocity_jump();
    // check if the given depth is exactly on the discontinuity
private:
    double d_earth_radius;
    std::vector<layer> d_layers;
    // profile
    //std::vector<double> d_profile_depth;
    //std::vector<double> d_profile_vp;
    //std::vector<double> d_profile_vs;
    //std::vector<double> d_profile_rho;
    // velocity jump
    std::vector<std::pair<int, float> >  d_layer_jump;
};

typedef struct s_pt3d
{
    /* data */
    double d_lon, d_lat, d_depth;
    double d_theta, d_phi, d_r;
    double d_x, d_y, d_z;
    /* method */
    int init(const double lon, const double lat, const double depth) {
        d_lon = lon;
        d_lat = lat;
        d_depth = depth;
        return 0;
    }
    int init(   const double lon, const double lat, const double depth, 
                const double theta, const double phi, const double r, 
                const double x, const double y, const double z) {
        d_lon = lon;
        d_lat = lat;
        d_depth = depth;
        d_r = r;
        d_theta = theta;
        d_phi = phi;
        d_x = x;
        d_y = y;
        d_z = z;
        return 0;
    }
} pt3d;


class earthmod3d {
public:
    earthmod3d() {};
    earthmod3d(earthmod1d * mod1d, const double lons[], const int nlon,  const double lats[], const int nlat, const double depth[], const int ndep)  {
        init(mod1d, lons, nlon, lats, nlat, depth, ndep);
    }
    earthmod3d(earthmod1d * mod1d, const double dlon, const double dlat, const double depth[], const int ndep)  {
        init(mod1d, dlon, dlat, depth, ndep);
    }
    ~earthmod3d() {
        if(!d_lon.empty() ) d_lon.clear();
        if(!d_lat.empty() ) d_lat.clear();
        if(!d_depth.empty() ) d_depth.clear();
        if(!d_vp.empty() ) d_vp.clear();
        if(!d_vs.empty() ) d_vs.clear();
        if(!d_rho.empty() ) d_rho.clear();
        if(!d_dvp.empty() ) d_dvp.clear();
        if(!d_dvs.empty() ) d_dvs.clear();
        if(!d_pts.empty() ) d_pts.clear();
        if(!d_slowness_p.empty()  ) d_slowness_p.clear();
        if(!d_slowness_s.empty()  ) d_slowness_s.clear();
        if(!d_dslowness_p.empty() ) d_dslowness_p.clear();
        if(!d_dslowness_s.empty() ) d_dslowness_s.clear();
        d_mod1d = NULL;
    }
    // init 3-D grid and coordinates transformation
    int init(earthmod1d * mod1d, const double dlon, const double dlat, const double depth[], const int ndep) ;
    int init(earthmod1d * mod1d, const double lons[], const int nlon,  const double lats[], const int nlat, const double depth[], const int ndep );
    int geo2sph(double lon, double lat, double depth, double *theta, double *phi, double *r)  { 
        *r =  d_mod1d->radius() - depth;
        *theta = lon;
        *phi = 90.0 - lat;
        return 0;
    }
    int sph2xyz(double theta, double phi, double r, double *x, double *y, double *z) 
    { 
        *x = r*sin(DEG2RAD(phi))*cos(DEG2RAD(theta));
        *y = r*sin(DEG2RAD(phi))*sin(DEG2RAD(theta));
        *z = r*cos(DEG2RAD(phi));
        return 0;
    }
    int xyz2sph(double x, double y, double z, double *theta, double *phi, double *r) {
        double tmp = x*x+y*y;
        *theta = RAD2DEG( atan2(y, x) );
        *phi   = RAD2DEG( atan2(sqrtf(tmp), z) );
        *r = sqrtf( tmp+z*z );
        return 0;
    }
    int sph2geo(double theta, double phi, double r, double *lon, double *lat, double *depth) {
        *lat = 90.0 - phi;
        *lon = ROUND_DEG360(theta); // longitude should be in the range of [0, 360) degree
        *depth = d_mod1d->radius() - r;
        return 0;
    } 
    int geo2xyz(double lon, double lat, double depth, double *x, double *y, double *z) {
        double theta, phi, r;
        geo2sph(lon, lat, depth, &theta, &phi, &r);
        sph2xyz(theta, phi, r, x, y, z);
        return 0;
    }
    int xyz2geo(double x, double y, double z, double *lon, double *lat, double *depth) {
        double theta, phi, r;
        xyz2sph(x, y, z, &theta, &phi, &r);
        sph2geo(theta, phi, r, lon, lat, depth);
        return 0;
    }
    // accessing 3-D grid 
    int output_grd_pts(const char * filename) ;
    double vp(int pt_idx) { return d_vp[pt_idx]; }
    double vs(int pt_idx) { return d_vs[pt_idx]; }
    double slowness_p(int pt_idx) { return d_slowness_p[pt_idx]; }
    double slowness_s(int pt_idx) { return d_slowness_s[pt_idx]; }
    pt3d*  point(int pt_idx) { return &(d_pts[pt_idx]); }
    double lon(int pt_idx)   { return d_pts[pt_idx].d_lon; }
    double lat(int pt_idx)   { return d_pts[pt_idx].d_lat; }
    double depth(int pt_idx) { return d_pts[pt_idx].d_depth; }
    long npts() { return d_npts; }
    double radius() { return d_mod1d->radius(); }
    earthmod1d * mod1d() { return d_mod1d; }
    // search for the location given a 3-D point
    inline int round_lon_index(const int ilon) {
        return ilon % d_lon.size();
        //return (ilon >= d_lon.size() ) ? (ilon-d_lon.size() ) : ilon;
    }
    inline int round_lat_index(const int ilat) {
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
    inline int round_depth_index(const int idep) {
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
    inline int point_index(const int ilon, const int ilat, const int idep) {
        return round_depth_index(idep) * d_nlonlat + round_lat_index(ilat)*d_nlon + round_lon_index(ilon);
    }inline int search_grd_lon(const double lon) {
        std::vector<double>::iterator ptr = std::upper_bound(d_lon.begin(), d_lon.end(), lon );
        int ilon = ptr-d_lon.begin()-1; // d_lon[ilon] <= lon < d_lon[ilon+1]
        return round_lon_index(ilon);
    }
    inline int search_grd_lat(const double lat) {
        std::vector<double>::iterator ptr = std::upper_bound(d_lat.begin(), d_lat.end(), lat );
        int ilat = ptr-d_lat.begin()-1; // d_lat[ilat] <= lat < d_lat[ilat+1]
        return round_lat_index(ilat);
    }
    inline int search_grd_dep(const double dep) {
        std::vector<double>::iterator ptr = std::upper_bound(d_depth.begin(), d_depth.end(), dep );
        int idep = ptr-d_depth.begin()-1; // d_dep[idep] <= dep < d_dep[idep+1]
        return round_depth_index(idep);
    }
    inline int search_grd(const double lon, const double lat, const double dep, int *ptr_ilon, int *ptr_ilat, int *ptr_idep) {
        // search for the point index in the lon-lat-depth grid given a point.
        int ilon, ilat, idep;
        *ptr_ilon = search_grd_lon(lon);
        *ptr_ilat = search_grd_lat(lat);
        *ptr_idep = search_grd_dep(dep);
        return 0;
    }
public:
    /*
        To set 3D velocity anomaly
    */
    int set_mod3d_cube(const double d0, const double d1, 
                        const double lon0, const double lon1, 
                        const double lat0, const double lat1,
                        const double dvp,  const double dvs ) 
    {
        double v0 = 1.0+dvp;
        double v1 = 1.0+dvs;
        double s0 = (1.0/v0);
        double s1 = (1.0/v1);
        for(int idx=0; idx<npts(); ++idx) {
            pt3d & pt = d_pts[idx];
            if (pt.d_depth >= d0 && pt.d_depth <= d1 &&
                pt.d_lon >= lon0 && pt.d_lon <= lon1 &&
                pt.d_lat >= lat0 && pt.d_lat <= lat1 )
            {
                d_dvp[idx] = v0;
                d_dvs[idx] = v1;
                d_dslowness_p[idx] = s0;
                d_dslowness_s[idx] = s1; 
            }
        }
        return 0;
    }
    int set_mod3d_cylinder(const double d0, const double d1, 
                           const double lon, const double lat,
                           const double radius,
                           const double dvp,  const double dvs ) 
    {
        double v0 = 1.0+dvp;
        double v1 = 1.0+dvs;
        double s0 = (1.0/v0);
        double s1 = (1.0/v1);
        for(int idx=0; idx<npts(); ++idx) {
            pt3d & pt = d_pts[idx];
            if (earthmod1d::great_cirle_distance(lon, lat, pt.d_lon, pt.d_lat) <= radius )
            {
                if (pt.d_depth >= d0 && pt.d_depth <= d1) {
                    //printf("%f %f %f %f %d\n",  lon, lat, pt.d_lon, pt.d_lat, idx);
                    d_dvp[idx] = v0;
                    d_dvs[idx] = v1;
                    d_dslowness_p[idx] = s0;
                    d_dslowness_s[idx] = s1; 
                }
            }
        }
        return 0;
    }
    int set_mod3d_smooth_sphere(double d_lonlat ) {
        std::vector<double> new_dvp(d_npts);
        std::vector<double> new_dvs(d_npts);
        int step = (int)( round(d_lonlat*0.5/(d_lon[1]-d_lon[0]) ) );
        if (step <= 0) {
            return 0;
        }
        /////////////////////////////////////////////////////////////////////
        double alpha = 1.0/(2*step+1)/(2*step+1);
        printf("%d %f\n", step, alpha);
        for (int idep=0; idep<d_depth.size(); ++idep) {
            for (int ilat=0; ilat<d_lat.size(); ++ilat) {
                for (int ilon=0; ilon<d_lon.size(); ++ilon) {
                    int idx_pt = point_index(ilon, ilat, idep);
                    new_dvp[idx_pt] = 0.0;
                    new_dvs[idx_pt] = 0.0;
                    // err here
                    int ja0 = round_lat_index(ilat-step), ja1 = round_lat_index(ilat+step);
                    int jo0 = round_lon_index(ilon-step), jo1 = round_lon_index(ilon+step);
                    int v = 0;
                    for(int ja=ja0; ja<=ja1; ++ja) {
                        for(int jo=jo0; jo<=jo1; ++jo) {
                            int jpt = point_index(jo, ja, idep);
                            new_dvp[idx_pt] += d_dvp[jpt];
                            new_dvs[idx_pt] += d_dvp[jpt];
                            ++v;
                        }
                    }
                    if (v>0) {
                        new_dvp[idx_pt] /= v;
                        new_dvs[idx_pt] /= v;
                    } else {
                        new_dvp[idx_pt] = d_dvp[idx_pt];
                        new_dvs[idx_pt] = d_dvp[idx_pt];
                    }
                }
            }
        }
        d_dvp.assign(new_dvp.begin(), new_dvp.end() );
        d_dvs.assign(new_dvs.begin(), new_dvs.end() );
        for(int idx=0; idx<d_npts; ++idx) {
            d_dslowness_p[idx] = 1.0/d_dvp[idx];
            d_dslowness_s[idx] = 1.0/d_dvs[idx];
        }
        return 0;
    }
public:
    int __interpolate_lonlat(const int i0, const int i1, const int i2, const int i3, 
                    const double lon, const double lat, 
                    double *coef0, double *coef1, double *coef2, double *coef3) {
        double x0=d_pts[i0].d_lat, x1=d_pts[i1].d_lat, x2=d_pts[i2].d_lat, x3=d_pts[i3].d_lat;
        double y0=d_pts[i0].d_lon, y1=d_pts[i1].d_lon, y2=d_pts[i2].d_lon, y3=d_pts[i3].d_lon;
        double xp = lat, yp = lon;
        double a, b, c, d, e, f;
        a = (x2-xp)/(x2-x0); b = (xp-x0)/(x2-x0);
        c = (x3-xp)/(x3-x1); d = (xp-x1)/(x3-x1);
        e = (y1-yp)/(y1-y0); f = (yp-y0)/(y1-y0);
        //
        *coef0 = a*e;
        *coef2 = b*e;
        *coef1 = c*f;
        *coef3 = d*f;
        return 0;
    }
    int __interpolate_dep(const int i0, const int i4, const double depth, double *g, double *h) {
        double z0=d_pts[i0].d_depth, z4=d_pts[i4].d_depth;
        double zp=depth;
        *g = (z4-zp)/(z4-z0);
        *h = (zp-z0)/(z4-z0);
        return 0;
    }
    int interpolate3d(const int i0, const int i1, const int i2, const int i3,
                        const int i4, const int i5, const int i6, const int i7,
                        const double lon, const double lat, const double depth,
                        double *c0, double *c1, double *c2, double *c3,
                        double *c4, double *c5, double *c6, double *c7    )
    {
        //
        double u0, u1, u2, u3;
        double d0, d1, d2, d3;
        double g, h;
        __interpolate_lonlat(i0, i1, i2, i3, lon, lat, &u0, &u1, &u2, &u3);
        __interpolate_lonlat(i4, i5, i6, i7, lon, lat, &d0, &d1, &d2, &d3);
        __interpolate_dep(i0, i4, depth, &g, &h);
        *c0 = g*u0;
        *c1 = g*u1;
        *c2 = g*u2;
        *c3 = g*u3;
        *c4 = h*d0;
        *c5 = h*d1;
        *c6 = h*d2;
        *c7 = h*d3;
        //printf("%f %f %f %f && %f %f %f %f && %f %f\n", u0, u1, u2, u3, d0, d1, d2, d3, g, h);
        //printf("%f %f %f %f    %f %f %f %f\n", *c0, *c1, *c2, *c3, *c4, *c5, *c6, *c7);
        //
        return 0;
    }
    int __interpolate_lonlat_grd3d( const int ilon, const int ilat, const double lon, const double lat, 
                                    double *coef0, double *coef1, double *coef2, double *coef3) {
        //
        double a, b, c, d, e, f;
        int ilon2 = round_lon_index(ilon+1);
        int ilat2 = round_lat_index(ilat+1);
        double xp = lat, yp = lon;
        ///
        if (ilon==ilon2) {
            e = 1.0;
            f = 0.0;
        } 
        else if(ilon2 == 0) { // 
            double y0=d_lon[ilon], y1=d_lon[ilon2]+360.0, y2=d_lon[ilon],  y3=d_lon[ilon2]+360.0;
            e = (y1-yp)/(y1-y0); f = (yp-y0)/(y1-y0);
        }
        else {
            double y0=d_lon[ilon], y1=d_lon[ilon2], y2=d_lon[ilon],  y3=d_lon[ilon2];
            e = (y1-yp)/(y1-y0); f = (yp-y0)/(y1-y0);
        }
        ///
        if (ilat == ilat2) {
            a = 1.0; c = 1.0;
            b = 0.0; d = 0.0;
        }
        else {
            double x0=d_lat[ilat], x1=d_lat[ilat],  x2=d_lat[ilat2], x3=d_lat[ilat2];
            a = (x2-xp)/(x2-x0); b = (xp-x0)/(x2-x0);
            c = (x3-xp)/(x3-x1); d = (xp-x1)/(x3-x1);
        }
        //
        *coef0 = a*e;
        *coef2 = b*e;
        *coef1 = c*f;
        *coef3 = d*f;
        return 0;
    }
    int __interpolate_dep_grd_3d(const int idep, const double depth, double *g, double *h) {
        int idep2 = round_depth_index(idep+1);
        if (idep == idep2) {
            *g = 1.0;
            *h = 0.0;
            return 0;
        }
        double z0=d_depth[idep], z4=d_depth[idep2];
        //printf("z0: %lf z4: %lf\n", z0, z4);
        double zp=depth;
        //printf("dep interpolate %d %f %f %f\n", idep, z0, z4, zp);
        *g = (z4-zp)/(z4-z0);
        *h = (zp-z0)/(z4-z0);
        return 0;
    }
    int interpolate_grd3d(  const int ilon, const int ilat, const int idep, 
                            const double lon, const double lat, const double depth,
                            double *c0, double *c1, double *c2, double *c3,
                            double *c4, double *c5, double *c6, double *c7    ) {
        //
        double u0, u1, u2, u3;
        double d0, d1, d2, d3;
        double g, h;
        __interpolate_lonlat_grd3d(ilon, ilat, lon, lat, &u0, &u1, &u2, &u3);
        __interpolate_lonlat_grd3d(ilon, ilat, lon, lat, &d0, &d1, &d2, &d3);
        __interpolate_dep_grd_3d(idep, depth, &g, &h);
        *c0 = g*u0;
        *c1 = g*u1;
        *c2 = g*u2;
        *c3 = g*u3;
        *c4 = h*d0;
        *c5 = h*d1;
        *c6 = h*d2;
        *c7 = h*d3;
        //printf("%f %f %f %f && %f %f %f %f && %f %f\n", u0, u1, u2, u3, d0, d1, d2, d3, g, h);
        //printf("%f %f %f %f    %f %f %f %f\n", *c0, *c1, *c2, *c3, *c4, *c5, *c6, *c7);
        //
        return 0;
    }
private:
    /* data */
    earthmod1d * d_mod1d;
    // grd lines
    int d_nlon, d_nlat, d_ndep, d_nlonlat;
    std::vector<double> d_lon; // should be values in [0, 360) degree
    std::vector<double> d_lat; // should be values in [-90, 90] degree
    std::vector<double> d_depth; // should be values in [0, 6371] km
    // grd points
    long d_npts;
    //
    std::vector<double> d_vp; // 3D vp = d_vp * d_dvp
    std::vector<double> d_vs;
    std::vector<double> d_rho;
    std::vector<double> d_dvp;
    std::vector<double> d_dvs;
    //
    std::vector<double> d_slowness_p; // 3D d_slowness_p = d_slowness_p * d_dslowness_p
    std::vector<double> d_slowness_s;
    std::vector<double> d_dslowness_p;
    std::vector<double> d_dslowness_s;
    //
    std::vector<pt3d>  d_pts;
    // 
};

/*
    A single raypath
*/
class raypath3d {
public:
    raypath3d() {}
    raypath3d(earthmod3d *mod3d, const int npts, const double lons[], const double lats[], const double depth[] ) {
        init(mod3d, npts, lons, lats, depth);
        //output("I2_old.txt");
        //adjust_avoid_discontinuity();
        //output("I2_new.txt");
    }
    raypath3d(earthmod3d *mod3d, const char *filename) {
        init(mod3d, filename);
        //adjust_avoid_discontinuity();
    }
    raypath3d(earthmod3d *mod3d, const int id, const char * filename, const char * tag) {
        init(mod3d, id, filename, tag);
    }
    raypath3d(earthmod3d * mod3d, double evdp, double evlo, double evla, double stlo, double stla, char * phase ) {
        init(mod3d, evdp, evlo, evla, stlo, stla, phase);
    }
    
    ~raypath3d() {
        if(!d_lon_list.empty() )     d_lon_list.clear();
        if(!d_lat_list.empty() )     d_lat_list.clear();
        if(!d_depth_list.empty() )   d_depth_list.clear();
        if(!d_sensitivity.empty() )  d_sensitivity.clear();
        //
        if(!d_lon.empty() )     d_lon.clear();
        if(!d_lat.empty() )     d_lat.clear();
        if(!d_depth.empty() )   d_depth.clear();
        if(!d_theta.empty() )   d_theta.clear();
        if(!d_phi.empty() )     d_phi.clear();
        if(!d_r.empty() )       d_r.clear();
        if(!d_x.empty() )       d_x.clear();
        if(!d_y.empty() )       d_y.clear();
        if(!d_z.empty() )       d_z.clear();
        
    }
    int npts() { return d_npts; }
    double traveltime1d() { return d_traveltime_1d; }
    std::vector<double> & sensitivity() { return d_sensitivity; }
    int inter_denser_raypath(double dl);
    int double_denser_raypath();
    int adjust_avoid_discontinuity();
    int grd_sensitivity();
    int output(const char * filename);
    static double length(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2) {
        double dx = x1-x2;
        double dy = y1-y2;
        double dz = z1-z2;
        return sqrt(dx*dx+dy*dy+dz*dz);
    }
public:
    int init(earthmod3d *mod3d, const int id, const char * filename, const char * tag);
    const char * filename() { return d_raypath_filename.c_str(); }
    const char * tag() { return d_tag.c_str(); }
private:
    int init(earthmod3d *mod3d, double evdp, double evlo, double evla, double stlo, double stla, char * phase );
    int init(earthmod3d *mod3d, const int npts, const double lons[], const double lats[], const double depth[]);
    int init(earthmod3d *mod3d, const std::list<double> lon, const std::list<double> lat, const std::list<double> depth );
    int init(earthmod3d *mod3d, const char *filename);
    int list2vec();
    int geo2xyz();
    int xyz2geo();
    int old_adjust_avoid_discontinuity();
private:
    int d_id;
    std::string d_raypath_filename;
    std::string d_tag;
private:
    // raypath coordinates
    int d_npts;
    std::list<double>   d_lon_list;   // list are used for easy manipulation of raypath points
    std::list<double>   d_lat_list;   //
    std::list<double>   d_depth_list; //
    // raypath sensitivity to a 3d earth model
    earthmod3d *d_mod3d;
    std::vector<double> d_sensitivity;
    // 
    double              d_traveltime_1d;
private:
    // not independent variables
    std::vector<double> d_lon;
    std::vector<double> d_lat;
    std::vector<double> d_depth;
    std::vector<double> d_theta;
    std::vector<double> d_phi;
    std::vector<double> d_r;
    std::vector<double> d_x;
    std::vector<double> d_y;
    std::vector<double> d_z;
};

/*
    A single cross-term
*/
class cross_term {
public:
    cross_term();
    cross_term(raypath3d *ray1, raypath3d *ray2);
    ~cross_term() {
        d_ray1 = NULL;
        d_ray2 = NULL;
    }
    int idcc() {return d_idcc; }
    int id1() {return d_id1; }
    int id2() {return d_id2; }
    const char * tag() { return d_tag.c_str(); }
    int run() {
        std::vector<double> & sens1 = d_ray1->sensitivity();
        std::vector<double> & sens2 = d_ray2->sensitivity();
        d_matrix_row.assign(d_mod3d->npts(), 0.0 );
        for (int idx=0; idx<d_mod3d->npts(); ++idx) {
            d_matrix_row[idx] = sens1[idx] - sens2[idx];
        }
        return 0;
    }
public:
    int init(earthmod3d * mod3d, const int idcc, const int id1, const int id2, raypath3d *ray1, raypath3d *ray2, const char * tag) {
        d_mod3d = mod3d;
        d_idcc  = idcc;
        d_id1   = id1;
        d_id2   = id2;
        d_tag   = tag;
        d_ray1 = ray1;
        d_ray2 = ray2;
        return 0;
    }
private:
    int          d_idcc;
    int          d_id1,  d_id2;
    std::string  d_tag;
    earthmod3d  *d_mod3d;
    raypath3d   *d_ray1, *d_ray2;
    std::vector<float> d_matrix_row;
};


/*
    An object for many cross terms. It is based on two files:
    1. A file that declare many raypaths. Each line of the file is:
        idx_ray(int)   raypath_fnm(string)   tag_str(string)
    2. A file that delcare many cross terms. Each line of the file is:
        idx_cc(int)   idx_ray1(int)   idx_ray2(int)   tag_str(string)
    
    A `cross_term_vol` instance can be used for forward modeling when 
    combined with a 3D model that is an `earthmod3d` object. 
    Also, a `cross_term_vol` instance can be used for inversion when
    combined with given time information.
*/
class cross_term_vol {
public:
    cross_term_vol() {}
    ~cross_term_vol() {}
private:
    int init(earthmod3d * mod3d, const char * fnm_ray, const char * fnm_cc) {
        d_mod3d = mod3d;
        d_fnm_ray_vol = fnm_ray;
        d_fnm_crossterm_vol = fnm_cc;
        return 0;
    }
    int init_rays(int nrays, const char *fnm) {
        #define MAXCHAR 4096
        d_ray_id.resize(nrays);
        d_rays = new raypath3d[nrays];
        //
        FILE *fid = fopen(fnm, "r");
        static char line[MAXCHAR];
        int tmp_id;
        static char tmp_fnm[MAXCHAR];
        static char tmp_tag[MAXCHAR];
        for(int iline=0; iline < nrays; ++iline ) {
            std::memset(line,    0, MAXCHAR);
            std::memset(tmp_fnm, 0, MAXCHAR);
            std::memset(tmp_tag, 0, MAXCHAR);
            char *c = fgets(line, MAXCHAR, fid);
            if (c != line) { fprintf(stderr, "Err in reading file %s\n", fnm); exit(-1); }
            sscanf(line, "%d %s %s",  &tmp_id, tmp_fnm, tmp_tag);
            d_ray_id[iline]  = tmp_id;
            d_rays[iline].init(d_mod3d, d_ray_id[iline], tmp_fnm, tmp_tag );
            d_ray_map[tmp_id] = &(d_rays[iline]);
        }
        fclose(fid);
        //
        return 0;
        #undef MAXCHAR
    }
    int output_rays(const char *fnm) {
        FILE *fid = fopen(fnm, "w");
        for(int idx=0; idx< d_ray_id.size(); ++idx ) {
            fprintf(fid, "%d %s %s\n", d_ray_id[idx], d_rays[idx].filename(), d_rays[idx].tag() );
        }
        fclose(fid);
        return 0;
    }
    int init_cross_terms(int n_cross_terms, const char *fnm) {
        #define MAXCHAR 4096
        d_cross_term_id.resize(n_cross_terms);
        d_cross_terms = new cross_term[n_cross_terms];
        //
        static char line[MAXCHAR];
        int tmp_idcc, tmp_id1, tmp_id2;
        static char tmp_tag[MAXCHAR];
        FILE *fid = fopen(fnm, "r");
        for(int iline=0; iline <n_cross_terms; ++iline) {
            std::memset(line, 0, MAXCHAR);
            std::memset(tmp_tag, 0, MAXCHAR);
            char *c = fgets(line, MAXCHAR, fid);
            if (c != line) { fprintf(stderr, "Err in reading file %s\n", fnm); exit(-1); }
            sscanf(line, "%d %d %d %s", &tmp_idcc, &tmp_id1, &tmp_id2, tmp_tag);
            //
            d_cross_term_id[iline] = tmp_idcc;
            d_cross_terms[iline].init(d_mod3d, tmp_idcc, tmp_id1, tmp_id2, d_ray_map[tmp_id1], d_ray_map[tmp_id2], tmp_tag);
            d_cross_term_map[tmp_idcc] = &(d_cross_terms[iline]);
        }
        fclose(fid);
        return 0;
        #undef MAXCHAR
    }
    int output_cross_terms(const char *fnm) {
        FILE *fid = fopen(fnm, "w");
        for(int idx=0; idx< d_ray_id.size(); ++idx ) {
            fprintf(fid, "%d %d %d %s\n", d_cross_term_id[idx], d_cross_terms[idx].id1(), d_cross_terms[idx].id2(), d_cross_terms[idx].tag() );
        }
        fclose(fid);
        return 0;
    }
private:
    earthmod3d * d_mod3d;
    // two important files
    std::string d_fnm_ray_vol;
    std::string d_fnm_crossterm_vol;
private:
    // data volume for ray paths
    std::vector<int> d_ray_id;
    raypath3d *d_rays;
    typedef raypath3d *ptr_raypath3d;
    std::unordered_map<int,  ptr_raypath3d> d_ray_map;
private:
    // data volume for cross terms
    std::vector<int> d_cross_term_id;
    cross_term *d_cross_terms;
    typedef cross_term *ptr_cross_term;
    std::unordered_map<int,  ptr_cross_term> d_cross_term_map;
};


namespace ak135 {
    extern earthmod1d model;
};
#endif