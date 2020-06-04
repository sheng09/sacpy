#ifndef TOMO_RAYPATH____
#define TOMO_RAYPATH____

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

#define DEG2RAD(x) ( (x)/180.0*3.1415926535 )
#define RAD2DEG(x) ( (x)*180.0/3.1415926535 )
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



/* Basic geometry functions
*/
inline int geo2xyz(double lon, double lat, double depth, double *x, double *y, double *z, double R0=6371.0) {
    double r =  R0 - depth;
    double theta = lon;
    double phi = 90.0 - lat;
    *x = r*sin(DEG2RAD(phi))*cos(DEG2RAD(theta));
    *y = r*sin(DEG2RAD(phi))*sin(DEG2RAD(theta));
    *z = r*cos(DEG2RAD(phi));
    return 0;
}
inline int xyz2geo(double x, double y, double z, double *lon, double *lat, double *depth, double R0= 6371.0) {
    double tmp = x*x+y*y;
    double theta = RAD2DEG( atan2(y, x) );
    double phi   = RAD2DEG( atan2(sqrtf(tmp), z) );
    double r = sqrtf( tmp+z*z );
    *lat = 90.0 - phi;
    *lon = ROUND_DEG360(theta); // longitude should be in the range of [0, 360) degree
    *depth = R0 - r;
    return 0;
}

/* Basic 3-D point and related functions
*/
class pt3d_less
{
public:
    /* method */
    pt3d_less() {}
    pt3d_less(double lo, double la, double dep) {
        init(lo, la, dep);
    }
    ~pt3d_less() {}
    //
    inline int init(double lo, double la, double dep) {
        lon = lo;
        lat = lat;
        depth = dep;
        return 0;
    }
public:
    /* data */
    double lon, lat, depth;
};

class pt3d
{
public:
    pt3d() {}
    pt3d(double lo, double la, double dep, double R0= 6371.0, bool xyz=false) {
        if (!xyz) 
        {
            init(lo, la, dep, R0);  
        }
        else
        {
            init_xyz(lo, la, dep, R0);
        }
    }
    pt3d(pt3d_less pt) {
        init(pt);
    }
    ~pt3d() {}
    /* method */
    inline int init(double lo, double la, double dep, double R0= 6371.0) {
        lon = lo;
        lat = la;
        depth = dep;
        geo2xyz(lon, lat, depth, &x, &y, &z, R0);
        return 0;
    }
    inline int init(pt3d_less pt) {
        init(pt.lon, pt.lat, pt.depth);
        return 0;
    }
    inline int init_xyz(double ix, double iy, double iz, double R0= 6371.0) {
        x = ix; y = iy; z= iz;
        xyz2geo(x, y, z, &lon, &lat, &depth, R0);
        return 0;
    }
    inline bool is_same_point(pt3d & pt) {
        if (ISEQUAL(lon, pt.lon) && ISEQUAL(lat, pt.lat) && ISEQUAL(depth, pt.depth) )
            return true;
        return false;
    }
public:
    /* data */
    double lon, lat, depth;
    double x, y, z;
};

// Great circle distance
inline double great_circle_distance(double lon1, double lat1, double lon2, double lat2) {
        double dlamda = DEG2RAD( lon1-lon2 );
        double p1 = DEG2RAD(lat1);
        double p2 = DEG2RAD(lat2);
        double v = acos(sin(p1)*sin(p2)+cos(p1)*cos(p2)*cos(dlamda) );
        return RAD2DEG(v) ;
}
inline double great_circle_distance(pt3d & pt1, pt3d & pt2) {
    double lon1 = pt1.lon;
    double lat1 = pt1.lat;
    double lon2 = pt2.lon;
    double lat2 = pt2.lat;
    return great_circle_distance(lon1, lat1, lon2, lat2);
}
// Line distance in 3-D
inline double distance(pt3d & pt1, pt3d & pt2) {
    double dx = pt1.x-pt2.x;
    double dy = pt1.y-pt2.y;
    double dz = pt1.z-pt2.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}
inline double distance(pt3d_less & pt1, pt3d_less & pt2) {
    pt3d q1, q2;
    q1.init(pt1);
    q2.init(pt2);
    return distance(q1, q2);
}
// Interpolate a new point between two points
inline pt3d interpolate3D_depth(pt3d & pt1, pt3d & pt2, double depth, double R0=6371.0) {
    double a = pt2.x - pt1.x;
    double b = pt2.y - pt1.y;
    double c = pt2.z - pt1.z;
    double r = R0 - depth;
    double A = a*a + b*b + c*c;
    double B = 2.0*(a*pt1.x + b*pt1.y + c*pt1.z);
    double C = pt1.x*pt1.x + pt1.y*pt1.y + pt1.z*pt1.z - r*r;
    double delta = std::sqrt(B*B-4*A*C);
    double l0 = (-B+delta)/A*0.5;
    double l1=  (-B-delta)/A*0.5;
    //
    pt3d q1(l0*a+pt1.x, l0*b+pt1.y, l0*c+pt1.z, R0, true);
    pt3d q2(l1*a+pt1.x, l1*b+pt1.y, l1*c+pt1.z, R0, true);
    double dis1 = distance(pt1, q1) + distance(pt2, q1);
    double dis2 = distance(pt2, q2) + distance(pt2, q2);
    if (dis1 < dis2) 
    {
        double lon, lat, junk;
        xyz2geo(q1.x, q1.y, q1.z, &lon, &lat, &junk, R0);
        return pt3d(lon, lat, depth, R0);
    }
    else 
    {
        double lon, lat, junk;
        xyz2geo(q2.x, q2.y, q2.z, &lon, &lat, &junk, R0);
        return pt3d(lon, lat, depth, R0);
    }
}
// check if a depth is between two points. That does not includes the depth exactly on either points.
inline bool cross_depth(pt3d &pt1, pt3d &pt2, double depth) {
    if (    ( (pt1.depth - depth)*(pt2.depth - depth) < 0 ) &&
            (!ISEQUAL(pt1.depth, depth)) &&
            (!ISEQUAL(pt2.depth, depth))   )
    {
        return true;
    }
    return false;
}
// Linear interpolation in a field V(x), V(x, y), or V(x, y, z), etc.
inline int interpolate1D_line(double x0, double x1, double xp, double *c0, double *c1) {
    //
    //    ---------------o--------*--------o----------------> x
    //                   x0       xp       x1
    //
    //    V(xp) = c0 V(x0) + c1 V(x1)
    //        
    //    c0 = (x1-xp)/(x1-x0)
    //    c1 = (xp-x0)/(x1-x0)
    //
    if (ISEQUAL(x0, x1) )
    {
        *c0 = 1.0;
        *c1 = 0.0;
    }
    else
    {
        *c0 = (x1-xp)/(x1-x0);
        *c1 = (xp-x0)/(x1-x0);
    }
    return 0;
}
inline int interpolate2D_rect(double x0, double y0, double x3, double y3, double xp, double yp, double *c0, double *c1, double *c2, double *c3)
{
    // A rectangel is described by two points (x0, y0) and (x3, y3).
    // Here, we look forward to linear interpolation given another point `p`, so
    // that V(p) = c0* V(p0) + c1* V(p1) + ... + c3* V(p3)
    //
    //     ^ Y
    //     |
    //     | p1 (x2,y2)   p3 (x3, y3)
    //     o---*--------o
    //     |     q'     |
    //     |            |
    //     |   * p(x,y) |
    //     |            |
    //     |     q      |
    //     o---*--------o-------> X
    //    p0 (x0,y0)     p1 (x1, y1)
    //
    //  V(q)  = a* V(p0) + b* V(p1)
    //  V(q') = a* V(p2) + b* V(p3)
    //  V(p)  = c* V(q)  + d* V(q')
    //        = c* [a* V(p0) + b* V(p1)] + d* [ a* V(p2) + b* V(p3) ]
    //        = ac V(p0) + bc V(p1) + ad V(p2) + bd V(p3)
    double a, b;
    interpolate1D_line(x0, x3, xp, &a, &b);
    double c, d;
    interpolate1D_line(y0, y3, yp, &c, &d);
    ///
    *c0 = a*c;
    *c1 = b*c;
    *c2 = a*d;
    *c3 = b*d;
    return 0;
}
inline int interpolate3D_cube(double x0, double y0, double z0,
    double x7, double y7, double z7,
    double xp, double yp, double zp,
    double *c0, double *c1, double *c2, double *c3, 
    double *c4, double *c5, double *c6, double *c7)
{
    //
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
    // The point (xp, yp, zp) has a projection point q on the top surface (z=z0), and another projection point q' on the bottom surface (z=z7)
    // 
    // V(p)  = g V(q)  + h V(q')
    // V(q)  = a V(p0) + b V(p1) + c V(p2) + d V(p3)
    // V(q') = a V(p4) + b V(p5) + c V(p6) + d V(p7)
    // 
    //
    double a, b, c, d;
    interpolate2D_rect(x0, y0, x7, y7, xp, yp, &a, &b, &c, &d);
    double g, h;
    interpolate1D_line(z0, z7, zp, &g, &h);
    //
    *c0 = a*g;
    *c1 = b*g;
    *c2 = c*g;
    *c3 = d*g;
    *c4 = a*h;
    *c5 = b*h;
    *c6 = c*h;
    *c7 = d*h;
    return 0;
}





/* 3-D Raypath that is composed of a list of `pt3d`, and built-in methods
*/
class raypath3d_segment : public std::list<pt3d> {
public:
    raypath3d_segment()  {}
    raypath3d_segment(char type, double R0, std::list<pt3d> & path) { set_path_type(type); set_R0(R0); set_path(path); }
    raypath3d_segment(std::list<pt3d> & path) { set_path(path); }
    //raypath3d_segment(raypath3d_segment & ray_path) { if (this != &ray_path) {  set_path(ray_path); }  }
    raypath3d_segment(std::list<pt3d>::iterator &it0, std::list<pt3d>::iterator &it1) { 
        this->assign(it0, it1);
    }
    raypath3d_segment(char type, double R0, std::list<pt3d>::iterator &it0, std::list<pt3d>::iterator &it1) { 
        set_path_type(type); 
        set_R0(R0);
        this->assign(it0, it1);
    }
    ~raypath3d_segment() {
        if (!this->empty() ) this->clear();
    }
    /* Set R0
    */
    inline int set_R0(double R0) { d_R0 = R0; }
    /* set and manipulate ray-path
    */
    inline int set_path_type(char type) { d_type = type; }
    inline int set_path(int npts, double * lon, double * lat, double *dep) 
    {
        for(int idx=0; idx<npts; ++idx, ++lon, ++lat, ++dep) 
        {
            this->push_back( pt3d(*lon, *lat, *dep, d_R0) );
        }
        return 0;
    }
    inline int set_path(std::list<pt3d> & path)
    {
        this->assign(path.begin(), path.end() );
        return 0;
    }
    int inter_denser_raypath(double dl)
    {
        std::list<pt3d> new_path;
        std::list<pt3d>::iterator point = this->begin();
        std::list<pt3d>::iterator next_point = point; ++next_point;
        //
        for(; next_point!=this->end(); ++point, ++next_point) {
            double  l = distance(*point, *next_point);
            if (l <= dl) 
            {
                new_path.push_back(*point);
            }
            else
            {
                int n = (int)(l/dl+1);
                double deltax = (next_point->x - point->x)/n;
                double deltay = (next_point->y - point->y)/n;
                double deltaz = (next_point->z - point->z)/n;
                for(int ipt=0; ipt<n; ++ipt) {
                    double x = (point->x + deltax*ipt);
                    double y = (point->y + deltay*ipt);
                    double z = (point->z + deltaz*ipt);
                    new_path.push_back(pt3d(x, y, z, d_R0, true) );
                }
            }
        }
        new_path.push_back(*point);
        set_path(new_path);
        if (!new_path.empty() ) { new_path.clear(); }
        return 0;
    }
    int double_denser_raypath() {
        std::list<pt3d>::iterator point = this->begin();
        std::list<pt3d>::iterator next_point = point; ++next_point;
        //
        for(; next_point!=this->end(); ++point, ++next_point) {
            double x = (next_point->x + point->x)*0.5;
            double y = (next_point->y + point->y)*0.5;
            double z = (next_point->z + point->z)*0.5;
            this->insert(next_point, pt3d(x, y, z, d_R0, true) );
            ++point;
        }
        return 0;
    }
    // insert points to set points exactly on the depth
    int insert_point_depth(double depth) {
        std::list<pt3d>::iterator point = this->begin();
        std::list<pt3d>::iterator next_point = point; ++next_point;
        for(;next_point!=this->end(); ++point, ++next_point )
        {
            if (cross_depth(*point, *next_point, depth) )
            {
                pt3d new_point = interpolate3D_depth(*point, *next_point, depth, d_R0);
                this->insert(next_point, new_point);
                ++point;
            }
        }
        return 0;
    }
    int insert_point_avoid_depth(double depth, int verbose= 0) {
        static double derr = 1.0e-6; // 1.0e-6 present very good performence
        std::list<pt3d>::iterator point0 = this->begin();
        std::list<pt3d>::iterator point1 = point0; ++point1;
        std::list<pt3d>::iterator point2 = point1; ++point2;
        // check the first point
        if ( ISEQUAL(point0->depth, depth) )
        {
            if (point0->depth < point1->depth)
            {
                ///
                //   * x0    here we need to move x0 lowerward a little bit
                //    \  
                //     \ 
                //      * x1
                ///
                point0->depth += derr;
                if (verbose)
                {
                    fprintf(stderr, "first point: depth x0(%lf) x1(%lf)\n", point0->depth, point1->depth);
                }
                point0->init(point0->lon, point0->lat, point0->depth, d_R0);
            }
            else
            {
                ///
                //       * x1
                //      /
                //     /
                //    * x0    here we need to move x0 upward a little bit
                ///
                point0->depth -= derr;
            }
            // update XYZ
            if (verbose)
            {
                fprintf(stderr, "first point: depth x0(%lf) x1(%lf)\n", point0->depth, point1->depth);
            }
            point0->init(point0->lon, point0->lat, point0->depth, d_R0);
            
        }
        // check the midder point
        for(;point2 != this->end(); ++point0, ++point1, ++point2 )
        {
            if ( ISEQUAL(point1->depth, depth) )
            {
                double x0 = point0->depth;
                double x1 = point1->depth;
                double x2 = point2->depth;
                if (x0 < x1 && x2 <= x1) 
                {
                    ///
                    //   * x0  * x2 
                    //    \   /
                    //     \ /
                    //      * x1 (itdep1)
                    ///
                    x1 -= derr;
                    point1->init(point1->lon, point1->lat, x1, d_R0);
                    if (verbose)
                    {
                        fprintf(stderr, "middle point: depth x0(%lf) x1(%lf) x2(%lf)\n", point0->depth, point1->depth, point2->depth);
                    }
                }
                else if (x0 < x1 && x2 > x1) 
                {
                    ///
                    //   * x0 (itdep0)         * x0 (itdep0)        
                    //    \                     \                   
                    //     \                     *  <== insert point here
                    //      \                     \                 
                    //       * x1 (itdep1)         * x1 (itdep1)    
                    //        \                     \               
                    //         * x2 (itdep2)         * x2 (itdep2)  
                    ///
                    if (verbose)
                    {
                        fprintf(stderr, "middle point: depth x0(%lf) x1(%lf) x2(%lf)\n", point0->depth, point1->depth, point2->depth);
                    }
                    this->insert(point1, *point1);
                    ++point0;
                    //
                    point0->init(point0->lon, point0->lat, x1-derr, d_R0);
                    point1->init(point1->lon, point1->lat, x1+derr, d_R0);
                }
                else if ( x0 > x1 && x2 >= x1) 
                {
                    ///
                    //      * x1 (itdep1)
                    //     / \
                    //    /   \
                    //   * x0  * x2 
                    ///
                    if (verbose)
                    {
                        fprintf(stderr, "middle point: depth x0(%lf) x1(%lf) x2(%lf)\n", point0->depth, point1->depth, point2->depth);
                    }
                    x1 += derr;
                    point1->init(point1->lon, point1->lat, x1, d_R0);
                }
                else if ( x0 > x1 && x2 < x1) 
                {
                    ///
                    //          * x2 (itdep2)            * x2 (itdep2)  
                    //         /                        /               
                    //        * x1 (itdep1)            * x1 (itdep1)    
                    //       /                        /                 
                    //      /                        * <== insert point here
                    //     /                        /                   
                    //    * x0 (itdep0)            * x0 (itdep0)        
                    ///
                    if (verbose)
                    {
                        fprintf(stderr, "middle point: depth x0(%lf) x1(%lf) x2(%lf)\n", point0->depth, point1->depth, point2->depth);
                    }
                    this->insert(point1, *point1);
                    ++point0;
                    //
                    point0->init(point0->lon, point0->lat, x1+derr, d_R0);
                    point1->init(point1->lon, point1->lat, x1-derr, d_R0);
                }
            }
        }
        // check for the last point
        point1 = this->end(); --point1;
        point0 = point1; --point0;
        if ( ISEQUAL(point1->depth, depth) )
        {
            if (point0->depth < point1->depth)
            {
                ///
                //   * x0    
                //    \  
                //     \ 
                //      * x1 here we need to move x1 upward a little bit
                ///
                point1->depth -= derr;
                if (verbose)
                {
                    fprintf(stderr, "last point: depth x0(%lf) x1(%lf)\n", point0->depth, point1->depth);
                }
            }
            else
            {
                ///
                //       * x1 here we need to move x0 lowerward a little bit
                //      /
                //     /
                //    * x0    
                ///
                point1->depth += derr;
                if (verbose)
                {
                    fprintf(stderr, "last point: depth x0(%lf) x1(%lf)\n", point0->depth, point1->depth);
                }
            }
            // update XYZ
            point1->init(point1->lon, point1->lat, point1->depth, d_R0);
        }
        return 0;
    }
    int remove_same_point() {
        std::list<pt3d>::iterator point = this->begin();
        std::list<pt3d>::iterator next_point = point; ++next_point;
        for(;next_point!=this->end(); ++point, ++next_point)
        {
            if (point->is_same_point(*next_point) )
                this->erase(next_point);
        }
        return 0;
    }
    /* access ray path
    */
    char type() { return d_type; }
    inline double R0 () { return d_R0; }
    inline int npts() const { return this->size(); }
    inline int set_time1d(double t) { d_time_1d = t; }
    inline double time1d() { return d_time_1d; }
    //std::list<pt3d> & raypath_points() { return d_path_pts; }
    //const std::list<pt3d> & const_raypath_points() const { return d_path_pts; }
private:
    /* 3-D Ray-path for S or P wave
    */
    char d_type; // can be 'S/P'
    //d_path_pts;
    /* Some background values
    */
    double d_R0 = 6371.0;
    double d_time_1d;
};


class raypath3d : public std::vector<raypath3d_segment>
{
public:
    raypath3d() {};
    ~raypath3d()
    {
        if (!d_phase.empty() ) d_phase.clear();
        if (!this->empty() )   this->clear();
    };
    /* Set the ray-path
    */
    int init(const char *phase, int npts, double *lon, double *lat, double *dep, int verbose=0) {
        d_phase = phase;
        d_whole_path.set_path(npts, lon, lat, dep);
        decipher_phase_path(verbose);
        return 0;
    }
    int init(const char *phase, const char * filename, int verbose= 0)
    {
        #define MAXNPTS 4194304
        //
        FILE *fid = fopen(filename, "rb");
        int npts;
        int junk;
        static double lons[MAXNPTS];
        static double lats[MAXNPTS];
        static double depth[MAXNPTS];
        junk = fread(&npts, sizeof(int), 1, fid);
        junk = fread(&d_traveltime_taup, sizeof(double), 1, fid);
        junk = fread(&d_rayparam_taup,   sizeof(double), 1, fid);
        if (npts > MAXNPTS) 
        { 
            fprintf(stderr, "Err. Too many points in the raypath. (%d, %s)\n", npts, filename);
        }
        if (junk != 1)    { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
        junk = fread(lons,  sizeof(double), npts, fid );
        if (junk != npts) { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
        junk = fread(lats,  sizeof(double), npts, fid );
        if (junk != npts) { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
        junk = fread(depth, sizeof(double), npts, fid );
        if (junk != npts) { fprintf(stderr, "Err in reading file %s\n", filename); exit(-1); }
        fclose(fid);
        //
        init(phase, npts, lons, lats, depth, verbose);
        return 0;
        #undef MAXNPTS
    }
    int output_plain_txt(const char *fnm)
    {
        FILE *fid = fopen(fnm, "w");
        for(std::vector<raypath3d_segment>::iterator it = this->begin(); it!=this->end(); ++it )
        {
            fprintf(fid, "# > %c %ld\n", it->type(), it->size() );
            for(auto point = it->begin(); point != it->end(); ++point )
            {
                fprintf(fid, "%lf %lf %lf %lf %lf %lf %c\n", point->lon, point->lat, point->depth, point->x, point->y, point->z, it->type() );
            }
        }
        fclose(fid);
        return 0;
    }
    inline double traveltime_taup() { return  d_traveltime_taup; }
    inline double rayparam_taup()   { return  d_rayparam_taup; }
private:
    // To cut whole path into peices so that each peice correspond to a single wave leg (P, S, K, I, J)
    int decipher_phase_path(int verbose= 0)
    {
        // update the whole path to make sure there are intersections with the CMB and ICB.
        d_whole_path.insert_point_depth(d_CMB);
        d_whole_path.insert_point_depth(d_ICB);
        d_whole_path.remove_same_point();
        raypath3d_segment::iterator it0 = d_whole_path.begin();
        raypath3d_segment::iterator it1 = d_whole_path.begin(); ++it1;
        auto itphase = d_phase.begin();
        int npts = 2;
        while( it1 != d_whole_path.end() )
        {
            if (ISEQUAL(it1->depth, 0.0) || ISEQUAL(it1->depth, d_CMB) || ISEQUAL(it1->depth, d_ICB) )
            {
                auto ittmp = it1; ++ittmp;
                if (*itphase == 'c') ++itphase; // there cannot be `cc`
                //raypath3d_segment *segment = new raypath3d_segment(*itphase, d_whole_path.R0(), it0, it1);
                this->push_back( raypath3d_segment(*itphase, d_whole_path.R0(), it0, ittmp) );
                //printf("%d \n", this->operator[](0).size() );
                if (verbose) 
                {
                    fprintf(stderr, ">>> Find one raypath segment: %c  depth(%lf->%lf) npts(%d)\n", *itphase, 
                        it0->depth, it1->depth, npts);
                }
                //
                it0 = it1; 
                ++it1;
                npts = 2;
                ++itphase;
            }
            else
            {
                ++it1;
                ++npts;
            }
        }
        if (npts>2) {
            --npts;
            if (*itphase == 'c') ++itphase; // there cannot be `cc`
            auto ittmp = it1; --ittmp;
            fprintf(stderr, ">>> Find one raypath segment: %c  depth(%lf->%lf) npts(%d)\n", *itphase, 
                        it0->depth, ittmp->depth, npts);
            this->push_back( raypath3d_segment(*itphase, d_whole_path.R0(), it0, it1) );
        }
    }
private:
    raypath3d_segment  d_whole_path;
    //
    std::string d_phase;
    double d_traveltime_taup;
    double d_rayparam_taup;
    /**/
    double  d_CMB = 2891.50; // depth in km
    double  d_ICB = 5153.50; // depth in km
};

#endif