#ifndef TOMO__SENSITIVITY___________
#define TOMO__SENSITIVITY___________

#include "raypath.hh"
#include "earthmodel.hh"

class sensitivity : public std::vector<double> 
{
public:
    sensitivity() {}
    ~sensitivity() {
        if (!this->empty() ) { this->clear(); }
    }

    int run(earthmod3d & mod3d, raypath3d & ray) 
    {
        // The whole sensitivity
        this->resize(mod3d.npts()); this->assign(this->size(), 0.0); 
        d_time_1d = 0.0;
        // The sensitivities for all path segments
        d_path_segments_sensitivity.resize(ray.size() );
        for(auto it = d_path_segments_sensitivity.begin(); it!=d_path_segments_sensitivity.end(); ++it)
        {
            it->resize(mod3d.npts() ); it->assign(it->size(), 0.0); 
        }
        //
        earthmod1d *mod1d = mod3d.mod1d();
        auto jumps = mod1d->discontinuous_layers();
        //for(auto segment =ray.begin(); segment != ray.end(); ++segment)
        for(int isegment = 0; isegment<ray.size(); ++isegment)
        {
            // adjust the ray to avoid discontinuities
            // fprintf(stderr, "\n\n\n");
            for(auto it=jumps.begin(); it!=jumps.end(); ++it)
            {
                ray[isegment].insert_point_avoid_depth(it->second, 0);
                //segment->insert_point_avoid_depth(it->second, 1);
            }
            // compute the sensitivity
            //run_ray_segment(mod3d, *segment);
            run_ray_segment(mod3d, ray[isegment], d_path_segments_sensitivity[isegment]);
        }
        return 0;
    }
    int run_ray_segment(earthmod3d & mod3d, raypath3d_segment & segment, std::vector<double> segment_sensitivity )
    {
        // Obtain associated length for each point
        //
        //     s0      s1       s2       s3       s4       s5                  s[N-1]
        //     p0      p1       p2       p3       p4       p5                  p[N-1]
        //     +--------+--------+--------+--------+--------+ ... ... +--------+
        //         l0       l1       l2       l3       l4               l[N-2]
        //   
        //          l0*(s0+s1)   l1*(s1+s2)   l2*(s2+s3)   l3*(s3+s4)          l[N-2]*(s[N-2]+s[N-1])
        //     T =  ---------- + ---------- + ---------- + ---------- + ... +  ----------------------
        //               2            2            2            2                         2
        //     
        //         l0       l0+l1      l1+l2      l2+l3      l3+l4              l[N-3]+l[N-2]          l[N-2]
        //     T = --- s0 + ----- s1 + ----- s2 + ----- s3 + ----- s4 + ...   + ------------- s[N-2] + ------ s[N-1]
        //          2         2          2          2          2                      2                   2  
        //   
        //     T =  u0*s0 +    u1*s1 +    u2*s2 +    u3*s3 +    u4*s4 + ...   +        u[N-2]*s[N-2] + u[N-1]*s[N-1]
        //       =  SUM_i u[i]*s[i]
        //       =  SUM_i u[i]*s(pi)
        //
        // As for the slowness s(pi), it depends on eight surrounding grid points declared in 3-D model. That is
        //     s(pi) = c0[i]*s(p0) + c1*s(p1) + c2*s(p2) + ... + c7*s(p7),
        //
        // where the geometry is
        //
        //                     ^ X (lat)
        //                    /        
        //                p1 o-------------o p3
        //                  /|            /|
        //                 / |           / |
        //                /  |          /  |
        //           p0  /   |      p2 /   |
        //              o-------------o---------->  Y (lon)
        //              | p5 o--------|----o p7
        //              |   /         |   /  
        //              |  /          |  /
        //              | /           | /
        //              |/            |/
        //              o-------------o
        //           p4 |           p6
        //              |
        //              V 
        //              Z (depth)
        //
        // In 3-D model, the point p0 is stored in position of i0, and p1 is i1, etc. That is
        //     s(p0) = M[i0], s(p1) = M[i1], ..., s(p7) = M[i7]
        // where M is the model parameter, and i0, i1, i2,...,i7 can be determined.
        // Therefore, the traveltime is
        //     T =  SUM_i u[i]*s(pi)
        //       =  SUM_i u[i]* { c0[i]*s(p0) + c1*s(p1) + c2*s(p2) + ... + c7*s(p7)  }
        //       =  SUM_i u[i]* { c0[i]*M[i0] + c1*M[i1] + c2*M[i2] + ... + c7*M[i7]  }
        //       =  SUM_i u[i]*c0[i]*M[i0] + u[i]*c1*M[i1]  + ... + u[i]*c[7]*M[i7]
        //       =  ...
        //       =  SUM_j G[j] * M[j]
        //
        // where u[i]*c0[i], u[i]*c1,... can be calculated for the ith point. If we repeat the computation
        // for all points, that is a loop for i:0-->N-1, then we obtain the sensitivity kerenel G[j] for 
        // the jth point.
        // 
        // We focus on model perturbations, that is M[j] = M0[j] * pert[j], and hence
        //    T  =  SUM_j G[j] * M0[j] * pert[j]
        //       =  SUM_j    G'[j]     * pert[j]
        //
        // Obtain the coefficient `u`.
        auto pt0 = segment.begin();
        auto pt1 = pt0; ++pt1;
        auto pt2 = pt1; ++pt2;
        std::vector<double> u(segment.npts() );
        u[0] = 0.5*distance(*pt0, *pt1);
        for(int idx=1; pt2!=segment.end(); ++pt0, ++pt1, ++pt2, ++idx)
        {
            u[idx] = 0.5* (distance(*pt0, *pt1) + distance(*pt1, *pt2) );
        }
        u[u.size()-1] = 0.5*distance(*pt0, *pt1);
        // A loop over all points. 
        // Calculate c0[i],c1[i],...,c7[i] and i0,i1,...,i7 for the ith point.
        int idx = 0;
        for(auto pt = segment.begin(); pt!=segment.end(); ++pt, ++idx)
        {
            int i0, i1, i2, i3, i4, i5, i6, i7;
            double c0, c1, c2, c3, c4, c5, c6, c7;
            double sum;
            mod3d.interpolate3D_coef(*pt, &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, 
                                          &i0, &i1, &i2, &i3, &i4, &i5, &i6, &i7 );
            sum = c0+c1+c2+c3+c4+c5+c6+c7;
            if ( fabs(sum-1.0)>1.0e-6 ) {
                fprintf(stderr, "sum %lf\n", sum);
            }
            //this->operator[](i0) += u[idx] * c0;
            //this->operator[](i1) += u[idx] * c1;
            //this->operator[](i2) += u[idx] * c2;
            //this->operator[](i3) += u[idx] * c3;
            //this->operator[](i4) += u[idx] * c4;
            //this->operator[](i5) += u[idx] * c5;
            //this->operator[](i6) += u[idx] * c6;
            segment_sensitivity[i0] += u[idx] * c0;
            segment_sensitivity[i1] += u[idx] * c1;
            segment_sensitivity[i2] += u[idx] * c2;
            segment_sensitivity[i3] += u[idx] * c3;
            segment_sensitivity[i4] += u[idx] * c4;
            segment_sensitivity[i5] += u[idx] * c5;
            segment_sensitivity[i6] += u[idx] * c6;
            segment_sensitivity[i7] += u[idx] * c7;
        }
        // Set sensitivity
        std::vector<double> & slowness = (segment.type() == 'S' || segment.type() == 'J' ) ? mod3d.slowness_s_vector() : mod3d.slowness_p_vector();
        double time1d = 0.0;
        int ncount = 0;
        for(int idx=0; idx<segment_sensitivity.size(); ++idx)
        {
            //if ( !ISEQUAL(this->operator[](idx), 0.0) ) { // the sensitivity are ZERO for very most of points.
            if ( !ISEQUAL(segment_sensitivity[idx], 0.0) ) { // the sensitivity are ZERO for very most of points.
                // this->operator[](idx) *= slowness[idx];
                // time1d += this->operator[](idx);
                segment_sensitivity[idx] *= slowness[idx];
                time1d += segment_sensitivity[idx];
                ++ncount;
            }
        }
        segment.set_time1d(time1d);
        // printf(">>> segment time: %lf %d\n", time1d, ncount);
        d_time_1d += time1d;
        //
        if ( !u.empty() ) { u.clear(); }
        return 0;
    }
    double time_1d() { return d_time_1d; }
private:   
    double d_time_1d; // this is in fact the sum of all sensitivities.
    std::vector<std::vector<double> > d_path_segments_sensitivity;
};

class sensitivity_zip : public std::pair< std::vector<int>, std::vector<double>  >  {
public:
    sensitivity_zip()  {}
    sensitivity_zip(std::vector<double> & row ) { from_vector(row); }
    ~sensitivity_zip() 
    {
        if (! this->first.empty()  ) this->first.clear();
        if (! this->second.empty() ) this->second.clear();
    }

    int from_vector(std::vector<double> & row)
    {
        std::vector<int>    & index = this->first;  index.resize(0);
        std::vector<double> & value = this->second; value.resize(0);
        for(int i=0; i<row.size(); ++i)
        {
            if (!ISEQUAL(row[i], 0.0) )
            {
                index.push_back(i);
                value.push_back(row[i] );
            }
        }
        return 0;
    }
};

#endif
