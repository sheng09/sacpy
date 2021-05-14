#include "signal_proc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double aimag(complexf c)
{
    return (c.im);
}
double cmplxabs(complexf c)
{
    return (sqrt((double)((c.re * c.re) + (c.im * c.im))));
}
double cmplxang(complexf c)
{
    double d;

    if (fabs(c.im) < 1.0e-14)
    { /* Real Number */
        if (c.re > 0)
        {
            d = 0;
        }
        else if (c.re < 0)
        {
            d = M_PI;
        }
        else
        {
            fprintf(stderr, "complex: Attempt to take angle (argument) of 0+0i\n");
            return 0.0;
        }
    }
    else if (fabs(c.re) < 1.0e-14)
    { /* Imaginary Number */
        if (c.im > 0)
        {
            d = M_PI_2;
        }
        else if (c.re < 0)
        {
            d = -M_PI_2;
        }
        else
        {
            fprintf(stderr, "complex: Attempt to take angle (argument) of 0+0i\n");
            return 0.0;
        }
    }
    else
    {
        d = atan(c.im / c.re);
        if (c.re < 0.0)
        {
            if (c.im < 0.0)
            {
                d -= M_PI;
            }
            else
            {
                d += M_PI;
            }
        }
    }

    return (d);
}
double cmplxtof(complexf c)
{
    return (c.re);
}

complexf cmplxadd(complexf c1, complexf c2)
{
    c1.re += c2.re;
    c1.im += c2.im;
    return (c1);
}
complexf cmplxcj(complexf c)
{
    c.im = -c.im;
    return (c);
}
complexf cmplxmul(complexf c1, complexf c2)
{
    complexf c3;
    c3.re = (c1.re * c2.re) - (c1.im * c2.im);
    c3.im = (c1.re * c2.im) + (c1.im * c2.re);
    return (c3);
}
complexf flttocmplx(double d1, double d2)
{
    complexf c;
    c.re = (float)d1;
    c.im = (float)d2;
    return (c);
}
complexf cmplxsub(complexf c1,  complexf c2)
{
    c1.re -= c2.re;
    c1.im -= c2.im;
    return (c1);
}

complexf cmplxsqrt(complexf c)
{

    double sqrtsave, angle;

    sqrtsave = sqrt(cmplxabs(c));
    angle = cmplxang(c);

    c.re = (float)(sqrtsave * cos(angle / 2.0));
    c.im = (float)(sqrtsave * sin(angle / 2.0));

    if (c.re < 0.0)
    {
        c.re = -c.re;
        c.im = -c.im;
    }
    else if (c.re < 1.0e-14 && c.re > -1.0e-14 && c.im < 0.0)
        c.im = -c.im;

    return (c);
}
complexf cmplxdiv(complexf c1, complexf c2)
{
    complexf c;
    float f;

    if (c2.re == 0.0 && c2.im == 0.0)
    {
        printf("complex divide by zero-cmplxdiv\n");
        exit(1);
    }

    f = c2.re * c2.re + c2.im * c2.im;

    c.re = (c1.re * c2.re + c1.im * c2.im) / f;
    c.im = (c2.re * c1.im - c1.re * c2.im) / f;

    return (c);
}
complexf cmplxlog(complexf c)
{
    complexf c1;

    c1.re = (float)log(cmplxabs(c));
    c1.im = (float)cmplxang(c);

    return (c1);
}
complexf cmplxexp(complexf c)
{
    double d;

    if (c.re == 0.0)
        d = 1.0;
    else
        d = exp(c.re);

    if (c.im == 0.0)
    {
        c.re = (float)d;
        return (c);
    }

    c.re = (float)(d * cos(c.im));
    c.im = (float)(d * sin(c.im));

    return (c);
}
complexf cmplxpow(complexf c, double d)
{
    if (c.re == 0.0 && c.im == 0.0)
        return (c);

    c = cmplxlog(c);
    c.re = (float)(d * c.re);
    c.im = (float)(d * c.im);

    return (cmplxexp(c));
}
complexf cmplxneg(complexf c)
{
    c.re = -c.re;
    c.im = -c.im;
    return (c);
}
double powi(double b, int x) {
	return pow(b, (double)x);
}


/** 
 * Design IIR Digital Filters from Analog Prototypes
 * 
 * @param iord 
 *    Filter Order, Maximum of 10
 * @param type 
 *    Filter Type, Character *2
 *      - 0 'LP'  Lowpass
 *      - 1 'HP'  Highpass
 *      - 2 'BP'  Bandpass
 *      - 4 'BR'  Bandreject
 * @param aproto 
 *    Analog Prototype, Character *2
 *      - 0 'BU'  Butterworth
 *      - 1 'BE'  Bessel
 *      - 2 'C1'  Cheyshev Type I
 *      - 4 'C2'  Cheyshev Type II
 * @param a 
 *    Chebyshev stopband Attenuation Factor
 * @param trbndw 
 *    Chebyshev transition bandwidth, fraction of lowpass prototype 
 *    passband width
 * @param fl 
 *    Low Frequency cutoff
 * @param fh 
 *    High Frequency cutoff
 * @param ts 
 *    Sampling Rate / Delta
 * @param sn 
 *    Array containing numerator coefficients of 2nd Order Sections
 *    Packed Head to Tail
 * @param sd 
 *    Array containing denominator coefficients of 2nd Order Sections
 *    Packed Head to Tail
 * @param nsects 
 *    Length of arrays \p sn and \p sd
 *
 *  @copyright 1990  Regents of the University of California                      
 *
 *  @author  Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 * 
 *  @date Documented/Reviewed
 *
 */
int design(int iord, int type, int aproto, double a, double trbndw,
            double fl, double fh, double ts, float *sn, float *sd, int *nsects)
{
    char stype[10][4];
    float dcvalue, eps, fhw, flw, omegar, ripple;
    complexf p[10], z[10];

    //std::cout << "b desin\n";
    /*  Analog prototype selection                                                   
	 * */

    if (aproto == IIR_BU)  // BU
    {
        buroots(p, (char *)stype, 4, &dcvalue, nsects, iord);
    }
    else if (aproto == IIR_BE)
    {
        beroots(p, (char *)stype, 4, &dcvalue, nsects, iord);
    }
    else if (aproto == IIR_C1)
    {
        chebparm(a, trbndw, iord, &eps, &ripple);
        c1roots(p, (char *)stype, 4, &dcvalue, nsects, iord, eps);
    }
    else if (aproto == IIR_C2)
    {
        omegar = 1. + trbndw;
        c2roots(p, z, (char *)stype, 4, &dcvalue, nsects, iord, a, omegar);
    }
    else
	{
        fprintf(stderr, "filter: Unknown Analog filter prototype: '%d'\n", aproto);
        fprintf(stderr, "        Expected: BU, BESSEL, C1, C2\n");
        return -1;
    }

    /*  Analog mapping selection                                                     
	 * */
    if (type == IIR_BP) // BP
    {
        flw = warp(fl * ts / 2., 2.);
        fhw = warp(fh * ts / 2., 2.);
        lptbp(p, z, (char *)stype, 4, dcvalue, nsects, flw, fhw, sn, sd);
    }
    else if (type == IIR_BR) // BR
    {
        flw = warp(fl * ts / 2., 2.);
        fhw = warp(fh * ts / 2., 2.);
        lptbr(p, z, (char *)stype, 4, dcvalue, nsects, flw, fhw, sn, sd);
    }
    else if (type == IIR_LP) // LP
    {
        fhw = warp(fh * ts / 2., 2.);
        lp(p, z, (char *)stype, 4, dcvalue, *nsects, sn, sd);
        cutoffs(sn, sd, *nsects, fhw);
    }
    else if (type == IIR_HP) // HP
    {
        flw = warp(fl * ts / 2., 2.);
        lpthp(p, z, (char *)stype, 4, dcvalue, *nsects, sn, sd);
        cutoffs(sn, sd, *nsects, flw);
    }

    /*  Bilinear analog to digital transformation                                    
	 * */
    bilin2(sn, sd, *nsects);
    return 0;
}

/** 
 * Compute the Butterworth Poles for a Normalized Low Pass (LP) Filter
 * 
 * @param p 
 *    Complex Array containing Poles. Contains only one of from each
 *     - Complex Conjugate Pole-Zero Pair
 *     - Complex Conjugate Pole Pair
 *     - Single Real Pole
 * @param rtype 
 *    Character Array indicating 2nd Order Section Types
 *     - 'CPZ' Complex Conjugate Pole-Zero Pair
 *     - 'CP'  Complex Conjugate Pole Pair
 *     - 'SP'  Single Real Pole
 * @param rtype_s 
 *     Length of string \p rtype
 * @param dcvalue 
 *     Magnitude of the filter at Zero Frequency
 * @param nsects 
 *     Number of 2nd Order Sections
 * @param iord 
 *     Desired Filter Order, Must be between 1 and 8
 *
 * @return Nothing
 *
 * @copyright 1990  Regents of the University of California
 *
 *
 * @author  Dave Harris                                                         
 *          Lawrence Livermore National Laboratory                              
 *          L-205                                                               
 *          P.O. Box 808                                                        
 *          Livermore, CA  94550                                                
 *          USA                                                                 
 *          (415) 423-0617                                                      
 *
 * \note
 *   \f[
 *     \begin{array}{rcl}
 *        n &=& iord \\
 *     H(s) &=& k_0 / \displaystyle\prod_{k=1}^n  s - s_k \\
 *      s_k &=& exp ( i \pi [0.5 + 2(2 k - 1)/2n ] ); k = 1, 2, ..., n \\
 *      k_0 &=& 1.0
 *     \end{array}
 *   \f]
 *
 * \date 900907   LAST MODIFIED
 * 
 */
void buroots(complexf *p, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord)
{
    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))
    int half, k;
    float angle, pi;

    complexf *const P = &p[0] - 1;

    pi = 3.14159265;

    half = iord / 2;

    /* TEST FOR ODD ORDER, AND ADD POLE AT -1                                        
	 * */
    *nsects = 0;
    if (2 * half < iord)
    {
        P[1] = flttocmplx(-1., 0.);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "SP", 2);
        *nsects = 1;
    }
    for (k = 1; k <= half; k++)
    {
        angle = pi * (.5 + (float)(2 * k - 1) / (float)(2 * iord));
        *nsects = *nsects + 1;
        P[*nsects] = flttocmplx(cos(angle), sin(angle));
        fstrncpy(RTYPE(*nsects - 1, 0), rtype_s - 1, "CP", 2);
    }
    *dcvalue = 1.0;

    return ;
    #undef RTYPE
}

/** 
 * Compute Bessel Poles For a Normalized Low Pass (LP) Filter
 * 
 * @param p 
 *    Complex Array containing Poles. Contains only one of from each
 *     - Complex Conjugate Pole-Zero Pair
 *     - Complex Conjugate Pole Pair
 *     - Single Real Pole
 * @param rtype 
 *    Character Array indicating 2nd Order Section Types
 *     - 'CPZ' Complex Conjugate Pole-Zero Pair
 *     - 'CP'  Complex Conjugate Pole Pair
 *     - 'SP'  Single Real Pole
 * @param rtype_s 
 *     Length of string \p rtype
 * @param dcvalue 
 *     Magnitude of the filter at Zero Frequency
 * @param nsects 
 *     Number of 2nd Order Sections
 * @param iord 
 *     Desired Filter Order, Must be between 1 and 8
 *
 * @return Nothing
 *
 * @copyright 1990  Regents of the University of California                      
 *
 * \bug Poles are defined explicitly within the routine. 
 *        Pull them out and define and document them.
 *
 * @author  Dave Harris                                                         
 *           Lawrence Livermore National Laboratory                              
 *           L-205                                                               
 *           P.O. Box 808                                                        
 *           Livermore, CA  94550                                                
 *           USA                                                                 
 *           (415) 423-0617                                                      
 *
 *
 * \date 920415   Changed P and RTYPE to adjustable array by 
 *                     using an "*" rather than a "1".     
 *
 */
void beroots(complexf *p, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord)
{
    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))

    complexf *const P = &p[0] - 1;

    if (iord == 1)
    {

        P[1] = flttocmplx(-1.0, 0.0);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "SP", 2);
    }
    else if (iord == 2)
    {

        P[1] = flttocmplx(-1.1016013, 0.6360098);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "CP", 2);
    }
    else if (iord == 3)
    {

        P[1] = flttocmplx(-1.0474091, 0.9992645);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "CP", 2);
        P[2] = flttocmplx(-1.3226758, 0.0);
        fstrncpy(RTYPE(1, 0), rtype_s - 1, "SP", 2);
    }
    else if (iord == 4)
    {

        P[1] = flttocmplx(-0.9952088, 1.2571058);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "CP", 2);
        P[2] = flttocmplx(-1.3700679, 0.4102497);
        fstrncpy(RTYPE(1, 0), rtype_s - 1, "CP", 2);
    }
    else if (iord == 5)
    {

        P[1] = flttocmplx(-0.9576766, 1.4711244);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "CP", 2);
        P[2] = flttocmplx(-1.3808774, 0.7179096);
        fstrncpy(RTYPE(1, 0), rtype_s - 1, "CP", 2);
        P[3] = flttocmplx(-1.5023160, 0.0);
        fstrncpy(RTYPE(2, 0), rtype_s - 1, "SP", 2);
    }
    else if (iord == 6)
    {

        P[1] = flttocmplx(-0.9306565, 1.6618633);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "CP", 2);
        P[2] = flttocmplx(-1.3818581, 0.9714719);
        fstrncpy(RTYPE(1, 0), rtype_s - 1, "CP", 2);
        P[3] = flttocmplx(-1.5714904, 0.3208964);
        fstrncpy(RTYPE(2, 0), rtype_s - 1, "CP", 2);
    }
    else if (iord == 7)
    {

        P[1] = flttocmplx(-0.9098678, 1.8364514);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "CP", 2);
        P[2] = flttocmplx(-1.3789032, 1.1915667);
        fstrncpy(RTYPE(1, 0), rtype_s - 1, "CP", 2);
        P[3] = flttocmplx(-1.6120388, 0.5892445);
        fstrncpy(RTYPE(2, 0), rtype_s - 1, "CP", 2);
        P[4] = flttocmplx(-1.6843682, 0.0);
        fstrncpy(RTYPE(3, 0), rtype_s - 1, "SP", 2);
    }
    else if (iord == 8)
    {

        P[1] = flttocmplx(-0.8928710, 1.9983286);
        fstrncpy(RTYPE(0, 0), rtype_s - 1, "CP", 2);
        P[2] = flttocmplx(-1.3738431, 1.3883585);
        fstrncpy(RTYPE(1, 0), rtype_s - 1, "CP", 2);
        P[3] = flttocmplx(-1.6369417, 0.8227968);
        fstrncpy(RTYPE(2, 0), rtype_s - 1, "CP", 2);
        P[4] = flttocmplx(-1.7574108, 0.2728679);
        fstrncpy(RTYPE(3, 0), rtype_s - 1, "CP", 2);
    }

    *nsects = iord - iord / 2;

    *dcvalue = 1.0;

    /*  DONE                                                                         
	 * */
    return;
    #undef RTYPE
}

/** 
 * Calculate Chebyshev Type I and II Design Parameters
 * 
 * @param a 
 *    Desired Stopband Attenuation, i.e. max stopband 
 *    amplitude is 1/ATTEN
 * @param trbndw 
 *    Transition bandwidth between stop and passband as a 
 *    fraction of the passband width
 * @param iord 
 *    Filter Order (number of Poles)
 * @param eps 
 *    Output Chebyshev passband parameter
 * @param ripple 
 *    Passband ripple
 *
 * @return Nothing
 *
 * \note
 *   \f[
 *     \begin{array}{rcl}
 *  \omega &=& 1.0 + trbndw \\
 *  \alpha &=& (\omega + \sqrt{\omega^2 - 1.0} )^{iord}\\
 *       g &=& \alpha^2 + 1 / 2\alpha \\
 *     eps &=& \sqrt{a^2 - 1.0 } / g \\
 *  ripple &=& 1 / \sqrt{ 1.0 + eps^2 } \\
 *     \end{array}
 *   \f]
 *
 * \copyright 1990  Regents of the University of California
 *
 * \author   Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *           (415) 423-0617
 */
void chebparm(double a, double trbndw, int iord, float *eps, float *ripple)
{
    float alpha, g, omegar;

    omegar = 1. + trbndw;
    alpha = powi(omegar + sqrt(powi(omegar, 2) - 1.), iord);
    g = (powi(alpha, 2) + 1.) / (2. * alpha);
    *eps = sqrt(powi(a, 2) - 1.) / g;
    *ripple = 1. / sqrt(1. + powi(*eps, 2));

    return;
}

/** 
 * Compute Chebyshev Type I Poles for a Normalized Low Pass (LP) Filter
 * 
 * @param p 
 *    Complex Array containing Poles. Contains only one of from each
 *     - Complex Conjugate Pole-Zero Pair
 *     - Complex Conjugate Pole Pair
 *     - Single Real Pole
 * @param rtype 
 *    Character Array indicating 2nd Order Section Types
 *     - 'CPZ' Complex Conjugate Pole-Zero Pair
 *     - 'CP'  Complex Conjugate Pole Pair
 *     - 'SP'  Single Real Pole
 * @param rtype_s 
 *     Length of string \p rtype
 * @param dcvalue 
 *     Magnitude of the filter at Zero Frequency
 * @param nsects 
 *     Number of 2nd Order Sections
 * @param iord 
 *     Desired Filter Order, Must be between 1 and 8
 * @param eps
 *     Output Chebyshev Parameter Related to Passband Ripple
 *
 * @return Nothing
 *
 * @copyright 1990  Regents of the University of California
 *
 * @author  Dave Harris                                                         
 *          Lawrence Livermore National Laboratory                              
 *          L-205                                                               
 *          P.O. Box 808                                                        
 *          Livermore, CA  94550                                                
 *          USA                                                                 
 *          (415) 423-0617                                                      
 *
 * \todo Documentation for chebyshev Type I Filter.  
 *          Find roots and multiply them together.
 *
 *
 * \date 900907    LAST MODIFIED
 *
 */
void c1roots(complexf *p, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord, double eps)
{
    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))
    int half, i, i_;
    float angle, c, gamma, omega, pi, s, sigma;

    complexf *const P = &p[0] - 1;

    pi = 3.14159265;
    half = iord / 2;

    /*  INTERMEDIATE DESIGN PARAMETERS                                               
	 * */
    gamma = (1. + sqrt(1. + eps * eps)) / eps;
    gamma = log(gamma) / (float)(iord);
    gamma = exp(gamma);
    s = .5 * (gamma - 1. / gamma);
    c = .5 * (gamma + 1. / gamma);

    /*  CALCULATE POLES                                                              
	 * */
    *nsects = 0;
    for (i = 1; i <= half; i++)
    {
        i_ = i - 1;
        fstrncpy(RTYPE(i_, 0), rtype_s - 1, "CP", 2);
        angle = (float)(2 * i - 1) * pi / (float)(2 * iord);
        sigma = -s * sin(angle);
        omega = c * cos(angle);
        P[i] = flttocmplx(sigma, omega);
        *nsects = *nsects + 1;
    }
    if (2 * half < iord)
    {
        fstrncpy(RTYPE(half, 0), rtype_s - 1, "SP", 2);
        P[half + 1] = flttocmplx(-s, 0.0);
        *nsects = *nsects + 1;
        *dcvalue = 1.0;
    }
    else
    {
        *dcvalue = 1. / sqrt(1 + powi(eps, 2));
    }

    /*  DONE                                                                         
	 * */
    return;
    #undef RTYPE
}

/** 
 * Compute root for normailzed Low Pass Chebyshev Type II Filter
 * 
 * @param p 
 *    Complex Array containing Poles. Contains only one of from each
 *     - Complex Conjugate Pole-Zero Pair
 *     - Complex Conjugate Pole Pair
 *     - Single Real Pole
 * @param z 
 *    Complex Array containing Zeros Contains only one of from each
 *     - Complex Conjugate Pole-Zero Pair
 *     - Complex Conjugate Pole Pair
 *     - Single Real Pole
 * @param rtype 
 *    Character Array indicating 2nd Order Section Types
 *     - 'CPZ' Complex Conjugate Pole-Zero Pair
 *     - 'CP'  Complex Conjugate Pole Pair
 *     - 'SP'  Single Real Pole
 * @param rtype_s 
 *    Length of string \p rtype
 * @param dcvalue 
 *    Magnitude of filter at Zero Frequency
 * @param nsects 
 *    Number of 2nd order sections
 * @param iord 
 *   Input Desired Filter Order
 * @param a 
 *   Input Stopband attenuation factor
 * @param omegar 
 *   Input Cutoff frequency of stopband passband cutoff is 1.0 Hz
 *
 * @return Nothing
 *
 * \copyright Copyright 1990  Regents of the University of California
 *
 * \author   Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *           (415) 423-0617
 *
 * \date 900907:  LAST MODIFIED
 *
 */
void c2roots(complexf *p, complexf *z, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord, double a, double omegar)
{
    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))
    int half, i, i_;
    float alpha, angle, beta, c, denom, gamma, omega, pi, s, sigma;

    complexf *const P = &p[0] - 1;
    complexf *const Z = &z[0] - 1;

    pi = 3.14159265;
    half = iord / 2;

    /*  INTERMEDIATE DESIGN PARAMETERS                                               
	 * */
    gamma = a + sqrt(a * a - 1.);
    gamma = log(gamma) / (float)(iord);
    gamma = exp(gamma);
    s = .5 * (gamma - 1. / gamma);
    c = .5 * (gamma + 1. / gamma);

    *nsects = 0;
    for (i = 1; i <= half; i++)
    {
        i_ = i - 1;

        /*  CALCULATE POLES                                                              
		 * */
        fstrncpy(RTYPE(i_, 0), rtype_s - 1, "CPZ", 2);

        angle = (float)(2 * i - 1) * pi / (float)(2 * iord);
        alpha = -s * sin(angle);
        beta = c * cos(angle);
        denom = alpha * alpha + beta * beta;
        sigma = omegar * alpha / denom;
        omega = -omegar * beta / denom;
        P[i] = flttocmplx(sigma, omega);

        /*  CALCULATE ZEROS                                                              
		 * */
        omega = omegar / cos(angle);
        Z[i] = flttocmplx(0.0, omega);

        *nsects = *nsects + 1;
    }

    /*  ODD-ORDER FILTERS                                                            
	 * */
    if (2 * half < iord)
    {
        fstrncpy(RTYPE(half, 0), rtype_s - 1, "SP", 2);
        P[half + 1] = flttocmplx(-omegar / s, 0.0);
        *nsects = *nsects + 1;
    }

    /*  DC VALUE                                                                     
	 * */
    *dcvalue = 1.0;

    /*  DONE                                                                         
	 * */
    return;
    #undef RTYPE
}

/** 
 *  Subroutine to generate second order section parameterization
 *    from an pole-zero description for lowpass filters.
 * 
 * 
 * @param p 
 *    Array of Poles
 * @param z 
 *    Array of Zeros
 * @param rtype 
 *    Character array containing root type information
 *      - "SP"  Single real pole
 *      - "CP"  Complex conjugate pole pair
 *      - "CPZ" Complex conjugate pole and zero pairs
 * @param rtype_s
 *     Length of \p rtype  
 * @param dcvalue 
 *     Zero-frequency value of prototype filter
 * @param nsects 
 *     Number of second-order sections
 * @param sn 
 *     Output Numerator polynomials for second order sections.
 * @param sd 
 *     Output Denominator polynomials for second order sections.
 *
 * @copyright  Copyright 1990  Regents of the University of California
 *
 * @author:  Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *           (415) 423-0617
 *
 * \date   020416: Changed SN and SD adjustable arrays to use 
 *                 "*" rather than "1". - wct
 */
void lp(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int nsects, float *sn, float *sd)
{

    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))

    int i, i_, iptr;
    float scale;

    complexf *const P = &p[0] - 1;
    float *const Sd = &sd[0] - 1;
    float *const Sn = &sn[0] - 1;
    complexf *const Z = &z[0] - 1;

    iptr = 1;
    for (i = 1; i <= nsects; i++)
    {
        i_ = i - 1;

        if (memcmp(RTYPE(i_, 0), "CPZ", 3) == 0)
        {

            scale = cmplxtof(cmplxmul(P[i], cmplxcj(P[i]))) / cmplxtof(cmplxmul(Z[i],
                                                                                cmplxcj(Z[i])));
            Sn[iptr] = cmplxtof(cmplxmul(Z[i], cmplxcj(Z[i]))) * scale;
            Sn[iptr + 1] = -2. * cmplxtof(Z[i]) * scale;
            Sn[iptr + 2] = 1. * scale;
            Sd[iptr] = cmplxtof(cmplxmul(P[i], cmplxcj(P[i])));
            Sd[iptr + 1] = -2. * cmplxtof(P[i]);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;
        }
        else if (memcmp(RTYPE(i_, 0), "CP", 2) == 0)
        {

            scale = cmplxtof(cmplxmul(P[i], cmplxcj(P[i])));
            Sn[iptr] = scale;
            Sn[iptr + 1] = 0.;
            Sn[iptr + 2] = 0.;
            Sd[iptr] = cmplxtof(cmplxmul(P[i], cmplxcj(P[i])));
            Sd[iptr + 1] = -2. * cmplxtof(P[i]);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;
        }
        else if (memcmp(RTYPE(i_, 0), "SP", 2) == 0)
        {

            scale = -cmplxtof(P[i]);
            Sn[iptr] = scale;
            Sn[iptr + 1] = 0.;
            Sn[iptr + 2] = 0.;
            Sd[iptr] = -cmplxtof(P[i]);
            Sd[iptr + 1] = 1.;
            Sd[iptr + 2] = 0.;
            iptr = iptr + 3;
        }
    }

    Sn[1] = dcvalue * Sn[1];
    Sn[2] = dcvalue * Sn[2];
    Sn[3] = dcvalue * Sn[3];

    return;
    #undef RTYPE
}

/** 
 *  Subroutine to convert an prototype lowpass filter to a bandpass filter via
 *    the analog polynomial transformation.  The lowpass filter is
 *    described in terms of its poles and zeros (as input to this routine).
 *    The output consists of the parameters for second order sections.
 * 
 * @param p 
 *    Array of Poles
 * @param z 
 *    Array of Zeros
 * @param rtype 
 *    Character array containing root type information
 *      - "SP"  Single real pole
 *      - "CP"  Complex conjugate pole pair
 *      - "CPZ" Complex conjugate pole and zero pairs
 * @param rtype_s 
 *     Length of \p rtype  
 * @param dcvalue 
 *     Zero-frequency value of prototype filter
 * @param nsects 
 *     Number of second-order sections.
 *     On output this subroutine doubles the number of            
 *     sections.                                        
 * @param fl 
 *     Low Frequency cutoff
 * @param fh 
 *     High Frequency cutoff
 * @param sn 
 *     Output Numerator polynomials for second order sections.
 * @param sd 
 *     Output Denominator polynomials for second order sections.
 *
 * \bug This routine defines PI and TWO PI, which are available in math.h
 *
 * @copyright  Copyright 1990  Regents of the University of California
 *
 * @author:  Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *           (415) 423-0617
 *
 * 
 */
void lptbp(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int *nsects, double fl, double fh, float *sn, float *sd)
{

    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))

    int idx, idx_, iptr, n;
    float a, b, pi, scale, twopi;
    complexf ctemp, h, p1, p2, s, z1, z2;

    complexf *const P = &p[0] - 1;
    float *const Sd = &sd[0] - 1;
    float *const Sn = &sn[0] - 1;
    complexf *const Z = &z[0] - 1;

    pi = 3.14159265;
    twopi = 2. * pi;
    a = twopi * twopi * fl * fh;
    b = twopi * (fh - fl);

    n = *nsects;
    *nsects = 0;
    iptr = 1;
    for (idx = 1; idx <= n; idx++)
    {
        idx_ = idx - 1;

        if (memcmp(RTYPE(idx_, 0), "CPZ", 3) == 0)
        {

            ctemp = cmplxsub(cmplxpow(
                                 (cmplxmul(flttocmplx(b, 0.), Z[idx])),
                                 (double)2),
                             flttocmplx(4. * a, 0.));

            ctemp = cmplxsqrt(ctemp);
            z1 = cmplxmul(flttocmplx(0.5, 0.),
                          (cmplxadd(cmplxmul(flttocmplx(b, 0.), Z[idx]),
                                    ctemp)));
            z2 = cmplxmul(flttocmplx(0.5, 0.),
                          (cmplxsub(cmplxmul(flttocmplx(b, 0.), Z[idx]),
                                    ctemp)));
            ctemp = cmplxsub(cmplxpow((cmplxmul(flttocmplx(b, 0.),
                                                P[idx])),
                                      (double)2),
                             flttocmplx(4. * a, 0.));
            ctemp = cmplxsqrt(ctemp);
            p1 = cmplxmul(flttocmplx(0.5, 0.),
                          (cmplxadd(cmplxmul(flttocmplx(b, 0.), P[idx]),
                                    ctemp)));
            p2 = cmplxmul(flttocmplx(0.5, 0.),
                          (cmplxsub(cmplxmul(flttocmplx(b, 0.), P[idx]),
                                    ctemp)));
            Sn[iptr] = cmplxtof(cmplxmul(z1, cmplxcj(z1)));
            Sn[iptr + 1] = -2. * cmplxtof(z1);
            Sn[iptr + 2] = 1.;
            Sd[iptr] = cmplxtof(cmplxmul(p1, cmplxcj(p1)));
            Sd[iptr + 1] = -2. * cmplxtof(p1);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;
            Sn[iptr] = cmplxtof(cmplxmul(z2, cmplxcj(z2)));
            Sn[iptr + 1] = -2. * cmplxtof(z2);
            Sn[iptr + 2] = 1.;
            Sd[iptr] = cmplxtof(cmplxmul(p2, cmplxcj(p2)));
            Sd[iptr + 1] = -2. * cmplxtof(p2);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;

            *nsects = *nsects + 2;
        }
        else if (memcmp(RTYPE(idx_, 0), "CP", 2) == 0)
        {

            ctemp = cmplxsub(cmplxpow(
                                 (cmplxmul(flttocmplx(b, 0.), P[idx])),
                                 (double)2),
                             flttocmplx(4. * a, 0.));
            ctemp = cmplxsqrt(ctemp);
            p1 = cmplxmul(flttocmplx(0.5, 0.),
                          (cmplxadd(cmplxmul(flttocmplx(b, 0.), P[idx]),
                                    ctemp)));
            p2 = cmplxmul(flttocmplx(0.5, 0.),
                          (cmplxsub(cmplxmul(flttocmplx(b, 0.), P[idx]),
                                    ctemp)));
            Sn[iptr] = 0.;
            Sn[iptr + 1] = b;
            Sn[iptr + 2] = 0.;
            Sd[iptr] = cmplxtof(cmplxmul(p1, cmplxcj(p1)));
            Sd[iptr + 1] = -2. * cmplxtof(p1);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;
            Sn[iptr] = 0.;
            Sn[iptr + 1] = b;
            Sn[iptr + 2] = 0.;
            Sd[iptr] = cmplxtof(cmplxmul(p2, cmplxcj(p2)));
            Sd[iptr + 1] = -2. * cmplxtof(p2);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;

            *nsects = *nsects + 2;
        }
        else if (memcmp(RTYPE(idx_, 0), "SP", 2) == 0)
        {

            Sn[iptr] = 0.;
            Sn[iptr + 1] = b;
            Sn[iptr + 2] = 0.;
            Sd[iptr] = a;
            Sd[iptr + 1] = -b * cmplxtof(P[idx]);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;

            *nsects = *nsects + 1;
        }
    }

    /*  Scaling - use the fact that the bandpass filter amplitude 
     *  at sqrt( omega_l 
     *  equals the amplitude of the lowpass prototype at d.c.
     * */
    s = flttocmplx(0., sqrt(a));
    h = flttocmplx(1., 0.);

    iptr = 1;
    for (idx = 1; idx <= *nsects; idx++)
    {
        h = cmplxdiv(cmplxmul(h, (cmplxadd(cmplxmul(
                                               (cmplxadd(cmplxmul(flttocmplx(Sn[iptr + 2], 0.), s),
                                                         flttocmplx(Sn[iptr + 1], 0.))),
                                               s),
                                           flttocmplx(Sn[iptr], 0.)))),
                     (cmplxadd(cmplxmul(
                                   (cmplxadd(cmplxmul(flttocmplx(Sd[iptr + 2], 0.), s),
                                             flttocmplx(Sd[iptr + 1], 0.))),
                                   s),
                               flttocmplx(Sd[iptr], 0.))));
        iptr = iptr + 3;
    }
    scale = cmplxtof(cmplxdiv(flttocmplx(dcvalue, 0.),
                              cmplxsqrt(cmplxmul(flttocmplx(cmplxtof(h), 0.),
                                                 cmplxcj(h)))));

    Sn[1] = Sn[1] * scale;
    Sn[2] = Sn[2] * scale;
    Sn[3] = Sn[3] * scale;

    return;
    #undef RTYPE
}

/** 
 * 
 *  Subroutine to convert a lowpass filter to a band reject filter
 *    via an analog polynomial transformation.  The lowpass filter is
 *    described in terms of its poles and zeros (as input to this routine).
 *    The output consists of the parameters for second order sections.
 * 
 * @param p 
 *    Array of Poles
 * @param z 
 *    Array of Zeros
 * @param rtype 
 *    Character array containing root type information
 *      - "SP"  Single real pole
 *      - "CP"  Complex conjugate pole pair
 *      - "CPZ" Complex conjugate pole and zero pairs
 * @param rtype_s 
 *     Length of \p rtype  
 * @param dcvalue 
 *     Zero-frequency value of prototype filter
 * @param nsects 
 *     Number of second-order sections.
 *     On output this subroutine doubles the number of            
 *     sections.                                        
 * @param fl 
 *     Low Frequency cutoff
 * @param fh 
 *     High Frequency cutoff
 * @param sn 
 *     Output Numerator polynomials for second order sections.
 * @param sd 
 *     Output Denominator polynomials for second order sections.
 *
 * \bug This routine defines PI and TWO PI, which are available in math.h
 *
 * @copyright  Copyright 1990  Regents of the University of California
 *
 * @author:  Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *           (415) 423-0617
 *
 */
void lptbr(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int *nsects, double fl, double fh, float *sn, float *sd)
{

    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))

    int i, i_, iptr, n;
    float a, b, h, pi, scale, twopi;
    complexf cinv, ctemp, p1, p2, z1, z2;

    complexf *const P = &p[0] - 1;
    float *const Sd = &sd[0] - 1;
    float *const Sn = &sn[0] - 1;
    complexf *const Z = &z[0] - 1;

    pi = 3.14159265;
    twopi = 2. * pi;
    a = twopi * twopi * fl * fh;
    b = twopi * (fh - fl);

    n = *nsects;
    *nsects = 0;
    iptr = 1;
    for (i = 1; i <= n; i++)
    {
        i_ = i - 1;

        if (memcmp(RTYPE(i_, 0), "CPZ", 3) == 0)
        {

            cinv = cmplxdiv(flttocmplx(1., 0.), Z[i]);
            ctemp = cmplxsub(cmplxpow((cmplxmul(flttocmplx(b, 0.), cinv)), (double)2),
                             flttocmplx(4. * a, 0.));
            ctemp = cmplxsqrt(ctemp);
            z1 = cmplxmul(flttocmplx(0.5, 0.), (cmplxadd(cmplxmul(flttocmplx(b, 0.), cinv),
                                                         ctemp)));
            z2 = cmplxmul(flttocmplx(0.5, 0.), (cmplxsub(cmplxmul(flttocmplx(b, 0.), cinv),
                                                         ctemp)));
            cinv = cmplxdiv(flttocmplx(1., 0.), P[i]);
            ctemp = cmplxsub(cmplxpow((cmplxmul(flttocmplx(b, 0.), cinv)), (double)2),
                             flttocmplx(4. * a, 0.));
            ctemp = cmplxsqrt(ctemp);
            p1 = cmplxmul(flttocmplx(0.5, 0.), (cmplxadd(cmplxmul(flttocmplx(b, 0.), cinv),
                                                         ctemp)));
            p2 = cmplxmul(flttocmplx(0.5, 0.), (cmplxsub(cmplxmul(flttocmplx(b, 0.), cinv),
                                                         ctemp)));
            Sn[iptr] = cmplxtof(cmplxmul(z1, cmplxcj(z1)));
            Sn[iptr + 1] = -2. * cmplxtof(z1);
            Sn[iptr + 2] = 1.;
            Sd[iptr] = cmplxtof(cmplxmul(p1, cmplxcj(p1)));
            Sd[iptr + 1] = -2. * cmplxtof(p1);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;
            Sn[iptr] = cmplxtof(cmplxmul(z2, cmplxcj(z2)));
            Sn[iptr + 1] = -2. * cmplxtof(z2);
            Sn[iptr + 2] = 1.;
            Sd[iptr] = cmplxtof(cmplxmul(p2, cmplxcj(p2)));
            Sd[iptr + 1] = -2. * cmplxtof(p2);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;

            *nsects = *nsects + 2;
        }
        else if (memcmp(RTYPE(i_, 0), "CP", 2) == 0)
        {

            cinv = cmplxdiv(flttocmplx(1., 0.), P[i]);
            ctemp = cmplxsub(cmplxpow((cmplxmul(flttocmplx(b, 0.), cinv)), (double)2),
                             flttocmplx(4. * a, 0.));
            ctemp = cmplxsqrt(ctemp);
            p1 = cmplxmul(flttocmplx(0.5, 0.), (cmplxadd(cmplxmul(flttocmplx(b, 0.), cinv),
                                                         ctemp)));
            p2 = cmplxmul(flttocmplx(0.5, 0.), (cmplxsub(cmplxmul(flttocmplx(b, 0.), cinv),
                                                         ctemp)));
            Sn[iptr] = a;
            Sn[iptr + 1] = 0.;
            Sn[iptr + 2] = 1.;
            Sd[iptr] = cmplxtof(cmplxmul(p1, cmplxcj(p1)));
            Sd[iptr + 1] = -2. * cmplxtof(p1);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;
            Sn[iptr] = a;
            Sn[iptr + 1] = 0.;
            Sn[iptr + 2] = 1.;
            Sd[iptr] = cmplxtof(cmplxmul(p2, cmplxcj(p2)));
            Sd[iptr + 1] = -2. * cmplxtof(p2);
            Sd[iptr + 2] = 1.;
            iptr = iptr + 3;

            *nsects = *nsects + 2;
        }
        else if (memcmp(RTYPE(i_, 0), "SP", 2) == 0)
        {

            Sn[iptr] = a;
            Sn[iptr + 1] = 0.;
            Sn[iptr + 2] = 1.;
            Sd[iptr] = -a * cmplxtof(P[i]);
            Sd[iptr + 1] = b;
            Sd[iptr + 2] = -cmplxtof(P[i]);
            iptr = iptr + 3;

            *nsects = *nsects + 1;
        }
    }

    /*  Scaling - use the fact that the bandreject filter amplitude at d.c.
	 *            equals the lowpass prototype amplitude at d.c.
	 * */
    h = 1.0;

    iptr = 1;
    for (i = 1; i <= *nsects; i++)
    {
        h = h * Sn[iptr] / Sd[iptr];
        iptr = iptr + 3;
    }
    scale = dcvalue / fabs(h);
    Sn[1] = Sn[1] * scale;
    Sn[2] = Sn[2] * scale;
    Sn[3] = Sn[3] * scale;

    return;
    #undef RTYPE
}

/*
 *  Subroutine to convert a lowpass filter to a highpass filter via
 *    an analog polynomial transformation.  The lowpass filter is
 *    described in terms of its poles and zeroes (as input to this routine).
 *    The output consists of the parameters for second order sections.
 *
 * @param p 
 *    Array of Poles
 * @param z 
 *    Array of Zeros
 * @param rtype 
 *    Character array containing root type information
 *      - "SP"  Single real pole
 *      - "CP"  Complex conjugate pole pair
 *      - "CPZ" Complex conjugate pole and zero pairs
 * @param rtype_s 
 *     Length of \p rtype  
 * @param dcvalue 
 *     Zero-frequency value of prototype filter
 * @param nsects 
 *     Number of second-order sections.
 * @param sn 
 *     Output Numerator polynomials for second order sections.
 * @param sd 
 *     Output Denominator polynomials for second order sections.
 *
 * \bug This routine defines PI and TWO PI, which are available in math.h
 *
 * @copyright  Copyright 1990  Regents of the University of California
 *
 * @author:  Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *           (415) 423-0617
 *
 * 
 */
void lpthp(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int nsects, float *sn, float *sd)
{

    #define RTYPE(I_, J_) (rtype + (I_) * (rtype_s) + (J_))

    int i, i_, iptr;
    float scale;

    complexf *const P = &p[0] - 1;
    float *const Sd = &sd[0] - 1;
    float *const Sn = &sn[0] - 1;
    complexf *const Z = &z[0] - 1;

    iptr = 1;
    for (i = 1; i <= nsects; i++)
    {
        i_ = i - 1;

        if (memcmp(RTYPE(i_, 0), "CPZ", 3) == 0)
        {

            scale = cmplxtof(cmplxmul(P[i], cmplxcj(P[i]))) / cmplxtof(cmplxmul(Z[i],
                                                                                cmplxcj(Z[i])));
            Sn[iptr] = 1. * scale;
            Sn[iptr + 1] = -2. * cmplxtof(Z[i]) * scale;
            Sn[iptr + 2] = cmplxtof(cmplxmul(Z[i], cmplxcj(Z[i]))) * scale;
            Sd[iptr] = 1.;
            Sd[iptr + 1] = -2. * cmplxtof(P[i]);
            Sd[iptr + 2] = cmplxtof(cmplxmul(P[i], cmplxcj(P[i])));
            iptr = iptr + 3;
        }
        else if (memcmp(RTYPE(i_, 0), "CP", 2) == 0)
        {

            scale = cmplxtof(cmplxmul(P[i], cmplxcj(P[i])));
            Sn[iptr] = 0.;
            Sn[iptr + 1] = 0.;
            Sn[iptr + 2] = scale;
            Sd[iptr] = 1.;
            Sd[iptr + 1] = -2. * cmplxtof(P[i]);
            Sd[iptr + 2] = cmplxtof(cmplxmul(P[i], cmplxcj(P[i])));
            iptr = iptr + 3;
        }
        else if (memcmp(RTYPE(i_, 0), "SP", 2) == 0)
        {

            scale = -cmplxtof(P[i]);
            Sn[iptr] = 0.;
            Sn[iptr + 1] = scale;
            Sn[iptr + 2] = 0.;
            Sd[iptr] = 1.;
            Sd[iptr + 1] = -cmplxtof(P[i]);
            Sd[iptr + 2] = 0.;
            iptr = iptr + 3;
        }
    }

    Sn[1] = Sn[1] * dcvalue;
    Sn[2] = Sn[2] * dcvalue;
    Sn[3] = Sn[3] * dcvalue;

    return;
    #undef RTYPE
}

/**
* Applies tangent frequency warping to compensate
*   for bilinear analog -> digital transformation
*
* @param f
*   Original Design Frequency Specification (Hz)
* @param ts
*   Sampling Internal (seconds)
*
* @return
*
* @date 200990  Last Modified:  September 20, 1990
*
* @opyright   1990  Regents of the University of California
*
* @author   Dave Harris
*           Lawrence Livermore National Laboratory
*           L-205
*           P.O. Box 808
*           Livermore, CA  94550
*           USA
*           (415) 423-0617
*
*/
double warp(double f, double ts)
{
	float angle, twopi, warp_v;

	twopi = 2.0 * M_PI;
	angle = twopi*f*ts / 2.;
	warp_v = 2.*tan(angle) / ts;
	warp_v = warp_v / twopi;

	return(warp_v);
}

/**
* Alter the cutoff of a filter.  Assumed that the filter
* is structured as 2nd order sections.  Changes the cutoffs
* of a normalized lowpass or highpass filters through a
* simple polynomial transformation.
*
* @param sn
*   Numerator polynomials for 2nd order sections
* @param sd
*   Denominator polynomials for 2nd order sections
* @param nsects
*   Number of 2nd order sections
* @param f
*   New cutoff frequency
*
* @return Nothing
*
* \copyright 1990  Regents of the University of California
*
* \author   Dave Harris
*           Lawrence Livermore National Laboratory
*           L-205
*           P.O. Box 808
*           Livermore, CA  94550
*           USA
*           (415) 423-0617
*
*/
void cutoffs(float *sn, float *sd, int nsects, double f)
{
	int i, iptr;
	float scale;

	float *const Sd = &sd[0] - 1;
	float *const Sn = &sn[0] - 1;

	scale = 2.*3.14159265*f;

	iptr = 1;
	for (i = 1; i <= nsects; i++) {
		Sn[iptr + 1] = Sn[iptr + 1] / scale;
		Sn[iptr + 2] = Sn[iptr + 2] / (scale*scale);
		Sd[iptr + 1] = Sd[iptr + 1] / scale;
		Sd[iptr + 2] = Sd[iptr + 2] / (scale*scale);
		iptr = iptr + 3;
	}

	return;
}

/** 
 * Transform an analog filter to a digital filter via the bilinear 
 *    transformation.  Assumes both filters are stored as 2nd Order
 *    sections and the transform is done in place.
 * 
 * @param sn 
 *    Array containing numerator polynomial coefficeients. Packed head to
 *      tail and using groups of 3.  Length is 3 * \p nsects
 * @param sd 
 *    Array containing demoninator polynomial coefficeients. Packed head to
 *      tail and using groups of 3.  Length is 3 * \p nsects
 * @param nsects 
 *    Number of 2nd order sections.
 * 
 * @return Nothing
 *
 * @copyright 1990  Regents of the University of California
 *
 * @author  Dave Harris                                                         
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *           (415) 423-0617
 *
 * \note
 *    - scale = a0 + a1 + a2
 *    - 
 *    - a_0 = 1.0
 *    - a_1 = 2.0 * (a_0 - a_2) / scale
 *    - a_2 = (a_2 - a_1 + a_0) / scale
 *    - 
 *    - b_0 = (b_2 + b_1 + b_0) / scale
 *    - b_1 = 2.0 * (b_0 - b_2) / scale
 *    - b_2 = (b_2 - b_1 + b_0) / scale
 *
 * \todo Further Documentation 
 */
void bilin2(float *sn, float *sd, int nsects)
{
    int i, iptr;
    double a0, a1, a2, scale;

    float *const Sd = &sd[0] - 1;
    float *const Sn = &sn[0] - 1;

    iptr = 1;
    for (i = 1; i <= nsects; i++)
    {

        a0 = Sd[iptr];
        a1 = Sd[iptr + 1];
        a2 = Sd[iptr + 2];

        scale = a2 + a1 + a0;
        Sd[iptr] = 1.;
        Sd[iptr + 1] = (2. * (a0 - a2)) / scale;
        Sd[iptr + 2] = (a2 - a1 + a0) / scale;

        a0 = Sn[iptr];
        a1 = Sn[iptr + 1];
        a2 = Sn[iptr + 2];

        Sn[iptr] = (a2 + a1 + a0) / scale;
        Sn[iptr + 1] = (2. * (a0 - a2)) / scale;
        Sn[iptr + 2] = (a2 - a1 + a0) / scale;

        iptr = iptr + 3;
    }

    return;
}

/**
* Copy a string from \p from to \p to.  Spaces are padded at the end
*    of the to string
*
* @param to
*    Where to place the string
* @param tolen
*    Length of \p to
* @param from
*    Where to copy the string from
* @param fromlen
*    Length of \p from
*
* @return
*    The copied string
*
* @bug Should be replaced with strncpy() or equivalent
*
*/
char *fstrncpy(char *to, int tolen, char *from,	int fromlen)
{
	int cpylen;

	if (to == NULL || from == NULL || tolen <= 0 || fromlen <= 0)
		return(NULL);

	cpylen = fromlen;
	if (fromlen > tolen) cpylen = tolen;

	memcpy(to, from, cpylen);
	if (cpylen < tolen)
		memset(to + cpylen, (int)' ', tolen - cpylen);
	to[tolen] = '\0';

	return(to);
}


/** 
 * Apply an IIR (Infinite Impulse Response) Filter to a data 
 *   sequence.  The filter is assumed to be stored as second
 *   order sections.  The filtering is done in place.  Zero-phase
 *   (forward plus reverse filtering) is an option.
 * 
 * @param data 
 *    Array containing data on input and filtered data on output
 * @param nsamps 
 *    Length of array \p data
 * @param zp 
 *    If Zero Phase filtering is requested
 *    - TRUE Zero phase Two pass filtering (forward + reverse filters)
 *    - FALSE Single Pass filtering
 * @param sn 
 *    Numerator polynomials for 2nd Order Sections
 * @param sd 
 *    Denominator polynomials for 2nd Order Sections
 * @param nsects 
 *    Number of 2nd Order Sections
 *
 * @return Nothing
 *
 *  @copyright 1990  Regents of the University of California                      
 *
 *
 *  @author  Dave Harris
 *           Lawrence Livermore National Laboratory
 *           L-205
 *           P.O. Box 808
 *           Livermore, CA  94550
 *           USA
 *
 * @note 
 *    y_n = \Sum_1^\p N (b_0 * x_n + b1 * x_{n-1} + b2 * x_{n-2} - 
 *                                   a1 * y_{n-1} + b2 * y_{n-2} )
 *    where 
 *       - N = \p nsamps
 *       - b = \p sn
 *       - a = \p sd
 *     
 */
int apply(float *data, int nsamps, bool zp, float *sn, float *sd, int nsects)
{
    int i, j, jptr;
    float a1, a2, b0, b1, b2, output, x1, x2, y1, y2;

    float *const Data = &data[0] - 1;
    float *const Sd = &sd[0] - 1;
    float *const Sn = &sn[0] - 1;

    jptr = 1;
    for (j = 1; j <= nsects; j++)
    {
        x1 = 0.0;
        x2 = 0.0;
        y1 = 0.0;
        y2 = 0.0;
        b0 = Sn[jptr];
        b1 = Sn[jptr + 1];
        b2 = Sn[jptr + 2];
        a1 = Sd[jptr + 1];
        a2 = Sd[jptr + 2];
        for (i = 1; i <= nsamps; i++)
        {

            output = b0 * Data[i] + b1 * x1 + b2 * x2;
            output = output - (a1 * y1 + a2 * y2);
            y2 = y1;
            y1 = output;
            x2 = x1;
            x1 = Data[i];
            Data[i] = output;
        }
        jptr = jptr + 3;
    }

    if (zp)
    {
        jptr = 1;
        for (j = 1; j <= nsects; j++)
        {

            x1 = 0.0;
            x2 = 0.0;
            y1 = 0.0;
            y2 = 0.0;
            b0 = Sn[jptr];
            b1 = Sn[jptr + 1];
            b2 = Sn[jptr + 2];
            a1 = Sd[jptr + 1];
            a2 = Sd[jptr + 2];

            for (i = nsamps; i >= 1; i--)
            {
                output = b0 * Data[i] + b1 * x1 + b2 * x2;
                output = output - (a1 * y1 + a2 * y2);
                y2 = y1;
                y1 = output;
                x2 = x1;
                x1 = Data[i];
                Data[i] = output;
            }
            jptr = jptr + 3;
        }
    }

    return 0;
}

/** 
 *  IIR filter design and implementation
 * 
 * @param data 
 *    real array containing sequence to be filtered
 *    original data destroyed, replaced by filtered data
 * @param nsamps 
 *    number of samples in data
 * @param aproto 
 *    int
 *      - 0 : '(BU)tter  ' -- butterworth filter
 *      - 1 : '(BE)ssel  ' -- bessel filter
 *      - 2 : 'C1      ' -- chebyshev type i
 *      - 3 : 'C2      ' -- chebyshev type ii
 * @param trbndw 
 *    transition bandwidth as fraction of lowpass
 *    prototype filter cutoff frequency.  used
 *    only by chebyshev filters.
 * @param a 
 *    attenuation factor.  equals amplitude
 *    reached at stopband edge.  used only by
 *    chebyshev filters.
 * @param iord 
 *    order (#poles) of analog prototype
 *    not to exceed 10 in this configuration.  4 - 5
 *    should be ample.
 * @param type 
 *    int
 *     - 0: 'LP' -- low pass
 *     - 1: 'HP' -- high pass
 *     - 2: 'BP' -- band pass
 *     - 3: 'BR' -- band reject
 * @param flo 
 *    low frequency cutoff of filter (hertz)
 *    ignored if type = 'lp'
 * @param fhi 
 *    high frequency cutoff of filter (hertz)
 *    ignored if type = 'hp'
 * @param ts 
 *    sampling interval (seconds)
 * @param passes 
 *    integer variable containing the number of passes
 *     - 1 -- forward filtering only
 *     - 2 -- forward and reverse (i.e. zero phase) filtering
 *
 * @author:  Dave B. Harris
 *
 * @date 120990 Last Modified:  September 12, 1990
 */
void xapiir(float *data, int nsamps, int aproto,
			double trbndw, double a, int iord, int type,
            double flo, double fhi, double ts, int passes) {
    int nsects;
    float sd[30], sn[30];

    design(iord, type, aproto, a, trbndw, flo, fhi, ts, sn, sd, &nsects);
    apply(data, nsamps, ((passes == 1) ? false : true), sn, sd, nsects);
    return;
}




int moving_average(const float *in, float *out, int size, int wdnsize, int type, bool scale) {
    /*******************************************************
    *   type : useless variable here
    *   scale: true(default) to sacle `out` by 1.0/wdnsize
    *
    *
    *   The algorithm
    *
    *   WDNSIZE = 2M+1
    *
    *   0   1   ...  M-2 M-1  M  M+1 M+2      2M-2     2M     2M+2
    *   x---x---...---x---x---x---x---x---...---x---x---x---x---x---x---...
    *                                             2M-1    2M+1    2M+3
    *
    *   out[i] =  in[i-M] +... + in[i+M]
    *          =  SUM{ [i-M, i+M+1) }
    *
    *   Head:   i= [0, M+1)
    *
    *           out[0] = in[0] +...+ in[M]   = SUM{ [0, M+1) }
    *           out[1] = in[0] +...+ in[M+1] = SUM{ [0, M+2) } = out[0] + in[M+1]
    *           out[2] = in[0] +...+ in[M+2] = SUM{ [0, M+3) } = out[1] + in[M+2]
    *
    *           out[i] = ...                 = ...             = out[i-1] + in[i+M]
    *
    *   Middle: i= [M+1, N-M)
    *
    *           out[i]   = in[i-M] + in[i-(M-1)] + in[i-(M-2)] +...+ in[i+(M-1)] + in[i+M]
    *           out[i+1] =           in[i-(M-1)] + in[i-(M-2)] +...+ in[i+(M-1)] + in[i+M] + in[i+M+1]             = out[i]  - in[i-M]     + in[i+M+1]
    *           out[i+2] =                         in[i-(M-2)] +...                        + in[i+M+1] + in[i+M+2] = out[i+1]- in[i-(M-1)] + in[i+M+2]
    *           ...
    *
    *           out[i+1] = out[i]   - in[i-M] + in[i+M+1]
    *             or
    *           out[i]   = out[i-1] - in[i-M-1] + in[i+M]
    *
    *   Tail:   [N-M, N)
    *           out[N-(M+1)] = in[N-(M+1)-M] +...+ in[N-(M+1)+M]
    *                        = in[N-2M-1] + in[N-2M] + in[N-2M+1] +...+ in[N-1]
    *           out[N-M]     =              in[N-2M] + in[N-2M+1] +...+ in[N-1] = out[N-(M+1)] - in[N-(M+1)-M]
    *           out[N-M+1]   =                         in[N-2M+1] +...+ in[N-1] = out[N-M]     - in[N-2M]
    *           ...
    *
    *           out[i+1]     = out[i] - in[i-M]
    *             or
    *           out[i]       = out[i-1] - in[i-M-1]
    */
    memset(out, 0, size*sizeof(float) );

    int M = wdnsize / 2;
    int N = size;
    // head
    int M_plus_one =M+1;
    double v = 0.0; // to keep the accuracy
    for(int idx=0; idx!=M_plus_one; ++idx) { v += in[idx]; } out[0] = v;                  // out[0] = SUM{ [0, M+1) }
    for(int idx=1; idx!=M_plus_one; ++idx) { v = out[idx-1] + in[idx+M];  out[idx] = v; } // out[i] = out[i-1] + in[i+M]

    // middle
    int N_minus_M=N-M;
    for(int idx=M_plus_one; idx!=N_minus_M; ++idx) {
        v += (in[idx+M] - in[idx-M_plus_one]);
        out[idx] = v;
        //out[idx] = out[idx-1] + (in[idx+M] - in[idx-M_plus_one]); // out[i]   = out[i-1] - in[i-M-1] + in[i+M]
    }

    // Tail
    for(int idx=N_minus_M; idx!=N; ++idx) {
        v -= in[idx-M_plus_one];
        out[idx] = v;
        //out[idx] = out[idx-1] - in[idx-M_plus_one];             // out[i]       = out[i-1] - in[i-M-1]
    }

    // scale
    if (scale) {
        float v = 1.0/(2*M+1);
        for(int idx=0; idx<N; ++idx) { out[idx] *= v; }
    }
    return 0;
}
