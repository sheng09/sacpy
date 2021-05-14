#ifndef __SIGNAL_PROC__
#define __SIGNAL_PROC__

#include <stdbool.h>
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

/************************************************************************
 * Functions for complexf
*************************************************************************/
typedef struct complexf_t
{
    float re; /** Real Part */
    float im; /** Imaginary Part */
} complexf;

double   aimag(complexf fc);
double   cmplxabs(complexf c);
double   cmplxang(complexf c);
double   cmplxtof(complexf c);

complexf cmplxadd(complexf c1, complexf c2);
complexf cmplxcj(complexf c);
complexf cmplxmul(complexf c1, complexf c2);
complexf flttocmplx(double d1, double d2);
complexf cmplxsub(complexf c1, complexf c2);

complexf cmplxsqrt(complexf c);
complexf cmplxdiv(complexf c1, complexf c2);
complexf cmplxlog(complexf c);
complexf cmplxexp(complexf c);
complexf cmplxpow(complexf c, double d);
complexf cmplxneg(complexf c);
double powi(double b, int x);

#define IIR_LP 0
#define IIR_HP 1
#define IIR_BP 2
#define IIR_BR 3

#define IIR_BU 0
#define IIR_BE 1
#define IIR_C1 2
#define IIR_C2 3
/************************************************************************
 * Functions for IIR filters
*************************************************************************/
int design(int iord, int type, int IIR_APROTO, double a, double trbndw,
            double fl, double fh, double ts, float *sn, float *sd, int *nsects);
void buroots(complexf *p, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord);
void beroots(complexf *p, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord);
void chebparm(double a, double trbndw, int iord, float *eps, float *ripple);
void c1roots(complexf *p, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord, double eps);
void c2roots(complexf *p, complexf *z, char *rtype, int rtype_s, float *dcvalue, int *nsects, int iord, double a, double omegar);
void lp(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int nsects, float *sn, float *sd);
void lptbp(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int *nsects, double fl, double fh, float *sn, float *sd);
void lptbr(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int *nsects, double fl, double fh, float *sn, float *sd);
void lpthp(complexf *p, complexf *z, char *rtype, int rtype_s, double dcvalue, int nsects, float *sn, float *sd);

double warp(double f, double ts);
void cutoffs(float *sn, float *sd, int nsects, double f);

void bilin2(float *sn, float *sd, int nsects);

char *fstrncpy(char *to, int tolen, char *from,	int fromlen);

int apply(float *data, int nsamps, bool zp, float *sn, float *sd, int nsects);
void xapiir(float *data, int nsamps, int aproto, double trbndw, double a, int iord, int type, double flo, double fhi, double ts, int passes);


/************************************************************************
 * Functions for LA
*************************************************************************/
int moving_average(const float *in, float *out, int size, int wdnsize, int type, bool scale);
#endif