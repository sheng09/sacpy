#ifndef __SAC_IO__
#define __SAC_IO__

#include <stdbool.h>

typedef struct S_SACHDR
{
    float delta;          /* RF time increment, sec    */
    float depmin;         /*    minimum amplitude      */
    float depmax;         /*    maximum amplitude      */
    float scale;          /*    amplitude scale factor */
    float odelta;         /*    observed time inc      */
    float b;          /* RD initial time - wrt nz* */
    float e;          /* RD end time               */
    float o;          /*    event start            */
    float a;          /*    1st arrival time       */
    float internal1;      /*    internal use           */
    float t0;         /*    user-defined time pick */
    float t1;         /*    user-defined time pick */
    float t2;         /*    user-defined time pick */
    float t3;         /*    user-defined time pick */
    float t4;         /*    user-defined time pick */
    float t5;         /*    user-defined time pick */
    float t6;         /*    user-defined time pick */
    float t7;         /*    user-defined time pick */
    float t8;         /*    user-defined time pick */
    float t9;         /*    user-defined time pick */
    float f;          /*    event end, sec > 0     */
    float resp0;          /*    instrument respnse parm*/
    float resp1;          /*    instrument respnse parm*/
    float resp2;          /*    instrument respnse parm*/
    float resp3;          /*    instrument respnse parm*/
    float resp4;          /*    instrument respnse parm*/
    float resp5;          /*    instrument respnse parm*/
    float resp6;          /*    instrument respnse parm*/
    float resp7;          /*    instrument respnse parm*/
    float resp8;          /*    instrument respnse parm*/
    float resp9;          /*    instrument respnse parm*/
    float stla;           /*  T station latititude     */
    float stlo;           /*  T station longitude      */
    float stel;           /*  T station elevation, m   */
    float stdp;           /*  T station depth, m      */
    float evla;           /*    event latitude         */
    float evlo;           /*    event longitude        */
    float evel;           /*    event elevation        */
    float evdp;           /*    event depth            */
    float mag;            /*    reserved for future use*/
    float user0;          /*    available to user      */
    float user1;          /*    available to user      */
    float user2;          /*    available to user      */
    float user3;          /*    available to user      */
    float user4;          /*    available to user      */
    float user5;          /*    available to user      */
    float user6;          /*    available to user      */
    float user7;          /*    available to user      */
    float user8;          /*    available to user      */
    float user9;          /*    available to user      */
    float dist;           /*    stn-event distance, km */
    float az;         /*    event-stn azimuth      */
    float baz;            /*    stn-event azimuth      */
    float gcarc;          /*    stn-event dist, degrees*/
    float internal2;      /*    internal use           */
    float internal3;      /*    internal use           */
    float depmen;         /*    mean value, amplitude  */
    float cmpaz;          /*  T component azimuth     */
    float cmpinc;         /*  T component inclination */
    float unused2;        /*    reserved for future use*/
    float unused3;        /*    reserved for future use*/
    float unused4;        /*    reserved for future use*/
    float unused5;        /*    reserved for future use*/
    float unused6;        /*    reserved for future use*/
    float unused7;        /*    reserved for future use*/
    float unused8;        /*    reserved for future use*/
    float unused9;        /*    reserved for future use*/
    float unused10;       /*    reserved for future use*/
    float unused11;       /*    reserved for future use*/
    float unused12;       /*    reserved for future use*/
    int   nzyear;         /*  F zero time of file, yr  */
    int   nzjday;         /*  F zero time of file, day */
    int   nzhour;         /*  F zero time of file, hr  */
    int   nzmin;          /*  F zero time of file, min */
    int   nzsec;          /*  F zero time of file, sec */
    int   nzmsec;         /*  F zero time of file, msec*/
    int   nvhdr;      /*  R header version number  */
    int   internal5;      /*    internal use           */
    int   internal6;      /*    internal use           */
    int   npts;           /* RF number of samples      */
    int   internal7;      /*    internal use           */
    int   internal8;      /*    internal use           */
    int   unused13;       /*    reserved for future use*/
    int   unused14;       /*    reserved for future use*/
    int   unused15;       /*    reserved for future use*/
    int   iftype;         /* RA type of file          */
    int   idep;           /*    type of amplitude      */
    int   iztype;         /*    zero time equivalence  */
    int   unused16;       /*    reserved for future use*/
    int   iinst;          /*    recording instrument   */
    int   istreg;         /*    stn geographic region  */
    int   ievreg;         /*    event geographic region*/
    int   ievtyp;         /*    event type             */
    int   iqual;          /*    quality of data        */
    int   isynth;         /*    synthetic data flag    */
    int   unused17;       /*    reserved for future use*/
    int   unused18;       /*    reserved for future use*/
    int   unused19;       /*    reserved for future use*/
    int   unused20;       /*    reserved for future use*/
    int   unused21;       /*    reserved for future use*/
    int   unused22;       /*    reserved for future use*/
    int   unused23;       /*    reserved for future use*/
    int   unused24;       /*    reserved for future use*/
    int   unused25;       /*    reserved for future use*/
    int   unused26;       /*    reserved for future use*/
    int   leven;          /* RA data-evenly-spaced flag*/
    int   lpspol;         /*    station polarity flag  */
    int   lovrok;         /*    overwrite permission   */
    int   lcalda;         /*    calc distance, azimuth */
    int   unused27;       /*    reserved for future use*/
    char  kstnm[8];       /*  F station name           */
    char  kevnm[16];      /*    event name             */
    char  khole[8];       /*    man-made event name    */
    char  ko[8];          /*    event origin time id   */
    char  ka[8];          /*    1st arrival time ident */
    char  kt0[8];         /*    time pick 0 ident      */
    char  kt1[8];         /*    time pick 1 ident      */
    char  kt2[8];         /*    time pick 2 ident      */
    char  kt3[8];         /*    time pick 3 ident      */
    char  kt4[8];         /*    time pick 4 ident      */
    char  kt5[8];         /*    time pick 5 ident      */
    char  kt6[8];         /*    time pick 6 ident      */
    char  kt7[8];         /*    time pick 7 ident      */
    char  kt8[8];         /*    time pick 8 ident      */
    char  kt9[8];         /*    time pick 9 ident      */
    char  kf[8];          /*    end of event ident     */
    char  kuser0[8];      /*    available to user      */
    char  kuser1[8];      /*    available to user      */
    char  kuser2[8];      /*    available to user      */
    char  kcmpnm[8];      /*  F component name         */
    char  knetwk[8];      /*    network name           */
    char  kdatrd[8];      /*    date data read         */
    char  kinst[8];       /*    instrument name        */
} SACHDR;

extern const SACHDR sachdr_null;

extern const  int   IREAL;
extern const  int   ITIME;
extern const  int   IRLIM;
extern const  int   IAMPH;
extern const  int   IXY  ;
extern const  int   IUNKN;
extern const  int  IDISP ;
extern const  int  IVEL  ;
extern const  int  IACC  ;
extern const  int  IB    ;
extern const  int  IDAY  ;
extern const  int  IO    ;
extern const  int  IA    ;
extern const  int  IT0   ;
extern const  int  IT1   ;
extern const  int  IT2   ;
extern const  int  IT3   ;
extern const  int  IT4   ;
extern const  int  IT5   ;
extern const  int  IT6   ;
extern const  int  IT7   ;
extern const  int  IT8   ;
extern const  int  IT9   ;
extern const  int  IRADNV;
extern const  int  ITANNV;
extern const  int  IRADEV;
extern const  int  ITANEV;
extern const  int  INORTH;
extern const  int  IEAST ;
extern const  int  IHORZA;
extern const  int  IDOWN ;
extern const  int  IUP   ;
extern const  int  ILLLBB;
extern const  int  IWWSN1;
extern const  int  IWWSN2;
extern const  int  IHGLP ;
extern const  int  ISRO  ;
extern const  int  INUCL ;
extern const  int  IPREN ;
extern const  int  IPOSTN;
extern const  int  IQUAKE;
extern const  int  IPREQ ;
extern const  int  IPOSTQ;
extern const  int  ICHEM ;
extern const  int  IOTHER;
extern const  int  IGOOD ;
extern const  int  IGLCH ;
extern const  int  IDROP ;
extern const  int  ILOWSN;
extern const  int  IRLDTA;
extern const  int  IVOLTS;
extern const  int  INIV51;
extern const  int  INIV52;
extern const  int  INIV53;
extern const  int  INIV54;
extern const  int  INIV55;
extern const  int  INIV56;
extern const  int  INIV57;
extern const  int  INIV58;
extern const  int  INIV59;
extern const  int  INIV60;
extern const  int   TMARK;
extern const  int   USERN;
extern const  int  HD_SIZE;



//    Read sac header.
//    filename: the input
//    hdr:      the obtained header
//
//    return: 0 for success or -1 for failure.
int   read_sachead(const char *filename, SACHDR *hdr);
//    Read sac file for both the header and the time series
//    filename: the input
//    hdr     : the obtained header
//    scale   : scale the time series by normalize the amplitude by setting the absolute
//              amplitude in `hdr.scale`.
//
//    return: None-NULL for the pointer that point to the memory for the trace.
//            Or NULL for the failure. The failure can be due to:
//            #1. Wrong sac header info or time series size.
//            #2. Nan numbers or purely zeros in the time series.
float* read_sac(const char *filename, SACHDR *hdr, bool scale);
//    Read sac file for both the header and the time series with a cutting window.
//    filename: the input
//    hdr     : the obtained header
//    tmark, t1, t2: the cutting window
//    scale   : scale the time series by normalize the amplitude by setting the absolute
//              amplitude in `hdr.scale`.
//
//    return: None-NULL for the pointer that point to the memory for the trace.
//            Or NULL for the failure. The failure can be due to:
//            #1. Wrong sac header info or time series size.
//            #2. Nan numbers or purely zeros in the time series.
//            #3. invalid cutting window
float* read_sac2(const char *filename, SACHDR *hdr, int tmarker, float t1, float t2, bool scale);


int   write_sac(const char *filename, const SACHDR *hdr, const float *ptr);
int   write_sac2(const char *filename, int npts, float b, float delta, const float *ptr);


SACHDR sachdr_time(float dt, int npts, float b);
int    swap4bytes(char *ptr, int size);

int get_absolute_time_index(float t, float delta, float b);
int get_valid_time_index(float t, float delta, float b, int npts);
int copy_sachdr(SACHDR *hdr1, SACHDR *hdr2);

void free(void *ptr);
#endif
