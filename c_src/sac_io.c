#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "sac_io.h"

const  int   IREAL =  0;       /* undocumented              */
const  int   ITIME =  1;       /* file: time series data    */
const  int   IRLIM =  2;       /* file: real&imag spectrum  */
const  int   IAMPH =  3;       /* file: ampl&phas spectrum  */
const  int   IXY   =  4;       /* file: gen'l x vs y data   */
const  int   IUNKN =  5;       /* x data: unknown type      */
                               /* zero time: unknown        */
                               /* event type: unknown       */
const  int  IDISP  =  6;       /* x data: displacement (nm) */
const  int  IVEL   =  7;       /* x data: velocity (nm/sec) */
const  int  IACC   =  8;       /* x data: accel (cm/sec/sec)*/
const  int  IB     =  9;       /* zero time: start of file  */
const  int  IDAY   = 10;       /* zero time: 0000 of GMT day*/
const  int  IO     = 11;       /* zero time: event origin   */
const  int  IA     = 12;       /* zero time: 1st arrival    */
const  int  IT0    = 13;       /* zero time: user timepick 0*/
const  int  IT1    = 14;       /* zero time: user timepick 1*/
const  int  IT2    = 15;       /* zero time: user timepick 2*/
const  int  IT3    = 16;       /* zero time: user timepick 3*/
const  int  IT4    = 17;       /* zero time: user timepick 4*/
const  int  IT5    = 18;       /* zero time: user timepick 5*/
const  int  IT6    = 19;       /* zero time: user timepick 6*/
const  int  IT7    = 20;       /* zero time: user timepick 7*/
const  int  IT8    = 21;       /* zero time: user timepick 8*/
const  int  IT9    = 22;       /* zero time: user timepick 9*/
const  int  IRADNV = 23;       /* undocumented              */
const  int  ITANNV = 24;       /* undocumented              */
const  int  IRADEV = 25;       /* undocumented              */
const  int  ITANEV = 26;       /* undocumented              */
const  int  INORTH = 27;       /* undocumented              */
const  int  IEAST  = 28;       /* undocumented              */
const  int  IHORZA = 29;       /* undocumented              */
const  int  IDOWN  = 30;       /* undocumented              */
const  int  IUP    = 31;       /* undocumented              */
const  int  ILLLBB = 32;       /* undocumented              */
const  int  IWWSN1 = 33;       /* undocumented              */
const  int  IWWSN2 = 34;       /* undocumented              */
const  int  IHGLP  = 35;       /* undocumented              */
const  int  ISRO   = 36;       /* undocumented              */
const  int  INUCL  = 37;       /* event type: nuclear shot  */
const  int  IPREN  = 38;       /* event type: nuke pre-shot */
const  int  IPOSTN = 39;       /* event type: nuke post-shot*/
const  int  IQUAKE = 40;       /* event type: earthquake    */
const  int  IPREQ  = 41;       /* event type: foreshock     */
const  int  IPOSTQ = 42;       /* event type: aftershock    */
const  int  ICHEM  = 43;       /* event type: chemical expl */
const  int  IOTHER = 44;       /* event type: other source  */
                               /* data quality: other problm*/
const  int  IGOOD  =  45;       /* data quality: good        */
const  int  IGLCH  =  46;       /* data quality: has glitches*/
const  int  IDROP  =  47;       /* data quality: has dropouts*/
const  int  ILOWSN =  48;       /* data quality: low s/n     */
const  int  IRLDTA =  49;       /* data is real data         */
const  int  IVOLTS =  50;       /* file: velocity (volts)    */
const  int  INIV51 =  51;       /* undocumented              */
const  int  INIV52 =  52;       /* undocumented              */
const  int  INIV53 =  53;       /* undocumented              */
const  int  INIV54 =  54;       /* undocumented              */
const  int  INIV55 =  55;       /* undocumented              */
const  int  INIV56 =  56;       /* undocumented              */
const  int  INIV57 =  57;       /* undocumented              */
const  int  INIV58 =  58;       /* undocumented              */
const  int  INIV59 =  59;       /* undocumented              */
const  int  INIV60 =  60;       /* undocumented              */

const  int   TMARK =  10;
const  int   USERN =  40;
/* number of bytes in header that need to be swapped on PC (int+float+long)*/
const  int  HD_SIZE = 440;

const SACHDR sachdr_null = {
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345., -12345., -12345., -12345., -12345.,
  -12345, -12345, -12345, -12345, -12345,
  -12345,      6, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  -12345, -12345, -12345, -12345, -12345,
  { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }, { '-','1','2','3','4','5',' ',' ' },
  { '-','1','2','3','4','5',' ',' ' }
};
int read_sachead(const char *name, SACHDR * hdr)
{
    FILE *fp = NULL;
    if ((fp = fopen(name, "rb")) == NULL)
    {
        fprintf(stderr, "Unable to open %s\n",name);
        return -1;
    }
    if (fread(hdr, sizeof(SACHDR), 1, fp) != 1)
    {
        fprintf(stderr, "Error in reading SAC header %s\n",name);
        fclose(fp);
        return -1;
    }
    if (hdr->nvhdr > 6 || hdr->nvhdr < 0)
        swap4bytes((char *) hdr, HD_SIZE);
    hdr->e = hdr->b + hdr->delta*(hdr->npts-1);
    fclose(fp);
    return 0;
}
float* read_sac(const char *name, SACHDR *hdr, bool scale)
{
    FILE  *fp=NULL;
    if ((fp = fopen(name, "rb")) == NULL)
    {
        fprintf(stderr, "Unable to open %s\n",name);
        return NULL;
    }
    if (fread(hdr, sizeof(SACHDR), 1, fp) != 1)
    {
        fprintf(stderr, "Error in reading SAC header %s\n",name);
        return NULL;
    }
    int swapflag = 0;
    if (hdr->nvhdr > 6 || hdr->nvhdr < 0)
    {
        swap4bytes((char *) hdr, HD_SIZE);
        swapflag = 1;
    }
    hdr->e = hdr->b + (hdr->npts-1)*hdr->delta; // update hdr.e in case of wrongness
    size_t size = hdr->npts;
    float *ptr = NULL;
    if ((ptr = (float *) malloc(size*sizeof(float) ) ) == NULL)
    {
        fprintf(stderr, "Error in allocating memory for reading %s\n",name);
        return NULL;
    }
    if (fread((char *) ptr, size*sizeof(float), 1, fp) != 1)
    {
        fprintf(stderr, "Error in reading SAC data %s\n",name);
        return NULL;
    }
    fclose(fp);
    if (swapflag)
        swap4bytes((char *) ptr, size*sizeof(float) );
    // Check for NAN numbers
    int nan_number = 0;
    for(size_t idx=0; idx<size; ++idx)
    {
        int tag = fpclassify(ptr[idx]);
        if ( tag == FP_NAN || tag == FP_INFINITE)
        {
            nan_number = 1;
            break;
        }
    }
    if (nan_number == 1)
    {
        free(ptr);
        return NULL;
    }
    // Scale and set the hdr.scale
    if (scale)
    {
        float max_v = 0.0;
        for(size_t idx=0; idx<size; ++idx)
        {
            if (ptr[idx] > max_v)
                max_v = ptr[idx];
            else if (-ptr[idx] > max_v )
                max_v = -ptr[idx];
        }

        if ( hdr->scale != -12345.0)
            hdr->scale *= max_v;
        else
            hdr->scale = max_v;

        max_v = 1.0/max_v;
        for(size_t idx=0; idx<size; ++idx)
            ptr[idx] *= max_v;
    }
    return ptr;
}
float* read_sac2(const char *name, SACHDR *hdr, int tmark, float t1, float t2, bool scale)
{
    if (t1>t2)
    {
        fprintf(stderr, "Err. Invalid cutting window in read_sac(...). t1(%f)>t2(%f) %s\n", t1, t2, name);
        return NULL;
    }
    //
    FILE  *fp = NULL;
    if ((fp = fopen(name, "rb")) == NULL)
    {
        fprintf(stderr, "Unable to open %s\n",name);
        return NULL;
    }
    // read sac hdr
    int swapflag = 0;
    if (fread(hdr, sizeof(SACHDR), 1, fp) != 1)
    {
        fprintf(stderr, "Error in reading SAC header %s\n",name);
        return NULL;
    }
    if (hdr->nvhdr > 6 || hdr->nvhdr < 0)
    {
        swap4bytes((char *) hdr, HD_SIZE);
        swapflag = 1;
    }
    hdr->e = hdr->b + (hdr->npts-1)*hdr->delta; // update hdr.e in case of wrongness
    // Obtain time window to read
    float tref = 0.0;
    if (tmark==-5 || tmark==-3 || tmark==-2 || (tmark>=0&&tmark<10) )
    {
        tref = *( (float *) hdr + 10 + tmark);
        if (tref==-12345.)
        {
            fprintf(stderr,"Time mark undefined in %s\n",name);
            return NULL;
        }
    }
    else
    {
        fprintf(stderr,"Wrong Time mark %d\n", tmark);
        return NULL;
    }
    t1 += tref; // t1 can be smaller than b. Zeros will be padded in prefix when necessary.
    t2 += tref; // t2 can be larger  than e. Zeros ...            in appendix ...
    // Check if the cutting window is valid
    if (t1> hdr->e || t2 < hdr->b)
    {
        fprintf(stderr, "Err. Invalid cutting window in read_sac(...). t1(%f)>t2(%f) %s\n", t1, t2, name);
        return NULL;
    }
    //
    size_t idx_t1 = (size_t) floorf((t1-hdr->b)/hdr->delta);     // this point is included
    size_t idx_t2 = (size_t) ceilf( (t2-hdr->b)/hdr->delta) + 1; // this point is not included
    // Compute new hdr values however don't set here.
    size_t  old_npts = hdr->npts;
    size_t  new_npts = idx_t2-idx_t1;
    float new_b    = hdr->b + (idx_t1  ) * hdr->delta;
    float new_e    = hdr->b + (idx_t2-1) * hdr->delta;
    //
    float * ptr = NULL;
    float * rd_ptr= NULL;
    if ( (ptr = (float *) calloc(new_npts, sizeof(float) ) ) ==NULL)
    {
         fprintf(stderr, "Error in allocating memory for reading %s n=%ld\n", name, new_npts);
         return NULL;
    }
    // Set parameter for reading
    int rd_idx_1 = (idx_t1>0)        ? idx_t1 : 0;          // position to start in reading from file
    int rd_idx_2 = (idx_t2<old_npts) ? idx_t2 : old_npts;   // position to stop  in reading from file
    int rd_npts = rd_idx_2- rd_idx_1;                        // size of block to read
    // double-check if the reading window is valid
    if (rd_idx_1> old_npts || rd_idx_2 < 0)
    {
        free(ptr);
        return NULL;
    }
    //printf("rd: %d %d, idx_t12 %d %d, t12 %f %f\n", rd_idx_1, rd_idx_2, idx_t1, idx_t2, t1, t2);
    //
    if (idx_t1>0)
    {
        if (fseek(fp,idx_t1*sizeof(float),SEEK_CUR) < 0)
        {
            fprintf(stderr, "error in seek %s\n",name);
            free(ptr);
            return NULL;
        }
        rd_ptr = ptr;
    }
    else
    {
        rd_ptr = ptr-idx_t1;
    }
    if (fread( rd_ptr, sizeof(float), rd_npts, fp) != rd_npts)
    {
        fprintf(stderr, "Error in reading SAC data %s\n",name);
        free(ptr);
        return NULL;
    }
    fclose(fp);
    if (swapflag)
    {
        swap4bytes((char *) ptr, rd_npts*sizeof(float) );
    }
    // Update hdr
    hdr->b = new_b;
    hdr->e = new_e;
    hdr->npts = new_npts;
    // Check for NAN numbers
    int nan_number = 0;
    for(size_t idx=0; idx<new_npts; ++idx)
    {
        int tag = fpclassify(ptr[idx]);
        if ( tag == FP_NAN || tag == FP_INFINITE)
        {
            nan_number = 1;
            break;
        }
    }
    if (nan_number == 1)
    {
        free(ptr);
        return NULL;
    }
    // Scale and set the hdr.scale
    if (scale)
    {
        float max_v = 0.0;
        for(size_t idx=0; idx<new_npts; ++idx)
        {
            if (ptr[idx] > max_v)
                max_v = ptr[idx];
            else if (-ptr[idx] > max_v )
                max_v = -ptr[idx];
        }

        if ( hdr->scale != -12345.0)
            hdr->scale *= max_v;
        else
            hdr->scale = max_v;

        max_v = 1.0/max_v;
        for(size_t idx=0; idx<new_npts; ++idx)
            ptr[idx] *= max_v;
    }
    return ptr;
}

int write_sac(const char *name, const SACHDR *hdr, const float *ptr)
{
    FILE *fp = NULL;
    unsigned size;
    int error = 0;
    int return_value;
    size = hdr->npts*sizeof(float);
    if (hdr->iftype == IXY) size *= 2;
    if ( !error && (fp = fopen(name, "wb")) == NULL )
    {
        fprintf(stderr,"Error in opening file for writing %s\n",name);
        exit(-1);
    }
    if ( !error && (return_value=fwrite(hdr, sizeof(SACHDR), 1, fp) )!= 1 )
    {
        fprintf(stderr,"Error in writing SAC header for writing %s return: %d memsize: %u\n",name, return_value, size);
        exit(-1);
    }
    if ( !error && (return_value=fwrite(ptr, size, 1, fp) ) != 1 )
    {
        fprintf(stderr,"Error in writing SAC data for writing in write_sac(...) %s return: %d memsize: %u\n",name, return_value, size);
        exit(-1);
    }
    fclose(fp);
    return (error==0) ? 0 : -1;
}
int write_sac2(const char *name, int npts, float b, float delta, const float *ptr)
{
    SACHDR hdr = sachdr_time(delta, npts, b);
    return write_sac(name, &hdr, ptr);
}

SACHDR sachdr_time(float dt, int npts, float b)
{
    SACHDR hdr = sachdr_null;
    hdr.npts = npts;
    hdr.delta = dt;
    hdr.b = b;
    //hdr.o = 0.;
    hdr.e = b+(npts-1)*hdr.delta;
    hdr.iztype = IO;
    hdr.iftype = ITIME;
    hdr.leven = 1;
    return hdr;
}
int swap4bytes(char *ptr, int size)
{
    int i;
    char temp;
    for(i=0;i<size;i+=4)
    {
        temp    = ptr[i+3];
        ptr[i+3] = ptr[i];
        ptr[i]   = temp;
        temp    = ptr[i+2];
        ptr[i+2] = ptr[i+1];
        ptr[i+1] = temp;
    }
    return 0;
}

int get_absolute_time_index(float t, float delta, float b)
{
    return (int)(roundf((t-b)/delta));
}

int get_valid_time_index(float t, float delta, float b, int npts)
{
    int idx = get_absolute_time_index(t, delta, b);
    if (idx < 0)
        idx = 0;
    else if (idx >= npts)
        idx = npts-1;
    return idx;
}

int copy_sachdr(SACHDR *hdr1, SACHDR *hdr2)
{
    memcpy(hdr2, hdr1, sizeof(SACHDR) );
    return 0;
}