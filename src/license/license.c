/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/license/license.c,v $
 | 
 | @last_commit: $Date: 2008/02/14 14:22:49 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

#if 0
static char rcsid[] = "$Id: license.c,v 1.3 2008/02/14 14:22:49 harald Exp $" UNUSED;
#endif

#include "license.h"
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#if !defined(hpux) && !defined(linux) && !defined(_AIX)
# ifdef HAVE_SYS_SYSTEMINFO_H
#  include <sys/systeminfo.h>
# endif
#endif

#if defined(linux)
# ifdef HAVE_SYS_TYPES_H
#  include <sys/types.h>
# endif
# ifdef HAVE_SYS_SOCKET_H
#  include <sys/socket.h>
# endif
# ifdef HAVE_NETINET_IN_H
#  include <netinet/in.h>
# endif
# ifdef HAVE_NETDB_H
#  include <netdb.h>
# endif
#endif

#include "serial_NUMBER.c"

static unsigned int I1, I2, b, Mask1 = 2147483647, Mask2 = 536870911;
static int Q1 = 13, Q2 = 2, S1 = 12, S2 = 17, P1mS1 = 19, P2mS2 = 12, P1mP2 =
  2;
static double Norm = 4.656612873e-10;

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static void srandomine (unsigned int seed1, unsigned int seed2)
{
  I1 = seed1 & Mask1;
  I2 = seed2 & Mask2;
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static float randomine ()
{
  b = ((I1 << Q1) ^ I1) & Mask1;
  I1 = ((I1 << S1) ^ (b >> P1mS1)) & Mask1;
  b = ((I2 << Q2) ^ I2) & Mask2;
  I2 = ((I2 << S2) ^ (b >> P2mS2)) & Mask2;
  return ((float) (I1 ^ (I2 << P1mP2)) * Norm);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static void print_key (unsigned int v, char *key)
{
  sprintf (key, "%s%.8x", key, v);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static FILE *OpenLicenseFile (char *filename)
{
  FILE *fd;

  fd = fopen (filename, "r");
  return (fd);
}

#if defined(DEAD_CODE)
/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static void CloseLicenseFile (FILE * fd)
{

  fclose (fd);
}

#endif

/******************************************************************************
**      Function name : readParaverLicense
**
**      Description :
**  
**        Reads the fields corresponding to the Paraver license from license
**     file. The Paraver license fields could be :
**   
**        - for "Demo" license type:   
**                   Host identificator     : <host_identificator>
**                   Hardware serial number : <serial_number>
**                   License Server Port    : <port_number>
**                   Version                : version.subversion
**                   License type           : Demo
**                   Licenses number        : <number_of_licenses>
**                   Limit date             : <expiration_date>
**                   License Key            : <license_key>
**   
**        - for "Site" license type:   
**                   Host identificator     : <host_identificator>
**                   Hardware serial number : <serial_number>
**                   License Server Port    : <port_number>
**                   Version                : version.subversion
**                   License type           : Site
**                   License Key            : <license_key>
**   
**        - for "Permanent" license type:   
**                   Host identificator     : <host_identificator>
**                   Hardware serial number : <serial_number>
**                   License Server Port    : <port_number>
**                   Version                : version.subversion
**                   License type           : Permanent
**                   Licenses number        : <number_of_licenses>
**                   License Key            : <license_key>
*******************************************************************************/

static void readParaverLicense (FILE * fd, char *key_file,
                                license_fields_t * fields)
{
  char line[MAX_KEY + 100];
  char lic_type[20];

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, HOSTID_LABEL, &(fields->hostid));

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, SERIAL_LABEL, &(fields->serial));

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, PORT_LABEL, &(fields->port));

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, VERSION_LABEL, &(fields->version), &(fields->subversion));

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, TYPE_LABEL, lic_type);
  fields->lic_type = licTypeStringToNumber (lic_type);

  if (fields->lic_type != SITE_TYPE_NUMBER)
  {
    fscanf (fd, "%[^\n]\n", line);
    sscanf (line, NUMBER_LABEL, &(fields->number_of_lic));
  }
  else
  {
    fields->number_of_lic = 0;
  }

  if (fields->lic_type == DEMO_TYPE_NUMBER)
  {
    fscanf (fd, "%[^\n]\n", line);
    sscanf (line, DATE_LABEL, fields->date);
  }

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, KEY_LABEL, key_file);
}

/******************************************************************************
**      Function name : readPackageLicense
**
**      Description :
**  
**        Reads the fields corresponding to the Packages license from license
**     file. The Packages license fields could be :
**   
**        - for "Demo" license type:   
**                   Host identificator     : <host_identificator>
**                   Hardware serial number : <serial_number>
**                   Version                : version.subversion
**                   License type           : Demo
**                   Limit date             : <expiration_date>
**                   License Key            : <license_key>
**   
**        - for "Site" license type:   
**                   Host identificator     : <host_identificator>
**                   Hardware serial number : <serial_number>
**                   Version                : version.subversion
**                   License type           : Site
**                   License Key            : <license_key>
*******************************************************************************/

static void readPackageLicense (FILE * fd, char *key_file,
                                license_fields_t * fields)
{
  char line[MAX_KEY + 100];
  char lic_type[20];

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, HOSTID_LABEL, &fields->hostid);

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, SERIAL_LABEL, &(fields->serial));

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, VERSION_LABEL, &(fields->version), &(fields->subversion));

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, TYPE_LABEL, lic_type);
  fields->lic_type = licTypeStringToNumber (lic_type);

  if (fields->lic_type == DEMO_TYPE_NUMBER)
  {
    fscanf (fd, "%[^\n]\n", line);
    sscanf (line, DATE_LABEL, fields->date);
  }

  fscanf (fd, "%[^\n]\n", line);
  sscanf (line, KEY_LABEL, key_file);
}

static void readDimemasLicense (FILE * fd)
{
  char line[MAX_KEY + 100];

  fscanf (fd, "%[^\n]\n", line);
  fscanf (fd, "%[^\n]\n", line);
  fscanf (fd, "%[^\n]\n", line);
  fscanf (fd, "%[^\n]\n", line);
  fscanf (fd, "%[^\n]\n", line);
  fscanf (fd, "%[^\n]\n", line);
}


static int readLicense (FILE * fd, char *key_file, license_fields_t * fields)
{
  char line[MAX_KEY + 100];

#if defined(DEBUG_LICENSE)
  fprintf (stderr, "readLicense called\n");
#endif
  while (!feof (fd))
  {
    fscanf (fd, "%[^\n]\n", line);
    if (strlen (line) == 0)
    {
      if (!feof (fd))
        line[50] = fgetc (fd);
    }
    else
      break;
  }
  if (feof (fd))
    return LIC_EOFERROR;

#if defined(DEBUG_LICENSE)
  fprintf (stderr, "readLicense: License Header %s\n", line);
#endif
  if (strcmp (line, PARAVER_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = PARAVER_LIC_NUM;
#if defined(DEBUG_LICENSE)
    fprintf (stderr, "readLicense: Is a Paraver license\n");
#endif
    readParaverLicense (fd, key_file, fields);
  }
  else if (strcmp (line, SCPUS_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = SCPUS_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, INFOPERFEX_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = INFOPERFEX_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, OMPTRACE_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = OMPTRACE_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, MPITRACE_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = MPITRACE_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, MPIDTRACE_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = MPIDTRACE_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, OMPITRACE_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = OMPITRACE_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, JIS_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = JIS_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, JOMP_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = JOMP_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, JVMPI_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = JVMPI_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, UTE2PAR_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = UTE2PAR_LIC_NUM;
    readPackageLicense (fd, key_file, fields);
  }
  else if (strcmp (line, DIMEMAS_LICENSE_PROGRAM) == 0)
  {
    fields->license_pack = DIMEMAS_LIC_NUM;

    /*
     * Jump the Dimemas license 
     */
    readDimemasLicense (fd);
    return (LIC_UNKNOWNLIC);
  }
  else
  {
    return (LIC_UNKNOWNLIC);
  }

  return (LIC_NOERROR);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

int licTypeStringToNumber (char *licType)
{
  if (strcmp (licType, DEMO_TYPE_LABEL) == 0)
    return DEMO_TYPE_NUMBER;
  else if (strcmp (licType, PERMANENT_TYPE_LABEL) == 0)
    return PERMANENT_TYPE_NUMBER;
  else if (strcmp (licType, SITE_TYPE_LABEL) == 0)
    return SITE_TYPE_NUMBER;
  return (-1);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

char *licTypeNumberToString (int licType)
{
  if (licType == DEMO_TYPE_NUMBER)
    return DEMO_TYPE_LABEL;
  else if (licType == PERMANENT_TYPE_NUMBER)
    return PERMANENT_TYPE_LABEL;
  else if (licType == SITE_TYPE_NUMBER)
    return SITE_TYPE_LABEL;
  return (NULL);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static void licenseGenerateSmallKey (int par1, int par2, int iters,
                                     char *new_key)
{
  int limit, i;

  limit = iters * randomine ();
  for (i = 0; i < limit; i++)
    randomine ();
  print_key (I1, new_key);
  print_key (I2, new_key);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static void computeParaverKey (license_fields_t * params, char *new_key)
{

  unsigned int limit, date;
  int dd, yy, mm;

  /* HSG sprintf (new_key, ""); */
	new_key[0] = (char) 0;

  if (params->lic_type == DEMO_TYPE_NUMBER)
  {
    if (((sscanf (params->date, "%d/%d/%d", &(dd), &(mm), &(yy)) != 3) ||
         (dd < 1) || (dd > 31) ||
         (mm < 1) || (mm > 12) || (yy < 1999) || (yy > 2025)))
    {
      fprintf (stderr, "Invalid format for date expression.(%d/%d/%d\n",
               dd, mm, yy);
      exit (1);
    }
    date = (yy - 1999) * 500 + mm * 40 + dd;

    srandomine (date, params->hostid);
  }
  else
  {
    srandomine (params->serial, params->hostid);
  }

  licenseGenerateSmallKey (I1, params->version + params->hostid, BUC1,
                           new_key);
  licenseGenerateSmallKey (I2, params->serial + I2, BUC2, new_key);

  licenseGenerateSmallKey (I2, params->hostid + I1, BUC2, new_key);

  limit = (BUC6 / 4) + params->port * 4 + params->port;
  licenseGenerateSmallKey (I1, params->port + I2, limit, new_key);

  limit = BUC2 + params->port * 8 + params->port * 3;
  licenseGenerateSmallKey (params->hostid + I1, params->port + I1, limit,
                           new_key);

  if (params->lic_type != SITE_TYPE_NUMBER)
  {
    limit = BUC6 * params->number_of_lic * 3;
    licenseGenerateSmallKey (I1, params->number_of_lic, limit, new_key);
  }
  else
  {
    limit = BUC6 * params->version * 3;
    licenseGenerateSmallKey (I1, I2 + params->version, limit, new_key);
  }

  limit = (unsigned int) (BUC3 * params->version * 3);
  licenseGenerateSmallKey (I1, params->version, limit, new_key);

  limit = (unsigned int) (BUC4 * params->subversion * params->version * 5);
  licenseGenerateSmallKey (I2, params->subversion, limit, new_key);

  licenseGenerateSmallKey (I1, params->lic_type, BUC5 + params->lic_type / 10,
                           new_key);

  licenseGenerateSmallKey (params->serial, I1, BUC1, new_key);
  licenseGenerateSmallKey (params->hostid, I2, BUC2, new_key);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static void computePackageKey (license_fields_t * params, char *new_key)
{

  unsigned int limit, date;
  int dd, yy, mm;

  /* HSG sprintf (new_key, ""); */
	new_key[0] = (char) 0;

  if (params->lic_type == DEMO_TYPE_NUMBER)
  {
    if (((sscanf (params->date, "%d/%d/%d", &(dd), &(mm), &(yy)) != 3) ||
         (dd < 1) || (dd > 31) ||
         (mm < 1) || (mm > 12) || (yy < 1999) || (yy > 2025)))
    {
      fprintf (stderr, "Invalid format for date expression.(%d/%d/%d\n",
               dd, mm, yy);
      exit (1);
    }
    date = (yy - 1999) * 500 + mm * 40 + dd;

    srandomine (date, params->hostid);
  }
  else
  {
    srandomine (params->serial, params->hostid);
  }

  licenseGenerateSmallKey (I1, params->version + params->hostid, BUC1,
                           new_key);
  licenseGenerateSmallKey (I2, params->serial + I2, BUC2, new_key);

  limit = (unsigned int) (BUC3 * params->version * 3);
  licenseGenerateSmallKey (I1, params->version, limit, new_key);

  limit = (unsigned int) (BUC4 * params->subversion * params->version * 5);
  licenseGenerateSmallKey (I2, params->subversion, limit, new_key);

  licenseGenerateSmallKey (I1, params->lic_type, BUC5 + params->lic_type / 10,
                           new_key);

  licenseGenerateSmallKey (params->serial, I1, BUC1, new_key);
  licenseGenerateSmallKey (params->hostid, I2, BUC6, new_key);
}


/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static void ComputeLicense (license_fields_t * fields, char *computedKey)
{
  if (fields->license_pack == PARAVER_LIC_NUM)
  {
    computeParaverKey (fields, computedKey);
  }
  else
  {
    computePackageKey (fields, computedKey);
  }
}

/******************************************************************************
**      Function name : 
**
**      Description : 
=*******************************************************************************/

static int runningOnCorrectHost (license_fields_t * fields)
{
  unsigned int machine;
  unsigned long serialid;

#if defined(linux)
  struct hostent *hostInfo;
  char hostName[512];
#endif

#if defined(sgi)
#define BUFSIZE_LIC 1024
  char buf[BUFSIZE_LIC];
#endif

#if defined(linux)
  gethostname (hostName, sizeof (hostName));
  hostInfo = gethostbyname (hostName);
  memcpy (&(machine), hostInfo->h_addr, hostInfo->h_length);

#if defined(DEBUG_LICENSE)
  fprintf (stderr, "Machine: hostid %x serial %lx\n", machine, serialid);
#endif
/*
        machine = gethostid();
        serialid = gethostid();
*/
  serialid = get_hardware_serial_number_Linux ();
#elif defined(hpux)
  machine = gethostid ();
  serialid = get_hardware_serial_number_HPUX ();
#elif defined(sun)
  machine = gethostid ();
  serialid = get_hardware_serial_number_SUN ();
#elif defined(sgi)
  machine = gethostid ();
  (void) sysinfo (SI_HW_SERIAL, buf, BUFSIZE_LIC);
  serialid = strtoul (buf, (char **) 0, 10);
#elif defined(__alpha)
  machine = gethostid ();
  serialid = get_hardware_serial_number_ALPHA ();
#elif defined(_AIX)
  machine = gethostid ();
  serialid = get_hardware_serial_number_IBM ();
#else
#     error 'Machine not defined'
#endif

#if defined(DEBUG_LICENSE)
  fprintf (stderr, "runningOnCorrectHost: MACHINE hostid %x serialid %lx\n",
           machine, serialid);
  fprintf (stderr, "runningOnCorrectHost: FILE hostid %x serialid %x\n",
           fields->hostid, fields->serial);
#endif
  if (machine != fields->hostid || serialid != (unsigned long) fields->serial)
    return LIC_ERR;

  return (LIC_OK);
}

static char *Print_NeededLicenseParameters (void)
{
  unsigned int machine;
  unsigned long serialid;

#if defined(linux)
  struct hostent *hostInfo;
  char hostName[512];
#endif
  char *Message;

#if defined(sgi)
#define BUFSIZE_LIC 1024
  char buf[BUFSIZE_LIC];
#endif

#if defined(linux)
  gethostname (hostName, sizeof (hostName));
  hostInfo = gethostbyname (hostName);
  memcpy (&(machine), hostInfo->h_addr, hostInfo->h_length);
  serialid = get_hardware_serial_number_Linux ();
#elif defined(hpux)
  machine = gethostid ();
  serialid = get_hardware_serial_number_HPUX ();
#elif defined(sun)
  machine = gethostid ();
  serialid = get_hardware_serial_number_SUN ();
#elif defined(sgi)
  machine = gethostid ();
  (void) sysinfo (SI_HW_SERIAL, buf, BUFSIZE_LIC);
  serialid = strtoul (buf, (char **) 0, 10);
#elif defined(__alpha)
  machine = gethostid ();
  serialid = get_hardware_serial_number_ALPHA ();
#elif defined(_AIX)
  machine = gethostid ();
  serialid = get_hardware_serial_number_IBM ();
#else
#     error 'Machine not defined'
#endif

  Message = (char *) malloc (sizeof (char) * 1024);
  sprintf (Message,
           "License not found. Needed a license for: HOSTID 0x%x SERIALID 0x%lx Version %d.%d\n",
           machine, serialid, TOOL_VERSION, TOOL_SUB_VERSION);

  return (Message);
}

#if defined(DEAD_CODE)
/******************************************************************************
**      Function name : 
**
**      Description : 
=*******************************************************************************/

static int runningNoServerHost (license_fields_t * fields)
{
  unsigned long serialid;

#if defined(sgi)
#define BUFSIZE_LIC 1024
  char buf[BUFSIZE_LIC];
#endif

#if defined(linux)

  serialid = get_hardware_serial_number_Linux ();
#elif defined(hpux)
  serialid = get_hardware_serial_number_HPUX ();
#elif defined(sun)
  serialid = get_hardware_serial_number_SUN ();
#elif defined(sgi)
  (void) sysinfo (SI_HW_SERIAL, buf, BUFSIZE_LIC);
  serialid = strtoul (buf, (char **) 0, 10);
#elif defined(__alpha)
  serialid = get_hardware_serial_number_ALPHA ();
#elif defined(_AIX)
  serialid = get_hardware_serial_number_IBM ();
#else
#     error 'Machine not defined'
#endif

  if (serialid != (unsigned long) fields->serial)
    return LIC_ERR;

  return (LIC_OK);
}

#endif

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static int dateNotExpired (license_fields_t * fields)
{
  int dd, yy, mm;
  int dd_today, yy_today, mm_today;
  time_t h;
  char today[20];

#if defined(DEBUG_LICENSE)
  fprintf (stderr, "dateNotExpired entry\n");
#endif
  if (fields->lic_type == DEMO_TYPE_NUMBER)
  {
    time (&h);
    strftime (today, 20, "%d/%m/%Y", localtime (&h));

    sscanf (fields->date, "%d/%d/%d", &(dd), &(mm), &(yy));
    sscanf (today, "%d/%d/%d", &(dd_today), &(mm_today), &(yy_today));

    if (yy_today > yy)
      return LIC_ERR;
    else if (yy_today < yy)
      return LIC_OK;

    if (mm_today > mm)
      return LIC_ERR;
    else if (mm_today < mm)
      return LIC_OK;

    if (dd_today > dd)
      return LIC_ERR;
    else if (dd_today < dd)
      return LIC_OK;
  }
#if defined(DEBUG_LICENSE)
  fprintf (stderr, "dateNotExpired: date OK\n");
#endif
  return LIC_OK;
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static int VerifyLicenseForTool (license_fields_t * fields)
{
#if defined(SCPUS_LICENSE)
  int tool = SCPUS_LIC_NUM;
#elif defined(INFOPERFEX_LICENSE)
  int tool = INFOPERFEX_LIC_NUM;
#elif defined(OMPTRACE_LICENSE)
  int tool = OMPTRACE_LIC_NUM;
#elif defined(MPITRACE_LICENSE)
  int tool = MPITRACE_LIC_NUM;
#elif defined(MPIDTRACE_LICENSE)
  int tool = MPIDTRACE_LIC_NUM;
#elif defined(OMPITRACE_LICENSE)
  int tool = OMPITRACE_LIC_NUM;
#elif defined(PARAVER_LICENSE)
  int tool = PARAVER_LIC_NUM;
#elif defined(UTE2PARAVER_LICENSE)
  int tool = UTE2PAR_LIC_NUM;
#elif defined(JIS_LICENSE)
  int tool = JIS_LIC_NUM;
#elif defined(JOMP_LICENSE)
  int tool = JOMP_LIC_NUM;
#elif defined(JVMPI_LICENSE)
  int tool = JVMPI_LIC_NUM;
#else
#    error 'Tool not defined'
#endif
#if defined(DEBUG_LICENSE)
  fprintf (stderr, "fields->license_pack %d tool %d\n", fields->license_pack,
           tool);
#endif

  return (fields->license_pack == tool);

}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static int correctToolVersion (license_fields_t * fields)
{
  if ((fields->version != TOOL_VERSION) ||
      (fields->subversion != TOOL_SUB_VERSION))
  {
    return LIC_ERR;
  }
  return LIC_OK;
}

#if defined(DEAD_CODE)
/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

static char *checkParaverLicense (int *jobs, int *port_server,
                                  int *hostid_file, int *serialid_file,
                                  int *licType)
{
  char *paraver_home;
  char license_file[MAX_LICENSE_PATH];
  license_fields_t fields;
  char fileKey[MAX_KEY], computedKey[MAX_KEY];
  FILE *fd;
  int LicenseForTool, error;

  if ((paraver_home = getenv (ENVIRONMENT_HOME)) == NULL)
  {
    return (ERROR_SETUP_PARAVER_HOME);
  }

  strcpy (license_file, paraver_home);
  strcat (license_file, LICENSE_FILE_NAME);

#if defined(DEBUG_LICENSE)
  fprintf (stderr, "Openning license file: %s\n", license_file);
#endif
  fd = OpenLicenseFile (license_file);
  if (fd == NULL)
  {
    return (ERROR_FILE_NOT_FOUND);
  }
#if defined(DEBUG_LICENSE)
  fprintf (stderr, "License File openned\n");
#endif
  fields.license_correct = 0;
  LicenseForTool = 0;

  error = LIC_NOERROR;

  /*
   * While not found a license for the tool read the file.
   * LicenseForTool values are  :
   * * 0 / LIC_NFND while no license is found.
   * * 1 / LIC_ERR  an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   * * 2/ LIC_FND   an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   */
  while (!LicenseForTool)
  {
    error = readLicense (fd, fileKey, &fields);
    if (error == LIC_EOFERROR)
    {
      LicenseForTool = LIC_ERR;
      continue;
    }
    if (error == LIC_UNKNOWNLIC)
      continue;

    /*
     * Verify if license read is for the correct tool.
     * If not, try to read another license in the file license. 
     */
#if defined(DEBUG_LICENSE)
    fprintf (stderr, "VerifyLicenseForTool: Going to verify license tool\n");
#endif
    if (!VerifyLicenseForTool (&(fields)))
      continue;
#if defined(DEBUG_LICENSE)
    fprintf (stderr,
             "VerifyLicenseForTool: License is for Paraver license\n");
#endif

    /*
     * Compute the license key to verify that file hasn't been
     * modified. The new generated key must be equal than key file. 
     */
    ComputeLicense (&(fields), computedKey);

#define VerifyLicenseModification(a,b) ( (strcmp((a),(b)) == 0) ? 1 : 0)
    if (VerifyLicenseModification (fileKey, computedKey))
    {
#if defined(DEBUG_LICENSE)
      fprintf (stderr, "VerifyLicenseModification: OK\n");
#endif

      /*
       * Verify if tool is running in licensed host.
       * If isn't the correct host, check for the next license. 
       */
      error = runningOnCorrectHost (&(fields));
      if (error)
        continue;
#if defined(DEBUG_LICENSE)
      fprintf (stderr, "runningOnCorrectHost: OK\n");
#endif

      /*
       * Verify if license is for current version.
       * If not, check for a license for the tool version. 
       */
      error = correctToolVersion (&(fields));
      if (error)
        continue;
#if defined(DEBUG_LICENSE)
      fprintf (stderr, "correctToolVersion: OK\n");
#endif

      /*
       * Verify if license hasn,t expired (Demo version).
       * If has expired, check for a license which hasn't expired.
       * 
       * Also, fills the field : fields->license_correct to
       * not only verify the loop condition. 
       */
      error = dateNotExpired (&(fields));
      if (error)
        continue;

      /*
       * A correct license has been found. 
       */
      fields.license_correct = 0xFFFF;
      LicenseForTool = LIC_FND;

      /*
       * Fill the requested parameters. 
       */
      *jobs = fields.number_of_lic;
      *port_server = fields.port;
      *hostid_file = fields.hostid;
      *serialid_file = fields.serial;
      *licType = fields.lic_type;
    }
    else
    {
      /*
       * If genetared license is different than file license for
       * the application --> the file has been modified ERROR. 
       */
#if defined(DEBUG_LICENSE)
      fprintf (stderr, "VerifyLicenseModification: KO\n");
#endif
      error = LIC_BADLICENSE;
      LicenseForTool = LIC_ERR;
      continue;
    }
  }

  if (LicenseForTool == LIC_FND)
  {
    if (fields.license_correct == 0xFFFF)
      return NULL;

    return (Print_NeededLicenseParameters ());
  }

  if (error == LIC_BADLICENSE)
    return (ERROR_INCORRECT_KEY);

  /*
   * Seleccionar el ERROR que sortira. 
   */
  return (Print_NeededLicenseParameters ());
}

#endif /* DEAD_CODE */

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

int verify_execution ()
{
  char *paraver_home;
  char license_file[MAX_LICENSE_PATH];
  license_fields_t fields;
  char fileKey[MAX_KEY], computedKey[MAX_KEY];
  FILE *fd;
  int LicenseForTool, error;

  if ((paraver_home = getenv (ENVIRONMENT_HOME)) == NULL)
  {
    fprintf (stderr, "Environment variable %s must be defined!!!\n",
             ENVIRONMENT_HOME);
    return (0);
  }
  if (strlen (paraver_home) > MAX_LICENSE_PATH)
  {
    fprintf (stderr, "Environment variable %s defined too long!!!\n",
             ENVIRONMENT_HOME);
    return (0);
  }

  strcpy (license_file, paraver_home);
  strcat (license_file, LICENSE_FILE_NAME);

  fd = OpenLicenseFile (license_file);
  if (fd == NULL)
  {
    fprintf (stderr, ERROR_PACKFILE_NOT_FOUND, ENVIRONMENT_HOME);
    fprintf (stderr, "%s = %s\n", ENVIRONMENT_HOME, paraver_home);
    fprintf (stderr, "License file used: %s\n", license_file);
    fprintf (stderr, "%s", Print_NeededLicenseParameters ());
    return (0);
  }

  LicenseForTool = 0;

  error = LIC_NOERROR;

  /*
   * While not found a license for the tool read the file.
   * LicenseForTool values are  :
   * * 0 / LIC_NFND while no license is found.
   * * 1 / LIC_ERR  an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   * * 2/ LIC_FND   an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   */
  while (!LicenseForTool)
  {
    error = readLicense (fd, fileKey, &(fields));
    if (error == LIC_EOFERROR)
    {
      LicenseForTool = LIC_ERR;
      continue;
    }
    if (error == LIC_UNKNOWNLIC)
      continue;
    /*
     * Verify if license read is for the correct tool.
     * If not, try to read another license in the file license. 
     */
    if (!VerifyLicenseForTool (&(fields)))
      continue;

    /*
     * Compute the license key to verify that file hasn't been
     * modified. The new generated key must be equal than key file. 
     */
    ComputeLicense (&(fields), computedKey);

#define VerifyLicenseModification(a,b) ( (strcmp((a),(b)) == 0) ? 1 : 0)

#if defined(DEBUG_LICENSE)
    fprintf (stderr, "strcmp: fileKey %s -- computedKey %s (%d)\n", fileKey,
             computedKey, VerifyLicenseModification (fileKey, computedKey));
#endif
    if (VerifyLicenseModification (fileKey, computedKey))
    {

      /*
       * Verify if tool is running in licensed host.
       * If isn't the correct host, check for the next license. 
       */
      error = runningOnCorrectHost (&(fields));
      if (error)
        continue;

      /*
       * Verify if license is for current version.
       * If not, check for a license for the tool version. 
       */
      error = correctToolVersion (&(fields));
      if (error)
        continue;

      /*
       * Verify if license hasn,t expired (Demo version).
       * If has expired, check for a license which hasn't expired.
       * 
       * Also, fills the field : fields->license_correct to
       * not only verify the loop condition. 
       */
      error = dateNotExpired (&(fields));
      if (error)
        continue;

      /*
       * A correct license has been found. 
       */
      fields.license_correct = 0xFFFF;
      LicenseForTool = LIC_FND;
    }
    else
    {
      /*
       * If genetared license is different than file license for
       * the application --> the file has been modified ERROR. 
       */
      error = LIC_BADLICENSE;
      LicenseForTool = LIC_ERR;
      continue;
    }
  }

  if (LicenseForTool == LIC_FND)
  {
    if (fields.license_correct == 0xFFFF)
      return (1);
  }

  switch (error)
  {
    case LIC_BADLICENSE:
      fprintf (stderr, ERROR_INCORRECT_KEY);
  }

  /*
   * Seleccionar el ERROR que sortira. 
   */
  fprintf (stderr, "%s = %s\n", ENVIRONMENT_HOME, paraver_home);
  fprintf (stderr, "License file used: %s\n", license_file);
  fprintf (stderr, "%s", Print_NeededLicenseParameters ());
  return (0);
}

/******************************************************************************
**      Function name : 
**
**      Description : 
**      Input : 
**      Ouput : 
*******************************************************************************/

#if defined(DEAD_CODE)

#define KARNAK_HOSTID     0x93532a1c
#define KARNAK_SERIALID   0xB0013457

static int runningOnKarnakHost (license_fields_t * fields)
{
  if (KARNAK_HOSTID != fields->hostid ||
      KARNAK_SERIALID != (unsigned long) fields->serial)
    return LIC_ERR;

  return (LIC_OK);
}

static void checkAllMachinesLicense ()
{
  char *paraver_home;
  char license_file[MAX_LICENSE_PATH];
  license_fields_t fields;
  char fileKey[MAX_KEY], computedKey[MAX_KEY];
  FILE *fd;
  int LicenseForTool, error;

  if ((paraver_home = getenv (ENVIRONMENT_HOME)) == NULL)
  {
    fprintf (stderr, "Environment variable %s must be defined!!!\n",
             ENVIRONMENT_HOME);
    exit (1);
  }
  if (strlen (paraver_home) > MAX_LICENSE_PATH)
  {
    fprintf (stderr, "Environment variable %s defined too long!!!\n",
             ENVIRONMENT_HOME);
    exit (1);
  }

  strcpy (license_file, paraver_home);
  strcat (license_file, LICENSE_FILE_NAME);

  fd = OpenLicenseFile (license_file);
  if (fd == NULL)
  {
    fprintf (stderr, ERROR_PACKFILE_NOT_FOUND, ENVIRONMENT_HOME);
    exit (1);
  }

  LicenseForTool = 0;

  error = LIC_NOERROR;

  /*
   * While not found a license for the tool read the file.
   * LicenseForTool values are  :
   * * 0 / LIC_NFND while no license is found.
   * * 1 / LIC_ERR  an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   * * 2/ LIC_FND   an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   */
  while (!LicenseForTool)
  {
    error = readLicense (fd, fileKey, &(fields));
    if (error == LIC_EOFERROR)
    {
      LicenseForTool = LIC_ERR;
      continue;
    }
    if (error == LIC_UNKNOWNLIC)
      continue;
    /*
     * Verify if license read is for the correct tool.
     * If not, try to read another license in the file license. 
     */
    if (!VerifyLicenseForTool (&(fields)))
      continue;

    /*
     * Compute the license key to verify that file hasn't been
     * modified. The new generated key must be equal than key file. 
     */
    ComputeLicense (&(fields), computedKey);

#define VerifyLicenseModification(a,b) ( (strcmp((a),(b)) == 0) ? 1 : 0)
    if (VerifyLicenseModification (fileKey, computedKey))
    {

      /*
       * Verify if tool is running in licensed host.
       * If isn't the correct host, check for the next license. 
       */
      error = runningOnKarnakHost (&(fields));
      if (error)
        continue;

      /*
       * Verify if license is for current version.
       * If not, check for a license for the tool version. 
       */
      error = correctToolVersion (&(fields));
      if (error)
        continue;

      /*
       * Verify if license hasn,t expired (Demo version).
       * If has expired, check for a license which hasn't expired.
       * 
       * Also, fills the field : fields->license_correct to
       * not only verify the loop condition. 
       */
      error = dateNotExpired (&(fields));
      if (error)
        continue;

      /*
       * A correct license has been found. 
       */
      fields.license_correct = 0xFFFF;
      LicenseForTool = LIC_FND;
    }
    else
    {
      /*
       * If genetared license is different than file license for
       * the application --> the file has been modified ERROR. 
       */
      error = LIC_BADLICENSE;
      LicenseForTool = LIC_ERR;
      continue;
    }
  }

  if (LicenseForTool == LIC_FND)
  {
    if (fields.license_correct == 0xFFFF)
      return;
    fprintf (stderr, "%s", Print_NeededLicenseParameters ());
    exit (1);
  }

  switch (error)
  {
    case LIC_BADLICENSE:
      fprintf (stderr, ERROR_INCORRECT_KEY);
      exit (1);
  }

  /*
   * Seleccionar el ERROR que sortira. 
   */
  fprintf (stderr, "%s", Print_NeededLicenseParameters ());
  exit (1);
}
#endif

#if defined(DEAD_CODE)
static void checkNoServerLicense ()
{
  char *paraver_home;
  char license_file[MAX_LICENSE_PATH];
  license_fields_t fields;
  char fileKey[MAX_KEY], computedKey[MAX_KEY];
  FILE *fd;
  int LicenseForTool, error;

  if ((paraver_home = getenv (ENVIRONMENT_HOME)) == NULL)
  {
    fprintf (stderr, "Environment variable %s must be defined!!!\n",
             ENVIRONMENT_HOME);
    exit (1);
  }
  if (strlen (paraver_home) > MAX_LICENSE_PATH)
  {
    fprintf (stderr, "Environment variable %s defined too long!!!\n",
             ENVIRONMENT_HOME);
    exit (1);
  }

  strcpy (license_file, paraver_home);
  strcat (license_file, LICENSE_FILE_NAME);

  fd = OpenLicenseFile (license_file);
  if (fd == NULL)
  {
    fprintf (stderr, ERROR_PACKFILE_NOT_FOUND, ENVIRONMENT_HOME);
    exit (1);
  }

  LicenseForTool = 0;

  error = LIC_NOERROR;

  /*
   * While not found a license for the tool read the file.
   * LicenseForTool values are  :
   * * 0 / LIC_NFND while no license is found.
   * * 1 / LIC_ERR  an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   * * 2/ LIC_FND   an error has been encountered when processing
   * a license : end of file, ...
   * The "error" variable contains the error type.  
   */
  while (!LicenseForTool)
  {
    error = readLicense (fd, fileKey, &(fields));
    if (error == LIC_EOFERROR)
    {
      LicenseForTool = LIC_ERR;
      continue;
    }
    if (error == LIC_UNKNOWNLIC)
      continue;
    /*
     * Verify if license read is for the correct tool.
     * If not, try to read another license in the file license. 
     */
    if (!VerifyLicenseForTool (&(fields)))
      continue;

    /*
     * Compute the license key to verify that file hasn't been
     * modified. The new generated key must be equal than key file. 
     */
    ComputeLicense (&(fields), computedKey);

#define VerifyLicenseModification(a,b) ( (strcmp((a),(b)) == 0) ? 1 : 0)
    if (VerifyLicenseModification (fileKey, computedKey))
    {

      /*
       * Verify if tool is running in licensed host.
       * If isn't the correct host, check for the next license. 
       */
      error = runningNoServerHost (&(fields));
      if (error)
        continue;

      /*
       * Verify if license is for current version.
       * If not, check for a license for the tool version. 
       */
      error = correctToolVersion (&(fields));
      if (error)
        continue;

      /*
       * Verify if license hasn,t expired (Demo version).
       * If has expired, check for a license which hasn't expired.
       * 
       * Also, fills the field : fields->license_correct to
       * not only verify the loop condition. 
       */
      error = dateNotExpired (&(fields));
      if (error)
        continue;

      /*
       * A correct license has been found. 
       */
      fields.license_correct = 0xFFFF;
      LicenseForTool = LIC_FND;
    }
    else
    {
      /*
       * If genetared license is different than file license for
       * the application --> the file has been modified ERROR. 
       */
      error = LIC_BADLICENSE;
      LicenseForTool = LIC_ERR;
      continue;
    }
  }

  if (LicenseForTool == LIC_FND)
  {
    if (fields.license_correct == 0xFFFF)
      return;
    fprintf (stderr, "%s", Print_NeededLicenseParameters ());
    exit (1);
  }

  switch (error)
  {
    case LIC_BADLICENSE:
      fprintf (stderr, ERROR_INCORRECT_KEY);
      exit (1);
  }

  /*
   * Seleccionar el ERROR que sortira. 
   */
  fprintf (stderr, "%s", Print_NeededLicenseParameters ());
  exit (1);
}
#endif

