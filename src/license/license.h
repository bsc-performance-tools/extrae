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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/license/license.h,v $
 | 
 | @last_commit: $Date: 2009/02/05 15:07:16 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _LICENSE_H
#define _LICENSE_H

#define MAX_KEY 1024


#define LIC_NOERROR      0
#define LIC_EOFERROR     1
#define LIC_BADLICENSE   2

#define  SCPUS_LICENSE_PROGRAM      "SCPUs License file"
#define  SCPUS_LIC_NUM 0

#define  INFOPERFEX_LICENSE_PROGRAM "infoPerfex License file"
#define  INFOPERFEX_LIC_NUM 1

#define  OMPTRACE_LICENSE_PROGRAM   "OMPtrace License file"
#define  OMPTRACE_LIC_NUM  2

#define  MPITRACE_LICENSE_PROGRAM   "MPItrace License file"
#define  MPITRACE_LIC_NUM  3

#define  OMPITRACE_LICENSE_PROGRAM  "OMPItrace License file"
#define  OMPITRACE_LIC_NUM 4

#define  PARAVER_LICENSE_PROGRAM    "Paraver License file"
#define  PARAVER_LIC_NUM   5

#define  MPIDTRACE_LICENSE_PROGRAM   "MPIDtrace License file"
#define  MPIDTRACE_LIC_NUM  6

#define  UTE2PAR_LICENSE_PROGRAM   "UTE2Paraver License file"
#define  UTE2PAR_LIC_NUM  7

#define  DIMEMAS_LICENSE_PROGRAM   "Dimemas License file"
#define  DIMEMAS_LIC_NUM  8

#define  JIS_LICENSE_PROGRAM   "JIS License file"
#define  JIS_LIC_NUM  9

#define  JOMP_LICENSE_PROGRAM   "JIS-JOMP License file"
#define  JOMP_LIC_NUM  10

#define  JVMPI_LICENSE_PROGRAM   "JIS-JVMPI License file"
#define  JVMPI_LIC_NUM  11

#if defined(SCPUS_LICENSE)
#     define  TOOL_VERSION   1
#     define  TOOL_SUB_VERSION   0
#     define BUC1 ('s'+9)
#     define BUC2 ('c'+5)
#     define BUC3 ('p'+7)
#     define BUC4 ('u'-5)
#     define BUC5 ('s'+3)
#     define BUC6 ('1'+2)
#     define ENVIRONMENT_HOME "SCPUS_HOME"
#elif defined(INFOPERFEX_LICENSE)
#     define  TOOL_VERSION   1
#     define  TOOL_SUB_VERSION   0
#     define BUC1 ('i'+6)
#     define BUC2 ('n'-7)
#     define BUC3 ('f'-4)
#     define BUC4 ('o'+1)
#     define BUC5 ('p'-1)
#     define BUC6 ('e'+3)
#     define ENVIRONMENT_HOME "INFOPERFEX_HOME"
#elif defined(OMPTRACE_LICENSE)
#     define  TOOL_VERSION   1
#     define  TOOL_SUB_VERSION   0
#     define BUC1 ('o'+9)
#     define BUC2 ('p'-7)
#     define BUC3 ('e'-5)
#     define BUC4 ('n'-3)
#     define BUC5 ('M'-7)
#     define BUC6 ('P'+6)
#     define ENVIRONMENT_HOME "OMPTRACE_HOME"
#elif defined(MPITRACE_LICENSE)
#     define  TOOL_VERSION   1
#     define  TOOL_SUB_VERSION   2
#     define BUC1 ('m'+6)
#     define BUC2 ('p'-8)
#     define BUC3 ('i'-4)
#     define BUC4 ('t'+4)
#     define BUC5 ('r'-2)
#     define BUC6 ('c'+4)
#     define ENVIRONMENT_HOME "MPITRACE_HOME"
#elif defined(MPIDTRACE_LICENSE)
#     define  TOOL_VERSION   1
#     define  TOOL_SUB_VERSION   1
#     define BUC1 ('m'+12)
#     define BUC2 ('p'-1)
#     define BUC3 ('i'-4)
#     define BUC4 ('d'+7)
#     define BUC5 ('t'-3)
#     define BUC6 ('r'+10)
#     define ENVIRONMENT_HOME "MPIDTRACE_HOME"
#elif defined(OMPITRACE_LICENSE)
#     define  TOOL_VERSION   1
#     define  TOOL_SUB_VERSION   1
#     define BUC1 ('o'+4)
#     define BUC2 ('m'-3)
#     define BUC3 ('p'+6)
#     define BUC4 ('M'-1)
#     define BUC5 ('P'+8)
#     define BUC6 ('I'-7)
#     define ENVIRONMENT_HOME "OMPITRACE_HOME"
#elif defined(PARAVER_LICENSE)
#     define  TOOL_VERSION   3
#     define  TOOL_SUB_VERSION   0
#     define BUC1 100
#     define BUC2  80
#     define BUC3  34
#     define BUC4  23
#     define BUC5  32
#     define BUC6 100
#     define ENVIRONMENT_HOME "PARAVER_HOME"
#elif defined(UTE2PARAVER_LICENSE)
#     define TOOL_VERSION   0
#     define TOOL_SUB_VERSION   9
#     define BUC1 ('u'+4)
#     define BUC2 ('t'+5)
#     define BUC3 ('e'+3)
#     define BUC4 ('2'+2)
#     define BUC5 ('p'+1)
#     define BUC6 ('a'+9)
#     define ENVIRONMENT_HOME "UTE2PARAVER_HOME"
#elif defined( JIS_LICENSE )
#     define TOOL_VERSION       1
#     define TOOL_SUB_VERSION   0
#     define BUC1 ('j'+4)
#     define BUC2 ('i'+5)
#     define BUC3 ('s'+3)
#     define BUC4 ('s'+2)
#     define BUC5 ('g'+1)
#     define BUC6 ('i'+9)
#     define ENVIRONMENT_HOME "JIS_LICENSE_HOME"
#elif defined( JOMP_LICENSE )
#     define TOOL_VERSION       1
#     define TOOL_SUB_VERSION   0
#     define BUC1 ('j'+4)
#     define BUC2 ('i'+5)
#     define BUC3 ('s'+3)
#     define BUC4 ('o'+2)
#     define BUC5 ('m'+1)
#     define BUC6 ('p'+9)
#     define ENVIRONMENT_HOME "JOMP_LICENSE_HOME"
#elif defined( JVMPI_LICENSE )
#     define TOOL_VERSION       1
#     define TOOL_SUB_VERSION   0
#     define BUC1 ('j'+4)
#     define BUC2 ('i'+5)
#     define BUC3 ('s'+3)
#     define BUC4 ('o'+2)
#     define BUC5 ('m'+1)
#     define BUC6 ('p'+9)
#     define ENVIRONMENT_HOME "JVMPI_LICENSE_HOME"
#else
#    error 'Tool not defined'
#endif

#define LIC_NFND         0
#define LIC_OK           0
#define LIC_ERR          1
#define LIC_FND          2
#define LIC_UNKNOWNLIC   3

typedef struct license_fields_t
{
  unsigned int license_pack;
  unsigned int hostid;
  unsigned int serial;
  unsigned int version, subversion;
  unsigned int lic_type;
  unsigned int port;
  unsigned int number_of_lic;
  unsigned int license_correct;
  char date[50];
}
license_fields_t;


#define LICENSE_OK            1
#define FILE_NOT_FOUND       -1
#define HOST_ID              -2
#define SC_EXPIRED           -3
#define DATE_EXPIRED         -4
#define INCORRECT_VERSION    -5

#define HOSTID_LABEL   "Host identificator     : %x\n"
#define SERIAL_LABEL   "Hardware serial number : %x\n"
#define PORT_LABEL     "License Server Port    : %d\n"
#define VERSION_LABEL  "Version                : %d.%d\n"
#define NUMBER_LABEL   "Licenses number        : %d\n"
#define DATE_LABEL     "Limit date             : %s\n"
#define TYPE_LABEL     "License type           : %s\n"
#define KEY_LABEL      "License Key            : %s\n"

#define DEMO_TYPE_NUMBER        7864
#define DEMO_TYPE_LABEL         "Demo"

#define PERMANENT_TYPE_NUMBER   3211
#define PERMANENT_TYPE_LABEL    "Permanent"

#define SITE_TYPE_NUMBER   4279
#define SITE_TYPE_LABEL    "Site"

#define ERROR_SETUP_PARAVER_HOME \
"Set up the PARAVER_HOME configuration file.\nLicense file can't be found.\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define ERROR_DATE_EXPIRED \
"Demo license has expired\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define ERROR_INCORRECT_VERSION \
"Incorrect product version for this license file\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define ERROR_INCORRECT_KEY \
"Bad license file\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define ERROR_BAD_MACHINE \
"Bad host machine for this license file\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define ERROR_FILE_NOT_FOUND \
"License file not found. Check PARAVER_HOME installation.\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define ERROR_PACKFILE_NOT_FOUND \
"License file not found. Check $%s/etc/license.dat license file.\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define ERROR_LICENSE_NOT_FOUND \
"License not found.\nPlease contact cepbatools@cepba.upc.es for further information\n"

#define LICENSE_FILE_NAME "/etc/license.dat"
#define MAX_LICENSE_PATH  7000

#if defined(DEAD_CODE)
static char *checkParaverLicense (int *jobs, int *port_server,
                                  int *hostid_file, int *serialid_file,
                                  int *licType);
#endif

int licTypeStringToNumber (char *licType);
char *licTypeNumberToString (int licType);

#endif
