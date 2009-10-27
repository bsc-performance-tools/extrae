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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/addr2info.h,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:06:55 $
 | @version:     $Revision: 1.8 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __ADDR2INFO_H__
#define __ADDR2INFO_H__

#include <config.h>

#ifdef HAVE_BFD_H
# include <bfd.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "common.h"

#define TARGET          "default"
#define CODE_SECTION    ".text"

#define MAX_ADDR_LENGTH 20
#define ADDR_UNRESOLVED "Unresolved"
#define ADDR_NOT_FOUND  "_NOT_Found"

/* Public routines */
void Address2Info_Initialize (char * binary);
int  Address2Info_Translate (UINT64 address, int event_type, int uniqueID);
void Address2Info_Write_MPI_Labels (FILE * pcf_fd, int uniqueid);
void Address2Info_Write_OMP_Labels (FILE * pcf_fd, int eventtype, int eventtype_line, int uniqueid);
void Address2Info_Write_UF_Labels (FILE * pcf_fd, int uniqueid);
void Address2Info_Write_Sample_Labels (FILE * pcf_fd, int uniqueid);
int Address2Info_AddSymbol (UINT64 address, int addr_type, char * funcname,
  char * filename, int line);

enum
{
	ADDR2OMP_FUNCTION,
	ADDR2OMP_LINE,
	ADDR2MPI_FUNCTION,
	ADDR2MPI_LINE,
	ADDR2UF_FUNCTION,
	ADDR2UF_LINE,
	ADDR2SAMPLE_FUNCTION,
	ADDR2SAMPLE_LINE,
	ADDR2_FUNCTION_UNIQUE,
	ADDR2_LINE_UNIQUE
};

enum
{
	OUTLINED_OPENMP_TYPE = 0,
	MPI_CALLER_TYPE,
	USER_FUNCTION_TYPE,
	SAMPLE_TYPE,
	UNIQUE_TYPE,
	COUNT_ADDRESS_TYPES /* Must be the very last entry */
};

struct address_info
{
	UINT64 address;
	int function_id;
	char * file_name;
	int line;
};

struct address_table
{
	struct address_info * address;
	int num_addresses;
};

struct function_table
{
	UINT64 *address_id;
	char ** function;
	int num_functions;
};

#define COPY_STRING(source, destination) {                         \
    if (source == NULL)                                            \
    {                                                              \
        destination = NULL;                                        \
    }                                                              \
    else                                                           \
    {                                                              \
        destination = (char *)malloc(strlen((const char *)source)+1);     \
        destination = strcpy ((char *)destination, (const char *)source); \
        if (destination == NULL)                                   \
        {                                                          \
            fprintf(stderr,                                        \
                    "Error while copying string '%s' into %p\n", \
                    source,                                        \
                    destination);                                  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }                                                              \
}


#if defined(PARALLEL_MERGE)
void Share_Callers_Usage (void);
#endif

#endif /* __ADDR2INFO_H__ */
