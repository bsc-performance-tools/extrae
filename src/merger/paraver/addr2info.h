/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

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
#include "labels.h"
#include "addr2types.h"

#define ADDR_UNRESOLVED "Unresolved"
#define ADDR_NOT_FOUND  "_NOT_Found"
#define UNRESOLVED_ID 0
#define NOT_FOUND_ID 1


/* Public routines */
void Address2Info_Initialize (char * binary);
int Address2Info_Initialized (void);
UINT64 Address2Info_Translate (unsigned ptask, unsigned task, UINT64 address,
	int event_type, int uniqueID);
UINT64 Address2Info_Translate_MemReference (unsigned ptask, unsigned task,
	UINT64 address, int event_type, UINT64 *calleraddresses);
void Address2Info_Write_CUDA_Labels (FILE * pcf_fd, int uniqueid);
void Address2Info_Write_MPI_Labels (FILE * pcf_fd, int uniqueid);
void Address2Info_Write_OMP_Labels (FILE * pcf_fd, int eventtype,
	char *eventtype_description, int eventtype_line,
	char *eventtype_line_description, int uniqueid);
void Address2Info_Write_UF_Labels (FILE * pcf_fd, int uniqueid);
void Address2Info_Write_OTHERS_Labels (FILE * pcf_fd, int uniqueid, int nlabels,
	codelocation_label_t *labels);
void Address2Info_Write_Sample_Labels (FILE * pcf_fd, int uniqueid);
void Address2Info_Write_MemReferenceCaller_Labels (FILE * pcf_fd);
void Address2Info_AddSymbol (UINT64 address, int addr_type, char * funcname,
  char * filename, int line);
void Address2Info_Sort (int unique_ids);

UINT64 Address2Info_GetLibraryID (unsigned ptask, unsigned task, UINT64 address);
void Address2Info_Write_LibraryIDs (FILE *pcf_fd);

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
	ADDR2CUDA_FUNCTION,
	ADDR2CUDA_LINE,
	ADDR2OTHERS_FUNCTION,
	ADDR2OTHERS_LINE,
	ADDR2_FUNCTION_UNIQUE,
	ADDR2_LINE_UNIQUE,
	MEM_REFERENCE_DYNAMIC,
	MEM_REFERENCE_STATIC,
};

enum
{
	OUTLINED_OPENMP_TYPE = 0,
	MPI_CALLER_TYPE,
	USER_FUNCTION_TYPE,
	SAMPLE_TYPE,
	CUDAKERNEL_TYPE,
	OTHER_FUNCTION_TYPE,
	UNIQUE_TYPE,
	COUNT_ADDRESS_TYPES /* Must be the very last entry */
};

struct address_info
{
	UINT64 address;
	int line;
	int function_id;
	char * file_name;
	char * module;
};

struct address_table
{
	struct address_info * address;
	int num_addresses;
};

struct address_object_info_st
{
	int is_static;
	int line;
	const char * file_name;
	const char * module;
	const char * name;
};

struct address_object_table_st
{
	struct address_object_info_st * objects;
	int num_objects;
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
