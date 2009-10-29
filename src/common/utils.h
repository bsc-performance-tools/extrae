/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __UTILS_H__
#define __UTILS_H__

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "debug.h"

#define xmalloc(ptr,size)             \
{                                     \
   ptr = malloc(size);                \
   ASSERT (                           \
      (ptr != NULL),                  \
      "Error allocating memory."      \
   );                                 \
}

#define xrealloc(ptr,src,size)        \
{                                     \
   ptr = realloc(src, size);          \
   ASSERT (                           \
      (ptr != NULL),                  \
      "Error allocating memory."      \
   );                                 \
}

#define xfree(ptr)                    \
{                                     \
   if (ptr != NULL)                   \
   {                                  \
      free(ptr);                      \
   }                                  \
} 

#if defined(__cplusplus)
extern "C" {
#endif

int explode (char *sourceStr, const char *delimiter, char ***tokenArray);
void rename_or_copy (char *origen, char *desti);
unsigned long long getTimeFromStr (char *time, char *envvar, int rank);
unsigned long long getFactorValue (char *value, char *ref, int rank);
int mkdir_recursive (char *path);

#if defined(__cplusplus)
}
#endif

#endif /* __UTILS_H__ */
