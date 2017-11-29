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
   ptr = NULL;                        \
} 

#if defined(__cplusplus)
extern "C" {
#endif

int __Extrae_Utils_is_Whitespace (char c);
int __Extrae_Utils_is_Alphabetic (char c);
char *__Extrae_Utils_trim (char *sourceStr);
int __Extrae_Utils_explode (char *sourceStr, const char *delimiter, char ***tokenArray);
int __Extrae_Utils_append_from_to_file (const char *source, const char *destination);
int __Extrae_Utils_rename_or_copy (char *origen, char *desti);
unsigned long long __Extrae_Utils_getTimeFromStr (const char *time, const char *envvar, int rank);
unsigned long long __Extrae_Utils_getFactorValue (const char *value, const char *ref, int rank);
int __Extrae_Utils_mkdir_recursive (const char *path);
int __Extrae_Utils_file_exists (const char *file);
int __Extrae_Utils_directory_exists (const char *file);
int __Extrae_Utils_shorten_string (unsigned nprefix, unsigned nsufix, const char *infix,
	unsigned __Extrae_Utils_buffersize, char *buffer, const char *string);

#if defined(__cplusplus)
}
#endif

#define STRINGIFY(s) #s

#endif /* __UTILS_H__ */
