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

#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED

#include <config.h>

#include "types.h"

#ifdef HAVE_UNUSED_ATTRIBUTE
# ifndef UNUSED
#  define UNUSED __attribute__ ((unused))
# endif
#else
# ifndef UNUSED
#  define UNUSED 
# endif
#endif

#define UNREFERENCED_PARAMETER(a) {(void) (a);}
#define UNREFERENCED(a)           {(void) (a);}

#if !defined(TRUE)
# define TRUE (1==1)
#endif

#if !defined(FALSE)
# define FALSE (1==0)
#endif

#define ENABLE   TRUE
#define ENABLED  TRUE

#define DISABLE  FALSE
#define DISABLED FALSE

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define MPTRACE_ERROR(x) \
     if ( (x) != 0 ) return (-1)

/*******************************************************************************
 * Tracefile Name format.    
 ******************************************************************************/

/* 
 * Tracefile format is :  applicationnamePPPPPPPPPPRRRRRRVVVVVV.mpit
 *
 * where  PPPPPPPPPP is the parent pid (10 digits)
 * where  RRR is the process rank in MPI group (6 digits)
 * where  VVV is the virtual processor identifier within MPI process (6 digits)
 */

#define DIGITS_PID   10
#define DIGITS_TASK   6
#define DIGITS_THREAD 6
#define MPIT_ID_LENGTH (DIGITS_PID + DIGITS_TASK + DIGITS_THREAD + 1)

#define TEMPLATE_NODE_SEPARATOR      "@"
#define TEMPLATE_NODE_SEPARATOR_CHAR '@'

#define TEMPLATE_DIGITS "%.10d%.6d%.6u"
#define TEMPLATE_PTT    "%s/%s"TEMPLATE_NODE_SEPARATOR"%s."TEMPLATE_DIGITS"%s"  // dir/app@hostname.digits.ext
#define TEMPLATE_P      "%s/%s%s"

/* Uses pid, task id and thread id to generate the name of the file */
#define FileName_PTT(name, path, prefix, hostname, pid, task, thread, ext) \
	snprintf(name, sizeof(name), TEMPLATE_PTT, path, prefix, hostname, pid, task, thread, ext);

/* Uses pid to generate the name of the file */
#define FileName_P(name, path, prefix, ext) \
	snprintf(name, sizeof(name), TEMPLATE_P, path, prefix, ext);

#define EXT_MPITS      ".mpits"
#define EXT_SPAWN      ".spawn"

#define EXT_TMP_MPIT   ".ttmp"
#define EXT_MPIT       ".mpit"

#define EXT_TMP_SAMPLE ".stmp"
#define EXT_SAMPLE     ".sample"

#define EXT_TMP_ONLINE ".otmp"
#define EXT_ONLINE     ".online"

#define EXT_SYM        ".sym"

#define EXT_CALLBACK   ".cbk"

#endif
