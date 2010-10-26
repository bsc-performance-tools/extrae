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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include "threadid.h"

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif

#if defined(OMP_SUPPORT)
extern int omp_get_thread_num(void);
#elif defined(SMPSS_SUPPORT)
extern int css_get_thread_num(void);
#elif defined(NANOS_SUPPORT)
extern unsigned int nanos_ompitrace_get_thread_num(void); 
#elif defined(PTHREAD_SUPPORT)
# include <pthread.h>
# include "pthread_wrapper.h"
# include "wrapper.h"
#elif defined(TRT_SUPPORT)
# include <pthread.h>
extern int threadGetID(void);
#elif defined(UPC_SUPPORT)
# include <external/upc.h>
#endif

unsigned get_trace_thread_number (void)
{
#if defined(OMP_SUPPORT)
	return omp_get_thread_num();
#elif defined(SMPSS_SUPPORT)
	return css_get_thread_num();
#elif defined(NANOS_SUPPORT)
	return nanos_ompitrace_get_thread_num();
#elif defined(PTHREAD_SUPPORT)
	return Backend_GetpThreadIdentifier();
#elif defined(TRT_SUPPORT)
	return threadGetID();
#elif defined(UPC_SUPPORT)
	return GetUPCthreadID();
#else
	return 0;
#endif
}

void * get_trace_thread_number_function (void)
{
#if defined(OMP_SUPPORT)
	return (void*) omp_get_thread_num;
#elif defined(SMPSS_SUPPORT)
	return css_get_thread_num;
#elif defined(NANOS_SUPPORT)
	return nanos_ompitrace_get_thread_num;
#elif defined(PTHREAD_SUPPORT)
	return (void*) pthread_self;
#elif defined(TRT_SUPPORT)
	/* TRT is based on pthreads */
	return (void*) pthread_self; 
#elif defined(UPC_SUPPORT)
	return (void*) GetUPCthreadID;
#else
	return NULL;
#endif
}
