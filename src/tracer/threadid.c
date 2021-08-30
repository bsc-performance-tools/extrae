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

#include "common.h"

#include "threadid.h"

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif

#if defined(OMPT_SUPPORT)
# include "ompt-wrapper.h"
#endif

/*
   Default return values
   1 thread in total, and thread id is always 0
*/

enum { Extrae_threadid_default = 0 };
enum { Extrae_num_threads_default = 1};

/* Callback definitions and API */

static unsigned (*get_thread_num) (void) = NULL;
static unsigned (*get_num_threads) (void) = NULL;

void Extrae_set_threadid_function (unsigned (*threadid_function)(void))
{
	get_thread_num = threadid_function;
}

void Extrae_set_numthreads_function (unsigned (*numthreads_function)(void))
{
	get_num_threads = numthreads_function;
}

/* Internal routines */

#if defined(OMP_SUPPORT)
extern int omp_get_thread_num(void);
extern int omp_get_num_threads(void);
#elif defined(SMPSS_SUPPORT)
extern int css_get_thread_num(void);
extern int css_get_max_threads();
#elif defined(NANOS_SUPPORT)
/* extern unsigned int nanos_extrae_get_thread_num(void); */ 
/* NANOS uses Extrae_set_threadid_function/Extrae_set_numthreads_function */
#elif defined(PTHREAD_SUPPORT)
# include <pthread.h>
# include "pthread_wrapper.h"
# include "wrapper.h"
#elif defined(UPC_SUPPORT)
# include <external/upc.h>
#endif

unsigned Extrae_get_thread_number (void)
{
	if (get_thread_num)
	{
		return get_thread_num();
	}
#if defined(OMP_SUPPORT) && !defined(OMPT_SUPPORT)
	return omp_get_thread_num();
#elif defined(SMPSS_SUPPORT)
	return css_get_thread_num();
#elif defined(PTHREAD_SUPPORT)
	return Backend_GetpThreadIdentifier();
#elif defined(UPC_SUPPORT)
	return GetUPCthreadID();
#else
	return Extrae_threadid_default;
#endif
}

void * Extrae_get_thread_number_function (void)
{
	if (get_thread_num)
	{
		return (void*) get_thread_num;
	}
#if defined(OMP_SUPPORT) && !defined(OMPT_SUPPORT)
	return (void*) omp_get_thread_num;
#elif defined(SMPSS_SUPPORT)
	return css_get_thread_num;
#elif defined(PTHREAD_SUPPORT)
	return (void*) pthread_self;
#elif defined(UPC_SUPPORT)
	return (void*) GetUPCthreadID;
#else
	return NULL;
#endif
}

unsigned Extrae_get_num_threads (void)
{
	if (get_num_threads)
	{
		return get_num_threads();
	}
#if defined(OMP_SUPPORT) && !defined(OMPT_SUPPORT)
	return omp_get_num_threads();
#elif defined(SMPSS_SUPPORT)
	return css_get_max_threads();
#elif defined(PTHREAD_SUPPORT)
	return Backend_getNumberOfThreads();
#elif defined(UPC_SUPPORT)
	return GetNumUPCthreads();
#else
	return Extrae_num_threads_default;
#endif
}
