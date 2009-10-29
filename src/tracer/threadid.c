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
#elif defined(PTHREAD_SUPPORT)
# include <pthread.h>
# include "pthread_wrapper.h"
# include "wrapper.h"
#elif defined(TRT_SUPPORT)
# include <pthread.h>
extern int threadGetID(void);
#endif

unsigned get_trace_thread_number (void)
{
#if defined(OMP_SUPPORT)
	return omp_get_thread_num();
#elif defined(SMPSS_SUPPORT)
	return css_get_thread_num();
#elif defined(PTHREAD_SUPPORT)
	return Backend_GetpThreadIdentifier();
#elif defined(TRT_SUPPORT)
	return threadGetID();
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
#elif defined(PTHREAD_SUPPORT)
	return (void*) pthread_self;
#elif defined(TRT_SUPPORT)
	/* TRT is based on pthreads */
	return (void*) pthread_self; 
#else
	return NULL;
#endif
}
