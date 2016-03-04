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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
# include <sys/resource.h>
#endif

#include "utils.h"

#include "clock.h"

#if defined(USE_GETTIMEOFDAY_CLOCK)
# include <gettimeofday_clock.h>
# define  GET_CLOCK    gettimeofday_getTime()
# define  INIT_CLOCK   gettimeofday_Initialize()
# define  INIT_CLOCK_T gettimeofday_Initialize_thread()
#elif defined(USE_POSIX_CLOCK)
# include <posix_clock.h>
# define  GET_CLOCK    posix_getTime()
# define  INIT_CLOCK   posix_Initialize()
# define  INIT_CLOCK_T posix_Initialize_thread()
#elif defined(IS_BGL_MACHINE)
# include <bgl_clock.h>
# define  GET_CLOCK    bgl_getTime()
# define  INIT_CLOCK   bgl_Initialize()
# define  INIT_CLOCK_T bgl_Initialize_thread()
#elif defined(IS_BGP_MACHINE)
# include <bgp_clock.h>
# define  GET_CLOCK    bgp_getTime()
# define  INIT_CLOCK   bgp_Initialize()
# define  INIT_CLOCK_T bgp_Initialize_thread()
#elif defined(IS_BGQ_MACHINE)
# include <bgq_clock.h>
# define  GET_CLOCK    bgq_getTime()
# define  INIT_CLOCK   bgq_Initialize()
# define  INIT_CLOCK_T bgq_Initialize_thread()
#elif (defined (OS_LINUX) || defined(OS_FREEBSD) || defined(OS_DARWIN) || defined(OS_SOLARIS)) && defined (ARCH_IA32)
# include <ia32_clock.h>
# define  GET_CLOCK    ia32_getTime()
# define  INIT_CLOCK   ia32_Initialize()
# define  INIT_CLOCK_T ia32_Initialize_thread()
#elif defined(OS_LINUX) && defined(ARCH_IA64)
# include <ia64_clock.h>
# define  GET_CLOCK    ia64_getTime()
# define  INIT_CLOCK   ia64_Initialize()
# define  INIT_CLOCK_T ia64_Initialize_thread()
#elif (defined(OS_LINUX) || defined(OS_AIX)) && defined(ARCH_PPC)
# include <ppc_clock.h>
# define  GET_CLOCK    ppc_getTime()
# define  INIT_CLOCK   ppc_Initialize()
# define  INIT_CLOCK_T ppc_Initialize_thread()
#else
# error "Unhandled clock type"
#endif

static UINT64 *_extrae_last_read_clock = NULL;
static unsigned ClockType = REAL_CLOCK;

void Clock_setType (unsigned type)
{
	ClockType = (type==REAL_CLOCK || type==USER_CLOCK)?type:ClockType;
}

unsigned Clock_getType (void)
{
	return ClockType;
}

/* We obtain the last read time */
UINT64 Clock_getLastReadTime (unsigned thread)
{
	return _extrae_last_read_clock[thread];
}

/* We obtain the current time, but we don't store it in the last read time */
UINT64 Clock_getCurrentTime_nstore (void)
{
	UINT64 tmp;

	if (ClockType == REAL_CLOCK)
	{
		tmp = GET_CLOCK;

/*  if no "nanosecond" clock is available 
		struct timeval aux;
		gettimeofday (&aux, NULL);
		return (((UINT64) aux.tv_sec) * 1000000 + aux.tv_usec);
*/
	}
	else
	{
		struct rusage aux;

		if (getrusage(RUSAGE_SELF,&aux) >= 0)
		{
			/* Get user time */
			tmp =  aux.ru_utime.tv_sec*1000000 + aux.ru_utime.tv_usec;
			/* Accumulate system time */
			tmp += aux.ru_stime.tv_sec*1000000 + aux.ru_stime.tv_usec;
		}
		else
			tmp = 0;

		tmp = tmp * 1000;
	}


	return tmp;
}

/* We obtain the current time and we store it in the last read time,
   for future reads of the same timestamp */
UINT64 Clock_getCurrentTime (unsigned thread)
{
	UINT64 tmp = Clock_getCurrentTime_nstore ();
	_extrae_last_read_clock[thread] = tmp;
	return tmp;
}

void Clock_AllocateThreads (unsigned numthreads)
{
	_extrae_last_read_clock = (UINT64*) realloc (_extrae_last_read_clock,
		sizeof(UINT64)*numthreads);
	if (_extrae_last_read_clock == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot allocate timing memory for %u threads\n", numthreads);
		exit (-1);
	}
}

void Clock_CleanUp (void)
{
	xfree (_extrae_last_read_clock);
}

void Clock_Initialize (unsigned numthreads)
{
	Clock_AllocateThreads (numthreads);

	INIT_CLOCK;
}

void Clock_Initialize_thread (void)
{
	INIT_CLOCK_T;
}
