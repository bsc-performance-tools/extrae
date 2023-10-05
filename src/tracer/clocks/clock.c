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

/* Needed by getenv() */
#include <string.h>

#include "utils.h"
#include "xalloc.h"

#include "clock.h"

#include <posix_clock.h>
#include <rusage_clock.h>

#if defined(USE_GETTIMEOFDAY_CLOCK)
# include <gettimeofday_clock.h>
# define  GET_CLOCK    gettimeofday_getTime
# define  INIT_CLOCK   gettimeofday_Initialize
#elif defined(USE_POSIX_CLOCK)
# define  GET_CLOCK    posix_getTime
# define  INIT_CLOCK   posix_Initialize
#elif defined(IS_BGL_MACHINE)
# include <bgl_clock.h>
# define  GET_CLOCK    bgl_getTime
# define  INIT_CLOCK   bgl_Initialize
#elif defined(IS_BGP_MACHINE)
# include <bgp_clock.h>
# define  GET_CLOCK    bgp_getTime
# define  INIT_CLOCK   bgp_Initialize
#elif defined(IS_BGQ_MACHINE)
# include <bgq_clock.h>
# define  GET_CLOCK    bgq_getTime
# define  INIT_CLOCK   bgq_Initialize
#elif (defined (OS_LINUX) || defined(OS_FREEBSD) || defined(OS_DARWIN) || defined(OS_SOLARIS)) && defined (ARCH_IA32)
# include <ia32_clock.h>
# define  GET_CLOCK    ia32_getTime
# define  INIT_CLOCK   ia32_Initialize
#elif defined(OS_LINUX) && defined(ARCH_IA64)
# include <ia64_clock.h>
# define  GET_CLOCK    ia64_getTime
# define  INIT_CLOCK   ia64_Initialize
#elif (defined(OS_LINUX) || defined(OS_AIX)) && defined(ARCH_PPC)
# include <ppc_clock.h>
# define  GET_CLOCK    ppc_getTime
# define  INIT_CLOCK   ppc_Initialize
#else
# error "Unhandled clock type"
#endif

static UINT64 *_extrae_last_read_clock = NULL;
static unsigned ClockType = REAL_CLOCK;
iotimer_t (*get_clock)();

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
	return (UINT64)get_clock();
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
	_extrae_last_read_clock = (UINT64*) xrealloc (_extrae_last_read_clock,
		sizeof(UINT64)*numthreads);
}

void Clock_CleanUp (void)
{
	xfree (_extrae_last_read_clock);
}

void Clock_Initialize (unsigned numthreads)
{
	void (*init_clock)();

	Clock_AllocateThreads (numthreads);
	if (ClockType == REAL_CLOCK)
	{
		char *use_posix_clock = NULL;

		use_posix_clock = getenv("EXTRAE_USE_POSIX_CLOCK");
		if (use_posix_clock != NULL && strcmp(use_posix_clock, "0") == 0)
		{
			init_clock = INIT_CLOCK;
			get_clock = GET_CLOCK;
		} else
		{
			init_clock = posix_Initialize;
			get_clock = posix_getTime;
		}

/*  if no "nanosecond" clock is available 
		struct timeval aux;
		gettimeofday (&aux, NULL);
		return (((UINT64) aux.tv_sec) * 1000000 + aux.tv_usec);
*/
	} else if (ClockType == USER_CLOCK)
	{
		init_clock = rusage_Initialize;
		get_clock = rusage_getTime;
	} else
	{
		fprintf (stderr, PACKAGE_NAME": Couldn't get clock type\n");
		exit (-1);
	}

	init_clock();
}
