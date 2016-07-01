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

#if defined(OS_LINUX) && defined(ARCH_PPC)

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#include "ppc_clock.h"

static unsigned long long proc_timebase_MHz;

static inline unsigned long long cpu_time (void)
{
#ifdef __powerpc64__
	unsigned long long res;

	asm volatile( "mftb  %0" : "=r"(res));

	return res;
#elif defined (__powerpc__) || defined (__POWERPC__)
	unsigned int HighB, HighA, Low;

	do
	{
		asm volatile( "mftbu %0" : "=r"(HighB) );
		asm volatile( "mftb  %0" : "=r"(Low)	);
		asm volatile( "mftbu %0" : "=r"(HighA) );
	}
	while (HighB != HighA);

	return ((unsigned long long)HighA<<32) | ((unsigned long long)Low);
#else
# error "Cannot determine the ABI"
#endif
}

unsigned long long proc_timebase (void)
{
	FILE *fp;
	char buffer[ 32768 ];
	size_t bytes_read;
	char* match;
	int res;
	unsigned long long clk;

	fp = fopen( "/proc/cpuinfo", "r" );
	bytes_read = fread( buffer, 1, sizeof( buffer ), fp );
	fclose( fp );

	if (bytes_read == 0)
		return 0ULL;

	buffer[ bytes_read ] = '\0';
	match = strstr( buffer, "timebase" );
	if (match == NULL)
		return 0ULL;

	res = sscanf (match, "timebase : %llu", &clk );

	return (res == 1) ? clk : 0;
}

iotimer_t ppc_getTime (void)
{
  return (cpu_time() * 1000) / proc_timebase_MHz; 
}


void ppc_Initialize (void)
{
	proc_timebase_MHz = proc_timebase() / 1000000;
}

#elif defined(OS_AIX) && defined(ARCH_PPC)

#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#include "ppc_clock.h"

static timebasestruct_t initial;

/* Maybe we could add switch clock support ? */

iotimer_t ppc_getTime (void)
{
	timebasestruct_t t;
	long long s, ns;
	long long high, ini_high;
	long long low, ini_low;
	unsigned long long tmp;

	read_real_time(&t, TIMEBASE_SZ);
	time_base_to_time(&t, TIMEBASE_SZ);

	high = (long long)t.tb_high;
	ini_high = (long long)initial.tb_high;
	low = (long long)t.tb_low;
	ini_low = (long long)initial.tb_low;

	s = high - ini_high;
	ns = low - ini_low;

	if (ns < 0)
	{
		s--;
		ns += 1000000000;
	}
	tmp = ((unsigned long long) s) * 1000000000 + ((unsigned long long) ns);

	return tmp;
}

void ppc_Initialize (void)
{
	read_real_time (&initial, TIMEBASE_SZ);
	time_base_to_time (&initial, TIMEBASE_SZ);
}

#endif /* PPC & LINUX */
