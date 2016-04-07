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

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

unsigned long long proc_timebase_MHz = 2000; /* Assume a 2.0GHz machine, in fact, it does not matter */

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

unsigned long long ppc_getTime (void)
{
  return (cpu_time() * 1000) / proc_timebase_MHz; 
}

int main (int argc, char *argv[])
{
	struct timespec start, stop;
	int i;
	unsigned n = 1000000;
	unsigned long long t1, t2, useless;

	clock_gettime (CLOCK_MONOTONIC, &start);
	for (i = 0; i < n; i++)
		useless = ppc_getTime();
	clock_gettime (CLOCK_MONOTONIC, &stop);
	t1 = start.tv_nsec;
	t1 += start.tv_sec * 1000000000;
	t2 = stop.tv_nsec;
	t2 += stop.tv_sec * 1000000000;
	printf ("RESULT : clock_gettime() %Lu ns\n", (t2 - t1) / n);

	return 0;
}
