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
#include <stdint.h>

#include "extrae_user_events.h"

extern uint64_t Extrae_get_caller (int);

void deeper (int n)
{
	int i;
	for (i = 0; i < n; i++)
		Extrae_get_caller (7);
}

void abc (int n)
{
	deeper (n);
}

void xyz (int n)
{
	abc (n);
}

void bar (int n)
{
	xyz (n);
}

void foo (int n)
{
	bar (n);
}

int main(int argc, char **argv)
{
	unsigned long long t1, t2;
	struct timespec start, stop;
	int n = 1000000, i;
	Extrae_init();
	clock_gettime (CLOCK_MONOTONIC, &start);
	foo (n);
	clock_gettime (CLOCK_MONOTONIC, &stop);
	t1 = start.tv_nsec;
	t1 += start.tv_sec * 1000000000ULL;
	t2 = stop.tv_nsec;
	t2 += stop.tv_sec * 1000000000ULL;
	printf ("RESULT : Extrae_eventandcounters() %Lu ns\n", (t2 - t1) / n);
	Extrae_fini();
}
