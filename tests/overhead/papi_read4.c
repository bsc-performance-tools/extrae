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
#include <papi.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
	long_long v[8];
	int retval;
	int e = PAPI_NULL;
	struct timespec start, stop;
	int i;
	unsigned long long t1, t2;
	unsigned n = 1000000;

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT && retval > 0)
	{
		fprintf(stderr,"PAPI library version mismatch!\en");
		exit(1);
	}

	if (PAPI_create_eventset (&e) != PAPI_OK)
	{
		fprintf (stderr, "Failed to create eventset\n");
		exit (1);
	}

#if PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5
	if (PAPI_add_named_event (e, "PAPI_TOT_INS") != PAPI_OK)
#else
	if (PAPI_add_event (e, PAPI_TOT_INS) != PAPI_OK)
#endif
	{
		fprintf (stderr, "Failed to add PAPI_TOT_INS\n");
		exit (1);
	}
#if PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5
	if (PAPI_add_named_event (e, "PAPI_TOT_CYC") != PAPI_OK)
#else
	if (PAPI_add_event (e, PAPI_TOT_CYC) != PAPI_OK)
#endif
	{
		fprintf (stderr, "Failed to add PAPI_TOT_CYC\n");
		exit (1);
	}
#if PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5
	if (PAPI_add_named_event (e, "PAPI_BR_INS") != PAPI_OK)
#else
	if (PAPI_add_event (e, PAPI_BR_INS) != PAPI_OK)
#endif
	{
		fprintf (stderr, "Failed to add PAPI_BR_INS\n");
		exit (1);
	}
#if PAPI_VERSION_MAJOR(PAPI_VERSION) >= 5
	if (PAPI_add_named_event (e, "PAPI_L1_DCM") != PAPI_OK)
#else
	if (PAPI_add_event (e, PAPI_L1_DCM) != PAPI_OK)
#endif
	{
		fprintf (stderr, "Failed to add PAPI_L1_DCM\n");
		exit (1);
	}

	if (PAPI_start (e) != PAPI_OK)
	{
		fprintf (stderr, "Failed to start the eventset\n");
		exit (1);
	}

	clock_gettime (CLOCK_MONOTONIC, &start);
	for (i = 0; i < n; i++)
		PAPI_read (e, v);
	clock_gettime (CLOCK_MONOTONIC, &stop);
	t1 = start.tv_nsec;
	t1 += start.tv_sec * 1000000000;
	t2 = stop.tv_nsec;
	t2 += stop.tv_sec * 1000000000;
	printf ("RESULT : papi_read1() %Lu ns\n", (t2 - t1) / n);

	return 0;
}
