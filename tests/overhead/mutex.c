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

#include <pthread.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int main (int argc, char *argv[])
{
	int i;
	struct timespec start, end;
	unsigned long long t;

	clock_gettime( CLOCK_REALTIME, &start);
	for (i = 0; i < 10000000; i++)
	{
		pthread_mutex_lock(&lock);
		pthread_mutex_unlock(&lock);
	}
	clock_gettime( CLOCK_REALTIME, &end);

	t = ((end.tv_sec * 1000000000ULL) + end.tv_nsec)
	    -((start.tv_sec * 1000000000ULL) + start.tv_nsec);

	printf ("time %llu ns aggregated, %llu per loop iteration\n", t, t/10000000ULL);

	return 0;
}

