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
#include <pthread.h>
#include <stdlib.h>

#include "extrae_user_events.h"

#define MAX_THREADS 8

/* Barrier variable */
pthread_barrier_t barrier;

void longExecution(long th_id)
{
    Extrae_user_function (1);
    printf ("Thread %08lx: Waiting 5 seconds\n", th_id);
    sleep(5);
    Extrae_user_function (0);
}

void *routine1 (void *parameters)
{
	long th_id = (long) parameters;
	Extrae_event (1, 1);
	if (th_id == 0)
	{
		printf ("routine1 thread 0 executing a long function\n");
		longExecution(th_id);
	}
	printf ("routine1 stopped for barrier : (thread=%08lx, param %p)\n", pthread_self(), parameters);
	// Synchronization point
	int rc = pthread_barrier_wait(&barrier);
	if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
	{
		printf("Could not wait on barrier\n");
		exit(-1);
	}
	printf ("routine1 exiting from barrier : (thread=%08lx, param %p)\n", pthread_self(), parameters);
	Extrae_event (1, 0);
}

void *routine2 (void *parameters)
{
	Extrae_event (2, 1);
	printf ("routine 2 : (thread=%08lx, param %p)\n", pthread_self(), parameters);
	Extrae_event (2, 0);
}


int main (int argc, char *argv[])
{
	pthread_t t[MAX_THREADS];
	int i;
    // Barrier initialization
    if(pthread_barrier_init(&barrier, NULL, MAX_THREADS))
    {
        printf("Could not create a barrier\n");
        return -1;
    }

	Extrae_init ();

	for (i = 0; i < MAX_THREADS; i++)
		pthread_create (&t[i], NULL, routine1, (void*) ((long) i));
	for (i = 0; i < MAX_THREADS; i++)
		pthread_join (t[i], NULL);

	sleep (1);

	for (i = 0; i < MAX_THREADS; i++)
		pthread_create (&t[i], NULL, routine2, NULL);
	for (i = 0; i < MAX_THREADS; i++)
		pthread_join (t[i], NULL);

	Extrae_fini();

	return 0;
}


