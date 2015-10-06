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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.5/src/tracer/wrappers/OMP/omp_wrapper.c $
 | @last_commit: $Date: 2014-02-20 16:48:43 +0100 (jue, 20 feb 2014) $
 | @version:     $Revision: 2487 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id: omp_wrapper.c 2487 2014-02-20 15:48:43Z harald $";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_ASSERT_H
# include <assert.h>
#endif

#include "ompt-helper.h"

// #define NEED_MUTEX_TO_GET_TASKFUNCTION // This should not be necessary.

/* Relation between parallel id and parallel function */
typedef struct ompt_parallel_id_pf_st
{
	ompt_parallel_id_t pid;	 /* Parallel ID */
	void *pf;                /* Parallel function */
} ompt_parallel_id_pf_t;

static ompt_parallel_id_pf_t *ompt_pids_pf = NULL;
static unsigned n_ompt_pids_pf = 0;
static unsigned n_allocated_ompt_pids_pf = 0;
#define N_ALLOCATE_OMPT_PIDS 128
static pthread_mutex_t mutex_id_pf = PTHREAD_MUTEX_INITIALIZER;

void Extrae_OMPT_register_ompt_parallel_id_pf (ompt_parallel_id_t ompt_pid, void *pf)
{
	unsigned u;

	pthread_mutex_lock (&mutex_id_pf);
	if (n_ompt_pids_pf == n_allocated_ompt_pids_pf)
	{
#if defined(DEBUG)
		fprintf (stdout, "OMPT: allocating container ompt_pid - pf for %u buckets\n", n_allocated_ompt_pids_pf+N_ALLOCATE_OMPT_PIDS);
#endif
		/* Allocate container */
		ompt_pids_pf = (ompt_parallel_id_pf_t*) realloc (ompt_pids_pf, 
		  (n_allocated_ompt_pids_pf+N_ALLOCATE_OMPT_PIDS)*sizeof(ompt_parallel_id_pf_t));
		assert (ompt_pids_pf != NULL);

		/* Clear data */
		for (u = n_allocated_ompt_pids_pf; u < n_allocated_ompt_pids_pf+N_ALLOCATE_OMPT_PIDS; u++)
		{
			ompt_pids_pf[u].pid = (ompt_parallel_id_t) 0;
			ompt_pids_pf[u].pf = NULL;
		}

		/* Set new limit */
		n_allocated_ompt_pids_pf += N_ALLOCATE_OMPT_PIDS;
	}

	/* Look for an empty space within container */
	for (u = 0; u < n_allocated_ompt_pids_pf; u++)
		if (ompt_pids_pf[u].pid == (ompt_parallel_id_t) 0)
		{
			ompt_pids_pf[n_ompt_pids_pf].pid = ompt_pid;
			ompt_pids_pf[n_ompt_pids_pf].pf = pf;
			n_ompt_pids_pf++;
#if defined(DEBUG)
			fprintf (stdout, "OMPT: registered pid = %lld to pf = %p in slot %u, n_ompt_pids_pf = %u\n", ompt_pid, pf, u, n_ompt_pids_pf);
#endif
			break;
		}
	pthread_mutex_unlock (&mutex_id_pf);
}

void Extrae_OMPT_unregister_ompt_parallel_id_pf (ompt_parallel_id_t ompt_pid)
{
	/* Extract ompt_pid - pf relation if it exists within the container */
	if (n_ompt_pids_pf > 0)
	{
		unsigned u;

		pthread_mutex_lock (&mutex_id_pf);

		for (u = 0; u < n_allocated_ompt_pids_pf; u++)
			if (ompt_pids_pf[u].pid == ompt_pid)
			{
				ompt_pids_pf[u].pid = (ompt_parallel_id_t) 0;
				ompt_pids_pf[u].pf = NULL;
				n_ompt_pids_pf--;
#if defined(DEBUG)
				fprintf (stdout, "OMPT: unregistered pid = %lld from slot %u, n_ompt_pids_pf = %u\n", ompt_pid, u, n_ompt_pids_pf);
#endif
				break;
			}
		pthread_mutex_unlock (&mutex_id_pf);
	}
}

void * Extrae_OMPT_get_pf_parallel_id (ompt_parallel_id_t ompt_pid)
{
	unsigned u;
	void *ptr = NULL;

	pthread_mutex_lock (&mutex_id_pf);
	for (u = 0; u < n_allocated_ompt_pids_pf; u++)
		if (ompt_pids_pf[u].pid == ompt_pid)
		{
			ptr = ompt_pids_pf[u].pf;
			break;
		}
	pthread_mutex_unlock (&mutex_id_pf);

	return ptr;
}

/* Relation between parallel id and parallel function */
typedef struct ompt_task_id_pf_st
{
	ompt_task_id_t tid;  /* Task ID */
	void *tf;            /* Task function */
	long long task_ctr;  /* Task counter */
	int implicit;        /* is implicit ? */
	int is_running;      /* is currently running? */
} ompt_task_id_tf_t;

static ompt_task_id_tf_t *ompt_tids_tf = NULL;
static unsigned n_ompt_tids_tf = 0;
static unsigned n_allocated_ompt_tids_tf = 0;
#define N_ALLOCATE_OMPT_TIDS 128
static pthread_mutex_t mutex_tid_tf = PTHREAD_MUTEX_INITIALIZER;
static long long __task_ctr = 1;

void Extrae_OMPT_register_ompt_task_id_tf (ompt_task_id_t ompt_tid, void *tf, int implicit)
{
	unsigned u;

	pthread_mutex_lock (&mutex_tid_tf);
	if (n_ompt_tids_tf == n_allocated_ompt_tids_tf)
	{
#if defined(DEBUG)
		fprintf (stdout, "OMPT: allocating container ompt_tid - pf for %u buckets\n", n_allocated_ompt_tids_tf+N_ALLOCATE_OMPT_TIDS);
#endif
		/* Allocate container */
		ompt_tids_tf = (ompt_task_id_tf_t*) realloc (ompt_tids_tf, 
		  (n_allocated_ompt_tids_tf+N_ALLOCATE_OMPT_TIDS)*sizeof(ompt_task_id_tf_t));
		assert (ompt_tids_tf != NULL);

		/* Clear data */
		for (u = n_allocated_ompt_tids_tf; u < n_allocated_ompt_tids_tf+N_ALLOCATE_OMPT_TIDS; u++)
		{
			ompt_tids_tf[u].tid = (ompt_task_id_t) 0;
			ompt_tids_tf[u].tf = NULL;
			ompt_tids_tf[u].implicit = FALSE;
			ompt_tids_tf[u].is_running = FALSE;
		}

		/* Set new limit */
		n_allocated_ompt_tids_tf += N_ALLOCATE_OMPT_TIDS;
	}

	/* Look for an empty space within container */
	for (u = 0; u < n_allocated_ompt_tids_tf; u++)
		if (ompt_tids_tf[u].tid == (ompt_task_id_t) 0)
		{
			ompt_tids_tf[n_ompt_tids_tf].tid = ompt_tid;
			ompt_tids_tf[n_ompt_tids_tf].tf = tf;
			ompt_tids_tf[n_ompt_tids_tf].implicit = implicit;
			ompt_tids_tf[n_ompt_tids_tf].task_ctr = __task_ctr++;
			ompt_tids_tf[u].is_running = FALSE;
			n_ompt_tids_tf++;
#if defined(DEBUG)
			fprintf (stdout, "OMPT: registered tid = %lld to tf = %p in slot %u, n_ompt_tids_tf = %u\n", ompt_tid, tf, u, n_ompt_tids_tf);
#endif
			break;
		}
	pthread_mutex_unlock (&mutex_tid_tf);
}

void Extrae_OMPT_unregister_ompt_task_id_tf (ompt_task_id_t ompt_tid)
{
	/* Extract ompt_tid - tf relation if it exists within the container */
	if (n_ompt_tids_tf > 0)
	{
		unsigned u;

		pthread_mutex_lock (&mutex_tid_tf);
		for (u = 0; u < n_allocated_ompt_tids_tf; u++)
			if (ompt_tids_tf[u].tid == ompt_tid)
			{
				ompt_tids_tf[u].tid = (ompt_task_id_t) 0;
				ompt_tids_tf[u].tf = NULL;
				ompt_tids_tf[u].is_running = FALSE;
				n_ompt_tids_tf--;
#if defined(DEBUG)
				fprintf (stdout, "OMPT: unregistered tid = %lld from slot %u, n_ompt_tids_tf = %u\n", ompt_tid, u, n_ompt_tids_tf);
#endif
				break;
			}
		pthread_mutex_unlock (&mutex_tid_tf);
	}
}

void * Extrae_OMPT_get_tf_task_id (ompt_task_id_t ompt_tid,
	int *is_implicit, long long *taskctr)
{
	unsigned u;
	void *ptr = NULL;

#if defined(NEED_MUTEX_TO_GET_TASKFUNCTION)
	pthread_mutex_lock (&mutex_tid_tf);
#endif
	for (u = 0; u < n_allocated_ompt_tids_tf; u++)
		if (ompt_tids_tf[u].tid == ompt_tid)
		{
			ptr = ompt_tids_tf[u].tf;
			if (is_implicit != NULL)
				*is_implicit = ompt_tids_tf[u].implicit;
			if (taskctr != NULL)
				*taskctr = ompt_tids_tf[u].task_ctr;
			break;
		}
#if defined(NEED_MUTEX_TO_GET_TASKFUNCTION)
	pthread_mutex_unlock (&mutex_tid_tf);
#endif

	return ptr;
}

void Extrae_OMPT_tf_task_id_set_running (ompt_task_id_t ompt_tid, int b)
{
	unsigned u;

#if defined(NEED_MUTEX_TO_GET_TASKFUNCTION)
	pthread_mutex_lock (&mutex_tid_tf);
#endif

	for (u = 0; u < n_allocated_ompt_tids_tf; u++)
		if (ompt_tids_tf[u].tid == ompt_tid)
		{
			ompt_tids_tf[u].is_running = b;
			break;
		}

#if defined(NEED_MUTEX_TO_GET_TASKFUNCTION)
	pthread_mutex_unlock (&mutex_tid_tf);
#endif
}


int Extrae_OMPT_tf_task_id_is_running (ompt_task_id_t ompt_tid)
{
	unsigned u;
	int res = FALSE;

#if defined(NEED_MUTEX_TO_GET_TASKFUNCTION)
	pthread_mutex_lock (&mutex_tid_tf);
#endif

	for (u = 0; u < n_allocated_ompt_tids_tf; u++)
		if (ompt_tids_tf[u].tid == ompt_tid)
		{
			res = ompt_tids_tf[u].is_running;
			break;
		}

#if defined(NEED_MUTEX_TO_GET_TASKFUNCTION)
	pthread_mutex_unlock (&mutex_tid_tf);
#endif

	return res;
}

