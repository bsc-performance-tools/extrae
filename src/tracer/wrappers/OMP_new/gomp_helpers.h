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

#pragma once

#include "common.h"

#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#include "wrapper.h"

/**
 * #pragma omp parallel num_threads(new_num_threads)
 *
 * When the OpenMP clause 'num_threads' is set in a parallel pragma, the number
 * of active threads for the scope of that parallel changes to the value specified
 * by the 'num_threads' clause (e.g. num_threads(16)). After the parallel is over,
 * the number of active threads returns to the former value. The following
 * macros are used to update the registered threads in Extrae.
 */
#define NO_OMP_CLAUSE_NUM_THREADS               0

/**
 * OMP_CLAUSE_NUM_THREADS_CHANGE
 *
 * Saves the current number of threads before a parallel with
 * 'num_threads' clause and registers the new threads in Extrae.
 *
 * @param new_num_threads The number of threads specified by the 'num_threads' clause
 * 
 * FIXME: Changing the number of threads with the num_threads clause within a parallel
 * region is only allowed in OpenMP tracing libraries not supporting other
 * runtimes that may also increase the number of threads. Example: An OpenMP +
 * CUDA application gets 20 buffers reserved during initialization: 16 for
 * OpenMP and 4 more for CUDA devices. If an OpenMP parallel region extends the
 * number of threads beyond 16, the first 4 new OpenMP threads will be mapped on
 * the 4 buffers reserved for CUDA, mixing data from the different runtimes in
 * the same Paraver line. If the application calls omp_set_num_threads
 * explicitly it will trigger the same problem, so the omp_set_num_threads
 * wrappers also use these macros to prevent increasing the number of threads
 * in mixed tracing libraries.
 * The call omp_set_num_threads is no longer traced. Any OMP call that generates 
 * any kind of work in the new threads after a call to omp_set_num_threads has
 * to hapen inside of a parallel construct(defined by the standard) 
 * and these calls do trigger this check.
 */
#if (defined(PTHREAD_SUPPORT) || defined(CUDA_SUPPORT) || defined(OPENCL_SUPPORT))
# define OMP_CLAUSE_NUM_THREADS_CHANGE(new_num_threads)                       \
	fprintf(stderr, PACKAGE_NAME": The application is explicitly changing the " \
	    "number of OpenMP threads and you are using a tracing library "         \
	    "supporting multiple runtimes. This is currently not supported and "    \
	    "may produce inconsistent results.");
#else
# define OMP_CLAUSE_NUM_THREADS_CHANGE(new_num_threads)           \
{                                                                 \
  __GOMP_save_num_threads_clause( Backend_getNumberOfThreads() ); \
  if (new_num_threads != NO_OMP_CLAUSE_NUM_THREADS)               \
  {                                                               \
    Backend_ChangeNumberOfThreads(new_num_threads);               \
  }                                                               \
}
#endif /* PTHREAD_SUPPORT || CUDA_SUPPORT || OPENCL_SUPPORT */

/**
 * OMP_CLAUSE_NUM_THREADS_RESTORE
 *
 * Restores the current number of threads to the previous value
 * after a parallel with 'num_threads' clause.
 */
#define OMP_CLAUSE_NUM_THREADS_RESTORE()                          \
{                                                                 \
  Backend_ChangeNumberOfThreads(                                  \
    __GOMP_retrieve_num_threads_clause()                          \
  );                                                              \
}


/*
 * Several data helpers to temporarily store the pointers to the real
 * outlined functions or tasks, and the real data arguments, while we
 * inject a fake task that will replace the original one to emit
 * instrumentation events, and then retrieve the original pointers 
 * through the helpers to end up calling the real functions.
 */
typedef struct parallel_helper_t
{
	void (*fn)(void *);
	void *data;
} parallel_helper_t;

#define PARALLEL_HELPER_INITIALIZER(fn,data) { fn, data }

struct helpers_pool_t
{
	struct parallel_helper_t *pool;
	int current_helper;
	int max_helpers;
	pthread_mutex_t mtx;
};

typedef struct task_helper_t
{
	void (*fn)(void *);
	void *data;
	void *buf;
	long long id;
} task_helper_t;


typedef struct taskloop_helper_t
{
	void *magicno;
	void (*fn)(void *);
	long long id;
} taskloop_helper_t;


/**************/
/* PROTOTYPES */
/**************/

void     __GOMP_save_doacross_ncounts (unsigned ncounts);
unsigned __GOMP_retrieve_doacross_ncounts (void);

void     __GOMP_save_num_threads_clause(unsigned num_threads);
unsigned __GOMP_retrieve_num_threads_clause();

void *   __GOMP_new_helper (void (*fn)(void *), void *data);
void     __GOMP_init_helpers (void);

