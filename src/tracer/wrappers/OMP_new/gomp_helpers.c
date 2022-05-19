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

/**
 * Instrumentation of some GOMP calls requires data that is available in a
 * different call. Passing data from one call to the other requires some 
 * helper structures that are implemented here. 
 *
 * Currently we're using two main strategies to pass data between calls.
 *
 * 1) Temporarily store data into the thread-local storage (TLS). We're using 
 * this approach to support the instrumentation of the 'num_threads' clause 
 * in parallel pragmas, as well as the doacross ordered loops. This is nice 
 * as we don't need to use mutexes nor to dynamically allocate memory for the
 * thread, BUT we might find systems where the TLS is small, and more importantly,
 * IF THE RUNTIME SHARES THE SAME PHYSICAL THREAD TO RUN DIFFERENT LOGICAL 
 * THREADS, WE WILL MIX DATA BETWEEN THREADS!!!
 * However, our tests so far indicate that this change only occurs when exiting a 
 * parallel region and opening another, and only if we decrease the number of 
 * threads involved and later increase it. For example, let's consider three 
 * parallel regions with 6, 4, and then 8 threads. Sometimes, the 5th and 6th 
 * threads will change physical threads.
 * However, all these helpers are only needed within a parallel region 
 * and not among them. Therefore, we consider it safe to store them in the TLS.
 *
 * 2) A large pool storage to support the instrumentation of concurrent and
 * nested parallel blocks (see __GOMP_init_helpers). This works as long as 
 * the storage is larger than the amount of concurrent open parallels. 
 * The downside is that this is memory consuming. 
 *
 * Since the pool in (2) is quite large and we've never seen it fill up
 * yet, we could possibly make it generic to stop using the TLS in (1) and 
 * store data for 'num_clause' and 'doacross' in the pool. This way, we get
 * rid of the risk of 1..N physical to logical thread mapping. 
 *
 * Going one step further, the best solution possibly is to implement 
 * a tree indexed by all ancestor thread id's (to support nested), and 
 * store thread data in the leave nodes. We could extend this to also
 * store private tracing buffers in each leave, which would enable real 
 * support for nested parallelism. 
 */

#include "common.h"

#ifdef HAVE_STDDEF_H
# include <stddef.h>
#endif
#include "gomp_helpers.h"
#include "omp_common.h"
#include "pdebug.h"
#include "utils.h"
#include "xalloc.h"

/******************************************************************************\
 *                              DOACROSS HELPER                               *
 ******************************************************************************
 * Instrumentation of doacross is split in several routines. In the *_start   *
 * routines (e.g. GOMP_loop_doacross_static_start) we need to save the        *
 * parameter 'ncounts', which is later needed in GOMP_doacross_wait. Since    *
 * the latter does not receive any parameter that we can replace with a data  *
 * helper, what we do is to store ncounts in the thread TLS. This can be done *
 * because all threads execute the *_start routine. This parameter is saved   *
 * in an array indexed by nesting level, to support nowait clauses and nested *
 * parallelism. This works under the assumption that every single thread can  *
 * be in multiple nesting levels, but different nested threads are always     *
 * executed by different pthreads, otherwise there'd be collisions in the     *
 * array index.                                                               *
 * Once again, our tests indicate that if a change of physical thread ever    *
 * occurs it never happens to the master thread. This change only affects     *
 * threads that are left unused when decreasing the number of threads in a    *
 * parallel region. But there can never be fewer than one thread,             *
 * which is always the master thread.                                         *
\******************************************************************************/

/**
 * __GOMP_doacross_ncounts
 *
 * Saves in TLS the 'ncounts' argument in an array indexed per nesting level
 */
static __thread unsigned *__GOMP_doacross_ncounts = NULL;

/**
 * __GOMP_save_doacross_ncounts
 *
 * Saves the ncounts parameter from the doacross *_start routine in TLS 
 * to be retrieved later in GOMP_doacross_wait. This is stored in an array
 * indexed per current nesting level, in order to allow nowait clauses and
 * nested loops.
 *
 * @param ncounts The argument fom GOMP_loop_doacross_*_start to save in the TLS
 */
void __GOMP_save_doacross_ncounts(unsigned ncounts)
{
	int level = xtr_omp_get_level();

  /* dynamic levels */
  if (XTR_MAX_NESTING_LEVEL == INFINITY) 
  {
    __GOMP_doacross_ncounts = xrealloc(__GOMP_doacross_ncounts, sizeof(unsigned) * level);
  }
  else /* statically allocated levels */
  {
    if (__GOMP_doacross_ncounts == NULL)
    {
      __GOMP_doacross_ncounts = xmalloc(sizeof(unsigned) * XTR_MAX_NESTING_LEVEL);
    }
  }
	__GOMP_doacross_ncounts[level] = ncounts;
}


/**
 * __GOMP_retrieve_doacross_ncounts
 *
 * Retrieve the ncounts parameter from the thread TLS (previously stored in the
 * GOMP_loop_doacross_*_start function). This is stored in an array indexed per
 * current nesting level, in order to allow nowait clauses and nested loops.
 *
 * @return the ncounts parameter previously saved in the TLS.
 */
unsigned __GOMP_retrieve_doacross_ncounts()
{
	int level = xtr_omp_get_level();

	return ((__GOMP_doacross_ncounts != NULL) ? __GOMP_doacross_ncounts[level] : 0);
}


/******************************************************************************\
 *                          NUM_THREADS CLAUSE HELPER                         *
 ******************************************************************************
 * Parallel pragmas may include a 'num_threads' clause that change the number *
 * of threads inside that particular parallel block. We need to update the    *  
 * registered threads in Extrae to allocate new tracing buffers in case the   *
 * number of threads grow, and return to the previous number after leaving    *
 * the parallel.                                                              *
\******************************************************************************/

/**
 * __GOMP_num_threads_clause
 *
 * Saves in TLS the current number of threads before a parallel with 'num_threads' 
 * clause. Since we are not capturing trace data from nested threads other than 
 * the master, there's no need for this to be an array indexed by nesting level 
 * (like doacross helper) or anything more sophisticated. A global TLS variable 
 * allows to save/restore data for the 1st level parallelism, which is enough 
 * for now. This needs to be extended if we want to capture nested threads:
 * currrently this is resilient to this case by checking we are in level 0
 * in order to interact with this variable.
 */
static __thread int __GOMP_num_threads_clause = 0;


void __GOMP_save_num_threads_clause(unsigned num_threads)
{
	__GOMP_num_threads_clause = num_threads;
}


unsigned __GOMP_retrieve_num_threads_clause()
{
	return __GOMP_num_threads_clause;
}


/******************************************************************************\
 *                   SPLIT PARALLEL START/END (GCC 4) HELPER                  * 
 ******************************************************************************
 * The instrumentation of the old GOMP_parallel_*_start() functions from GCC4 *
 * requires to store the pointer to the outlined function, as well as the     *
 * pointer to the function data. This information needs to be accessible      *
 * until the runtime invokes the outlined function that we have wrapped,      *
 * and there, we recover the original outlined pointer and data to call the   *
 * real outlined function. Since the GOMP_parallel_*_start routines return    *
 * immediately after calling the runtime, we need a safe structure that       *
 * supports nested threads and concurrency to store this information.         *
 *                                                                            *
 * Historically, a simple global variable was used, but this approach doesn't *
 * work when there's concurrent parallels ongoing. Later, this was changed    *
 * into a global array indexed by nesting level, but this doesn't work either *
 * because when there's nested paralellism different threads share the same   *
 * omp_get_thread_num identifier, hence colliding into the same array index.  *
 *                                                                            *
 * The current solution consists in having a large circular pool of data      *
 * helpers. For every new parallel block, the next helper of the pool is      *
 * acquired (under a mutex) and data is stored there. The helpers are never   *
 * free'd, because we can't safely pass a pointer to the exact helper neither *
 * to the GOMP_parallel_end, nor the wrapped outlined to delete the structure *
 * once is no longer needed. Instead, we start reusing the helpers when the   *
 * structure is full.                                                         *
 *                                                                            *
 * This works because the structure is large enough and typically there   *
 * won't be so many open parallel blocks concurrently. However, you can make  *
 * it crash by opening enough parallel blocks, and this solution is overkill  *
 * in memory consumption.                                                     *
 *                                                                            *
 * The size of the pool can be increased from DEFAULT_OPENMP_HELPERS to the   *
 * value set by the environment variable: EXTRAE_OPENMP_HELPERS.              *
\******************************************************************************/
 
/**
 * __GOMP_helpers 
 *
 * Holds a pool of all active data helpers to pass data from
 * GOMP_parallel_*_start routines to the outlined function.
 */
struct helpers_pool_t __GOMP_helpers = { NULL, 0, 0, PTHREAD_MUTEX_INITIALIZER };


/**
 * __GOMP_new_helper
 *
 * Returns a new data helper in the __GOMP_helpers pool. When we run out of
 * helpers, we start reusing. As long as the corresponding parallel region already
 * finished, there will be no problems. But if that parallel region is still
 * active, we'll be corrupting the pointers to fn and data. A warning is shown
 * the first time the helpers are reused, and if the application is corrupted,
 * we need to increase the number of helpers setting EXTRAE_OPENMP_HELPERS.
 *
 * @param fn The pointer to the real outlined function or task.
 * @param data The pointer to the data passed to the real routine.
 *
 * @return The pointer to the pool slot where the data helper is stored.
 */
void *__GOMP_new_helper(void (*fn)(void *), void *data)
{
  int idx = 0;
  void *helper_ptr = NULL;
  static int warning_displayed = 0;

  pthread_mutex_lock(&__GOMP_helpers.mtx);

  /* Pick a slot in the pool */
  idx = __GOMP_helpers.current_helper;
  __GOMP_helpers.current_helper = (__GOMP_helpers.current_helper + 1) % __GOMP_helpers.max_helpers;

  pthread_mutex_unlock(&__GOMP_helpers.mtx);

  /* Save the pointers to fn and data */
  __GOMP_helpers.pool[idx].fn = fn;
  __GOMP_helpers.pool[idx].data = data;

  /* Return the pointer to the slot for this data helper */
  helper_ptr = &(__GOMP_helpers.pool[idx]);

#if defined(DEBUG)
  THREAD_DBG("__GOMP_new_helper: Registering helper #%d helper_ptr=%p fn=%p data=%p\n", idx, helper_ptr, fn, data);
#endif

  if (__GOMP_helpers.current_helper < idx)
  {
    /*
     * Display a warning (once) when we start reusing pool slots for helpers.
     * Could appear more than once in the event that two concurrent threads
     * evaluate warning_displayed simultaneously, but this is extremely
     * unlikely and it's better not to use a mutex here to minimize overhead.
     */
    if (!warning_displayed)
    {
      THREAD_WARN("I have run out of allocations for data helpers. If the application starts crashing or producing wrong results, please try increasing %s over %d until this warning disappears\n", ENV_VAR_EXTRAE_OPENMP_HELPERS, __GOMP_helpers.max_helpers);
      warning_displayed = 1;
    }
  }

  return helper_ptr;
}


/**
 * __GOMP_init_helpers
 *
 * Initializes the pool of data helpers __GOMP_helpers.
 *
 */
void __GOMP_init_helpers()
{
  int num_helpers = 0;
  char *env_helpers = NULL;

  pthread_mutex_lock(&__GOMP_helpers.mtx);

	/*
	 * If the environment variable ENV_VAR_EXTRAE_OPENMP_HELPERS is defined, this
	 * will be the size of the pool. Otherwise, DEFAULT_OPENMP_HELPERS is used.
	 */
	env_helpers = getenv(ENV_VAR_EXTRAE_OPENMP_HELPERS);
	if (env_helpers != NULL)
	{
		num_helpers = atoi(env_helpers);
	}
	if (num_helpers <= 0)
	{
		num_helpers = DEFAULT_OPENMP_HELPERS;
	}

#if defined(DEBUG)
	THREAD_DBG("Allocating %d data helpers\n", num_helpers);
#endif

	__GOMP_helpers.current_helper = 0;
	__GOMP_helpers.max_helpers = num_helpers;
	__GOMP_helpers.pool = (struct parallel_helper_t *)malloc(sizeof(struct parallel_helper_t) * num_helpers);
	if (__GOMP_helpers.pool == NULL)
	{
		THREAD_ERROR("Invalid initialization of '__GOMP_helpers.pool' (%d helpers)\n", num_helpers);
		exit(-1);
	}

	pthread_mutex_unlock(&__GOMP_helpers.mtx);
}

