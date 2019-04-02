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

#ifndef OMP_COMMON_H_
#define OMP_COMMON_H_

#include <omp.h>

#include "config.h"
#include "ompt-wrapper.h"

#define INSTRUMENT_OMP_WRAPPER(func) ((func != NULL) && (EXTRAE_INITIALIZED()) && (EXTRAE_ON()))

#if defined(OMPT_SUPPORT)
#define TRACE(func) (INSTRUMENT_OMP_WRAPPER(func) && (!ompt_enabled))
#else
#define TRACE(func) (INSTRUMENT_OMP_WRAPPER(func))
#endif

#define ENV_VAR_EXTRAE_OPENMP_HELPERS "EXTRAE_OPENMP_HELPERS"
#define DEFAULT_OPENMP_HELPERS        100000

#define MAX_NESTING_LEVEL             64
#define MAX_DOACROSS_ARGS             64

#define CHECK_NESTING_LEVEL(level)                                             \
{                                                                              \
	if ((level < 0) || (level > MAX_NESTING_LEVEL))                            \
	{                                                                          \
		fprintf(stderr, PACKAGE_NAME": ERROR! Current nesting level (%d) "     \
		    "is out of bounds (maximum supported is %d). Please recompile "    \
		    PACKAGE_NAME" increasing the value of MAX_NESTING_LEVEL in "       \
		    "src/tracer/wrappers/OMP/omp-common.h\n",                          \
		    level, MAX_NESTING_LEVEL);                                         \
		exit(0);                                                               \
	}                                                                          \
}

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

/*
 * Changing the number of threads with the num_threads clause within a parallel
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
 */
#if (defined(PTHREAD_SUPPORT) || defined(CUDA_SUPPORT) || defined(OPENCL_SUPPORT))
# define OMP_CLAUSE_NUM_THREADS_SAVE(var)
# define OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads)                            \
	fprintf(stderr, PACKAGE_NAME": The application is explicitly changing the "\
	    "number of OpenMP threads and you are using a tracing library "        \
	    "supporting multiple runtimes. This is currently not supported and "   \
	    "may produce inconsistent results.");
#else
# define OMP_CLAUSE_NUM_THREADS_SAVE(var) unsigned var = omp_get_num_threads();
# define OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads)                            \
	Backend_ChangeNumberOfThreads(num_threads);
#endif

extern int omp_get_max_threads(void);
extern int omp_get_level(void);
extern int omp_get_ancestor_thread_num(int level);

#define THREAD_LEVEL_LBL " [THD:%d LVL:%d] "
#define THREAD_LEVEL_VAR THREADID, omp_get_level()

void Extrae_OpenMP_init(int me);

#endif /* OMP_COMMON_H_ */
