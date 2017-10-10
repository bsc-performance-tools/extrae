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
                                                                                
#define CHECK_NESTING_LEVEL(level)                                                               \
{                                                                                                \
	  if ((level < 0) || (level > MAX_NESTING_LEVEL))                                              \
	  {                                                                                            \
			    fprintf (stderr, PACKAGE_NAME": ERROR! Current nesting level (%d) "                    \
					                      "is out of bounds (maximum supported is %d). Please recompile "  \
					                      PACKAGE_NAME" increasing the value of MAX_NESTING_LEVEL in "     \
					                      "src/tracer/wrappers/OMP/omp-common.h\n",                        \
					                      level, MAX_NESTING_LEVEL);                                       \
			    exit (0);                                                                              \
			    }                                                                                      \
}                                                                               
                                                                                
#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)                

extern int omp_get_max_threads(void);                                                  
extern int omp_get_level(void);                                                        
extern int omp_get_ancestor_thread_num(int level);                                     

#define THREAD_LEVEL_LBL " [THD:%d LVL:%d] "                                    
#define THREAD_LEVEL_VAR THREADID, omp_get_level()                              

/*
 * This helper structure is used to pass information 
 * from master threads at the start of the parallel region to
 * worker threads in a deeper nesting level.
 */
struct thread_helper_t
{
	  void *par_uf;
};

struct thread_helper_t * get_thread_helper();
struct thread_helper_t * get_parent_thread_helper();

void Extrae_OpenMP_init(int me);

void allocate_nested_helpers();

#endif /* OMP_COMMON_H_ */
