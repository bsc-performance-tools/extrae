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
#include <omp.h>
#include "wrap_macros.h"
#include "nested_id.h"
#include "symptr.h"

extern int xtr_OMP_tracing_config;

#define OMP_ENABLED                               1 << 0
#define OMP_COUNTERS_ENABLED                      1 << 1
#define OMP_LOCKS_ENABLED                         1 << 2
#define OMP_TASKLOOP_ENABLED                      1 << 3
#define OMP_MASTER_IN_NESTED                      1 << 4
#define OMP_ANNOTATE_CPU                          1 << 5
#define OMP_TASK_DEPENDENCY_LINE_ENABLED          1 << 6
#define OMP_TASKLOOP_DEPENDENCY_LINE_ENABLED      1 << 7

#define xtr_OMP_config_enable(flags)  xtr_OMP_tracing_config |= (flags)
#define xtr_OMP_config_disable(flags) xtr_OMP_tracing_config &= ~(flags)
#define xtr_OMP_config_toggle(flags)  xtr_OMP_tracing_config ^= (flags)
#define xtr_OMP_config_enable_conditional(flags, bool) \
{                                                      \
  if (bool)                                            \
    xtr_OMP_config_enable(flags);                      \
  else                                                 \
    xtr_OMP_config_disable(flags);                     \
}
#define xtr_OMP_check_config(flags) ((xtr_OMP_tracing_config & (flags)) == (flags))

#define TRACE_MASTER_IN_NESTED (xtr_OMP_check_config(OMP_MASTER_IN_NESTED) && I_am_master_in_nested())

#define NOT_IN_PARALLEL        (omp_get_level() == 0)
#define IN_PARALLEL(level)     (omp_get_level() == level)
#define IN_NESTED              (omp_get_level() > 1)
#define NOT_IN_NESTED         ((omp_get_level() <= 1) || (TRACE_MASTER_IN_NESTED))

#define ENV_VAR_EXTRAE_OPENMP_HELPERS "EXTRAE_OPENMP_HELPERS"
#define DEFAULT_OPENMP_HELPERS        100000

#define MAX_DOACROSS_ARGS 64

/**
 * OMP_HOOK_INIT
 *
 * Search the address of the wrapped symbol 'real_sym' through dlsym
 * and stores it in a function pointer variable that is named as
 * the concatenation of the wrapped symbol token and a suffix (_real),
 * e.g. (GOMP_barrier_real).
 *
 * @param real_sym Token of the wrapped symbol
 * @return a counter that increments by 1 each time the address resolution succeeds
 */
#define OMP_HOOK_INIT(real_sym, count)                         \
{                                                              \
  REAL_SYMBOL_PTR(real_sym) = XTR_FIND_SYMBOL(#real_sym);       \
                                                               \
  if (REAL_SYMBOL_PTR(real_sym) != NULL)                       \
  {                                                            \
    count ++;                                                  \
  }                                                            \
}

void Extrae_OpenMP_init(int me);

