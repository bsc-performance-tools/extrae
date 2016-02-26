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

#include "common.h"
#include "openshmem_events.h"

char *openshmem_events_labels[COUNT_OPENSHMEM_EVENTS] = {
  "start_pes",
  "shmem_my_pe",
  "_my_pe",
  "shmem_n_pes",
  "_num_pes",
  "shmem_pe_accessible",
  "shmem_addr_accessible",
  "shmem_ptr",
  "shmalloc",
  "shfree",
  "shrealloc",
  "shmemalign",
  "shmem_double_put",
  "shmem_float_put",
  "shmem_int_put",
  "shmem_long_put",
  "shmem_longdouble_put",
  "shmem_longlong_put",
  "shmem_put32",
  "shmem_put64",
  "shmem_put128",
  "shmem_putmem",
  "shmem_short_put",
  "shmem_char_p",
  "shmem_short_p",
  "shmem_int_p",
  "shmem_long_p",
  "shmem_longlong_p",
  "shmem_float_p",
  "shmem_double_p",
  "shmem_longdouble_p",
  "shmem_double_iput",
  "shmem_float_iput",
  "shmem_int_iput",
  "shmem_iput32",
  "shmem_iput64",
  "shmem_iput128",
  "shmem_long_iput",
  "shmem_longdouble_iput",
  "shmem_longlong_iput",
  "shmem_short_iput",
  "shmem_double_get",
  "shmem_float_get",
  "shmem_get32",
  "shmem_get64",
  "shmem_get128",
  "shmem_getmem",
  "shmem_int_get",
  "shmem_long_get",
  "shmem_longdouble_get",
  "shmem_longlong_get",
  "shmem_short_get",
  "shmem_char_g",
  "shmem_short_g",
  "shmem_int_g",
  "shmem_long_g",
  "shmem_longlong_g",
  "shmem_float_g",
  "shmem_double_g",
  "shmem_longdouble_g",
  "shmem_double_iget",
  "shmem_float_iget",
  "shmem_iget32",
  "shmem_iget64",
  "shmem_iget128",
  "shmem_int_iget",
  "shmem_long_iget",
  "shmem_longdouble_iget",
  "shmem_longlong_iget",
  "shmem_short_iget",
  "shmem_int_add",
  "shmem_long_add",
  "shmem_longlong_add",
  "shmem_int_cswap",
  "shmem_long_cswap",
  "shmem_longlong_cswap",
  "shmem_double_swap",
  "shmem_float_swap",
  "shmem_int_swap",
  "shmem_long_swap",
  "shmem_longlong_swap",
  "shmem_swap",
  "shmem_int_finc",
  "shmem_long_finc",
  "shmem_longlong_finc",
  "shmem_int_inc",
  "shmem_long_inc",
  "shmem_longlong_inc",
  "shmem_int_fadd",
  "shmem_long_fadd",
  "shmem_longlong_fadd",
  "shmem_barrier_all",
  "shmem_barrier",
  "shmem_broadcast32",
  "shmem_broadcast64",
  "shmem_collect32",
  "shmem_collect64",
  "shmem_fcollect32",
  "shmem_fcollect64",
  "shmem_int_and_to_all",
  "shmem_long_and_to_all",
  "shmem_longlong_and_to_all",
  "shmem_short_and_to_all",
  "shmem_double_max_to_all",
  "shmem_float_max_to_all",
  "shmem_int_max_to_all",
  "shmem_long_max_to_all",
  "shmem_longdouble_max_to_all",
  "shmem_longlong_max_to_all",
  "shmem_short_max_to_all",
  "shmem_double_min_to_all",
  "shmem_int_wait",
  "shmem_int_wait_until",
  "shmem_long_wait",
  "shmem_long_wait_until",
  "shmem_longlong_wait",
  "shmem_longlong_wait_until",
  "shmem_short_wait",
  "shmem_short_wait_until",
  "shmem_wait",
  "shmem_wait_until",
  "shmem_fence",
  "shmem_quiet",
  "shmem_clear_lock",
  "shmem_set_lock",
  "shmem_test_lock",
  "shmem_clear_cache_inv",
  "shmem_set_cache_inv",
  "shmem_clear_cache_line_inv",
  "shmem_set_cache_line_inv",
  "shmem_udcflush",
  "shmem_udcflush_line"
};

char *GetOPENSHMEMLabel( int openshmem_event )
{
  return openshmem_events_labels[ openshmem_event ];
}
