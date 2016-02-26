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
#include "openshmem_probes.h"
#include "openshmem_trace_macros.h"
#include "taskid.h"
#include "wrapper.h"

void PROBE_start_pes_ENTRY (int npes)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, START_PES_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_start_pes_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, START_PES_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_my_pe_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_MY_PE_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_my_pe_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_MY_PE_EV, EVT_END, EMPTY);
  }
}

void PROBE__my_pe_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, _MY_PE_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE__my_pe_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, _MY_PE_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_n_pes_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_N_PES_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_n_pes_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_N_PES_EV, EVT_END, EMPTY);
  }
}

void PROBE__num_pes_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, _NUM_PES_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE__num_pes_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, _NUM_PES_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_pe_accessible_ENTRY (int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_PE_ACCESSIBLE_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_pe_accessible_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_PE_ACCESSIBLE_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_addr_accessible_ENTRY (void *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_ADDR_ACCESSIBLE_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_addr_accessible_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_ADDR_ACCESSIBLE_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_ptr_ENTRY (void *target, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_PTR_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_ptr_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_PTR_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmalloc_ENTRY (size_t size)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMALLOC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmalloc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMALLOC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shfree_ENTRY (void *ptr)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHFREE_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shfree_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHFREE_EV, EVT_END, EMPTY);
  }
}

void PROBE_shrealloc_ENTRY (void *ptr, size_t size)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHREALLOC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shrealloc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHREALLOC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmemalign_ENTRY (size_t alignment, size_t size)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEMALIGN_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmemalign_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEMALIGN_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_put_ENTRY (double *target, const double *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_PUT_EV, EVT_BEGIN, (len * sizeof(double)));
  }
}

void PROBE_shmem_double_put_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_PUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_put_ENTRY (float *target, const float *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_PUT_EV, EVT_BEGIN, (len * sizeof(float)));
  }
}

void PROBE_shmem_float_put_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_PUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_put_ENTRY (int *target, const int *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_PUT_EV, EVT_BEGIN, (len * sizeof(int)));
  }
}

void PROBE_shmem_int_put_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_PUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_put_ENTRY (long *target, const long *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_PUT_EV, EVT_BEGIN, (len * sizeof(long)));
  }
}

void PROBE_shmem_long_put_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_PUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longdouble_put_ENTRY (long double *target, const long double *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGDOUBLE_PUT_EV, EVT_BEGIN, (len * sizeof(long double)));
  }
}

void PROBE_shmem_longdouble_put_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGDOUBLE_PUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_put_ENTRY (long long *target, const long long *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_PUT_EV, EVT_BEGIN, (len * sizeof(long long)));
  }
}

void PROBE_shmem_longlong_put_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_PUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_put32_ENTRY (void *target, const void *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_PUT32_EV, EVT_BEGIN, (len * 4));
  }
}

void PROBE_shmem_put32_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_PUT32_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_put64_ENTRY (void *target, const void *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_PUT64_EV, EVT_BEGIN, (len * 8));
  }
}

void PROBE_shmem_put64_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_PUT64_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_put128_ENTRY (void *target, const void *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_PUT128_EV, EVT_BEGIN, (len * 16));
  }
}

void PROBE_shmem_put128_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_PUT128_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_putmem_ENTRY (void *target, const void *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_PUTMEM_EV, EVT_BEGIN, len);
  }
}

void PROBE_shmem_putmem_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_PUTMEM_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_put_ENTRY (short *target, const short *source, size_t len, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_PUT_EV, EVT_BEGIN, (len * sizeof(short)));
  }
}

void PROBE_shmem_short_put_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_PUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_char_p_ENTRY (char *addr, char value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_CHAR_P_EV, EVT_BEGIN, sizeof(char));
  }
}

void PROBE_shmem_char_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_CHAR_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_p_ENTRY (short *addr, short value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_P_EV, EVT_BEGIN, sizeof(short));
  }
}

void PROBE_shmem_short_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_p_ENTRY (int *addr, int value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_P_EV, EVT_BEGIN, sizeof(int));
  }
}

void PROBE_shmem_int_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_p_ENTRY (long *addr, long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_P_EV, EVT_BEGIN, sizeof(long));
  }
}

void PROBE_shmem_long_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_p_ENTRY (long long *addr, long long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_P_EV, EVT_BEGIN, sizeof(long long));
  }
}

void PROBE_shmem_longlong_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_p_ENTRY (float *addr, float value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_P_EV, EVT_BEGIN, sizeof(float));
  }
}

void PROBE_shmem_float_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_p_ENTRY (double *addr, double value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_P_EV, EVT_BEGIN, sizeof(double));
  }
}

void PROBE_shmem_double_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longdouble_p_ENTRY (long double *addr, long double value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGDOUBLE_P_EV, EVT_BEGIN, sizeof(long double));
  }
}

void PROBE_shmem_longdouble_p_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGDOUBLE_P_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_iput_ENTRY (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_IPUT_EV, EVT_BEGIN, (nelems * sizeof(double)));
  }
}

void PROBE_shmem_double_iput_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_IPUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_iput_ENTRY (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_IPUT_EV, EVT_BEGIN, (nelems * sizeof(float)));
  }
}

void PROBE_shmem_float_iput_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_IPUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_iput_ENTRY (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_IPUT_EV, EVT_BEGIN, (nelems * sizeof(int)));
  }
}

void PROBE_shmem_int_iput_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_IPUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_iput32_ENTRY (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_IPUT32_EV, EVT_BEGIN, (nelems * 4));
  }
}

void PROBE_shmem_iput32_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_IPUT32_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_iput64_ENTRY (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_IPUT64_EV, EVT_BEGIN, (nelems * 8));
  }
}

void PROBE_shmem_iput64_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_IPUT64_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_iput128_ENTRY (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_IPUT128_EV, EVT_BEGIN, (nelems * 16));
  }
}

void PROBE_shmem_iput128_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_IPUT128_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_iput_ENTRY (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_IPUT_EV, EVT_BEGIN, (nelems * sizeof(long)));
  }
}

void PROBE_shmem_long_iput_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_IPUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longdouble_iput_ENTRY (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGDOUBLE_IPUT_EV, EVT_BEGIN, (nelems * sizeof(long double)));
  }
}

void PROBE_shmem_longdouble_iput_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGDOUBLE_IPUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_iput_ENTRY (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_IPUT_EV, EVT_BEGIN, (nelems * sizeof(long long)));
  }
}

void PROBE_shmem_longlong_iput_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_IPUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_iput_ENTRY (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_IPUT_EV, EVT_BEGIN, (nelems * sizeof(short)));
  }
}

void PROBE_shmem_short_iput_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_IPUT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_get_ENTRY (double *target, const double *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_GET_EV, EVT_BEGIN, (nelems * sizeof(double)));
  }
}

void PROBE_shmem_double_get_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_GET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_get_ENTRY (float *target, const float *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_GET_EV, EVT_BEGIN, (nelems * sizeof(float)));
  }
}

void PROBE_shmem_float_get_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_GET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_get32_ENTRY (void *target, const void *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_GET32_EV, EVT_BEGIN, (nelems * 4));
  }
}

void PROBE_shmem_get32_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_GET32_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_get64_ENTRY (void *target, const void *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_GET64_EV, EVT_BEGIN, (nelems * 8));
  }
}

void PROBE_shmem_get64_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_GET64_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_get128_ENTRY (void *target, const void *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_GET128_EV, EVT_BEGIN, (nelems * 16));
  }
}

void PROBE_shmem_get128_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_GET128_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_getmem_ENTRY (void *target, const void *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_GETMEM_EV, EVT_BEGIN, nelems);
  }
}

void PROBE_shmem_getmem_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_GETMEM_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_get_ENTRY (int *target, const int *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_GET_EV, EVT_BEGIN, (nelems * sizeof(int)));
  }
}

void PROBE_shmem_int_get_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_GET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_get_ENTRY (long *target, const long *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_GET_EV, EVT_BEGIN, (nelems * sizeof(long)));
  }
}

void PROBE_shmem_long_get_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_GET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longdouble_get_ENTRY (long double *target, const long double *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGDOUBLE_GET_EV, EVT_BEGIN, (nelems * sizeof(long double)));
  }
}

void PROBE_shmem_longdouble_get_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGDOUBLE_GET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_get_ENTRY (long long *target, const long long *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_GET_EV, EVT_BEGIN, (nelems * sizeof(long long)));
  }
}

void PROBE_shmem_longlong_get_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_GET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_get_ENTRY (short *target, const short *source, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_GET_EV, EVT_BEGIN, (nelems * sizeof(short)));
  }
}

void PROBE_shmem_short_get_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_GET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_char_g_ENTRY (char *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_CHAR_G_EV, EVT_BEGIN, sizeof(char));
  }
}

void PROBE_shmem_char_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_CHAR_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_g_ENTRY (short *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_G_EV, EVT_BEGIN, sizeof(short));
  }
}

void PROBE_shmem_short_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_g_ENTRY (int *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_G_EV, EVT_BEGIN, sizeof(int));
  }
}

void PROBE_shmem_int_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_g_ENTRY (long *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_G_EV, EVT_BEGIN, sizeof(long));
  }
}

void PROBE_shmem_long_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_g_ENTRY (long long *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_G_EV, EVT_BEGIN, sizeof(long long));
  }
}

void PROBE_shmem_longlong_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_g_ENTRY (float *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_G_EV, EVT_BEGIN, sizeof(float));
  }
}

void PROBE_shmem_float_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_g_ENTRY (double *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_G_EV, EVT_BEGIN, sizeof(double));
  }
}

void PROBE_shmem_double_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longdouble_g_ENTRY (long double *addr, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGDOUBLE_G_EV, EVT_BEGIN, sizeof(long double));
  }
}

void PROBE_shmem_longdouble_g_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGDOUBLE_G_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_iget_ENTRY (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_IGET_EV, EVT_BEGIN, (nelems * sizeof(double)));
  }
}

void PROBE_shmem_double_iget_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_IGET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_iget_ENTRY (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_IGET_EV, EVT_BEGIN, (nelems * sizeof(float)));
  }
}

void PROBE_shmem_float_iget_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_IGET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_iget32_ENTRY (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_IGET32_EV, EVT_BEGIN, (nelems * 4));
  }
}

void PROBE_shmem_iget32_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_IGET32_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_iget64_ENTRY (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_IGET64_EV, EVT_BEGIN, (nelems * 8));
  }
}

void PROBE_shmem_iget64_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_IGET64_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_iget128_ENTRY (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_IGET128_EV, EVT_BEGIN, (nelems * 16));
  }
}

void PROBE_shmem_iget128_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_IGET128_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_iget_ENTRY (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_IGET_EV, EVT_BEGIN, (nelems * sizeof(int)));
  }
}

void PROBE_shmem_int_iget_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_IGET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_iget_ENTRY (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_IGET_EV, EVT_BEGIN, (nelems * sizeof(long)));
  }
}

void PROBE_shmem_long_iget_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_IGET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longdouble_iget_ENTRY (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGDOUBLE_IGET_EV, EVT_BEGIN, (nelems * sizeof(long double)));
  }
}

void PROBE_shmem_longdouble_iget_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGDOUBLE_IGET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_iget_ENTRY (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_IGET_EV, EVT_BEGIN, (nelems * sizeof(long long)));
  }
}

void PROBE_shmem_longlong_iget_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_IGET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_iget_ENTRY (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_IGET_EV, EVT_BEGIN, (nelems * sizeof(short)));
  }
}

void PROBE_shmem_short_iget_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_IGET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_add_ENTRY (int *target, int value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_ADD_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_add_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_ADD_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_add_ENTRY (long *target, long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_ADD_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_add_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_ADD_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_add_ENTRY (long long *target, long long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_ADD_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_add_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_ADD_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_cswap_ENTRY (int *target, int cond, int value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_CSWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_cswap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_CSWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_cswap_ENTRY (long *target, long cond, long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_CSWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_cswap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_CSWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_cswap_ENTRY (long long *target, long long cond, long long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_CSWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_cswap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_CSWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_swap_ENTRY (double *target, double value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_SWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_double_swap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_SWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_swap_ENTRY (float *target, float value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_SWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_float_swap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_SWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_swap_ENTRY (int *target, int value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_SWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_swap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_SWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_swap_ENTRY (long *target, long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_SWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_swap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_SWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_swap_ENTRY (long long *target, long long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_SWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_swap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_SWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_swap_ENTRY (long *target, long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SWAP_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_swap_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SWAP_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_finc_ENTRY (int *target, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_FINC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_finc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_FINC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_finc_ENTRY (long *target, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_FINC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_finc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_FINC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_finc_ENTRY (long long *target, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_FINC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_finc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_FINC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_inc_ENTRY (int *target, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_INC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_inc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_INC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_inc_ENTRY (long *target, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_INC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_inc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_INC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_inc_ENTRY (long long *target, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_INC_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_inc_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_INC_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_fadd_ENTRY (int *target, int value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_FADD_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_fadd_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_FADD_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_fadd_ENTRY (long *target, long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_FADD_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_fadd_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_FADD_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_fadd_ENTRY (long long *target, long long value, int pe)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_FADD_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_fadd_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_FADD_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_barrier_all_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_BARRIER_ALL_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_barrier_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_BARRIER_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_barrier_ENTRY (int PE_start, int logPE_stride, int PE_size, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_BARRIER_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_barrier_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_BARRIER_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_broadcast32_ENTRY (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    int bytes = ((TASKID == PE_root) ? (nlong * 4 * (PE_size / (1 << logPE_stride))) : 0);
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_BROADCAST32_EV, EVT_BEGIN, bytes);
  }
}

void PROBE_shmem_broadcast32_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_BROADCAST32_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_broadcast64_ENTRY (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    int bytes = ((TASKID == PE_root) ? (nlong * 8 * (PE_size / (1 << logPE_stride))) : 0);
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_BROADCAST64_EV, EVT_BEGIN, bytes);
  }
}

void PROBE_shmem_broadcast64_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_BROADCAST64_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_collect32_ENTRY (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_COLLECT32_EV, EVT_BEGIN, (nelems * 4 * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_collect32_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_COLLECT32_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_collect64_ENTRY (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_COLLECT64_EV, EVT_BEGIN, (nelems * 8 * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_collect64_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_COLLECT64_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_fcollect32_ENTRY (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FCOLLECT32_EV, EVT_BEGIN, (nelems * 4 * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_fcollect32_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FCOLLECT32_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_fcollect64_ENTRY (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FCOLLECT64_EV, EVT_BEGIN, (nelems * 8 * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_fcollect64_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FCOLLECT64_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_and_to_all_ENTRY (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_AND_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(int) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_int_and_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_AND_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_and_to_all_ENTRY (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_AND_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(long) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_long_and_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_AND_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_and_to_all_ENTRY (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_AND_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(long long) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_longlong_and_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_AND_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_and_to_all_ENTRY (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_AND_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(short) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_short_and_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_AND_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_max_to_all_ENTRY (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_MAX_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(double) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_double_max_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_MAX_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_float_max_to_all_ENTRY (float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FLOAT_MAX_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(float) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_float_max_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FLOAT_MAX_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_max_to_all_ENTRY (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_MAX_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(int) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_int_max_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_MAX_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_max_to_all_ENTRY (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_MAX_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(long) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_long_max_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_MAX_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longdouble_max_to_all_ENTRY (long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGDOUBLE_MAX_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(long double) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_longdouble_max_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGDOUBLE_MAX_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_max_to_all_ENTRY (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_MAX_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(long long) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_longlong_max_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_MAX_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_max_to_all_ENTRY (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_MAX_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(short) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_short_max_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_MAX_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_double_min_to_all_ENTRY (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_DOUBLE_MIN_TO_ALL_EV, EVT_BEGIN, (nreduce * sizeof(double) * (PE_size / (1 << logPE_stride))));
  }
}

void PROBE_shmem_double_min_to_all_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_DOUBLE_MIN_TO_ALL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_wait_ENTRY (int *ivar, int cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_WAIT_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_wait_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_WAIT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_int_wait_until_ENTRY (int *ivar, int cmp, int cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_INT_WAIT_UNTIL_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_int_wait_until_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_INT_WAIT_UNTIL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_wait_ENTRY (long *ivar, long cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_WAIT_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_wait_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_WAIT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_long_wait_until_ENTRY (long *ivar, int cmp, long cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONG_WAIT_UNTIL_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_long_wait_until_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONG_WAIT_UNTIL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_wait_ENTRY (long long *ivar, long long cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_WAIT_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_wait_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_WAIT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_longlong_wait_until_ENTRY (long long *ivar, int cmp, long long cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_LONGLONG_WAIT_UNTIL_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_longlong_wait_until_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_LONGLONG_WAIT_UNTIL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_wait_ENTRY (short *ivar, short cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_WAIT_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_short_wait_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_WAIT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_short_wait_until_ENTRY (short *ivar, int cmp, short cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SHORT_WAIT_UNTIL_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_short_wait_until_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SHORT_WAIT_UNTIL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_wait_ENTRY (long *ivar, long cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_WAIT_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_wait_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_WAIT_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_wait_until_ENTRY (long *ivar, int cmp, long cmp_value)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_WAIT_UNTIL_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_wait_until_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_WAIT_UNTIL_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_fence_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_FENCE_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_fence_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_FENCE_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_quiet_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_QUIET_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_quiet_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_QUIET_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_clear_lock_ENTRY (long *lock)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_CLEAR_LOCK_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_clear_lock_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_CLEAR_LOCK_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_set_lock_ENTRY (long *lock)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SET_LOCK_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_set_lock_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SET_LOCK_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_test_lock_ENTRY (long *lock)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_TEST_LOCK_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_test_lock_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_TEST_LOCK_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_clear_cache_inv_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_CLEAR_CACHE_INV_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_clear_cache_inv_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_CLEAR_CACHE_INV_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_set_cache_inv_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SET_CACHE_INV_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_set_cache_inv_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SET_CACHE_INV_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_clear_cache_line_inv_ENTRY (void *target)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_CLEAR_CACHE_LINE_INV_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_clear_cache_line_inv_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_CLEAR_CACHE_LINE_INV_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_set_cache_line_inv_ENTRY (void *target)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_SET_CACHE_LINE_INV_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_set_cache_line_inv_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_SET_CACHE_LINE_INV_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_udcflush_ENTRY (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_UDCFLUSH_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_udcflush_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_UDCFLUSH_EV, EVT_END, EMPTY);
  }
}

void PROBE_shmem_udcflush_line_ENTRY (void *target)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(LAST_READ_TIME, SHMEM_UDCFLUSH_LINE_EV, EVT_BEGIN, EMPTY);
  }
}

void PROBE_shmem_udcflush_line_EXIT (void)
{
  DEBUG_PROBES();
  if (EXTRAE_ON())
  {
    TRACE_OPENSHMEM_EVENT_AND_COUNTERS(TIME, SHMEM_UDCFLUSH_LINE_EV, EVT_END, EMPTY);
  }
}

