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

#include <shmem.h>

#ifndef __OPENSHMEM__WRAPPERS_H__
#define __OPENSHMEM__WRAPPERS_H__

void start_pes (int npes);
int shmem_my_pe (void);
int _my_pe (void);
int shmem_n_pes (void);
int _num_pes (void);
int shmem_pe_accessible (int pe);
int shmem_addr_accessible (void *addr, int pe);
void * shmem_ptr (void *target, int pe);
void * shmalloc (size_t size);
void shfree (void *ptr);
void * shrealloc (void *ptr, size_t size);
void * shmemalign (size_t alignment, size_t size);
void shmem_double_put (double *target, const double *source, size_t len, int pe);
void shmem_float_put (float *target, const float *source, size_t len, int pe);
void shmem_int_put (int *target, const int *source, size_t len, int pe);
void shmem_long_put (long *target, const long *source, size_t len, int pe);
void shmem_longdouble_put (long double *target, const long double *source, size_t len,int pe);
void shmem_longlong_put (long long *target, const long long *source, size_t len, int pe);
void shmem_put32 (void *target, const void *source, size_t len, int pe);
void shmem_put64 (void *target, const void *source, size_t len, int pe);
void shmem_put128 (void *target, const void *source, size_t len, int pe);
void shmem_putmem (void *target, const void *source, size_t len, int pe);
void shmem_short_put (short*target, const short*source, size_t len, int pe);
void shmem_char_p (char *addr, char value, int pe);
void shmem_short_p (short *addr, short value, int pe);
void shmem_int_p (int *addr, int value, int pe);
void shmem_long_p (long *addr, long value, int pe);
void shmem_longlong_p (long long *addr, long long value, int pe);
void shmem_float_p (float *addr, float value, int pe);
void shmem_double_p (double *addr, double value, int pe);
void shmem_longdouble_p (long double *addr, long double value, int pe);
void shmem_double_iput (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_float_iput (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_int_iput (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_iput32 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_iput64 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_iput128 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_long_iput (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_longdouble_iput (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_longlong_iput (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_short_iput (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_double_get (double *target, const double *source, size_t nelems, int pe);
void shmem_float_get (float *target, const float *source, size_t nelems, int pe);
void shmem_get32 (void *target, const void *source, size_t nelems, int pe);
void shmem_get64 (void *target, const void *source, size_t nelems, int pe);
void shmem_get128 (void *target, const void *source, size_t nelems, int pe);
void shmem_getmem (void *target, const void *source, size_t nelems, int pe);
void shmem_int_get (int *target, const int *source, size_t nelems, int pe);
void shmem_long_get (long *target, const long *source, size_t nelems, int pe);
void shmem_longdouble_get (long double *target, const long double *source, size_t nelems, int pe);
void shmem_longlong_get (long long *target, const long long *source, size_t nelems, int pe);
void shmem_short_get (short *target, const short *source, size_t nelems, int pe);
char shmem_char_g (char *addr, int pe);
short shmem_short_g (short *addr, int pe);
int shmem_int_g (int *addr, int pe);
long shmem_long_g (long *addr, int pe);
long long shmem_longlong_g (long long *addr, int pe);
float shmem_float_g (float *addr, int pe);
double shmem_double_g (double *addr, int pe);
long double shmem_longdouble_g (long double *addr, int pe);
void shmem_double_iget (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_float_iget (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_iget32 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_iget64 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_iget128 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_int_iget (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_long_iget (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_longdouble_iget (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_longlong_iget (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_short_iget (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe);
void shmem_int_add (int *target, int value, int pe);
void shmem_long_add (long *target, long value, int pe);
void shmem_longlong_add (long long *target, long long value, int pe);
int shmem_int_cswap (int *target, int cond, int value, int pe);
long shmem_long_cswap (long *target, long cond, long value, int pe);
long long shmem_longlong_cswap (long long *target, long long cond, long long value, int pe);
double shmem_double_swap (double *target, double value, int pe);
float shmem_float_swap (float *target, float value, int pe);
int shmem_int_swap (int *target, int value, int pe);
long shmem_long_swap (long *target, long value, int pe);
long long shmem_longlong_swap (long long *target, long long value, int pe);
long shmem_swap (long *target, long value, int pe);
int shmem_int_finc (int *target, int pe);
long shmem_long_finc (long *target, int pe);
long long shmem_longlong_finc (long long *target, int pe);
void shmem_int_inc (int *target, int pe);
void shmem_long_inc (long *target, int pe);
void shmem_longlong_inc (long long *target, int pe);
int shmem_int_fadd (int *target, int value, int pe);
long shmem_long_fadd (long *target, long value, int pe);
long long shmem_longlong_fadd (long long *target, long long value, int pe);
void shmem_barrier_all (void);
void shmem_barrier (int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_broadcast32 (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_broadcast64 (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_collect32 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_collect64 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_fcollect32 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_fcollect64 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);
void shmem_int_and_to_all (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_and_to_all (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longlong_and_to_all (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);
void shmem_short_and_to_all (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_double_max_to_all (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
void shmem_float_max_to_all (float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
void shmem_int_max_to_all (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
void shmem_long_max_to_all (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
void shmem_longdouble_max_to_all (long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);
void shmem_longlong_max_to_all (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);
void shmem_short_max_to_all (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
void shmem_double_min_to_all (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
void shmem_int_wait (int *ivar, int cmp_value);
void shmem_int_wait_until (int *ivar, int cmp, int cmp_value);
void shmem_long_wait (long *ivar, long cmp_value);
void shmem_long_wait_until (long *ivar, int cmp, long cmp_value);
void shmem_longlong_wait (long long *ivar, long long cmp_value);
void shmem_longlong_wait_until (long long *ivar, int cmp, long long cmp_value);
void shmem_short_wait (short *ivar, short cmp_value);
void shmem_short_wait_until (short *ivar, int cmp, short cmp_value);
void shmem_wait (long *ivar, long cmp_value);
void shmem_wait_until (long *ivar, int cmp, long cmp_value);
void shmem_fence (void);
void shmem_quiet (void);
void shmem_clear_lock (long *lock);
void shmem_set_lock (long *lock);
int shmem_test_lock (long *lock);
void shmem_clear_cache_inv (void);
void shmem_set_cache_inv (void);
void shmem_clear_cache_line_inv (void *target);
void shmem_set_cache_line_inv (void *target);
void shmem_udcflush (void);
void shmem_udcflush_line (void *target);

void Extrae_OPENSHMEM_init(void);

#endif /* __OPENSHMEM__WRAPPERS_H__ */
