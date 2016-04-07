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

#ifndef __OPENSHMEM_EVENTS_H__
#define __OPENSHMEM_EVENTS_H__

#include "common.h"

#define OPENSHMEM_BASE_EVENT   52000000
#define OPENSHMEM_SENDBYTES_EV 52100000
#define OPENSHMEM_RECVBYTES_EV 52200000

#define COUNT_OPENSHMEM_EVENTS 132

typedef enum {
  START_PES_EV = OPENSHMEM_BASE_EVENT,
  SHMEM_MY_PE_EV,
  _MY_PE_EV,
  SHMEM_N_PES_EV,
  _NUM_PES_EV,
  SHMEM_PE_ACCESSIBLE_EV,
  SHMEM_ADDR_ACCESSIBLE_EV,
  SHMEM_PTR_EV,
  SHMALLOC_EV,
  SHFREE_EV,
  SHREALLOC_EV,
  SHMEMALIGN_EV,
  SHMEM_DOUBLE_PUT_EV,
  SHMEM_FLOAT_PUT_EV,
  SHMEM_INT_PUT_EV,
  SHMEM_LONG_PUT_EV,
  SHMEM_LONGDOUBLE_PUT_EV,
  SHMEM_LONGLONG_PUT_EV,
  SHMEM_PUT32_EV,
  SHMEM_PUT64_EV,
  SHMEM_PUT128_EV,
  SHMEM_PUTMEM_EV,
  SHMEM_SHORT_PUT_EV,
  SHMEM_CHAR_P_EV,
  SHMEM_SHORT_P_EV,
  SHMEM_INT_P_EV,
  SHMEM_LONG_P_EV,
  SHMEM_LONGLONG_P_EV,
  SHMEM_FLOAT_P_EV,
  SHMEM_DOUBLE_P_EV,
  SHMEM_LONGDOUBLE_P_EV,
  SHMEM_DOUBLE_IPUT_EV,
  SHMEM_FLOAT_IPUT_EV,
  SHMEM_INT_IPUT_EV,
  SHMEM_IPUT32_EV,
  SHMEM_IPUT64_EV,
  SHMEM_IPUT128_EV,
  SHMEM_LONG_IPUT_EV,
  SHMEM_LONGDOUBLE_IPUT_EV,
  SHMEM_LONGLONG_IPUT_EV,
  SHMEM_SHORT_IPUT_EV,
  SHMEM_DOUBLE_GET_EV,
  SHMEM_FLOAT_GET_EV,
  SHMEM_GET32_EV,
  SHMEM_GET64_EV,
  SHMEM_GET128_EV,
  SHMEM_GETMEM_EV,
  SHMEM_INT_GET_EV,
  SHMEM_LONG_GET_EV,
  SHMEM_LONGDOUBLE_GET_EV,
  SHMEM_LONGLONG_GET_EV,
  SHMEM_SHORT_GET_EV,
  SHMEM_CHAR_G_EV,
  SHMEM_SHORT_G_EV,
  SHMEM_INT_G_EV,
  SHMEM_LONG_G_EV,
  SHMEM_LONGLONG_G_EV,
  SHMEM_FLOAT_G_EV,
  SHMEM_DOUBLE_G_EV,
  SHMEM_LONGDOUBLE_G_EV,
  SHMEM_DOUBLE_IGET_EV,
  SHMEM_FLOAT_IGET_EV,
  SHMEM_IGET32_EV,
  SHMEM_IGET64_EV,
  SHMEM_IGET128_EV,
  SHMEM_INT_IGET_EV,
  SHMEM_LONG_IGET_EV,
  SHMEM_LONGDOUBLE_IGET_EV,
  SHMEM_LONGLONG_IGET_EV,
  SHMEM_SHORT_IGET_EV,
  SHMEM_INT_ADD_EV,
  SHMEM_LONG_ADD_EV,
  SHMEM_LONGLONG_ADD_EV,
  SHMEM_INT_CSWAP_EV,
  SHMEM_LONG_CSWAP_EV,
  SHMEM_LONGLONG_CSWAP_EV,
  SHMEM_DOUBLE_SWAP_EV,
  SHMEM_FLOAT_SWAP_EV,
  SHMEM_INT_SWAP_EV,
  SHMEM_LONG_SWAP_EV,
  SHMEM_LONGLONG_SWAP_EV,
  SHMEM_SWAP_EV,
  SHMEM_INT_FINC_EV,
  SHMEM_LONG_FINC_EV,
  SHMEM_LONGLONG_FINC_EV,
  SHMEM_INT_INC_EV,
  SHMEM_LONG_INC_EV,
  SHMEM_LONGLONG_INC_EV,
  SHMEM_INT_FADD_EV,
  SHMEM_LONG_FADD_EV,
  SHMEM_LONGLONG_FADD_EV,
  SHMEM_BARRIER_ALL_EV,
  SHMEM_BARRIER_EV,
  SHMEM_BROADCAST32_EV,
  SHMEM_BROADCAST64_EV,
  SHMEM_COLLECT32_EV,
  SHMEM_COLLECT64_EV,
  SHMEM_FCOLLECT32_EV,
  SHMEM_FCOLLECT64_EV,
  SHMEM_INT_AND_TO_ALL_EV,
  SHMEM_LONG_AND_TO_ALL_EV,
  SHMEM_LONGLONG_AND_TO_ALL_EV,
  SHMEM_SHORT_AND_TO_ALL_EV,
  SHMEM_DOUBLE_MAX_TO_ALL_EV,
  SHMEM_FLOAT_MAX_TO_ALL_EV,
  SHMEM_INT_MAX_TO_ALL_EV,
  SHMEM_LONG_MAX_TO_ALL_EV,
  SHMEM_LONGDOUBLE_MAX_TO_ALL_EV,
  SHMEM_LONGLONG_MAX_TO_ALL_EV,
  SHMEM_SHORT_MAX_TO_ALL_EV,
  SHMEM_DOUBLE_MIN_TO_ALL_EV,
  SHMEM_INT_WAIT_EV,
  SHMEM_INT_WAIT_UNTIL_EV,
  SHMEM_LONG_WAIT_EV,
  SHMEM_LONG_WAIT_UNTIL_EV,
  SHMEM_LONGLONG_WAIT_EV,
  SHMEM_LONGLONG_WAIT_UNTIL_EV,
  SHMEM_SHORT_WAIT_EV,
  SHMEM_SHORT_WAIT_UNTIL_EV,
  SHMEM_WAIT_EV,
  SHMEM_WAIT_UNTIL_EV,
  SHMEM_FENCE_EV,
  SHMEM_QUIET_EV,
  SHMEM_CLEAR_LOCK_EV,
  SHMEM_SET_LOCK_EV,
  SHMEM_TEST_LOCK_EV,
  SHMEM_CLEAR_CACHE_INV_EV,
  SHMEM_SET_CACHE_INV_EV,
  SHMEM_CLEAR_CACHE_LINE_INV_EV,
  SHMEM_SET_CACHE_LINE_INV_EV,
  SHMEM_UDCFLUSH_EV,
  SHMEM_UDCFLUSH_LINE_EV
} openshmem_event_t;


char *GetOPENSHMEMLabel( int openshmem_event );

#endif /* __OPENSHMEM_EVENTS_H__ */
