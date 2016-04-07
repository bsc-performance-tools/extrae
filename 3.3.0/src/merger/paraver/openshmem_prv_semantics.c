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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "openshmem_events.h"
#include "openshmem_prv_semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"

//#define DEBUG

static int Get_State (unsigned int EvType)
{
	switch (EvType)
	{
	  case SHMEM_DOUBLE_PUT_EV:
	  case SHMEM_FLOAT_PUT_EV:
	  case SHMEM_INT_PUT_EV:
	  case SHMEM_LONG_PUT_EV:
	  case SHMEM_LONGDOUBLE_PUT_EV:
	  case SHMEM_LONGLONG_PUT_EV:
	  case SHMEM_PUT32_EV:
	  case SHMEM_PUT64_EV:
	  case SHMEM_PUT128_EV:
	  case SHMEM_PUTMEM_EV:
	  case SHMEM_SHORT_PUT_EV:
	  case SHMEM_CHAR_P_EV:
	  case SHMEM_SHORT_P_EV:
	  case SHMEM_INT_P_EV:
	  case SHMEM_LONG_P_EV:
	  case SHMEM_LONGLONG_P_EV:
	  case SHMEM_FLOAT_P_EV:
	  case SHMEM_DOUBLE_P_EV:
	  case SHMEM_LONGDOUBLE_P_EV:
	  case SHMEM_DOUBLE_IPUT_EV:
	  case SHMEM_FLOAT_IPUT_EV:
	  case SHMEM_INT_IPUT_EV:
	  case SHMEM_IPUT32_EV:
	  case SHMEM_IPUT64_EV:
	  case SHMEM_IPUT128_EV:
	  case SHMEM_LONG_IPUT_EV:
	  case SHMEM_LONGDOUBLE_IPUT_EV:
	  case SHMEM_LONGLONG_IPUT_EV:
	  case SHMEM_SHORT_IPUT_EV:
	  case SHMEM_DOUBLE_GET_EV:
	  case SHMEM_FLOAT_GET_EV:
	  case SHMEM_GET32_EV:
	  case SHMEM_GET64_EV:
	  case SHMEM_GET128_EV:
	  case SHMEM_GETMEM_EV:
	  case SHMEM_INT_GET_EV:
	  case SHMEM_LONG_GET_EV:
	  case SHMEM_LONGDOUBLE_GET_EV:
	  case SHMEM_LONGLONG_GET_EV:
	  case SHMEM_SHORT_GET_EV:
	  case SHMEM_CHAR_G_EV:
	  case SHMEM_SHORT_G_EV:
	  case SHMEM_INT_G_EV:
	  case SHMEM_LONG_G_EV:
	  case SHMEM_LONGLONG_G_EV:
	  case SHMEM_FLOAT_G_EV:
	  case SHMEM_DOUBLE_G_EV:
	  case SHMEM_LONGDOUBLE_G_EV:
	  case SHMEM_DOUBLE_IGET_EV:
	  case SHMEM_FLOAT_IGET_EV:
	  case SHMEM_IGET32_EV:
	  case SHMEM_IGET64_EV:
	  case SHMEM_IGET128_EV:
	  case SHMEM_INT_IGET_EV:
	  case SHMEM_LONG_IGET_EV:
	  case SHMEM_LONGDOUBLE_IGET_EV:
	  case SHMEM_LONGLONG_IGET_EV:
	  case SHMEM_SHORT_IGET_EV:
		return STATE_REMOTE_MEM_ACCESS;
	  
	  case SHMEM_INT_ADD_EV:
	  case SHMEM_LONG_ADD_EV:
	  case SHMEM_LONGLONG_ADD_EV:
	  case SHMEM_INT_CSWAP_EV:
	  case SHMEM_LONG_CSWAP_EV:
	  case SHMEM_LONGLONG_CSWAP_EV:
	  case SHMEM_DOUBLE_SWAP_EV:
	  case SHMEM_FLOAT_SWAP_EV:
	  case SHMEM_INT_SWAP_EV:
	  case SHMEM_LONG_SWAP_EV:
	  case SHMEM_LONGLONG_SWAP_EV:
	  case SHMEM_SWAP_EV:
	  case SHMEM_INT_FINC_EV:
	  case SHMEM_LONG_FINC_EV:
	  case SHMEM_LONGLONG_FINC_EV:
	  case SHMEM_INT_INC_EV:
	  case SHMEM_LONG_INC_EV:
	  case SHMEM_LONGLONG_INC_EV:
	  case SHMEM_INT_FADD_EV:
	  case SHMEM_LONG_FADD_EV:
	  case SHMEM_LONGLONG_FADD_EV:
		return STATE_ATOMIC_MEM_OP;

	  case SHMEM_BARRIER_ALL_EV:
	  case SHMEM_BARRIER_EV:
		return STATE_BARRIER;

	  case SHMEM_BROADCAST32_EV:
	  case SHMEM_BROADCAST64_EV:
	  case SHMEM_COLLECT32_EV:
	  case SHMEM_COLLECT64_EV:
	  case SHMEM_FCOLLECT32_EV:
	  case SHMEM_FCOLLECT64_EV:
	  case SHMEM_INT_AND_TO_ALL_EV:
	  case SHMEM_LONG_AND_TO_ALL_EV:
	  case SHMEM_LONGLONG_AND_TO_ALL_EV:
	  case SHMEM_SHORT_AND_TO_ALL_EV:
	  case SHMEM_DOUBLE_MAX_TO_ALL_EV:
	  case SHMEM_FLOAT_MAX_TO_ALL_EV:
	  case SHMEM_INT_MAX_TO_ALL_EV:
	  case SHMEM_LONG_MAX_TO_ALL_EV:
	  case SHMEM_LONGDOUBLE_MAX_TO_ALL_EV:
	  case SHMEM_LONGLONG_MAX_TO_ALL_EV:
	  case SHMEM_SHORT_MAX_TO_ALL_EV:
	  case SHMEM_DOUBLE_MIN_TO_ALL_EV:
		return STATE_SYNC;

	  case SHMEM_INT_WAIT_EV:
	  case SHMEM_INT_WAIT_UNTIL_EV:
	  case SHMEM_LONG_WAIT_EV:
	  case SHMEM_LONG_WAIT_UNTIL_EV:
	  case SHMEM_LONGLONG_WAIT_EV:
	  case SHMEM_LONGLONG_WAIT_UNTIL_EV:
	  case SHMEM_SHORT_WAIT_EV:
	  case SHMEM_SHORT_WAIT_UNTIL_EV:
	  case SHMEM_WAIT_EV:
	  case SHMEM_WAIT_UNTIL_EV:
		return STATE_WAITMESS;

	  case SHMEM_FENCE_EV:
	  case SHMEM_QUIET_EV:
		return STATE_MEM_ORDERING;

	  case SHMEM_CLEAR_LOCK_EV:
	  case SHMEM_SET_LOCK_EV:
	  case SHMEM_TEST_LOCK_EV:
		return STATE_LOCKING;

	  default:
		return STATE_OTHERS;
	}
}

/******************************************************************************
 ***  Other_OPENSHMEM_Event:
 ******************************************************************************/

static int Other_OPENSHMEM_Event (event_t * current_event,
        unsigned long long current_time, unsigned int cpu, unsigned int ptask,
        unsigned int task, unsigned int thread, FileSet_t *fset)
{
        UNREFERENCED_PARAMETER(fset);
        unsigned int  EvType  = Get_EvEvent(current_event);
        unsigned long EvValue = ((Get_EvValue (current_event) > 0) ? EvType - OPENSHMEM_BASE_EVENT + 1 : 0);

        Switch_State (Get_State(EvType), (EvValue != EVT_END), ptask, task, thread);

        trace_paraver_state (cpu, ptask, task, thread, current_time);
        trace_paraver_event (cpu, ptask, task, thread, current_time, OPENSHMEM_BASE_EVENT, EvValue);

	return 0;
}


/******************************************************************************
 ***  Any_Outgoing_OPENSHMEM_Event:
 ******************************************************************************/

static int Any_Outgoing_OPENSHMEM_Event (event_t * current_event,
        unsigned long long current_time, unsigned int cpu, unsigned int ptask,
        unsigned int task, unsigned int thread, FileSet_t *fset)
{
        UNREFERENCED_PARAMETER(fset);
        unsigned int  EvType  = Get_EvEvent(current_event);
        unsigned long EvValue = ((Get_EvValue (current_event) > 0) ? EvType - OPENSHMEM_BASE_EVENT + 1 : 0);
	unsigned int  EvSize  = Get_EvSize  (current_event);

        Switch_State (Get_State(EvType), (EvValue != EVT_END), ptask, task, thread);

        trace_paraver_state (cpu, ptask, task, thread, current_time);
        trace_paraver_event (cpu, ptask, task, thread, current_time, OPENSHMEM_BASE_EVENT, EvValue);
        trace_paraver_event (cpu, ptask, task, thread, current_time, OPENSHMEM_SENDBYTES_EV, EvSize);

	return 0;
}



/******************************************************************************
 ***  Any_Incoming_OPENSHMEM_Event:
 ******************************************************************************/

static int Any_Incoming_OPENSHMEM_Event (event_t * current_event,
        unsigned long long current_time, unsigned int cpu, unsigned int ptask,
        unsigned int task, unsigned int thread, FileSet_t *fset)
{
        UNREFERENCED_PARAMETER(fset);
        unsigned int  EvType  = Get_EvEvent(current_event);
        unsigned long EvValue = ((Get_EvValue (current_event) > 0) ? EvType - OPENSHMEM_BASE_EVENT + 1 : 0);
	unsigned int  EvSize  = Get_EvSize  (current_event);

        Switch_State (Get_State(EvType), (EvValue != EVT_END), ptask, task, thread);

        trace_paraver_state (cpu, ptask, task, thread, current_time);
        trace_paraver_event (cpu, ptask, task, thread, current_time, OPENSHMEM_BASE_EVENT, EvValue);
        trace_paraver_event (cpu, ptask, task, thread, current_time, OPENSHMEM_RECVBYTES_EV, EvSize);

	return 0;
}






SingleEv_Handler_t PRV_OPENSHMEM_Event_Handlers[] = {
  { START_PES_EV, Other_OPENSHMEM_Event },
  { SHMEM_MY_PE_EV, Other_OPENSHMEM_Event },
  { _MY_PE_EV, Other_OPENSHMEM_Event },
  { SHMEM_N_PES_EV, Other_OPENSHMEM_Event },
  { _NUM_PES_EV, Other_OPENSHMEM_Event },
  { SHMEM_PE_ACCESSIBLE_EV, Other_OPENSHMEM_Event },
  { SHMEM_ADDR_ACCESSIBLE_EV, Other_OPENSHMEM_Event },
  { SHMEM_PTR_EV, Other_OPENSHMEM_Event },
  { SHMALLOC_EV, Other_OPENSHMEM_Event },
  { SHFREE_EV, Other_OPENSHMEM_Event },
  { SHREALLOC_EV, Other_OPENSHMEM_Event },
  { SHMEMALIGN_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_PUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_FLOAT_PUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_INT_PUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONG_PUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_PUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONGLONG_PUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_PUT32_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_PUT64_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_PUT128_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_PUTMEM_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_SHORT_PUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_CHAR_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_SHORT_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_INT_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONG_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONGLONG_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_FLOAT_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_DOUBLE_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_P_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_DOUBLE_IPUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_FLOAT_IPUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_INT_IPUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_IPUT32_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_IPUT64_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_IPUT128_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONG_IPUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_IPUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_LONGLONG_IPUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_SHORT_IPUT_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_DOUBLE_GET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_FLOAT_GET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_GET32_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_GET64_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_GET128_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_GETMEM_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_INT_GET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONG_GET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_GET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGLONG_GET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_SHORT_GET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_CHAR_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_SHORT_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_INT_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONG_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGLONG_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_FLOAT_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_DOUBLE_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_G_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_DOUBLE_IGET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_FLOAT_IGET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_IGET32_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_IGET64_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_IGET128_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_INT_IGET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONG_IGET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_IGET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGLONG_IGET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_SHORT_IGET_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_INT_ADD_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_ADD_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_ADD_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_CSWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_CSWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_CSWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_SWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_SWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_SWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_SWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_SWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_SWAP_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_FINC_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_FINC_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_FINC_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_INC_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_INC_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_INC_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_FADD_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_FADD_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_FADD_EV, Other_OPENSHMEM_Event },
  { SHMEM_BARRIER_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_BARRIER_EV, Other_OPENSHMEM_Event },
  { SHMEM_BROADCAST32_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_BROADCAST64_EV, Any_Outgoing_OPENSHMEM_Event },
  { SHMEM_COLLECT32_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_COLLECT64_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_FCOLLECT32_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_FCOLLECT64_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_INT_AND_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONG_AND_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGLONG_AND_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_SHORT_AND_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_DOUBLE_MAX_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_FLOAT_MAX_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_INT_MAX_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONG_MAX_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_MAX_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_LONGLONG_MAX_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_SHORT_MAX_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_DOUBLE_MIN_TO_ALL_EV, Any_Incoming_OPENSHMEM_Event },
  { SHMEM_INT_WAIT_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_WAIT_UNTIL_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_WAIT_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_WAIT_UNTIL_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_WAIT_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_WAIT_UNTIL_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_WAIT_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_WAIT_UNTIL_EV, Other_OPENSHMEM_Event },
  { SHMEM_WAIT_EV, Other_OPENSHMEM_Event },
  { SHMEM_WAIT_UNTIL_EV, Other_OPENSHMEM_Event },
  { SHMEM_FENCE_EV, Other_OPENSHMEM_Event },
  { SHMEM_QUIET_EV, Other_OPENSHMEM_Event },
  { SHMEM_CLEAR_LOCK_EV, Other_OPENSHMEM_Event },
  { SHMEM_SET_LOCK_EV, Other_OPENSHMEM_Event },
  { SHMEM_TEST_LOCK_EV, Other_OPENSHMEM_Event },
  { SHMEM_CLEAR_CACHE_INV_EV, Other_OPENSHMEM_Event },
  { SHMEM_SET_CACHE_INV_EV, Other_OPENSHMEM_Event },
  { SHMEM_CLEAR_CACHE_LINE_INV_EV, Other_OPENSHMEM_Event },
  { SHMEM_SET_CACHE_LINE_INV_EV, Other_OPENSHMEM_Event },
  { SHMEM_UDCFLUSH_EV, Other_OPENSHMEM_Event },
  { SHMEM_UDCFLUSH_LINE_EV, Other_OPENSHMEM_Event },
  { NULL_EV, NULL }
};

