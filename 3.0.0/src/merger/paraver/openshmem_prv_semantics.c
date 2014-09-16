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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/merger/paraver/mpi_prv_semantics.c $
 | @last_commit: $Date: 2014-06-18 12:53:30 +0200 (mié, 18 jun 2014) $
 | @version:     $Revision: 2760 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: mpi_prv_semantics.c 2760 2014-06-18 10:53:30Z harald $";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "openshmem_events.h"
#include "openshmem_prv_semantics.h"
#include "paraver_state.h"

//#define DEBUG

static int Get_State (unsigned int EvType)
{
	int state = 0;
	
	switch (EvType)
	{
		default:
			state = STATE_OTHERS;
			break;
	}
	return state;
}

/******************************************************************************
 ***  Other_OPENSHMEM_Event:
 ******************************************************************************/

static int Other_OPENSHMEM_Event (event_t * current_event,
        unsigned long long current_time, unsigned int cpu, unsigned int ptask,
        unsigned int task, unsigned int thread, FileSet_t *fset)
{
	int EntryOrExit;
        unsigned int EvType;
        unsigned long EvValue;
        UNREFERENCED_PARAMETER(fset);

        EvType      = OPENSHMEM_EVENT_TYPE;
        EntryOrExit = Get_EvMiscParam (current_event);
        if (EntryOrExit)
	  EvValue = Get_EvValue (current_event) + 1;
	else
	  EvValue = 0;

        Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);

        trace_paraver_state (cpu, ptask, task, thread, current_time);
        trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
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
  { SHMEM_DOUBLE_PUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_PUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_PUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_PUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_PUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_PUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_PUT32_EV, Other_OPENSHMEM_Event },
  { SHMEM_PUT64_EV, Other_OPENSHMEM_Event },
  { SHMEM_PUT128_EV, Other_OPENSHMEM_Event },
  { SHMEM_PUTMEM_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_PUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_CHAR_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_P_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_IPUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_IPUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_IPUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_IPUT32_EV, Other_OPENSHMEM_Event },
  { SHMEM_IPUT64_EV, Other_OPENSHMEM_Event },
  { SHMEM_IPUT128_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_IPUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_IPUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_IPUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_IPUT_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_GET_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_GET_EV, Other_OPENSHMEM_Event },
  { SHMEM_GET32_EV, Other_OPENSHMEM_Event },
  { SHMEM_GET64_EV, Other_OPENSHMEM_Event },
  { SHMEM_GET128_EV, Other_OPENSHMEM_Event },
  { SHMEM_GETMEM_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_GET_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_GET_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_GET_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_GET_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_GET_EV, Other_OPENSHMEM_Event },
  { SHMEM_CHAR_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_G_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_IGET_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_IGET_EV, Other_OPENSHMEM_Event },
  { SHMEM_IGET32_EV, Other_OPENSHMEM_Event },
  { SHMEM_IGET64_EV, Other_OPENSHMEM_Event },
  { SHMEM_IGET128_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_IGET_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_IGET_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_IGET_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_IGET_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_IGET_EV, Other_OPENSHMEM_Event },
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
  { SHMEM_BROADCAST32_EV, Other_OPENSHMEM_Event },
  { SHMEM_BROADCAST64_EV, Other_OPENSHMEM_Event },
  { SHMEM_COLLECT32_EV, Other_OPENSHMEM_Event },
  { SHMEM_COLLECT64_EV, Other_OPENSHMEM_Event },
  { SHMEM_FCOLLECT32_EV, Other_OPENSHMEM_Event },
  { SHMEM_FCOLLECT64_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_AND_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_AND_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_AND_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_AND_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_MAX_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_FLOAT_MAX_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_INT_MAX_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONG_MAX_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGDOUBLE_MAX_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_LONGLONG_MAX_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_SHORT_MAX_TO_ALL_EV, Other_OPENSHMEM_Event },
  { SHMEM_DOUBLE_MIN_TO_ALL_EV, Other_OPENSHMEM_Event },
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

