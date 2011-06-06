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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/merger/paraver/trt_prv_semantics.c $
 | @last_commit: $Date: 2011-02-01 16:38:52 +0100 (dt, 01 feb 2011) $
 | @version:     $Revision: 537 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: trt_prv_semantics.c 537 2011-02-01 15:38:52Z harald $";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "file_set.h"
#include "object_tree.h"
#include "trace_to_prv.h"
#include "trt_prv_events.h"
#include "semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "addresses.h"
#include "options.h"

#include "record.h"
#include "events.h"

static int CUDA_Call (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset )
{
	unsigned int state;
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	switch (EvType)
	{
		case CUDALAUNCH_EV:
			state = STATE_OVHD;
			break;
		case CUDABARRIER_EV:
			state = STATE_BARRIER;
			break;
		case CUDAMEMCPY_EV:
			state = STATE_OVHD;
			break;
	}

	Switch_State (state, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	if (EvValue != EVT_END)
		trace_paraver_event (cpu, ptask, task, thread, current_time, CUDACALL_EV, EvType - CUDABASE_EV);
	else
		trace_paraver_event (cpu, ptask, task, thread, current_time, CUDACALL_EV, EVT_END);

	if (EvType == CUDAMEMCPY_EV && EvValue > 0)
		trace_paraver_event (cpu, ptask, task, thread, current_time, CUDAMEMCPY_SIZE_EV, EvValue);

	return 0;
}

SingleEv_Handler_t PRV_CUDA_Event_Handlers[] = {
	{ CUDALAUNCH_EV, CUDA_Call },
	{ CUDABARRIER_EV, CUDA_Call },
	{ CUDAMEMCPY_EV, CUDA_Call },
	{ NULL_EV, NULL }
};

