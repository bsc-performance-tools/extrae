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

#include "events.h"
#include "file_set.h"
#include "paraver_generator.h"
#include "paraver_state.h"
#include "record.h"
#include "semantics.h"

static int
GPI_Event(event_t * current_event, unsigned long long current_time, unsigned cpu,
    unsigned ptask, unsigned task, unsigned thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	switch (EvType)
	{
		case GPI_INIT_EV:
		case GPI_TERM_EV:
		Switch_State(STATE_INITFINI, (EvValue != EVT_END), ptask, task, thread);
		trace_paraver_state (cpu, ptask, task, thread, current_time);
		break;
	}

	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

SingleEv_Handler_t PRV_GPI_Event_Handlers[] =
{
	{GPI_INIT_EV, GPI_Event},
	{GPI_TERM_EV, GPI_Event},
	{ NULL_EV, NULL }
};
