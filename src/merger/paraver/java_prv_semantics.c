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

#include "file_set.h"
#include "object_tree.h"
#include "trace_to_prv.h"
#include "semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "addresses.h"
#include "options.h"

#include "record.h"
#include "events.h"

static int JAVA_JVMTI_call (event_t* event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset )
{
	unsigned state;
	unsigned EvType;
	unsigned long long EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (event);
	EvValue = Get_EvValue (event);

	switch (EvType)
	{
		case JAVA_JVMTI_GARBAGECOLLECTOR_EV:
		case JAVA_JVMTI_EXCEPTION_EV:
			state = STATE_OTHERS;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType,
	  EvValue);

	return 0;
}

SingleEv_Handler_t PRV_Java_Event_Handlers[] = {
	{ JAVA_JVMTI_GARBAGECOLLECTOR_EV, JAVA_JVMTI_call },
	{ JAVA_JVMTI_EXCEPTION_EV, JAVA_JVMTI_call },
	{ JAVA_JVMTI_OBJECT_ALLOC_EV, JAVA_JVMTI_call },
	{ JAVA_JVMTI_OBJECT_FREE_EV, JAVA_JVMTI_call },
	{ NULL_EV, NULL }
};

