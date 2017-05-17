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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "utils.h"
#include "semantics.h"
#include "events.h"

#include "misc_prv_semantics.h"
#include "mpi_prv_semantics.h"
#include "omp_prv_semantics.h"
#include "pthread_prv_semantics.h"
#include "cuda_prv_semantics.h"
#include "opencl_prv_semantics.h"
#include "openshmem_prv_semantics.h"
#include "java_prv_semantics.h"

#include "mpi_trf_semantics.h"
#include "misc_trf_semantics.h"

int num_Registered_Handlers = 0;
RangeEv_Handler_t *Event_Handlers = NULL;

static void Register_Handler (int range_min, int range_max, Ev_Handler_t *handler);
static void Register_Event_Handlers (SingleEv_Handler_t list[]);
static void Register_Range_Handlers  (RangeEv_Handler_t list[]);

int SkipHandler (event_t *p1, unsigned long long p2, unsigned int p3, unsigned int p4, unsigned int p5, unsigned int p6, FileSet_t *p7)
{
	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);
	UNREFERENCED_PARAMETER(p3);
	UNREFERENCED_PARAMETER(p4);
	UNREFERENCED_PARAMETER(p5);
	UNREFERENCED_PARAMETER(p6);
	UNREFERENCED_PARAMETER(p7);

	/* Do NOTHING */
	return 0;
}

void Semantics_Initialize (int output_format)
{
	switch (output_format)
	{
		case TRF_SEMANTICS:
			Register_Event_Handlers (TRF_MISC_Event_Handlers);
			Register_Range_Handlers (TRF_MISC_Range_Handlers);
			Register_Event_Handlers (TRF_MPI_Event_Handlers);
			break;
		case PRV_SEMANTICS:
		default:
			Register_Event_Handlers (PRV_MISC_Event_Handlers);
			Register_Range_Handlers (PRV_MISC_Range_Handlers);
			Register_Event_Handlers (PRV_MPI_Event_Handlers);
			Register_Event_Handlers (PRV_OMP_Event_Handlers);
			Register_Event_Handlers (PRV_pthread_Event_Handlers);
			Register_Event_Handlers (PRV_CUDA_Event_Handlers);
			Register_Range_Handlers (PRV_OpenCL_Event_Handlers);
			Register_Event_Handlers (PRV_OPENSHMEM_Event_Handlers);
			Register_Event_Handlers (PRV_Java_Event_Handlers);
			break;
	}
}

static void Register_Handler (int range_min, int range_max, Ev_Handler_t *handler)
{
	num_Registered_Handlers ++;

	xrealloc(Event_Handlers, Event_Handlers, num_Registered_Handlers * sizeof(RangeEv_Handler_t));
	Event_Handlers[num_Registered_Handlers - 1].range_min = range_min;
	Event_Handlers[num_Registered_Handlers - 1].range_max = range_max;
	Event_Handlers[num_Registered_Handlers - 1].handler = handler;
}

static void Register_Event_Handlers (SingleEv_Handler_t list[])
{
	int i = 0;
	
	while (list[i].event != NULL_EV)
	{
		Register_Handler (list[i].event, list[i].event, list[i].handler);
		i ++;
	}
}

static void Register_Range_Handlers (RangeEv_Handler_t list[])
{
	int i = 0;
	
	while (list[i].range_min != NULL_EV)
	{
		Register_Handler (list[i].range_min, list[i].range_max, list[i].handler);
		i ++;
	}
}

Ev_Handler_t * Semantics_getEventHandler (int event)
{
	int i = 0;

	while (i < num_Registered_Handlers)
	{	
		if ((event >= Event_Handlers[i].range_min) && (event <= Event_Handlers[i].range_max))
		{
			return Event_Handlers[i].handler;
		}
		i ++;
	}
	return NULL;
}
