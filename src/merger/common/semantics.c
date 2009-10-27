/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/common/semantics.c,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:06:55 $
 | @version:     $Revision: 1.5 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: semantics.c,v 1.5 2009/05/28 13:06:55 harald Exp $";

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
#include "trt_prv_semantics.h"

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
			Register_Event_Handlers (TRF_MPI_Event_Handlers);
			Register_Event_Handlers (TRF_MISC_Event_Handlers);
			break;
		case PRV_SEMANTICS:
		default:
			Register_Event_Handlers (PRV_MISC_Event_Handlers);
			Register_Range_Handlers (PRV_MISC_Range_Handlers);
			Register_Event_Handlers (PRV_MPI_Event_Handlers);
			Register_Event_Handlers (PRV_OMP_Event_Handlers);
			Register_Event_Handlers (PRV_pthread_Event_Handlers);
			Register_Event_Handlers (PRV_TRT_Event_Handlers);
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
