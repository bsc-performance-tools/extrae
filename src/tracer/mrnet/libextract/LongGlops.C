/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/libextract/LongGlops.C,v $
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "LongGlops.h"
#include "events.h"
#include "record.h"
#include "trace_buffers.h"
#include "utils.h"
#include <mpi.h>

int IsGlobalCollective (event_t *current, int this_value)
{
	int type = Get_EvEvent (current);
	int value = Get_EvValue (current);
	MPI_Comm comm = (MPI_Comm) Get_EvComm (current);

	return ((IsMPICollective(type)) && (value == this_value) && (Is_MPI_World_Comm(comm)));
}


static int FindNextCollective (BufferIterator_t *it, event_t **glop_begin, event_t **glop_end)
{
    event_t *evt_begin = NULL;
    event_t *evt_end = NULL;

	while ((!BIT_OutOfBounds(it)) && (!IsGlobalCollective(BIT_GetEvent(it), EVT_BEGIN)))
	{
		BIT_Next(it);
	}
    if (!BIT_OutOfBounds(it))
    {
        /* Found the start of a global operation */
        evt_begin = BIT_GetEvent(it);

        while ((!BIT_OutOfBounds(it)) && (!IsGlobalCollective(BIT_GetEvent(it), EVT_END)))
        {
			BIT_Next(it);
        }
        if (!BIT_OutOfBounds(it))
        {
            evt_end = BIT_GetEvent(it);
        }
    }

    *glop_begin = evt_begin;
    *glop_end = evt_end;

	return ((evt_begin != NULL) && (evt_end != NULL));
}


int Extract_LongGlops (int task_id, int thread_id, unsigned long long **io_Glops_Durations, int *io_firstGlopID, int *io_lastGlopID)
{
    event_t *glop_begin = NULL, *glop_end = NULL;
    int max_events = 0;
    unsigned long long *Glops_Durations = NULL;
    int NumGlops = 0;
    int firstGlopID = 0, lastGlopID = 0;
    Buffer_t *buffer = TRACING_BUFFER(thread_id);
	BufferIterator_t *it = BIT_NewForward(buffer);

	max_events = Buffer_GetFillCount (buffer);

    /* Allocate enough space (we're allocating much more than needed!) */
    Glops_Durations = (unsigned long long *)malloc(max_events * sizeof(unsigned long long));
    memset((void *)Glops_Durations, 0, max_events * sizeof(unsigned long long));

    /* Search for the next global operation events */
    while ((FindNextCollective (it, &glop_begin, &glop_end)) > 0)
    {
        /* Write the duration of the global operation */
        Glops_Durations[NumGlops] = (unsigned long long)(glop_end->time - glop_begin->time);

        /* Update the first and last global operation identifiers */
        if (NumGlops == 0) firstGlopID = glop_end->param.mpi_param.aux;
        lastGlopID = glop_end->param.mpi_param.aux;

        NumGlops ++;
    }
    fprintf(stderr,"[T: %d] Get_Long_Glops: NumGlops=%d, FirstID=%d, LastID=%d\n", task_id, NumGlops, firstGlopID, lastGlopID);
    fflush(stderr);

    *io_Glops_Durations = Glops_Durations;
    *io_firstGlopID = firstGlopID;
    *io_lastGlopID = lastGlopID;
	return NumGlops;
}

void Filter_LongGlops (int task_id, int thread_id, int commonFirstGlop, int commonLastGlop, unsigned int *Selected_Glops)
{
    event_t *glop_begin = NULL, *glop_end = NULL;
    Buffer_t *buffer = TRACING_BUFFER(thread_id);
	BufferIterator_t *it = BIT_NewForward(buffer);

	NewMask_SetRegion(buffer, Buffer_GetHead(buffer), Buffer_GetTail(buffer), MASK_NOFLUSH);

    /* Search for the next global operation events */
    while ((FindNextCollective (it, &glop_begin, &glop_end)) > 0)
    {
        int GlopID = Get_EvAux(glop_end);

        if ((GlopID >= commonFirstGlop) && (GlopID <= commonLastGlop) && (Selected_Glops[GlopID - commonFirstGlop]))
        {
			NewMask_UnsetRegion (buffer, glop_begin, glop_end, MASK_NOFLUSH);
        }
    }
}

