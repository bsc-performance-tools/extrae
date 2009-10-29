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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "events.h"
#include "record.h"
#include "file_set.h"
#include "HardwareCounters.h"

static void show_current (event_t * c)
{
#if SIZEOF_LONG == 8
	fprintf (stdout, "EV: %d VAL: %lu TIME: %lu", c->event, c->value, c->time);
#else
	fprintf (stdout, "EV: %d VAL: %llu TIME: %llu", c->event, c->value, c->time);
#endif

	if (c->event == IRECV_EV || c->event == IRECVED_EV ||
	    c->event == SENDRECV_EV || c->event == SENDRECV_REPLACE_EV ||
	    c->event == PERSIST_REQ_EV)
	{
		fprintf (stdout, " TARGET:%u SIZE:%d TAG:%d COMM:%d AUX:%d\n",
		  c->param.mpi_param.target,
		  c->param.mpi_param.size, c->param.mpi_param.tag,
		  c->param.mpi_param.comm, c->param.mpi_param.aux);
	}
#if USE_HARDWARE_COUNTERS
#if defined(DEAD_CODE)
	else if (c->event == HWC_CHANGE_EV || c->event == HWC_SET_OVERFLOW_EV)
	{
		int i;

		fprintf (stdout, " %s HWC { ", (c->event == HWC_CHANGE_EV)?"new":"sample");
		for (i = 0; i < MAX_HWC; i++)
			fprintf (stdout, "0x%llx ", c->HWCValues[i]);
		fprintf (stdout, "}\n");
	}
#endif /* DEAD_CODE */
	else if (c->event == HWC_DEF_EV)
	{
		int i;

		fprintf (stdout, " HWC definition { ");
        for (i = 0; i < MAX_HWC; i++)
            fprintf (stdout, "0x%llx ", c->HWCValues[i]);
		fprintf (stdout, "}\n");
	}
#endif
  else
    fprintf (stdout, "\n");

#if USE_HARDWARE_COUNTERS
  if (Get_EvHWCRead (c))
		HardwareCounters_Show (c);
#endif
}

void make_dump (FileSet_t *fset)
{
#if USE_HARDWARE_COUNTERS
	UINT64 last_counters[MAX_HWC], current_counters[MAX_HWC];
	int j;
#endif
	UINT64 last_time;
	int i = 0;
	event_t *e;

	while (i < fset->nfiles)
	{
		last_time = 0;
#if USE_HARDWARE_COUNTERS
		for (j = 0; j < MAX_HWC; j++)
			last_counters[j] = 0;
#endif
		fprintf (stdout, "File %d\n", i);
		e = Current_FS (&fset->files[i]);
		while (e != NULL)
		{
			if (Get_EvTime(e) < last_time)
				fprintf (stdout, "** WARNING clock went backwards?\n");
			show_current (e);
#if USE_HARDWARE_COUNTERS
			if (Get_EvHWCRead(e))
			{
				HardwareCounters_Get (e, current_counters);
				for (j = 0; j < MAX_HWC; j++)
					if (current_counters[j] < last_counters[j])
						fprintf (stdout, "** WARNING counter %d went backwards?\n", j);
				for (j = 0; j < MAX_HWC; j++)
					last_counters[j] = current_counters[j];
			}
#endif
			StepOne_FS (&fset->files[i]);
			last_time = Get_EvTime(e);
			e = Current_FS (&fset->files[i]);
		}
		i++;
	}
	exit (0);
}
