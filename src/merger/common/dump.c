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
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
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

#if USE_HARDWARE_COUNTERS
static int num_counters = 0;
#endif

static void show_current (event_t * c, UINT64 max_time)
{
	if (c->time < max_time) /* Check whether this event is back in time */
	{
#if SIZEOF_LONG == 8
		fprintf (stdout, "EV: %d VAL: %lu [0x%lx] TIME: %lu (delta = %lu) ", c->event, c->value, c->value, c->time, max_time-c->time);
#else
		fprintf (stdout, "EV: %d VAL: %llu [0x%llx] TIME: %llu (delta = %llu) ", c->event, c->value, c->value, c->time, max_time-c->time);
#endif
	}
	else 
	{
#if SIZEOF_LONG == 8
		fprintf (stdout, "EV: %d VAL: %lu [0x%lx] TIME: %lu %s", c->event, c->value, c->value, c->time, (c->time==max_time)?"+ ":"");
#else
		fprintf (stdout, "EV: %d VAL: %llu [0x%llx] TIME: %llu %s", c->event, c->value, c->value, c->time, (c->time==max_time)?"+ ":"");
#endif
	}

	if (c->event == MPI_IRECV_EV || c->event == MPI_IRECVED_EV || c->event == MPI_RECV_EV ||
	    c->event == MPI_SENDRECV_EV || c->event == MPI_SENDRECV_REPLACE_EV ||
	    c->event == MPI_PERSIST_REQ_EV ||
	    c->event == MPI_SEND_EV || c->event == MPI_ISEND_EV ||
	    c->event == MPI_SSEND_EV || c->event == MPI_ISSEND_EV ||
	    c->event == MPI_BSEND_EV || c->event == MPI_IBSEND_EV ||
	    c->event == MPI_RSEND_EV || c->event == MPI_IRSEND_EV)
	{
		fprintf (stdout, "TARGET:%u SIZE:%d TAG:%d COMM:%d AUX:%lld\n",
		  c->param.mpi_param.target,
		  c->param.mpi_param.size, c->param.mpi_param.tag,
		  c->param.mpi_param.comm, c->param.mpi_param.aux);
	}
	else if (c->event == USER_SEND_EV || c->event == USER_RECV_EV)
	{
		fprintf (stdout, "TARGET:%u SIZE:%d TAG:%d AUX:%lld\n",
		  c->param.mpi_param.target, c->param.mpi_param.size,
			c->param.mpi_param.tag, c->param.mpi_param.aux);
	}
	else if (c->event == MPI_COMM_SPLIT_EV || c->event == MPI_COMM_DUP_EV ||
	         c->event == MPI_COMM_CREATE_EV || c->event == MPI_CART_CREATE_EV ||
	         c->event == MPI_CART_SUB_EV)
	{
		fprintf (stdout, "COMM DEF SIZE: %d COMM: %d TRACE? %d\n",
		  c->param.mpi_param.size, c->param.mpi_param.comm, c->param.mpi_param.aux);
	}
	else if (c->event == MPI_INIT_EV && c->value == EVT_END)
	{
		fprintf (stdout, "OPTIONS: 0x%08llx\n", c->param.mpi_param.aux);
	}
	else if (c->event == USER_EV)
	{
		fprintf (stdout, "USER_EV value: %llu [0x%llx]\n", c->param.misc_param.param, c->param.misc_param.param);
	}
#if USE_HARDWARE_COUNTERS
	else if (c->event == HWC_DEF_EV)
	{
		int def_num_counters = 0;
		int i;

		fprintf (stdout, "HWC definition { ");
		for (i = 0; i < MAX_HWC; i++)
		{
			fprintf (stdout, "0x%llx ", c->HWCValues[i]);
			if (c->HWCValues[i] != NO_COUNTER)
				def_num_counters++;
		}
		fprintf (stdout, "}\n");

		num_counters = MAX (def_num_counters, num_counters);
	}
#endif
  else
    fprintf (stdout, "\n");

#if USE_HARDWARE_COUNTERS
  if (Get_EvHWCRead (c))
		HardwareCounters_Show (c, num_counters);
#endif
}

void make_dump (FileSet_t *fset)
{
	UINT64 max_time;
	UINT64 last_time;
	unsigned i = 0;
	event_t *e;

	while (i < fset->nfiles)
	{
		last_time = max_time = 0;
		fprintf (stdout, "File %d\n", i);
		e = Current_FS (&fset->files[i]);
		while (e != NULL)
		{
			if (Get_EvTime(e) < last_time)
				fprintf (stdout, "** WARNING clock went backwards?\n");
			show_current (e, max_time);

			StepOne_FS (&fset->files[i]);
			last_time = Get_EvTime(e);
			max_time = MAX(Get_EvTime(e), max_time);
			e = Current_FS (&fset->files[i]);
		}
		i++;
	}
	exit (0);
}
