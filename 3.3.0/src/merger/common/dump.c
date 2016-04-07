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
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "options.h"
#include "events.h"
#include "record.h"
#include "file_set.h"
#include "HardwareCounters.h"
#include "mpi_comunicadors.h"

#if USE_HARDWARE_COUNTERS
static int num_counters = 0;
#endif

static void show_current (const event_t * c, UINT64 max_time)
{
	int dump_time = get_option_dump_Time();

	if (c->time < max_time) /* Check whether this event is back in time */
	{
		if (dump_time)
			fprintf (stdout, "TIME: %lu (delta = %lu) EV: %d VAL: %lu [0x%lx] ", c->time, max_time-c->time, c->event, c->value, c->value);
		else
			fprintf (stdout, "TIME: - (delta = -) EV: %d VAL: %lu [0x%lx] ", c->event, c->value, c->value);
	}
	else 
	{
		if (dump_time)
		{
			char *clock_append = (c->time==max_time)?"+ ":"";
			fprintf (stdout, "TIME: %lu %s EV: %d VAL: %lu [0x%lx] ", c->time, clock_append, c->event, c->value, c->value);
		}
		else
			fprintf (stdout, "TIME: - EV: %d VAL: %lu [0x%lx] ", c->event, c->value, c->value);
	}

	if (c->event == MPI_IRECV_EV || c->event == MPI_IRECVED_EV || c->event == MPI_RECV_EV ||
	    c->event == MPI_SENDRECV_EV || c->event == MPI_SENDRECV_REPLACE_EV ||
	    c->event == MPI_PERSIST_REQ_EV ||
	    c->event == MPI_SEND_EV || c->event == MPI_ISEND_EV ||
	    c->event == MPI_SSEND_EV || c->event == MPI_ISSEND_EV ||
	    c->event == MPI_BSEND_EV || c->event == MPI_IBSEND_EV ||
	    c->event == MPI_RSEND_EV || c->event == MPI_IRSEND_EV)
	{
		fprintf (stdout, "TARGET:%u SIZE:%d TAG:%d COMM:%d AUX:%ld\n",
		  c->param.mpi_param.target,
		  c->param.mpi_param.size, c->param.mpi_param.tag,
		  c->param.mpi_param.comm, c->param.mpi_param.aux);
	}
	else if (c->event == USER_SEND_EV || c->event == USER_RECV_EV)
	{
		fprintf (stdout, "TARGET:%u SIZE:%d TAG:%d AUX:%ld\n",
		  c->param.mpi_param.target, c->param.mpi_param.size,
			c->param.mpi_param.tag, c->param.mpi_param.aux);
	}
	else if (c->event == MPI_INIT_EV && c->value == EVT_END)
	{
		fprintf (stdout, "OPTIONS: 0x%"PRIx64"\n", c->param.mpi_param.aux);
	}
	else if (c->event == MPI_ALIAS_COMM_CREATE_EV)
	{
		if (c->param.mpi_param.target == MPI_NEW_INTERCOMM_ALIAS)
		{
			if (c->value == EVT_BEGIN)
				fprintf (stdout, "InterCommunicator Alias: input id=%d [0x%x] (part %d, leader %d)\n",
				  c->param.mpi_param.comm, c->param.mpi_param.comm, c->param.mpi_param.size, c->param.mpi_param.tag);
			else
				fprintf (stdout, "InterCommunicator Alias: output id=%d [0x%x]\n",
				  c->param.mpi_param.comm, c->param.mpi_param.comm);
		}
		else
			fprintf (stdout, "Communicator Alias: id=%d [0x%x] ", c->param.mpi_param.comm, c->param.mpi_param.comm);

		if (c->param.mpi_param.target != MPI_NEW_INTERCOMM_ALIAS)
		{
			if (c->param.mpi_param.target == MPI_COMM_WORLD_ALIAS)
				fprintf (stdout, "MPI_COMM_WORLD alias\n");
			else if (c->param.mpi_param.target == MPI_COMM_SELF_ALIAS)
				fprintf (stdout, "MPI_COMM_SELF alias\n");
			else
				fprintf (stdout, "partners=%d\n",  c->param.mpi_param.size);
		}
	}
	else if (c->event == USER_EV)
	{
		fprintf (stdout, "USER EVENT value: %lu [0x%lx]\n", c->param.misc_param.param, c->param.misc_param.param);
	}
	else if (c->event == SAMPLING_ADDRESS_LD_EV)
	{
		fprintf (stdout, "SAMPLING_ADDRESS EVENT (load) value: %lu [0x%lx]\n", c->param.misc_param.param, c->param.misc_param.param);
	}
	else if (c->event == SAMPLING_ADDRESS_ST_EV)
	{
		fprintf (stdout, "SAMPLING_ADDRESS EVENT (store) value: %lu [0x%lx]\n", c->param.misc_param.param, c->param.misc_param.param);
	}
	else if (c->event == SAMPLING_ADDRESS_MEM_LEVEL_EV)
	{
		fprintf (stdout, "SAMPLING_ADDRESS_MEM_LEVEL_EV EVENT value: %lu [0x%lx]\n",
		  c->param.misc_param.param, c->param.misc_param.param);
	}
	else if (c->event == SAMPLING_ADDRESS_TLB_LEVEL_EV)
	{
		fprintf (stdout, "SAMPLING_ADDRESS_TLB_LEVEL_EV EVENT value: %lu [0x%lx]\n",
		  c->param.misc_param.param, c->param.misc_param.param);
	}
	else if (c->event == NAMEDCRIT_EV && (c->value == LOCKED_VAL || c->value == UNLOCKED_VAL))
	{
		fprintf (stdout, "NAMED CRITICAL ADDRESS: %lu [0x%lx]\n", c->param.omp_param.param[0], c->param.omp_param.param[0]);
	}
	else if (c->event == MALLOC_EV || c->event == REALLOC_EV)
	{
		if (c->value == EVT_BEGIN)
			fprintf (stdout, "%s SIZE: %lu\n", c->event==MALLOC_EV?"malloc()":"realloc()",
			  c->param.misc_param.param);
		else if (c->value == EVT_END)
			fprintf (stdout, "%s ADDRESS: %lu\n", c->event==MALLOC_EV?"malloc()":"realloc()",
			  c->param.misc_param.param);
	}
	else if (c->event == FREE_EV && c->value == EVT_BEGIN)
	{
		fprintf (stdout, "free() ADDRESS: %lu\n", c->param.misc_param.param);
	}
	else if (c->event == OMPT_TASKFUNC_EV)
	{
		fprintf (stdout, "OMPT TASK FUNCTION <%lx>\n",
		  c->param.omp_param.param[0]);
	}
	else if (c->event == OMPT_DEPENDENCE_EV)
	{
		fprintf (stdout, "OMPT TASK DEPENDENCE <%lx,%lx>\n",
		  c->param.omp_param.param[0], c->param.omp_param.param[1]);
	}
	else if (c->event == OMP_STATS_EV)
	{	fprintf (stdout, "OMP STATS: category %lu, value %lu\n",
		  c->value, c->param.omp_param.param[0]);
	}
#if USE_HARDWARE_COUNTERS
	else if (c->event == HWC_DEF_EV)
	{
		int def_num_counters = 0;
		int i;

		fprintf (stdout, "HWC definition { ");
		for (i = 0; i < MAX_HWC; i++)
		{
			fprintf (stdout, "0x%08llx ", c->HWCValues[i]);
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

void make_dump (const FileSet_t *fset)
{
	UINT64 max_time;
	UINT64 last_time;
	unsigned i = 0;
	event_t *e;

	while (i < fset->nfiles)
	{
		last_time = max_time = 0;
		fprintf (stdout, "File %d (object %u.%u.%u)\n", i, fset->files[i].ptask,
		   fset->files[i].task, fset->files[i].thread);
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
