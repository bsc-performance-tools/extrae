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

#include "events.h"
#include "labels.h"

#include "gpi_prv_events.h"

struct GPI_event_label_t
{
	unsigned  eventtype;
	unsigned  present;
	char     *description;
	int       eventval;
};

#define MAX_GPI_EVENT_TYPE_ENTRIES 31

static struct GPI_event_label_t GPI_event_type_label[MAX_GPI_EVENT_TYPE_ENTRIES] =
{
	{GPI_INIT_EV,                 FALSE, "gaspi_proc_init",              1},
	{GPI_CONNECT_EV,              FALSE, "gaspi_connect",                2},
	{GPI_DISCONNECT_EV,           FALSE, "gaspi_disconnect",             3},
	{GPI_GROUP_CREATE_EV,         FALSE, "gaspi_group_create",         100},
	{GPI_GROUP_ADD_EV,            FALSE, "gaspi_group_add",            101},
	{GPI_GROUP_COMMIT_EV,         FALSE, "gaspi_group_commit",         102},
	{GPI_GROUP_DELETE_EV,         FALSE, "gaspi_group_delete",         103},
	{GPI_SEGMENT_ALLOC_EV,        FALSE, "gaspi_segment_alloc",        200},
	{GPI_SEGMENT_REGISTER_EV,     FALSE, "gaspi_segment_register",     201},
	{GPI_SEGMENT_CREATE_EV,       FALSE, "gaspi_segment_create",       202},
	{GPI_SEGMENT_BIND_EV,         FALSE, "gaspi_segment_bind",         203},
	{GPI_SEGMENT_USE_EV,          FALSE, "gaspi_segment_use",          204},
	{GPI_SEGMENT_DELETE_EV,       FALSE, "gaspi_segment_delete",       205},
	{GPI_WRITE_EV,                FALSE, "gaspi_write",                300},
	{GPI_READ_EV,                 FALSE, "gaspi_read",                 301},
	{GPI_WAIT_EV,                 FALSE, "gaspi_wait",                 302},
	{GPI_NOTIFY_EV,               FALSE, "gaspi_notify",               303},
	{GPI_NOTIFY_WAITSOME_EV,      FALSE, "gaspi_notify_waitsome",      304},
	{GPI_NOTIFY_RESET_EV,         FALSE, "gaspi_notify_reset",         305},
	{GPI_WRITE_NOTIFY_EV,         FALSE, "gaspi_write_notify",         306},
	{GPI_WRITE_LIST_EV,           FALSE, "gaspi_write_list",           307},
	{GPI_WRITE_LIST_NOTIFY_EV,    FALSE, "gaspi_write_list_notify",    308},
	{GPI_READ_LIST_EV,            FALSE, "gaspi_read_list",            309},
	{GPI_PASSIVE_SEND_EV,         FALSE, "gaspi_passive_send",         400},
	{GPI_PASSIVE_RECEIVE_EV,      FALSE, "gaspi_passive_receive",      401},
	{GPI_ATOMIC_FETCH_ADD_EV,     FALSE, "gaspi_atomic_fetch_add",     500},
	{GPI_ATOMIC_COMPARE_SWAP_EV,  FALSE, "gaspi_atomic_compare_swap",  501},
	{GPI_BARRIER_EV,              FALSE, "gaspi_barrier",              600},
	{GPI_ALLREDUCE_EV,            FALSE, "gaspi_allreduce",            601},
	{GPI_ALLREDUCE_USER_EV,       FALSE, "gaspi_allreduce_user",       602},
	{GPI_TERM_EV,                 FALSE, "gaspi_proc_term",           9999}
};

void
Enable_GPI_Operation(unsigned evttype)
{
	unsigned u;

	for (u = 0; u < MAX_GPI_EVENT_TYPE_ENTRIES; u++)
	{
		if (GPI_event_type_label[u].eventtype == evttype)
		{
			GPI_event_type_label[u].present = TRUE;
			break;
		}
	}
}

int
Translate_GPI_Operation(unsigned in_evttype, unsigned long long in_evtvalue,
    unsigned *out_evttype, unsigned long long *out_evtvalue)
{
	unsigned u;
	unsigned out_type = GPI_BASE_EV;

	for (u = 0; u < MAX_GPI_EVENT_TYPE_ENTRIES; u++)
		if (GPI_event_type_label[u].eventtype == in_evttype)
		{
			*out_evttype = out_type;
			if (in_evtvalue != 0)
				*out_evtvalue = GPI_event_type_label[u].eventval;
			else
				*out_evtvalue = 0;
			return TRUE;
		}

	return FALSE;
}

void
WriteEnabled_GPI_Operations(FILE * fd)
{
	unsigned u;
#if 0
	int anypresent = FALSE;
	int memtransfersizepresent = FALSE;
	int clfinishpresent = FALSE;

	for (u = 0; u < MAX_TYPE_ENTRIES; u++)
	{
		anypresent = GPI_event_type_label[u].present || anypresent;

		if (GPI_event_type_label[u].present && (
		      GPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEREADBUFFER_EV ||
		      GPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEREADBUFFERRECT_EV ||
		      GPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEWRITEBUFFER_EV ||
		      GPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEWRITEBUFFERRECT_EV )
		   )
			memtransfersizepresent = TRUE;

		if (OpenCL_event_presency_label_host[u].present && (
		     OpenCL_event_presency_label_host[u].eventtype == OPENCL_CLFINISH_EV))
			clfinishpresent = TRUE;
	}
#endif

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GPI_BASE_EV, "GPI call");
	fprintf (fd, "VALUES\n");
	fprintf (fd, "0 Outside GPI\n");

	for (u = 0; u < MAX_GPI_EVENT_TYPE_ENTRIES; u++)
	{
		if (GPI_event_type_label[u].present)
		{
			fprintf (fd, "%d %s\n",
			    GPI_event_type_label[u].eventval,
			    GPI_event_type_label[u].description);
		}
	}
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GPI_SIZE_EV, "GPI size");
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GPI_RANK_EV, "GPI rank");
	LET_SPACES(fd);
}
