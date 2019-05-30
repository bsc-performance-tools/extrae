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

#include "gaspi_prv_events.h"

struct GASPI_event_label_t
{
	unsigned  eventtype;
	unsigned  present;
	char     *description;
	int       eventval;
};

#define MAX_GASPI_EVENT_TYPE_ENTRIES 31

static struct GASPI_event_label_t GASPI_event_type_label[MAX_GASPI_EVENT_TYPE_ENTRIES] =
{
	{GASPI_INIT_EV,                 FALSE, "gaspi_proc_init",              1},
	{GASPI_CONNECT_EV,              FALSE, "gaspi_connect",                2},
	{GASPI_DISCONNECT_EV,           FALSE, "gaspi_disconnect",             3},
	{GASPI_GROUP_CREATE_EV,         FALSE, "gaspi_group_create",         100},
	{GASPI_GROUP_ADD_EV,            FALSE, "gaspi_group_add",            101},
	{GASPI_GROUP_COMMIT_EV,         FALSE, "gaspi_group_commit",         102},
	{GASPI_GROUP_DELETE_EV,         FALSE, "gaspi_group_delete",         103},
	{GASPI_SEGMENT_ALLOC_EV,        FALSE, "gaspi_segment_alloc",        200},
	{GASPI_SEGMENT_REGISTER_EV,     FALSE, "gaspi_segment_register",     201},
	{GASPI_SEGMENT_CREATE_EV,       FALSE, "gaspi_segment_create",       202},
	{GASPI_SEGMENT_BIND_EV,         FALSE, "gaspi_segment_bind",         203},
	{GASPI_SEGMENT_USE_EV,          FALSE, "gaspi_segment_use",          204},
	{GASPI_SEGMENT_DELETE_EV,       FALSE, "gaspi_segment_delete",       205},
	{GASPI_WRITE_EV,                FALSE, "gaspi_write",                300},
	{GASPI_READ_EV,                 FALSE, "gaspi_read",                 301},
	{GASPI_WAIT_EV,                 FALSE, "gaspi_wait",                 302},
	{GASPI_NOTIFY_EV,               FALSE, "gaspi_notify",               303},
	{GASPI_NOTIFY_WAITSOME_EV,      FALSE, "gaspi_notify_waitsome",      304},
	{GASPI_NOTIFY_RESET_EV,         FALSE, "gaspi_notify_reset",         305},
	{GASPI_WRITE_NOTIFY_EV,         FALSE, "gaspi_write_notify",         306},
	{GASPI_WRITE_LIST_EV,           FALSE, "gaspi_write_list",           307},
	{GASPI_WRITE_LIST_NOTIFY_EV,    FALSE, "gaspi_write_list_notify",    308},
	{GASPI_READ_LIST_EV,            FALSE, "gaspi_read_list",            309},
	{GASPI_PASSIVE_SEND_EV,         FALSE, "gaspi_passive_send",         400},
	{GASPI_PASSIVE_RECEIVE_EV,      FALSE, "gaspi_passive_receive",      401},
	{GASPI_ATOMIC_FETCH_ADD_EV,     FALSE, "gaspi_atomic_fetch_add",     500},
	{GASPI_ATOMIC_COMPARE_SWAP_EV,  FALSE, "gaspi_atomic_compare_swap",  501},
	{GASPI_BARRIER_EV,              FALSE, "gaspi_barrier",              600},
	{GASPI_ALLREDUCE_EV,            FALSE, "gaspi_allreduce",            601},
	{GASPI_ALLREDUCE_USER_EV,       FALSE, "gaspi_allreduce_user",       602},
	{GASPI_TERM_EV,                 FALSE, "gaspi_proc_term",           9999}
};

void
Enable_GASPI_Operation(unsigned evttype)
{
	unsigned u;

	for (u = 0; u < MAX_GASPI_EVENT_TYPE_ENTRIES; u++)
	{
		if (GASPI_event_type_label[u].eventtype == evttype)
		{
			GASPI_event_type_label[u].present = TRUE;
			break;
		}
	}
}

int
Translate_GASPI_Operation(unsigned in_evttype, unsigned long long in_evtvalue,
    unsigned *out_evttype, unsigned long long *out_evtvalue)
{
	unsigned u;
	unsigned out_type = GASPI_BASE_EV;

	for (u = 0; u < MAX_GASPI_EVENT_TYPE_ENTRIES; u++)
		if (GASPI_event_type_label[u].eventtype == in_evttype)
		{
			*out_evttype = out_type;
			if (in_evtvalue != 0)
				*out_evtvalue = GASPI_event_type_label[u].eventval;
			else
				*out_evtvalue = 0;
			return TRUE;
		}

	return FALSE;
}

void
WriteEnabled_GASPI_Operations(FILE * fd)
{
	unsigned u;
#if 0
	int anypresent = FALSE;
	int memtransfersizepresent = FALSE;
	int clfinishpresent = FALSE;

	for (u = 0; u < MAX_TYPE_ENTRIES; u++)
	{
		anypresent = GASPI_event_type_label[u].present || anypresent;

		if (GASPI_event_type_label[u].present && (
		      GASPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEREADBUFFER_EV ||
		      GASPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEREADBUFFERRECT_EV ||
		      GASPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEWRITEBUFFER_EV ||
		      GASPI_event_type_label[u].eventtype == OPENCL_CLENQUEUEWRITEBUFFERRECT_EV )
		   )
			memtransfersizepresent = TRUE;

		if (OpenCL_event_presency_label_host[u].present && (
		     OpenCL_event_presency_label_host[u].eventtype == OPENCL_CLFINISH_EV))
			clfinishpresent = TRUE;
	}
#endif

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_BASE_EV, "GASPI call");
	fprintf (fd, "VALUES\n");
	fprintf (fd, "0 Outside GASPI\n");

	for (u = 0; u < MAX_GASPI_EVENT_TYPE_ENTRIES; u++)
	{
		if (GASPI_event_type_label[u].present)
		{
			fprintf (fd, "%d %s\n",
			    GASPI_event_type_label[u].eventval,
			    GASPI_event_type_label[u].description);
		}
	}
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_SIZE_EV, "GASPI size");
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_RANK_EV, "GASPI rank");
	LET_SPACES(fd);
}
