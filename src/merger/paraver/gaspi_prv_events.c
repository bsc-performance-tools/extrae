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

#include "labels.h"

#include "gaspi_prv_events.h"

struct GASPI_event_label_t GASPI_event_type_label[GASPI_MAX_VAL] =
{
	{GASPI_INIT_EV,                  FALSE, "gaspi_proc_init"},
	{GASPI_TERM_VAL,                 FALSE, "gaspi_proc_term"},
	{GASPI_CONNECT_VAL,              FALSE, "gaspi_connect"},
	{GASPI_DISCONNECT_VAL,           FALSE, "gaspi_disconnect"},
	{GASPI_GROUP_CREATE_VAL,         FALSE, "gaspi_group_create"},
	{GASPI_GROUP_ADD_VAL,            FALSE, "gaspi_group_add"},
	{GASPI_GROUP_COMMIT_VAL,         FALSE, "gaspi_group_commit"},
	{GASPI_GROUP_DELETE_VAL,         FALSE, "gaspi_group_delete"},
	{GASPI_SEGMENT_ALLOC_VAL,        FALSE, "gaspi_segment_alloc"},
	{GASPI_SEGMENT_REGISTER_VAL,     FALSE, "gaspi_segment_register"},
	{GASPI_SEGMENT_CREATE_VAL,       FALSE, "gaspi_segment_create"},
	{GASPI_SEGMENT_BIND_VAL,         FALSE, "gaspi_segment_bind"},
	{GASPI_SEGMENT_USE_VAL,          FALSE, "gaspi_segment_use"},
	{GASPI_SEGMENT_DELETE_VAL,       FALSE, "gaspi_segment_delete"},
	{GASPI_WRITE_VAL,                FALSE, "gaspi_write"},
	{GASPI_READ_VAL,                 FALSE, "gaspi_read"},
	{GASPI_WAIT_VAL,                 FALSE, "gaspi_wait"},
	{GASPI_NOTIFY_VAL,               FALSE, "gaspi_notify"},
	{GASPI_NOTIFY_WAITSOME_VAL,      FALSE, "gaspi_notify_waitsome"},
	{GASPI_NOTIFY_RESET_VAL,         FALSE, "gaspi_notify_reset"},
	{GASPI_WRITE_NOTIFY_VAL,         FALSE, "gaspi_write_notify"},
	{GASPI_WRITE_LIST_VAL,           FALSE, "gaspi_write_list"},
	{GASPI_WRITE_LIST_NOTIFY_VAL,    FALSE, "gaspi_write_list_notify"},
	{GASPI_READ_LIST_VAL,            FALSE, "gaspi_read_list"},
	{GASPI_READ_NOTIFY_VAL,          FALSE, "gaspi_read_notify"},
	{GASPI_READ_LIST_NOTIFY_VAL,     FALSE, "gaspi_read_list_notify"},
	{GASPI_PASSIVE_SEND_VAL,         FALSE, "gaspi_passive_send"},
	{GASPI_PASSIVE_RECEIVE_VAL,      FALSE, "gaspi_passive_receive"},
	{GASPI_ATOMIC_FETCH_ADD_VAL,     FALSE, "gaspi_atomic_fetch_add"},
	{GASPI_ATOMIC_COMPARE_SWAP_VAL,  FALSE, "gaspi_atomic_compare_swap"},
	{GASPI_BARRIER_VAL,              FALSE, "gaspi_barrier"},
	{GASPI_ALLREDUCE_VAL,            FALSE, "gaspi_allreduce"},
	{GASPI_ALLREDUCE_USER_VAL,       FALSE, "gaspi_allreduce_user"},
	{GASPI_QUEUE_CREATE_VAL,         FALSE, "gaspi_queue_create"},
	{GASPI_QUEUE_DELETE_VAL,         FALSE, "gaspi_queue_delete"}
};

struct GASPI_event_label_t GASPI_param_type_label[MAX_GASPI_PARAM_TYPE_ENTRIES] =
{
	{GASPI_RANK_EV,                 FALSE, "gaspi_rank"},
	{GASPI_NOTIFICATION_ID_EV,      FALSE, "gaspi_notification_id"},
	{GASPI_QUEUE_ID_EV,             FALSE, "gaspi_queue_id"}
};

void
Enable_GASPI_Operation(unsigned evttype, unsigned evtvalue)
{
	unsigned u;

	for (u = 0; u < GASPI_MAX_VAL; u++)
	{
		if (GASPI_event_type_label[u].eventtype == evttype)
		{
			GASPI_event_type_label[u].present = TRUE;
			break;
		} else if (GASPI_event_type_label[u].eventtype == evtvalue)
		{
			GASPI_event_type_label[u].present = TRUE;
			break;
		}
	}

	for (u = 0; u < MAX_GASPI_PARAM_TYPE_ENTRIES; u++)
	{
		if (GASPI_param_type_label[u].eventtype == evttype)
		{
			if (GASPI_param_type_label[u].present < evtvalue)
			{
				GASPI_param_type_label[u].present = evtvalue;
			}
			break;
		}
	}
}

void
WriteEnabled_GASPI_Operations(FILE * fd)
{
	unsigned u;

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_EV, "GASPI call");
	fprintf (fd, "VALUES\n");
	fprintf (fd, "0 Outside GASPI\n");

	for (u = 0; u < GASPI_MAX_VAL; u++)
	{
		if (GASPI_event_type_label[u].present)
		{
			if (GASPI_event_type_label[u].eventtype == GASPI_INIT_EV)
			{
				fprintf (fd, "%d %s\n",
				    1,
				    GASPI_event_type_label[u].description);
			} else
			{
				fprintf (fd, "%d %s\n",
				    GASPI_event_type_label[u].eventtype,
				    GASPI_event_type_label[u].description);
			}
		}
	}
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_SIZE_EV, "GASPI size");
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_RANK_EV, "GASPI rank");
	fprintf(fd, "VALUES\n");
	for (u = 0; u < GASPI_param_type_label[0].present; u++)
	{
		fprintf(fd, "%u %u\n", u+1, u);
	}
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_NOTIFICATION_ID_EV, "GASPI notification_id");
	fprintf(fd, "VALUES\n");
	for (u = 0; u < GASPI_param_type_label[1].present; u++)
	{
		fprintf(fd, "%u %u\n", u+1, u);
	}
	LET_SPACES(fd);

	fprintf (fd, "EVENT_TYPE\n");
	fprintf (fd, "%d    %d    %s\n", 0, GASPI_QUEUE_ID_EV, "GASPI queue");
	fprintf(fd, "VALUES\n");
	for (u = 0; u < GASPI_param_type_label[2].present; u++)
	{
		fprintf(fd, "%u %u\n", u+1, u);
	}
	LET_SPACES(fd);
}
