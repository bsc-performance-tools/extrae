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

#include "openacc_prv_events.h"
#include "labels.h"

int OPENACC_Present = FALSE;

struct OPENACC_event_label_st
{
	int eventval;
	char * description;
};

static struct
OPENACC_event_label_st OPENACC_event_label[OPENACC_MAX_VAL] =
{
	{ OPENACC_INIT_VAL, "acc_ev_device_init_start" },
	{ OPENACC_SHUTDOWN_VAL, "acc_ev_device_shutdown_start" },
	{ OPENACC_ENTER_DATA_VAL, "acc_ev_enter_data_start" },
	{ OPENACC_EXIT_DATA_VAL, "acc_ev_exit_data_start" },
	{ OPENACC_UPDATE_VAL, "acc_ev_update_start" },
	{ OPENACC_COMPUTE_VAL, "acc_ev_compute_construct_start" },
	{ OPENACC_ENQUEUE_KERNEL_LAUNCH_VAL, "acc_ev_enqueue_launch_start" },
	{ OPENACC_ENQUEUE_UPLOAD_VAL, "acc_ev_enqueue_upload_start" },
	{ OPENACC_ENQUEUE_DOWNLOAD_VAL, "acc_ev_enqueue_download_start" },
	{ OPENACC_WAIT_VAL, "acc_ev_wait_start" }
};

static struct
OPENACC_event_label_st OPENACC_data_event_label[OPENACC_DATA_MAX_VAL] =
{
	{ OPENACC_CREATE_VAL, "acc_ev_create" },
	{ OPENACC_DELETE_VAL, "acc_ev_delete" },
	{ OPENACC_ALLOC_VAL, "acc_ev_alloc" },
	{ OPENACC_FREE_VAL, "acc_ev_free" }
};

void
Enable_OPENACC_Operation(int Op)
{
	UNREFERENCED_PARAMETER(Op);
	OPENACC_Present = TRUE;
}

void
WriteEnabled_OPENACC_Operations(FILE *fd)
{
	unsigned u = 0;

	if (OPENACC_Present)
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d    %d    %s\n", 0, OPENACC_EV, "OpenACC");
		fprintf (fd, "VALUES\n");
		fprintf (fd, "0 End\n");
		for (u=0; u<OPENACC_MAX_VAL; u++)
		{
			fprintf(fd, "%d %s\n", OPENACC_event_label[u].eventval, OPENACC_event_label[u].description);
		}
		LET_SPACES(fd);

		fprintf(fd, "EVENT_TYPE\n");
		fprintf(fd, "%d    %d    %s\n", 0, OPENACC_DATA_EV, "OpenACC Data");
		fprintf(fd, "VALUES\n");
		fprintf(fd, "0 End\n");
		for (u=0; u<OPENACC_DATA_MAX_VAL; u++)
		{
			fprintf(fd, "%d %s\n", OPENACC_data_event_label[u].eventval, OPENACC_data_event_label[u].description);
		}
		LET_SPACES(fd);
	}
}
