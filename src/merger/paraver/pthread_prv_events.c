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

#ifdef HAVE_BFD
# include "addr2info.h"
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "events.h"
#include "omp_prv_events.h"
#include "mpi2out.h"
#include "options.h"
#include "labels.h"

struct pthread_event_presency_label_st
{
	unsigned eventtype;
	unsigned present;
	char * description;
	int eventval;
}; 

#define MAX_PTHREAD_TYPE_ENTRIES 13

static struct pthread_event_presency_label_st
 pthread_event_presency_label[MAX_PTHREAD_TYPE_ENTRIES] = 
{
 { PTHREAD_EXIT_EV, FALSE, "pthread_exit", 1 },
 { PTHREAD_CREATE_EV, FALSE, "pthread_create", 2 },
 { PTHREAD_JOIN_EV, FALSE, "pthread_join", 3 },
 { PTHREAD_DETACH_EV, FALSE, "pthread_detach", 4 },
 { PTHREAD_RWLOCK_WR_EV, FALSE, "pthread_rwlock_*wrlock", 5 },
 { PTHREAD_RWLOCK_RD_EV, FALSE, "pthread_rwlock_*rdlock", 6 },
 { PTHREAD_RWLOCK_UNLOCK_EV, FALSE, "pthread_rwlock_unlock", 7 },
 { PTHREAD_MUTEX_LOCK_EV, FALSE, "pthread_mutex_*lock", 8 },
 { PTHREAD_MUTEX_UNLOCK_EV, FALSE, "pthread_mutex_unlock", 9 },
 { PTHREAD_COND_SIGNAL_EV, FALSE, "pthread_cond_signal", 10 },
 { PTHREAD_COND_BROADCAST_EV, FALSE, "pthread_cond_broadcast", 11 },
 { PTHREAD_COND_WAIT_EV, FALSE, "pthread_cond_*wait", 12 },
 { PTHREAD_BARRIER_WAIT_EV, FALSE, "pthread_barrier_wait", 13 } 
};

int Translate_pthread_Operation (unsigned in_evttype, 
	unsigned long long in_evtvalue, unsigned *out_evttype,
	unsigned long long *out_evtvalue)
{
	unsigned u;

	for (u = 0; u < MAX_PTHREAD_TYPE_ENTRIES; u++)
		if (pthread_event_presency_label[u].eventtype == in_evttype)
		{
			*out_evttype = PTHREAD_BASE_EV;
			if (in_evtvalue != 0)
				*out_evtvalue = pthread_event_presency_label[u].eventval;
			else
				*out_evtvalue = 0;
			return TRUE;
		}

	return FALSE;
}

void Enable_pthread_Operation (unsigned evttype)
{
	unsigned u;

	for (u = 0; u < MAX_PTHREAD_TYPE_ENTRIES; u++)
		if (pthread_event_presency_label[u].eventtype == evttype)
		{
			pthread_event_presency_label[u].present = TRUE;
			break;
		}
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_pthread_Operations (void)
{
	int res;
	int i, tmp_in[MAX_PTHREAD_TYPE_ENTRIES], tmp_out[MAX_PTHREAD_TYPE_ENTRIES];

	for (i = 0; i < MAX_PTHREAD_TYPE_ENTRIES; i++)
		tmp_in[i] = pthread_event_presency_label[i].present;

	res = MPI_Reduce (tmp_in, tmp_out, MAX_PTHREAD_TYPE_ENTRIES, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing pthread enabled operations");

	for (i = 0; i < MAX_PTHREAD_TYPE_ENTRIES; i++)
		pthread_event_presency_label[i].present = tmp_out[i];
}

#endif

void WriteEnabled_pthread_Operations (FILE * fd)
{
	unsigned u;
	int anypresent = FALSE;
#if defined(HAVE_BFD)
	int createpresent = FALSE;
#endif

	for (u = 0; u < MAX_PTHREAD_TYPE_ENTRIES; u++)
	{
		anypresent = anypresent || pthread_event_presency_label[u].present;
#if defined(HAVE_BFD)
		if (pthread_event_presency_label[u].eventtype == PTHREAD_CREATE_EV)
			createpresent = TRUE;
#endif
	}

	if (anypresent)
	{

		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d    %d    %s\n", 0, PTHREAD_BASE_EV, "pthread call");
		fprintf (fd, "VALUES\n");
		fprintf (fd, "0 Outside pthread call\n");
		for (u = 0; u < MAX_PTHREAD_TYPE_ENTRIES; u++)
			if (pthread_event_presency_label[u].present)
				fprintf (fd, "%d %s\n", 
					pthread_event_presency_label[u].eventval,
					pthread_event_presency_label[u].description);
		LET_SPACES(fd);
	}

#if defined(HAVE_BFD)
	/* Hey, pthread & OpenMP share the same labels? */
	if (createpresent)
		Address2Info_Write_OMP_Labels (fd, PTHREAD_FUNC_EV, "pthread function",
			PTHREAD_FUNC_LINE_EV, "pthread function line and file",
			get_option_merge_UniqueCallerID());
#endif
}

