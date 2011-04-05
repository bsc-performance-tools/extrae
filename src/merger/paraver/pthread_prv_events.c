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

#define PTHD_CREATE_INDEX       0  /* pthread_create index */
#define PTHD_JOIN_INDEX         1  /* pthread_join index */
#define PTHD_DETACH_INDEX       2  /* pthread_detach index */
#define PTHD_USRF_INDEX         3  /* pthread_create @ target address index */
#define PTHD_WRRD_LOCK_INDEX    4  /* pthread_rwlock* functions */
#define PTHD_MUTEX_LOCK_INDEX   5  /* pthread_mutex* functions */
#define PTHD_COND_INDEX         6  /* pthread_cond* functions */

#define MAX_PTHD_INDEX		7

static int inuse[MAX_PTHD_INDEX] = { FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE };

void Enable_pthread_Operation (int tipus)
{
	if (tipus == PTHREADCREATE_EV)
		inuse[PTHD_CREATE_INDEX] = TRUE;
	else if (tipus == PTHREADJOIN_EV)
		inuse[PTHD_JOIN_INDEX] = TRUE;
	else if (tipus == PTHREADDETACH_EV)
		inuse[PTHD_DETACH_INDEX] = TRUE;
	else if (tipus == PTHREADFUNC_EV)
		inuse[PTHD_USRF_INDEX] = TRUE;
	else if (tipus == PTHREAD_RWLOCK_WR_EV || tipus == PTHREAD_RWLOCK_RD_EV ||
	  tipus == PTHREAD_RWLOCK_UNLOCK_EV)
		inuse[PTHD_WRRD_LOCK_INDEX] = TRUE;
	else if (tipus == PTHREAD_MUTEX_UNLOCK_EV || tipus == PTHREAD_MUTEX_LOCK_EV)
		inuse[PTHD_MUTEX_LOCK_INDEX] = TRUE;
	else if (tipus == PTHREAD_COND_SIGNAL_EV || tipus == PTHREAD_COND_WAIT_EV ||
	  tipus == PTHREAD_COND_BROADCAST_EV)
		inuse[PTHD_COND_INDEX] = TRUE;
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_pthread_Operations (void)
{
	int res, i, tmp[MAX_PTHD_INDEX];

	res = MPI_Reduce (inuse, tmp, MAX_PTHD_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing pthread enabled operations");

	for (i = 0; i < MAX_PTHD_INDEX; i++)
		inuse[i] = tmp[i];
}

#endif

void pthreadEvent_WriteEnabledOperations (FILE * fd)
{
	if (inuse[PTHD_CREATE_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_create\n", 0, PTHREADCREATE_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[PTHD_JOIN_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_join\n", 0, PTHREADJOIN_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[PTHD_DETACH_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_detach\n", 0, PTHREADDETACH_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[PTHD_WRRD_LOCK_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_rw_lock_wr\n"
		             "%d   %d    pthread_rw_lock_rd\n"
		             "%d   %d    pthread_rw_unlock\n",
		         0, PTHREAD_RWLOCK_WR_EV,
		         0, PTHREAD_RWLOCK_RD_EV,
		         0, PTHREAD_RWLOCK_UNLOCK_EV);
	}
	if (inuse[PTHD_MUTEX_LOCK_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_mutex_lock\n"
		             "%d   %d    pthread_mutex_unlock\n",
		         0, PTHREAD_MUTEX_LOCK_EV,
		         0, PTHREAD_MUTEX_UNLOCK_EV);
	}
	if (inuse[PTHD_COND_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_cond_wait\n"
		             "%d   %d    pthread_cond_signal\n"
		             "%d   %d    pthread_cond_broadcast\n",
		         0, PTHREAD_COND_WAIT_EV,
		         0, PTHREAD_COND_SIGNAL_EV,
		         0, PTHREAD_COND_BROADCAST_EV);
	}

#if defined(HAVE_BFD)
	/* Hey, pthread & OpenMP share the same labels? */
	if (inuse[PTHD_USRF_INDEX])
		Address2Info_Write_OMP_Labels (fd, PTHREADFUNC_EV, PTHREADFUNC_LINE_EV, get_option_merge_UniqueCallerID());
#endif
}
