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
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_FILE_H
# include <sys/file.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include <mpi.h>
#include "extrae_mpif.h"
#include "utils.h"
#include "utils_mpi.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"
#include "mpi_wrapper.h"
#include "wrapper.h"
#include "clock.h"
#include "signals.h"
#include "misc_wrapper.h"
#include "mpi_interface.h"
#include "mode.h"
#include "threadinfo.h"

#include "hash_table.h"

#if defined(C_SYMBOLS) && defined(FORTRAN_SYMBOLS)
# define COMBINED_SYMBOLS
#endif

#if defined(HAVE_MRNET)
# include "mrnet_be.h"
#endif

#include "misc_wrapper.h"
#include "mpi_stats.h"


#define MPI_CHECK(mpi_error, routine) \
	if (mpi_error != MPI_SUCCESS) \
	{ \
		fprintf (stderr, "Error in MPI call %s (file %s, line %d, routine %s) returned %d\n", \
			#routine, __FILE__, __LINE__, __func__, mpi_error); \
		fflush (stderr); \
		exit (1); \
	}

#if defined(FORTRAN_SYMBOLS)

/******************************************************************************
 ***  PMPI_Reduce_Wrapper
 ******************************************************************************/

void PMPI_Reduce_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= *count;

	/*
	*   event : REDUCE_EV                    value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send (non root) /received (root)
	*   tag : rank                           commid: communicator Id
	*   aux : root rank
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_REDUCE_EV, EVT_BEGIN, *op, size, me, c, *root);

	CtoF77 (pmpi_reduce) (sendbuf, recvbuf, count, datatype, op, root, comm,
	  ierror);

	/*
	*   event : REDUCE_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, size, 0);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, size);
	}

	TRACE_MPIEVENT (current_time, MPI_REDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_AllReduce_Wrapper
 ******************************************************************************/

void PMPI_AllReduce_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= *count;

	/*
	*   event : ALLREDUCE_EV                 value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send and received
	*   tag : rank                           commid: communicator Id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_ALLREDUCE_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_allreduce) (sendbuf, recvbuf, count, datatype, op, comm,
	  ierror);

	/*
	*   event : ALLREDUCE_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, size, size);

	TRACE_MPIEVENT (current_time, MPI_ALLREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Barrier_Wrapper
 ******************************************************************************/

void PMPI_Barrier_Wrapper (MPI_Fint *comm, MPI_Fint *ierror)
{
  MPI_Comm c = PMPI_Comm_f2c (*comm);
  int me, ret, csize;

  CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
  MPI_CHECK(ret, pmpi_comm_rank);

  CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  /*
   *   event : BARRIER_EV                    value : EVT_BEGIN
   *   target : ---                          size  : ---
   *   tag : rank                            commid: communicator id
   *   aux : ---
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_BARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
                    EMPTY);
  }
#else
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_BARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
                  EMPTY);
#endif

  CtoF77 (pmpi_barrier) (comm, ierror);

  /*
   *   event : BARRIER_EV                   value : EVT_END
   *   target : ---                         size  : size of the communicator
   *   tag : ---                            commid: communicator id
   *   aux : global op counter
   */

  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, 0, 0);

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    TRACE_MPIEVENT (current_time, MPI_BARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
  }
#else
  TRACE_MPIEVENT (current_time, MPI_BARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
#endif

}

/******************************************************************************
 ***  PMPI_BCast_Wrapper
 ******************************************************************************/

void PMPI_BCast_Wrapper (void *buffer, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= *count;

	/*
	*   event : BCAST_EV                     value : EVT_BEGIN
	*   target : root_rank                   size  : message size
	*   tag : rank                           commid: communicator identifier
	*   aux : ---
	*/

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 1;
#endif

	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_BCAST_EV, EVT_BEGIN, *root, size, me, c, 
	  EMPTY);

	CtoF77 (pmpi_bcast) (buffer, count, datatype, root, comm, ierror);

	/*
	*   event : BCAST_EV                    value : EVT_END
	*   target : ---                        size  : size of the communicator
	*   tag : ---                           commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, size);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, size, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_BCAST_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_AllToAll_Wrapper
 ******************************************************************************/

void PMPI_AllToAll_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, sendsize, recvsize, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : ALLTOALL_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_ALLTOALL_EV, EVT_BEGIN, EMPTY,
	  *sendcount * sendsize, me, c, *recvcount * recvsize * csize);

	CtoF77 (pmpi_alltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, comm, ierror);

	/*
	*   event : ALLTOALL_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * csize,  *sendcount * sendsize);

	TRACE_MPIEVENT (current_time, MPI_ALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_AllToAllV_Wrapper
 ******************************************************************************/

void PMPI_AllToAllV_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, sendsize, recvsize, csize;
	int proc, sendc = 0, recvc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (sendcount != NULL)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (recvcount != NULL)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;
		
	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		if (sendcount != NULL)
			sendc += sendcount[proc];
		if (recvcount != NULL)
			recvc += recvcount[proc];
	}

	/*
	*   event : ALLTOALLV_EV                 value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_ALLTOALLV_EV, EVT_BEGIN, EMPTY,
	  sendsize * sendc, me, c, recvsize * recvc);

	CtoF77 (pmpi_alltoallv) (sendbuf, sendcount, sdispls, sendtype,
	  recvbuf, recvcount, rdispls, recvtype, comm, ierror);

	/*
	*   event : ALLTOALLV_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, sendc * sendsize);

	TRACE_MPIEVENT (current_time, MPI_ALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}



/******************************************************************************
 ***  PMPI_Allgather_Wrapper
 ******************************************************************************/

void PMPI_Allgather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : ALLGATHER_EV                 value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_ALLGATHER_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
	  me, c, *recvcount * recvsize * csize);

	CtoF77 (pmpi_allgather) (sendbuf, sendcount, sendtype, recvbuf,
	  recvcount, recvtype, comm, ierror);

	/*
	*   event : ALLGATHER_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * csize, *sendcount * sendsize);

	TRACE_MPIEVENT (current_time, MPI_ALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Allgatherv_Wrapper
 ******************************************************************************/

void PMPI_Allgatherv_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (recvcounts != NULL)
	{	
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	if (recvcounts != NULL)
		for (proc = 0; proc < csize; proc++)
			recvc += recvcounts[proc];

	/*
	*   event : ALLGATHERV_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_ALLGATHERV_EV, EVT_BEGIN, EMPTY,
	  *sendcount * sendsize, me, c, recvsize * recvc);

	CtoF77 (pmpi_allgatherv) (sendbuf, sendcount, sendtype,
	  recvbuf, recvcounts, displs, recvtype, comm, ierror);

	/*
	*   event : ALLGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, *sendcount * sendsize);

	TRACE_MPIEVENT (current_time, MPI_ALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Gather_Wrapper
 ******************************************************************************/

void PMPI_Gather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : GATHER_EV                    value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
  TRACE_MPIEVENT (begin_time, MPI_GATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, *recvcount * recvsize * csize);
	}
	else
	{
  TRACE_MPIEVENT (begin_time, MPI_GATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, 0);
	}

	CtoF77 (pmpi_gather) (sendbuf, sendcount, sendtype, recvbuf,
	  recvcount, recvtype, root, comm, ierror);

	/*
	*   event : GATHER_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * csize, 0);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *sendcount * sendsize);
	}

	TRACE_MPIEVENT (current_time, MPI_GATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}



/******************************************************************************
 ***  PMPI_GatherV_Wrapper
 ******************************************************************************/

void PMPI_GatherV_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (recvcount != NULL)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : GATHERV_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
                if (recvcount != NULL)
                        for (proc = 0; proc < csize; proc++)
                                recvc += recvcount[proc];

  TRACE_MPIEVENT (begin_time, MPI_GATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, recvsize * recvc);
	}
	else
	{
  TRACE_MPIEVENT (begin_time, MPI_GATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, 0);
	}

	CtoF77 (pmpi_gatherv) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  displs, recvtype, root, comm, ierror);

	/*
	*   event : GATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, 0);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *sendcount * sendsize);
	}

	TRACE_MPIEVENT (current_time, MPI_GATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}



/******************************************************************************
 ***  PMPI_Scatter_Wrapper
 ******************************************************************************/

void PMPI_Scatter_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : SCATTER_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
    TRACE_MPIEVENT (begin_time, MPI_SCATTER_EV, EVT_BEGIN, *root,
		  *sendcount * sendsize * csize, me, c,
		  *recvcount * recvsize);
	}
	else
	{
    TRACE_MPIEVENT (begin_time, MPI_SCATTER_EV, EVT_BEGIN, *root, 0, me, c,
		  *recvcount * recvsize);
	}

	CtoF77 (pmpi_scatter) (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, recvtype, root, comm, ierror);

	/*
	*   event : SCATTER_EV                   value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *sendcount * sendsize * csize);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_SCATTER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_ScatterV_Wrapper
 ******************************************************************************/

void PMPI_ScatterV_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, recvsize, me, csize;
	int proc, sendsize, sendc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (sendcount != NULL)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event :  SCATTERV_EV                 value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
		if (sendcount != NULL)
			for (proc = 0; proc < csize; proc++)
				sendc += sendcount[proc];

    TRACE_MPIEVENT (begin_time, MPI_SCATTERV_EV, EVT_BEGIN, *root, sendsize * sendc, me,
		  c, *recvcount * recvsize);
	}
	else
	{
    TRACE_MPIEVENT (begin_time, MPI_SCATTERV_EV, EVT_BEGIN, *root, 0, me, c,
		  *recvcount * recvsize);
	}

	CtoF77 (pmpi_scatterv) (sendbuf, sendcount, displs, sendtype,
	  recvbuf, recvcount, recvtype, root, comm, ierror);

	/*
	*   event : SCATTERV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, sendc * sendsize);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_SCATTERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Reduce_Scatter_Wrapper
 ******************************************************************************/

void PMPI_Reduce_Scatter_Wrapper (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror)
{
	int me, size;
	int i;
	int sendcount = 0;
	int csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (recvcounts != NULL)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;


	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	if (recvcounts != NULL)
		for (i = 0; i < csize; i++)
			sendcount += recvcounts[i];

	/*
	*   type : REDUCESCAT_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_REDUCESCAT_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_reduce_scatter) (sendbuf, recvbuf, recvcounts, datatype,
		op, comm, ierror);

	/*
	*   event : REDUCESCAT_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == 0)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, sendcount * size, sendcount * size);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, recvcounts[me] * size, sendcount * size);
	}

	TRACE_MPIEVENT (current_time, MPI_REDUCESCAT_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Scan_Wrapper
 ******************************************************************************/

void PMPI_Scan_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	/*
	*   type : SCAN_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_SCAN_EV, EVT_BEGIN, *op, *count * size, me, c, EMPTY);

	CtoF77 (pmpi_scan) (sendbuf, recvbuf, count, datatype, op, comm, ierror);

	/*
	*   event : SCAN_EV                      value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm  : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me != csize - 1)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *count * size);
	}
	if (me != 0)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *count * size, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_SCAN_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Exscan_Wrapper
 ******************************************************************************/

void PMPI_Exscan_Wrapper(void *sendbuf, void *recvbuf, MPI_Fint *count,
                         MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
                         MPI_Fint *ierror)
{
	int me, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	/*
	*   type : EXSCAN_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
	TRACE_MPIEVENT(begin_time, MPI_EXSCAN_EV, EVT_BEGIN, *op, *count * size, me, c, EMPTY);

	CtoF77 (pmpi_exscan) (sendbuf, recvbuf, count, datatype, op, comm, ierror);

	/*
	*   event : EXSCAN_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm  : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;
	TRACE_MPIEVENT(current_time, MPI_EXSCAN_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me != csize - 1)
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *count * size);
	if (me != 0)
		xtr_stats_MPI_update_collective(begin_time, current_time, *count * size, 0);
}

#if defined(MPI3)

/******************************************************************************
 ***  PMPI_Ireduce_Wrapper
 ******************************************************************************/

void PMPI_Ireduce_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= *count;

	/*
	*   event : IREDUCE_EV                    value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send (non root) /received (root)
	*   tag : rank                           commid: communicator Id
	*   aux : root rank
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IREDUCE_EV, EVT_BEGIN, *op, size, me, c, *root);

	CtoF77 (pmpi_ireduce) (sendbuf, recvbuf, count, datatype, op, root, comm,
	  req, ierror);

	/*
	*   event : REDUCE_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, size, 0);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, size);
	}

	TRACE_MPIEVENT (current_time, MPI_IREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_IallReduce_Wrapper
 ******************************************************************************/

void PMPI_IallReduce_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= *count;

	/*
	*   event : IALLREDUCE_EV                 value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send and received
	*   tag : rank                           commid: communicator Id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IALLREDUCE_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_iallreduce) (sendbuf, recvbuf, count, datatype, op, comm,
	  req, ierror);

	/*
	*   event : IALLREDUCE_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, size, size);

	TRACE_MPIEVENT (current_time, MPI_IALLREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Ibarrier_Wrapper
 ******************************************************************************/

void PMPI_Ibarrier_Wrapper (MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
  MPI_Comm c = PMPI_Comm_f2c (*comm);
  int me, ret, csize;

  CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
  MPI_CHECK(ret, pmpi_comm_rank);

  CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  /*
   *   event : IBARRIER_EV                    value : EVT_BEGIN
   *   target : ---                          size  : ---
   *   tag : rank                            commid: communicator id
   *   aux : ---
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IBARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
                    EMPTY);
  }
#else
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IBARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
                  EMPTY);
#endif

  CtoF77 (pmpi_ibarrier) (comm, req, ierror);

  /*
   *   event : IBARRIER_EV                   value : EVT_END
   *   target : ---                         size  : size of the communicator
   *   tag : ---                            commid: communicator id
   *   aux : global op counter
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    iotimer_t current_time = TIME;
  TRACE_MPIEVENT (current_time, MPI_IBARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c,
                    Extrae_MPI_getCurrentOpGlobal());
  }
#else
  iotimer_t current_time = TIME;
#endif

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, 0, 0);

  TRACE_MPIEVENT (current_time, MPI_IBARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_BCast_Wrapper
 ******************************************************************************/

void PMPI_Ibcast_Wrapper (void *buffer, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= *count;

	/*
	*   event : IBCAST_EV                     value : EVT_BEGIN
	*   target : root_rank                   size  : message size
	*   tag : rank                           commid: communicator identifier
	*   aux : ---
	*/

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 1;
#endif

	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IBCAST_EV, EVT_BEGIN, *root, size, me, c, 
	  EMPTY);

	CtoF77 (pmpi_ibcast) (buffer, count, datatype, root, comm, req, ierror);

	/*
	*   event : IBCAST_EV                    value : EVT_END
	*   target : ---                        size  : size of the communicator
	*   tag : ---                           commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, size);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, size, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_IBCAST_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_IallToAll_Wrapper
 ******************************************************************************/

void PMPI_IallToAll_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, sendsize, recvsize, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : IALLTOALL_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IALLTOALL_EV, EVT_BEGIN, EMPTY,
	  *sendcount * sendsize, me, c, *recvcount * recvsize * csize);

	CtoF77 (pmpi_ialltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, comm, req, ierror);

	/*
	*   event : IALLTOALL_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * csize, *sendcount * sendsize);

	TRACE_MPIEVENT (current_time, MPI_IALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_IallToAllV_Wrapper
 ******************************************************************************/

void PMPI_IallToAllV_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *req,
	MPI_Fint *ierror)
{
	int me, ret, sendsize, recvsize, csize;
	int proc, sendc = 0, recvc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (sendcount != NULL)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (recvcount != NULL)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;
		
	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		if (sendcount != NULL)
			sendc += sendcount[proc];
		if (recvcount != NULL)
			recvc += recvcount[proc];
	}

	/*
	*   event : IALLTOALLV_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IALLTOALLV_EV, EVT_BEGIN, EMPTY,
	  sendsize * sendc, me, c, recvsize * recvc);

	CtoF77 (pmpi_ialltoallv) (sendbuf, sendcount, sdispls, sendtype,
	  recvbuf, recvcount, rdispls, recvtype, comm, req, ierror);

	/*
	*   event : IALLTOALLV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, sendc * sendsize);

	TRACE_MPIEVENT (current_time, MPI_IALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}



/******************************************************************************
 ***  PMPI_Iallgather_Wrapper
 ******************************************************************************/

void PMPI_Iallgather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : IALLGATHER_EV                 value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IALLGATHER_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
	  me, c, *recvcount * recvsize * csize);

	CtoF77 (pmpi_iallgather) (sendbuf, sendcount, sendtype, recvbuf,
	  recvcount, recvtype, comm, req, ierror);

	/*
	*   event : IALLGATHER_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * csize, *sendcount * sendsize);

	TRACE_MPIEVENT (current_time, MPI_IALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Iallgatherv_Wrapper
 ******************************************************************************/

void PMPI_Iallgatherv_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (recvcounts != NULL)
	{	
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	if (recvcounts != NULL)
		for (proc = 0; proc < csize; proc++)
			recvc += recvcounts[proc];

	/*
	*   event : IALLGATHERV_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IALLGATHERV_EV, EVT_BEGIN, EMPTY,
	  *sendcount * sendsize, me, c, recvsize * recvc);

	CtoF77 (pmpi_iallgatherv) (sendbuf, sendcount, sendtype,
	  recvbuf, recvcounts, displs, recvtype, comm, req, ierror);

	/*
	*   event : IALLGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, *sendcount * sendsize);

	TRACE_MPIEVENT (current_time, MPI_IALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Igather_Wrapper
 ******************************************************************************/

void PMPI_Igather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : IGATHER_EV                    value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
  TRACE_MPIEVENT (begin_time, MPI_IGATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, *recvcount * recvsize * csize);
	}
	else
	{
  TRACE_MPIEVENT (begin_time, MPI_IGATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, 0);
	}

	CtoF77 (pmpi_igather) (sendbuf, sendcount, sendtype, recvbuf,
	  recvcount, recvtype, root, comm, req, ierror);

	/*
	*   event : IGATHER_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * csize, 0);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *sendcount * sendsize);
	} 

	TRACE_MPIEVENT (current_time, MPI_IGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}



/******************************************************************************
 ***  PMPI_IgatherV_Wrapper
 ******************************************************************************/

void PMPI_IgatherV_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req,
	MPI_Fint *ierror)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (recvcount != NULL)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : IGATHERV_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
                if (recvcount != NULL)
                        for (proc = 0; proc < csize; proc++)
                                recvc += recvcount[proc];

  TRACE_MPIEVENT (begin_time, MPI_IGATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, recvsize * recvc);
	}
	else
	{
  TRACE_MPIEVENT (begin_time, MPI_IGATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, 0);
	}

	CtoF77 (pmpi_igatherv) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  displs, recvtype, root, comm, req, ierror);

	/*
	*   event : IGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, 0);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *sendcount * sendsize);
	}

	TRACE_MPIEVENT (current_time, MPI_GATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}



/******************************************************************************
 ***  PMPI_Iscatter_Wrapper
 ******************************************************************************/

void PMPI_Iscatter_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (*sendcount != 0)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event : ISCATTER_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
  TRACE_MPIEVENT (begin_time, MPI_ISCATTER_EV, EVT_BEGIN, *root,
		  *sendcount * sendsize * csize, me, c,
		  *recvcount * recvsize);
	}
	else
	{
  TRACE_MPIEVENT (begin_time, MPI_ISCATTER_EV, EVT_BEGIN, *root, 0, me, c,
		  *recvcount * recvsize);
	}

	CtoF77 (pmpi_iscatter) (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, recvtype, root, comm, req, ierror);

	/*
	*   event : ISCATTER_EV                   value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *sendcount * sendsize * csize);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_ISCATTER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_IscatterV_Wrapper
 ******************************************************************************/

void PMPI_IscatterV_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, recvsize, me, csize;
	int proc, sendsize, sendc = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	if (sendcount != NULL)
	{
		CtoF77 (pmpi_type_size) (sendtype, &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		sendsize = 0;

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (recvtype, &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		recvsize = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	/*
	*   event :  ISCATTERV_EV                 value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
  iotimer_t begin_time = LAST_READ_TIME;
	if (me == *root)
	{
		if (sendcount != NULL)
			for (proc = 0; proc < csize; proc++)
				sendc += sendcount[proc];

  TRACE_MPIEVENT (begin_time, MPI_ISCATTERV_EV, EVT_BEGIN, *root, sendsize * sendc, me,
		  c, *recvcount * recvsize);
	}
	else
	{
  TRACE_MPIEVENT (begin_time, MPI_ISCATTERV_EV, EVT_BEGIN, *root, 0, me, c,
		  *recvcount * recvsize);
	}

	CtoF77 (pmpi_iscatterv) (sendbuf, sendcount, displs, sendtype,
	  recvbuf, recvcount, recvtype, root, comm, req, ierror);

	/*
	*   event : ISCATTERV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == *root)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, sendc * sendsize);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_ISCATTERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Ireduce_Scatter_Wrapper
 ******************************************************************************/

void PMPI_Ireduce_Scatter_Wrapper (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
{
	int me, size;
	int i;
	int sendcount = 0;
	int csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (recvcounts != NULL)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;


	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	if (recvcounts != NULL)
		for (i = 0; i < csize; i++)
			sendcount += recvcounts[i];

	/*
	*   type : IREDUCESCAT_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IREDUCESCAT_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_ireduce_scatter) (sendbuf, recvbuf, recvcounts, datatype,
		op, comm, req, ierror);

	/*
	*   event : IREDUCESCAT_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == 0)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, sendcount * size, sendcount * size);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, recvcounts[me] * size, sendcount * size);
	}

	TRACE_MPIEVENT (current_time, MPI_IREDUCESCAT_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Iscan_Wrapper
 ******************************************************************************/

void PMPI_Iscan_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req,
	MPI_Fint *ierror)
{
	int me, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	/*
	*   type : ISCAN_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_ISCAN_EV, EVT_BEGIN, *op, *count * size, me, c, EMPTY);

	CtoF77 (pmpi_iscan) (sendbuf, recvbuf, count, datatype, op, comm, req, ierror);

	/*
	*   event : ISCAN_EV                      value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm  : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me != csize - 1)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *count * size);
	}
	if (me != 0)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *count * size, 0);
	}

	TRACE_MPIEVENT (current_time, MPI_ISCAN_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Iscan_Wrapper
 ******************************************************************************/

void PMPI_Iexscan_Wrapper(void *sendbuf, void *recvbuf, MPI_Fint *count,
                          MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
                          MPI_Fint *req, MPI_Fint *ierror)
{
	int me, size, csize;
	MPI_Comm c = PMPI_Comm_f2c(*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;

	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	/*
	*   type : IEXSCAN_EV                   value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
	TRACE_MPIEVENT (begin_time, MPI_IEXSCAN_EV, EVT_BEGIN, *op, *count * size, me, c, EMPTY);

	CtoF77 (pmpi_iexscan) (sendbuf, recvbuf, count, datatype, op, comm, req, ierror);

	/*
	*   event : IEXSCAN_EV                   value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm  : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;
	TRACE_MPIEVENT (current_time, MPI_IEXSCAN_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me != csize - 1)
		xtr_stats_MPI_update_collective(begin_time, current_time, 0, *count * size);
	if (me != 0)
		xtr_stats_MPI_update_collective(begin_time, current_time, *count * size, 0);
}

/******************************************************************************
 ***  PMPI_Reduce_Scatter_Block_Wrapper
 ******************************************************************************/

void PMPI_Reduce_Scatter_Block_Wrapper (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror)
{
	int me, size;
	int sendcount = 0;
	int csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (*recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;


	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	if (*recvcount != 0)
			sendcount += *recvcount;

	/*
	*   type : REDUCE_SCATTER_BLOCK_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_REDUCE_SCATTER_BLOCK_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_reduce_scatter_block) (sendbuf, recvbuf, recvcount, datatype,
		op, comm, ierror);

	/*
	*   event : REDUCE_SCATTER_BLOCK_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == 0)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, sendcount * size, sendcount * size);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * size, sendcount * size);
	}

	TRACE_MPIEVENT (current_time, MPI_REDUCE_SCATTER_BLOCK_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Ireduce_Scatter_Block_Wrapper
 ******************************************************************************/

void PMPI_Ireduce_Scatter_Block_Wrapper (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, size;
	int sendcount = 0;
	int csize;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

	CtoF77 (pmpi_comm_rank) (comm, &me, ierror);
	MPI_CHECK(*ierror, pmpi_comm_rank);

	if (recvcount != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, ierror);
		MPI_CHECK(*ierror, pmpi_type_size);
	}
	else
		size = 0;


	CtoF77 (pmpi_comm_size) (comm, &csize, ierror);
	MPI_CHECK(*ierror, pmpi_comm_size);

	if (*recvcount != 0)
			sendcount += *recvcount;

	/*
	*   type : IREDUCE_SCATTER_BLOCK_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IREDUCE_SCATTER_BLOCK_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_ireduce_scatter_block) (sendbuf, recvbuf, recvcount, datatype,
		op, comm, req, ierror);

	/*
	*   event : IREDUCE_SCATTER_BLOCK_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	if (me == 0)
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, sendcount * size, sendcount * size);
	}
	else
	{
		xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * size, sendcount * size);
	}

	TRACE_MPIEVENT (current_time, MPI_IREDUCE_SCATTER_BLOCK_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_AllToAllW_Wrapper
 ******************************************************************************/

void PMPI_AllToAllW_Wrapper (void *sendbuf, MPI_Fint *sendcounts,
	MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *rdispls, MPI_Fint *recvtypes,	MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, csize;
	int proc, sendbytes = 0, recvbytes = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

		
	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		int sendsize;			
		CtoF77 (pmpi_type_size) (&sendtypes[proc], &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);

		int recvsize;
		CtoF77 (pmpi_type_size) (&recvtypes[proc], &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	

		if (sendcounts != NULL)
			sendbytes += sendcounts[proc] * sendsize;
		if (recvcounts != NULL)
			recvbytes += recvcounts[proc] * recvsize;
	}

	/*
	*   event : ALLTOALLW_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_ALLTOALLW_EV, EVT_BEGIN, EMPTY,
	  sendbytes, me, c, recvbytes);

	CtoF77 (pmpi_alltoallw) (sendbuf, sendcounts, sdispls, sendtypes,
	  recvbuf, recvcounts, rdispls, recvtypes, comm, ierror);

	/*
	*   event : ALLTOALLW_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, recvbytes, sendbytes);

	TRACE_MPIEVENT (current_time, MPI_ALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_IallToAllW_Wrapper
 ******************************************************************************/

void PMPI_IallToAllW_Wrapper (void *sendbuf, MPI_Fint *sendcounts,
	MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *rdispls, MPI_Fint *recvtypes,	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, csize;
	int proc, sendbytes = 0, recvbytes = 0;
	MPI_Comm c = PMPI_Comm_f2c (*comm);

		
	CtoF77 (pmpi_comm_size) (comm, &csize, &ret);
	MPI_CHECK(ret, pmpi_comm_size);

	CtoF77 (pmpi_comm_rank) (comm, &me, &ret);
	MPI_CHECK(ret, pmpi_comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		int sendsize;			
		CtoF77 (pmpi_type_size) (&sendtypes[proc], &sendsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);

		int recvsize;
		CtoF77 (pmpi_type_size) (&recvtypes[proc], &recvsize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	

		if (sendcounts != NULL)
			sendbytes += sendcounts[proc] * sendsize;
		if (recvcounts != NULL)
			recvbytes += recvcounts[proc] * recvsize;
	}

	/*
	*   event : IALLTOALLW_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_IALLTOALLW_EV, EVT_BEGIN, EMPTY,
	  sendbytes, me, c, recvbytes);

	CtoF77 (pmpi_ialltoallw) (sendbuf, sendcounts, sdispls, sendtypes,
	  recvbuf, recvcounts, rdispls, recvtypes, comm, req, ierror);

	/*
	*   event : IALLTOALLW_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	iotimer_t current_time = TIME;

	/* MPI Stats */
	xtr_stats_MPI_update_collective(begin_time, current_time, recvbytes, sendbytes);

	TRACE_MPIEVENT (current_time, MPI_IALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/**
 * xtr_mpi_comm_neighbors_count 
 * 
 * Returns the number of neighbors of the given process topology. If the communicator 'comm' was created with...
 * 1) MPI_Graph_create, both 'indegree' and 'outdegree' are set to the number of neighbors retrieved by MPI_Graph_neighbors_count
 * 2) MPI_Cart_create, both 'indegree' and 'outdegree' are set to (2 * ndims), where ndims is retrieved by MPI_Cartdim_get
 * 3) MPI_Dist_graph_create, 'indegree' and 'outdegree' are set to the corresponding return value of MPI_Dist_graph_neighbors_count
 * 4) In any other case, 'indegree' and 'outdegree' are set to 0
 * 
 * @return the rank of the current task
 */
int xtr_mpi_comm_neighbors_count(MPI_Fint *comm, int *indegree, int *outdegree)
{
  int ret = 0, me = 0, status = 0, nneighbors = 0, ndims = 0;

  CtoF77(pmpi_comm_rank) (comm, &me, &ret);
  MPI_CHECK(ret, pmpi_comm_rank);

  CtoF77(pmpi_topo_test) (comm, &status, &ret);
  MPI_CHECK(ret, pmpi_topo_test);

  switch (status)
  {
    case MPI_GRAPH:
    {
      CtoF77(pmpi_graph_neighbors_count) (comm, &me, &nneighbors, &ret);
      MPI_CHECK(ret, pmpi_graph_neighbors_count);
      if (indegree  != NULL) *indegree  = nneighbors;
      if (outdegree != NULL) *outdegree = nneighbors;
      break;
    }
    case MPI_CART:
    {
      CtoF77(pmpi_cartdim_get) (comm, &ndims, &ret);
      MPI_CHECK(ret, pmpi_cartdim_get);
      if (indegree  != NULL) *indegree  = (2 * ndims);
      if (outdegree != NULL) *outdegree = (2 * ndims);
      break;
    }
    case MPI_DIST_GRAPH:
    {
      int local_indegree, local_outdegree, weighted; 
      CtoF77(pmpi_dist_graph_neighbors_count) (comm, &local_indegree, &local_outdegree, &weighted, &ret);
      MPI_CHECK(ret, pmpi_dist_graph_neighbors_count)
		if (indegree  != NULL) *indegree  = local_indegree;
		if (outdegree != NULL) *outdegree = local_outdegree;
      break;
    }
    case MPI_UNDEFINED:
    default:
    {
      if (indegree  != NULL) *indegree  = 0;
      if (outdegree != NULL) *outdegree = 0;
      break;
    }
  }
  return me;
}

/******************************************************************************
 ***  PMPI_Graph_create_Wrapper
 ******************************************************************************/

void PMPI_Graph_create_Wrapper (MPI_Fint *comm_old, MPI_Fint *nnodes, MPI_Fint *index, MPI_Fint *edges, MPI_Fint *reorder, MPI_Fint *comm_graph, MPI_Fint *ierr)
{
        MPI_Fint cnull;

        cnull = MPI_Comm_c2f(MPI_COMM_NULL);

	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_GRAPH_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	CtoF77 (pmpi_graph_create) (comm_old, nnodes, index, edges, reorder, comm_graph, ierr);

        if (*comm_graph != cnull && *ierr == MPI_SUCCESS)
        {
                MPI_Comm comm_id = PMPI_Comm_f2c(*comm_graph);
                Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
        }

	iotimer_t current_time = TIME;
	xtr_stats_MPI_update_other(begin_time, current_time);
  TRACE_MPIEVENT (current_time, MPI_GRAPH_CREATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

}

/******************************************************************************
 ***  PMPI_Dist_graph_create_Wrapper
 ******************************************************************************/

void PMPI_Dist_graph_create_Wrapper (MPI_Fint *comm_old, MPI_Fint *n, MPI_Fint *sources, MPI_Fint *degrees, MPI_Fint *destinations, MPI_Fint *weights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr)
{
	MPI_Fint cnull;

	cnull = MPI_Comm_c2f(MPI_COMM_NULL);

	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_DIST_GRAPH_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	CtoF77 (pmpi_dist_graph_create) (comm_old, n, sources, degrees, destinations, weights, info, reorder, comm_dist_graph, ierr);

	if (*comm_dist_graph != cnull && *ierr == MPI_SUCCESS)
	{
		MPI_Comm comm_id = PMPI_Comm_f2c(*comm_dist_graph);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	iotimer_t current_time = TIME;
	xtr_stats_MPI_update_other(begin_time, current_time);
  TRACE_MPIEVENT (current_time, MPI_DIST_GRAPH_CREATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

}

/******************************************************************************
 ***  PMPI_Dist_graph_create_adjacent_Wrapper
 ******************************************************************************/

void PMPI_Dist_graph_create_adjacent_Wrapper (MPI_Fint *comm_old, MPI_Fint *indegree, MPI_Fint *sources, MPI_Fint *sourceweights, MPI_Fint *outdegree, MPI_Fint *destinations, MPI_Fint *destweights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr)
{
	MPI_Fint cnull;

	cnull = MPI_Comm_c2f(MPI_COMM_NULL);

	iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_DIST_GRAPH_CREATE_ADJACENT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	CtoF77 (pmpi_dist_graph_create_adjacent) (comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph, ierr);

	if (*comm_dist_graph != cnull && *ierr == MPI_SUCCESS)
	{
		MPI_Comm comm_id = PMPI_Comm_f2c(*comm_dist_graph);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	iotimer_t current_time = TIME;
	xtr_stats_MPI_update_other(begin_time, current_time);
  TRACE_MPIEVENT (current_time, MPI_DIST_GRAPH_CREATE_ADJACENT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

}

/******************************************************************************
 ***  PMPI_Neighbor_allgather_Wrapper
 ******************************************************************************/

void PMPI_Neighbor_allgather_Wrapper (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
{
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (*sendcount != 0)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  if (*recvcount != 0)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, NULL);

  /*
   *   event  : NEIGHBOR_ALLGATHER_EV       value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_NEIGHBOR_ALLGATHER_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
    me, c, *recvcount * recvsize * indegree);                                

  CtoF77(pmpi_neighbor_allgather) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

  /*
   *   event  : NEIGHBOR_ALLGATHER_EV       value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * indegree, *sendcount * sendsize);

  TRACE_MPIEVENT (current_time, MPI_NEIGHBOR_ALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Ineighbor_allgather_Wrapper
 ******************************************************************************/

void PMPI_Ineighbor_allgather_Wrapper (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
{
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (*sendcount != 0)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  if (*recvcount != 0)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, NULL);

  /*
   *   event  : INEIGHBOR_ALLGATHER_EV      value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_INEIGHBOR_ALLGATHER_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
    me, c, *recvcount * recvsize * indegree);

  CtoF77(pmpi_ineighbor_allgather) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

  /*
   *   event  : INEIGHBOR_ALLGATHER_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * indegree, *sendcount * sendsize);

  TRACE_MPIEVENT (current_time, MPI_INEIGHBOR_ALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}

/******************************************************************************
 ***  PMPI_Neighbor_allgatherv_Wrapper
 ******************************************************************************/

void PMPI_Neighbor_allgatherv_Wrapper (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
{
  int ret = 0, proc = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0, recvc = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (*sendcount != 0)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  if (recvcounts != NULL)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, NULL);

  if (recvcounts != NULL)
  {
    for (proc = 0; proc < indegree; proc ++)
    {
      recvc += recvcounts[proc];
    }
  }

  /*
   *   event  : NEIGHBOR_ALLGATHERV_EV      value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_NEIGHBOR_ALLGATHERV_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
    me, c, recvsize * recvc);

  CtoF77(pmpi_neighbor_allgatherv) (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);

  /*
   *   event  : NEIGHBOR_ALLGATHERV_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, *sendcount * sendsize);

  TRACE_MPIEVENT (current_time, MPI_NEIGHBOR_ALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Ineighbor_allgatherv_Wrapper
 ******************************************************************************/

void PMPI_Ineighbor_allgatherv_Wrapper (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
{
  int ret = 0, proc = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0, recvc = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (*sendcount != 0)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcounts != NULL)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, NULL);

  if (recvcounts != NULL)
  {
    for (proc = 0; proc < indegree; proc ++)
    {
      recvc += recvcounts[proc];
    }
  }

  /*
   *   event  : INEIGHBOR_ALLGATHERV_EV     value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_INEIGHBOR_ALLGATHERV_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
    me, c, recvsize * recvc);

  CtoF77(pmpi_ineighbor_allgatherv) (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);

  /*
   *   event  : INEIGHBOR_ALLGATHERV_EV     value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, *sendcount * sendsize);

  TRACE_MPIEVENT (current_time, MPI_INEIGHBOR_ALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Neighbor_alltoall_Wrapper
 ******************************************************************************/

void PMPI_Neighbor_alltoall_Wrapper (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
{ 
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (*sendcount != 0)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  if (*recvcount != 0)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, NULL);

  /*
   *   event  : NEIGHBOR_ALLTOALL_EV        value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_NEIGHBOR_ALLTOALL_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
    me, c, *recvcount * recvsize * indegree);

  CtoF77(pmpi_neighbor_alltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

  /*
   *   event  : NEIGHBOR_ALLTOALL_EV        value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * indegree, *sendcount * sendsize);
  TRACE_MPIEVENT (current_time, MPI_NEIGHBOR_ALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Ineighbor_alltoall_Wrapper
 ******************************************************************************/

void PMPI_Ineighbor_alltoall_Wrapper (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
{
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (*sendcount != 0)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  if (*recvcount != 0)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, NULL);

  /*
   *   event  : INEIGHBOR_ALLTOALL_EV       value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_INEIGHBOR_ALLTOALL_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
    me, c, *recvcount * recvsize * indegree);

  CtoF77(pmpi_ineighbor_alltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

  /*
   *   event  : INEIGHBOR_ALLTOALL_EV        value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, *recvcount * recvsize * indegree, *sendcount * sendsize);
  TRACE_MPIEVENT (current_time, MPI_INEIGHBOR_ALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Neighbor_alltoallv_Wrapper
 ******************************************************************************/

void PMPI_Neighbor_alltoallv_Wrapper (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
{
  int proc = 0, ret = 0, me = 0, sendsize = 0, sendc = 0, recvsize = 0, recvc = 0, csize = 0, indegree = 0, outdegree = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (sendcounts != NULL)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  if (recvcounts != NULL)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, &outdegree);

  if (sendcounts != NULL)
  {
    for (proc = 0; proc < outdegree; proc++)
    {
      sendc += sendcounts[proc];
    }
  }
  if (recvcounts != NULL)
  {
    for (proc = 0; proc < indegree; proc++)
    {
      recvc += recvcounts[proc];
    }               
  }

  /*
   *   event  : NEIGHBOR_ALLTOALLV_EV       value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_NEIGHBOR_ALLTOALLV_EV, EVT_BEGIN, EMPTY, sendsize * sendc,
    me, c, recvsize * recvc);

  CtoF77(pmpi_neighbor_alltoallv) (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);

  /*
   *   event  : NEIGHBOR_ALLTOALLV_EV       value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, sendc * sendsize);

  TRACE_MPIEVENT (current_time, MPI_NEIGHBOR_ALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Ineighbor_alltoallv_Wrapper
 ******************************************************************************/

void PMPI_Ineighbor_alltoallv_Wrapper (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
{
  int proc = 0, ret = 0, me = 0, sendsize = 0, sendc = 0, recvsize = 0, recvc = 0, csize = 0, indegree = 0, outdegree = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  if (sendcounts != NULL)
  {
    CtoF77(pmpi_type_size) (sendtype, &sendsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  if (recvcounts != NULL)
  {
    CtoF77(pmpi_type_size) (recvtype, &recvsize, &ret);
    MPI_CHECK(ret, pmpi_type_size);
  }

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, &outdegree);

  if (sendcounts != NULL)
  {
    for (proc = 0; proc < outdegree; proc++)
    {
      sendc += sendcounts[proc];
    }
  }
  if (recvcounts != NULL)
  {
    for (proc = 0; proc < indegree; proc++)
    {
      recvc += recvcounts[proc];
    }               
  }

  /*
   *   event  : INEIGHBOR_ALLTOALLV_EV      value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_INEIGHBOR_ALLTOALLV_EV, EVT_BEGIN, EMPTY, sendsize * sendc,
    me, c, recvsize * recvc);

  CtoF77(pmpi_ineighbor_alltoallv) (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);

  /*
   *   event  : INEIGHBOR_ALLTOALLV_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, recvc * recvsize, sendc * sendsize);
  TRACE_MPIEVENT (current_time, MPI_INEIGHBOR_ALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Neighbor_alltoallw_Wrapper
 ******************************************************************************/

void PMPI_Neighbor_alltoallw_Wrapper (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierr)
{
  int ret = 0, proc = 0, me = 0, csize = 0, indegree = 0, outdegree = 0, sendbytes = 0, recvbytes = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, &outdegree);

  for (proc = 0; proc < outdegree; proc++)
  {
    int sendsize = 0;

    if (sendtypes != NULL)
    {
      CtoF77(pmpi_type_size) (&sendtypes[proc], &sendsize, &ret);
      MPI_CHECK(ret, pmpi_type_size);

      if (sendcounts != NULL)
      {
        sendbytes += sendcounts[proc] * sendsize;
      }
    }
  }

  for (proc = 0; proc < indegree; proc++)
  {
    int recvsize = 0;

    if (recvtypes != NULL)
    {
      CtoF77(pmpi_type_size) (&recvtypes[proc], &recvsize, &ret);
      MPI_CHECK(ret, pmpi_type_size);

      if (recvcounts != NULL)
      {
        recvbytes += recvcounts[proc] * recvsize;
      }
    }
  }

  /*
   *   event  : NEIGHBOR_ALLTOALLW_EV       value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_NEIGHBOR_ALLTOALLW_EV, EVT_BEGIN, EMPTY, sendbytes,
    me, c, recvbytes);

  CtoF77(pmpi_neighbor_alltoallw) (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);

  /*
   *   event  : NEIGHBOR_ALLTOALLW_EV       value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, recvbytes, sendbytes);
  TRACE_MPIEVENT (current_time, MPI_NEIGHBOR_ALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());
}


/******************************************************************************
 ***  PMPI_Ineighbor_alltoallw_Wrapper
 ******************************************************************************/

void PMPI_Ineighbor_alltoallw_Wrapper (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
{
  int proc = 0, me = 0, ret = 0, csize = 0, indegree = 0, outdegree = 0, sendbytes = 0, recvbytes = 0;
  MPI_Comm c = PMPI_Comm_f2c (*comm);

  CtoF77(pmpi_comm_size) (comm, &csize, &ret);
  MPI_CHECK(ret, pmpi_comm_size);

  me = xtr_mpi_comm_neighbors_count (comm, &indegree, &outdegree);

  for (proc = 0; proc < outdegree; proc++)
  {
    int sendsize = 0;

    if (sendtypes != NULL)
    {
      CtoF77(pmpi_type_size) (&sendtypes[proc], &sendsize, &ret);
      MPI_CHECK(ret, pmpi_type_size);

      if (sendcounts != NULL)
      {
        sendbytes += sendcounts[proc] * sendsize;
      }
    }
  }

  for (proc = 0; proc < indegree; proc++)
  {
    int recvsize = 0;

    if (recvtypes != NULL)
    {
      CtoF77(pmpi_type_size) (&recvtypes[proc], &recvsize, &ret);
      MPI_CHECK(ret, pmpi_type_size);

      if (recvcounts != NULL)
      {
        recvbytes += recvcounts[proc] * recvsize;
      }
    }
  }

  /*
   *   event  : INEIGHBOR_ALLTOALLW_EV      value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  iotimer_t begin_time = LAST_READ_TIME;
  TRACE_MPIEVENT (begin_time, MPI_INEIGHBOR_ALLTOALLW_EV, EVT_BEGIN, EMPTY, sendbytes,
    me, c, recvbytes);

  CtoF77(pmpi_ineighbor_alltoallw) (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);

  /*
   *   event  : INEIGHBOR_ALLTOALLW_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  iotimer_t current_time = TIME;

  /* MPI Stats */
  xtr_stats_MPI_update_collective(begin_time, current_time, recvbytes, sendbytes);
  TRACE_MPIEVENT (current_time, MPI_INEIGHBOR_ALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, c,Extrae_MPI_getCurrentOpGlobal());
}

#endif /* MPI3 */

#endif /* defined(FORTRAN_SYMBOLS) */

