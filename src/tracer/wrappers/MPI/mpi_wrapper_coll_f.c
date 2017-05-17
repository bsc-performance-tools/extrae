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
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REDUCE_EV, EVT_BEGIN, *op, size, me, c, *root);

	CtoF77 (pmpi_reduce) (sendbuf, recvbuf, count, datatype, op, root, comm,
	  ierror);

	/*
	*   event : REDUCE_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_REDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, size, 0);
	else
		updateStats_COLLECTIVE(global_mpi_stats, 0, size);
}


/******************************************************************************
 ***  PMPI_AllReduce_Wrapper
 ******************************************************************************/

void PMPI_AllReduce_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLREDUCE_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_allreduce) (sendbuf, recvbuf, count, datatype, op, comm,
	  ierror);

	/*
	*   event : ALLREDUCE_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, size, size);
}


/******************************************************************************
 ***  PMPI_Barrier_Wrapper
 ******************************************************************************/

void PMPI_Barrier_Wrapper (MPI_Fint *comm, MPI_Fint *ierror)
{
  MPI_Comm c = MPI_Comm_f2c (*comm);
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
    TRACE_MPIEVENT (LAST_READ_TIME, MPI_BARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
                    EMPTY);
  }
#else
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_BARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
                  EMPTY);
#endif

  CtoF77 (pmpi_barrier) (comm, ierror);

  /*
   *   event : BARRIER_EV                   value : EVT_END
   *   target : ---                         size  : size of the communicator
   *   tag : ---                            commid: communicator id
   *   aux : global op counter
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    TRACE_MPIEVENT (TIME, MPI_BARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c,
                    Extrae_MPI_getCurrentOpGlobal());
  }
#else
  TRACE_MPIEVENT (TIME, MPI_BARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c,
                  Extrae_MPI_getCurrentOpGlobal());
#endif

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, 0, 0);
}

/******************************************************************************
 ***  PMPI_BCast_Wrapper
 ******************************************************************************/

void PMPI_BCast_Wrapper (void *buffer, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BCAST_EV, EVT_BEGIN, *root, size, me, c, 
	  EMPTY);

	CtoF77 (pmpi_bcast) (buffer, count, datatype, root, comm, ierror);

	/*
	*   event : BCAST_EV                    value : EVT_END
	*   target : ---                        size  : size of the communicator
	*   tag : ---                           commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_BCAST_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, 0, size);
	else
		updateStats_COLLECTIVE(global_mpi_stats, size, 0);
}

/******************************************************************************
 ***  PMPI_AllToAll_Wrapper
 ******************************************************************************/

void PMPI_AllToAll_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, ret, sendsize, recvsize, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	*   target : received size               size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLTOALL_EV, EVT_BEGIN, *recvcount * recvsize,
	  *sendcount * sendsize, me, c, EMPTY);

	CtoF77 (pmpi_alltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, comm, ierror);

	/*
	*   event : ALLTOALL_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize,  *sendcount * sendsize);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	*   event : ALLTOALLV_EV                  value : EVT_BEGIN
	*   target : received size               size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLTOALLV_EV, EVT_BEGIN, recvsize * recvc,
	  sendsize * sendc, me, c, EMPTY);

	CtoF77 (pmpi_alltoallv) (sendbuf, sendcount, sdispls, sendtype,
	  recvbuf, recvcount, rdispls, recvtype, comm, ierror);

	/*
	*   event : ALLTOALLV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendc * sendsize);
}



/******************************************************************************
 ***  PMPI_Allgather_Wrapper
 ******************************************************************************/

void PMPI_Allgather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLGATHER_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
	  me, c, *recvcount * recvsize * csize);

	CtoF77 (pmpi_allgather) (sendbuf, sendcount, sendtype, recvbuf,
	  recvcount, recvtype, comm, ierror);

	/*
	*   event : ALLGATHER_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize * csize, *sendcount * sendsize);
}


/******************************************************************************
 ***  PMPI_Allgatherv_Wrapper
 ******************************************************************************/

void PMPI_Allgatherv_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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

	if (recvcount != NULL)
		for (proc = 0; proc < csize; proc++)
			recvc += recvcount[proc];

	/*
	*   event : ALLGATHERV_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLGATHERV_EV, EVT_BEGIN, EMPTY,
	  *sendcount * sendsize, me, c, recvsize * recvc);

	CtoF77 (pmpi_allgatherv) (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, displs, recvtype, comm, ierror);

	/*
	*   event : ALLGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY,
	  c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, *sendcount * sendsize);
}


/******************************************************************************
 ***  PMPI_Gather_Wrapper
 ******************************************************************************/

void PMPI_Gather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, *recvcount * recvsize * csize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
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
	TRACE_MPIEVENT (TIME, MPI_GATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize * csize, 0);
	else 
		updateStats_COLLECTIVE(global_mpi_stats, 0, *sendcount * sendsize);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
                if (recvcount != NULL)
                        for (proc = 0; proc < csize; proc++)
                                recvc += recvcount[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, recvsize * recvc);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
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
	TRACE_MPIEVENT (TIME, MPI_GATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, 0);
	else
		updateStats_COLLECTIVE(global_mpi_stats, 0, *sendcount * sendsize);
}



/******************************************************************************
 ***  PMPI_Scatter_Wrapper
 ******************************************************************************/

void PMPI_Scatter_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTER_EV, EVT_BEGIN, *root,
		  *sendcount * sendsize * csize, me, c,
		  *recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTER_EV, EVT_BEGIN, *root, 0, me, c,
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
	TRACE_MPIEVENT (TIME, MPI_SCATTER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, 0, *sendcount * sendsize * csize);
	else
		updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize, 0);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
		if (sendcount != NULL)
			for (proc = 0; proc < csize; proc++)
				sendc += sendcount[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTERV_EV, EVT_BEGIN, *root, sendsize * sendc, me,
		  c, *recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTERV_EV, EVT_BEGIN, *root, 0, me, c,
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
	TRACE_MPIEVENT (TIME, MPI_SCATTERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, 0, sendc * sendsize);
	else
		updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize, 0);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REDUCESCAT_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_reduce_scatter) (sendbuf, recvbuf, recvcounts, datatype,
		op, comm, ierror);

	/*
	*   event : REDUCESCAT_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_REDUCESCAT_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == 0) 
        updateStats_COLLECTIVE(global_mpi_stats, sendcount * size, sendcount * size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcounts[me] * size, sendcount * size);
}

/******************************************************************************
 ***  PMPI_Scan_Wrapper
 ******************************************************************************/

void PMPI_Scan_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
{
	int me, size, csize;
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCAN_EV, EVT_BEGIN, *op, *count * size, me, c, EMPTY);

	CtoF77 (pmpi_scan) (sendbuf, recvbuf, count, datatype, op, comm, ierror);

	/*
	*   event : SCAN_EV                      value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm  : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_SCAN_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me != csize - 1)
        updateStats_COLLECTIVE(global_mpi_stats, 0, *count * size);
	if (me != 0)
        updateStats_COLLECTIVE(global_mpi_stats, *count * size, 0);
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
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IREDUCE_EV, EVT_BEGIN, *op, size, me, c, *root);

	CtoF77 (pmpi_ireduce) (sendbuf, recvbuf, count, datatype, op, root, comm,
	  req, ierror);

	/*
	*   event : REDUCE_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, size, 0);
	else
		updateStats_COLLECTIVE(global_mpi_stats, 0, size);
}


/******************************************************************************
 ***  PMPI_IallReduce_Wrapper
 ******************************************************************************/

void PMPI_IallReduce_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLREDUCE_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_iallreduce) (sendbuf, recvbuf, count, datatype, op, comm,
	  req, ierror);

	/*
	*   event : IALLREDUCE_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, size, size);
}


/******************************************************************************
 ***  PMPI_Ibarrier_Wrapper
 ******************************************************************************/

void PMPI_Ibarrier_Wrapper (MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
  MPI_Comm c = MPI_Comm_f2c (*comm);
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
    TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
                    EMPTY);
  }
#else
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, c,
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
    TRACE_MPIEVENT (TIME, MPI_IBARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c,
                    Extrae_MPI_getCurrentOpGlobal());
  }
#else
  TRACE_MPIEVENT (TIME, MPI_IBARRIER_EV, EVT_END, EMPTY, csize, EMPTY, c,
                  Extrae_MPI_getCurrentOpGlobal());
#endif

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, 0, 0);
}

/******************************************************************************
 ***  PMPI_BCast_Wrapper
 ******************************************************************************/

void PMPI_Ibcast_Wrapper (void *buffer, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, size, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBCAST_EV, EVT_BEGIN, *root, size, me, c, 
	  EMPTY);

	CtoF77 (pmpi_ibcast) (buffer, count, datatype, root, comm, req, ierror);

	/*
	*   event : IBCAST_EV                    value : EVT_END
	*   target : ---                        size  : size of the communicator
	*   tag : ---                           commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IBCAST_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, 0, size);
	else
		updateStats_COLLECTIVE(global_mpi_stats, size, 0);
}

/******************************************************************************
 ***  PMPI_IallToAll_Wrapper
 ******************************************************************************/

void PMPI_IallToAll_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int me, ret, sendsize, recvsize, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	*   target : received size               size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLTOALL_EV, EVT_BEGIN, *recvcount * recvsize,
	  *sendcount * sendsize, me, c, EMPTY);

	CtoF77 (pmpi_ialltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, comm, req, ierror);

	/*
	*   event : IALLTOALL_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize,  *sendcount * sendsize);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	*   target : received size               size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLTOALLV_EV, EVT_BEGIN, recvsize * recvc,
	  sendsize * sendc, me, c, EMPTY);

	CtoF77 (pmpi_ialltoallv) (sendbuf, sendcount, sdispls, sendtype,
	  recvbuf, recvcount, rdispls, recvtype, comm, req, ierror);

	/*
	*   event : IALLTOALLV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendc * sendsize);
}



/******************************************************************************
 ***  PMPI_Iallgather_Wrapper
 ******************************************************************************/

void PMPI_Iallgather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLGATHER_EV, EVT_BEGIN, EMPTY, *sendcount * sendsize,
	  me, c, *recvcount * recvsize * csize);

	CtoF77 (pmpi_iallgather) (sendbuf, sendcount, sendtype, recvbuf,
	  recvcount, recvtype, comm, req, ierror);

	/*
	*   event : IALLGATHER_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize * csize, *sendcount * sendsize);
}


/******************************************************************************
 ***  PMPI_Iallgatherv_Wrapper
 ******************************************************************************/

void PMPI_Iallgatherv_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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

	if (recvcount != NULL)
		for (proc = 0; proc < csize; proc++)
			recvc += recvcount[proc];

	/*
	*   event : IALLGATHERV_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLGATHERV_EV, EVT_BEGIN, EMPTY,
	  *sendcount * sendsize, me, c, recvsize * recvc);

	CtoF77 (pmpi_iallgatherv) (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, displs, recvtype, comm, req, ierror);

	/*
	*   event : IALLGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY,
	  c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, *sendcount * sendsize);
}


/******************************************************************************
 ***  PMPI_Igather_Wrapper
 ******************************************************************************/

void PMPI_Igather_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, *recvcount * recvsize * csize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHER_EV, EVT_BEGIN, *root, *sendcount * sendsize,
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
	TRACE_MPIEVENT (TIME, MPI_IGATHER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize * csize, 0);
	else 
		updateStats_COLLECTIVE(global_mpi_stats, 0, *sendcount * sendsize);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
                if (recvcount != NULL)
                        for (proc = 0; proc < csize; proc++)
                                recvc += recvcount[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
		  me, c, recvsize * recvc);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHERV_EV, EVT_BEGIN, *root, *sendcount * sendsize,
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
	TRACE_MPIEVENT (TIME, MPI_GATHERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, 0);
	else
		updateStats_COLLECTIVE(global_mpi_stats, 0, *sendcount * sendsize);
}



/******************************************************************************
 ***  PMPI_Iscatter_Wrapper
 ******************************************************************************/

void PMPI_Iscatter_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
{
	int ret, sendsize, recvsize, me, csize;
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTER_EV, EVT_BEGIN, *root,
		  *sendcount * sendsize * csize, me, c,
		  *recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTER_EV, EVT_BEGIN, *root, 0, me, c,
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
	TRACE_MPIEVENT (TIME, MPI_ISCATTER_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, 0, *sendcount * sendsize * csize);
	else
		updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize, 0);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	if (me == *root)
	{
		if (sendcount != NULL)
			for (proc = 0; proc < csize; proc++)
				sendc += sendcount[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTERV_EV, EVT_BEGIN, *root, sendsize * sendc, me,
		  c, *recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTERV_EV, EVT_BEGIN, *root, 0, me, c,
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
	TRACE_MPIEVENT (TIME, MPI_ISCATTERV_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == *root)
		updateStats_COLLECTIVE(global_mpi_stats, 0, sendc * sendsize);
	else
		updateStats_COLLECTIVE(global_mpi_stats, *recvcount * recvsize, 0);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IREDUCESCAT_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_ireduce_scatter) (sendbuf, recvbuf, recvcounts, datatype,
		op, comm, req, ierror);

	/*
	*   event : IREDUCESCAT_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IREDUCESCAT_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == 0) 
        updateStats_COLLECTIVE(global_mpi_stats, sendcount * size, sendcount * size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcounts[me] * size, sendcount * size);
}

/******************************************************************************
 ***  PMPI_Iscan_Wrapper
 ******************************************************************************/

void PMPI_Iscan_Wrapper (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req,
	MPI_Fint *ierror)
{
	int me, size, csize;
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCAN_EV, EVT_BEGIN, *op, *count * size, me, c, EMPTY);

	CtoF77 (pmpi_iscan) (sendbuf, recvbuf, count, datatype, op, comm, req, ierror);

	/*
	*   event : ISCAN_EV                      value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm  : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ISCAN_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me != csize - 1)
        updateStats_COLLECTIVE(global_mpi_stats, 0, *count * size);
	if (me != 0)
        updateStats_COLLECTIVE(global_mpi_stats, *count * size, 0);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REDUCE_SCATTER_BLOCK_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_reduce_scatter_block) (sendbuf, recvbuf, recvcount, datatype,
		op, comm, ierror);

	/*
	*   event : REDUCE_SCATTER_BLOCK_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_REDUCE_SCATTER_BLOCK_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == 0) 
        updateStats_COLLECTIVE(global_mpi_stats, sendcount * size, sendcount * size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, *recvcount * size, sendcount * size);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IREDUCE_SCATTER_BLOCK_EV, EVT_BEGIN, *op, size, me, c, EMPTY);

	CtoF77 (pmpi_ireduce_scatter_block) (sendbuf, recvbuf, recvcount, datatype,
		op, comm, req, ierror);

	/*
	*   event : IREDUCE_SCATTER_BLOCK_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            com   : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IREDUCE_SCATTER_BLOCK_EV, EVT_END, EMPTY, csize, EMPTY, c, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == 0) 
        updateStats_COLLECTIVE(global_mpi_stats, sendcount * size, sendcount * size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, *recvcount * size, sendcount * size);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

		
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
	*   target : received size               size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLTOALLW_EV, EVT_BEGIN, recvbytes,
	  sendbytes, me, c, EMPTY);

	CtoF77 (pmpi_alltoallw) (sendbuf, sendcounts, sdispls, sendtypes,
	  recvbuf, recvcounts, rdispls, recvtypes, comm, ierror);

	/*
	*   event : ALLTOALLW_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, recvbytes, sendbytes);
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
	MPI_Comm c = MPI_Comm_f2c (*comm);

		
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
	*   target : received size               size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLTOALLW_EV, EVT_BEGIN, recvbytes,
	  sendbytes, me, c, EMPTY);

	CtoF77 (pmpi_ialltoallw) (sendbuf, sendcounts, sdispls, sendtypes,
	  recvbuf, recvcounts, rdispls, recvtypes, comm, req, ierror);

	/*
	*   event : IALLTOALLW_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, c,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, recvbytes, sendbytes);
}


#endif /* MPI3 */

#endif /* defined(FORTRAN_SYMBOLS) */

