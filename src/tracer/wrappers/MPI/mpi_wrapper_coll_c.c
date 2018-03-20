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
#include "utils.h"
#include "utils_mpi.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"
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


/******************************************************************************
 ***  MPI_Reduce_C_Wrapper
 ******************************************************************************/

int MPI_Reduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
	int me, ret, size, csize;

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	size *= count;

	/*
	*   event : REDUCE_EV                    value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send (non root) /received (root)
	*   tag : rank                           commid: communicator Id
	*   aux : root rank
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REDUCE_EV, EVT_BEGIN, op, size, me, comm, root);

	ret = PMPI_Reduce (sendbuf, recvbuf, count, datatype, op, root, comm);

	/*
	*   event : REDUCE_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_REDUCE_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
		updateStats_COLLECTIVE(global_mpi_stats, size, 0);
	else
		updateStats_COLLECTIVE(global_mpi_stats, 0, size);

	return ret;
}


/******************************************************************************
 ***  MPI_Allreduce_C_Wrapper
 ******************************************************************************/

int MPI_Allreduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int me, ret, size, csize;

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	size *= count;

	/*
	*   event : ALLREDUCE_EV                 value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send and received
	*   tag : rank                           commid: communicator Id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLREDUCE_EV, EVT_BEGIN, op, size, me, comm, EMPTY);

	ret = PMPI_Allreduce (sendbuf, recvbuf, count, datatype, op, comm);

	/*
	*   event : ALLREDUCE_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, size, size);

	return ret;
}


/******************************************************************************
 ***  MPI_Barrier_C_Wrapper
 ******************************************************************************/

int MPI_Barrier_C_Wrapper (MPI_Comm comm)
{
  int me, ret, csize;

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	ret = PMPI_Comm_size (comm, &csize);
        MPI_CHECK(ret, PMPI_Comm_size);

  /*
   *   event : BARRIER_EV                    value : EVT_BEGIN
   *   target : ---                          size  : ---
   *   tag : rank                            commid: communicator identifier
   *   aux : ---
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    TRACE_MPIEVENT (LAST_READ_TIME, MPI_BARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, comm, EMPTY);
  }
#else
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_BARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, comm, EMPTY);
#endif

  ret = PMPI_Barrier (comm);

  /*
   *   event : BARRIER_EV                   value : EVT_END
   *   target : ---                         size  : size of the communicator
   *   tag : ---                            commid: communicator identifier
   *   aux : global op counter
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    TRACE_MPIEVENT (TIME, MPI_BARRIER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());
  }
#else
  TRACE_MPIEVENT (TIME, MPI_BARRIER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());
#endif

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, 0, 0);

  return ret;
}


/******************************************************************************
 ***  MPI_BCast_C_Wrapper
 ******************************************************************************/

int MPI_BCast_C_Wrapper (void *buffer, int count, MPI_Datatype datatype, int root,
                         MPI_Comm comm)
{
	int me, ret, size, csize;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
		
	size *= count;

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	/*
	*   event : BCAST_EV                     value : EVT_BEGIN
	*   target : root_rank                   size  : message size
	*   tag : rank                           commid: communicator identifier
	*   aux : ---
	*/
#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 1;
#endif

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BCAST_EV, EVT_BEGIN, root, size, me, comm, EMPTY);

	ret = PMPI_Bcast (buffer, count, datatype, root, comm);

	/*
	*   event : BCAST_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                           commid : communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_BCAST_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, 0, size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, size, 0);

	return ret;
}



/******************************************************************************
 ***  MPI_Alltoall_C_Wrapper
 ******************************************************************************/

int MPI_Alltoall_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
	int me, ret, sendsize, recvsize, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : ALLTOALL_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLTOALL_EV, EVT_BEGIN, EMPTY,
	  sendcount * sendsize, me, comm, recvcount * recvsize * csize);

	ret = PMPI_Alltoall (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, comm);

	/*
	*   event : ALLTOALL_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
        updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Alltoallv_C_Wrapper
 ******************************************************************************/

int MPI_Alltoallv_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
  MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls,
  MPI_Datatype recvtype, MPI_Comm comm)
{
	int me, ret, sendsize, recvsize, csize;
	int proc, sendc = 0, recvc = 0;

	if (sendcounts != NULL)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcounts != NULL)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		if (sendcounts != NULL)
			sendc += sendcounts[proc];
		if (recvcounts != NULL)
			recvc += recvcounts[proc];
	}

	/*
	*   event : ALLTOALLV_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLTOALLV_EV, EVT_BEGIN, EMPTY,
	  sendsize * sendc, me, comm, recvsize * recvc);

	ret = PMPI_Alltoallv (sendbuf, sendcounts, sdispls, sendtype,
	  recvbuf, recvcounts, rdispls, recvtype, comm);

	/*
	*   event : ALLTOALLV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendc * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Allgather_C_Wrapper
 ******************************************************************************/

int MPI_Allgather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
	int ret, sendsize, recvsize, me, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : ALLGATHER_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLGATHER_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
	  me, comm, recvcount * recvsize * csize);

	ret = PMPI_Allgather (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, recvtype, comm);

	/*
	*   event : ALLGATHER_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

	return ret;
}


/******************************************************************************
 ***  MPI_Allgatherv_C_Wrapper
 ******************************************************************************/

int MPI_Allgatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcounts != NULL)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	if (recvcounts != NULL)
		for (proc = 0; proc < csize; proc++)
			recvc += recvcounts[proc];

	/*
	*   event : ALLGATHERV_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLGATHERV_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
	  me, comm, recvsize * recvc);

	ret = PMPI_Allgatherv (sendbuf, sendcount, sendtype,
	  recvbuf, recvcounts, displs, recvtype, comm);

	/*
	*   event : ALLGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Gather_C_Wrapper
 ******************************************************************************/

int MPI_Gather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
	int ret, sendsize, recvsize, me, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : GATHER_EV                    value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHER_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, recvcount * recvsize * csize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHER_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, 0);
	}

	ret = PMPI_Gather (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, recvtype, root, comm);

	/*
	*   event : GATHER_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_GATHER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, 0);
	else
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Gatherv_C_Wrapper
 ******************************************************************************/

int MPI_Gatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root,
  MPI_Comm comm)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcounts != NULL)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : GATHERV_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		if (recvcounts != NULL)
			for (proc = 0; proc < csize; proc++)
				recvc += recvcounts[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHERV_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, recvsize * recvc);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_GATHERV_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, 0);
	}

	ret = PMPI_Gatherv (sendbuf, sendcount, sendtype,
	  recvbuf, recvcounts, displs, recvtype, root, comm);

	/*
	*   event : GATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_GATHERV_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, 0);
	else
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Scatter_C_Wrapper
 ******************************************************************************/

int MPI_Scatter_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
	int ret, sendsize, recvsize, me, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : SCATTER_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTER_EV, EVT_BEGIN, root,
		  sendcount * sendsize * csize, me, comm, recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTER_EV, EVT_BEGIN, root, 0, me, comm,
		  recvcount * recvsize);
	}

	ret = PMPI_Scatter (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, root, comm);

	/*
	*   event : SCATTER_EV                   value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_SCATTER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendcount * sendsize * csize);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize, 0);

	return ret;
}



/******************************************************************************
 ***  MPI_Scatterv_C_Wrapper
 ******************************************************************************/

int MPI_Scatterv_C_Wrapper (void *sendbuf, int *sendcounts, int *displs,
  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
  int root, MPI_Comm comm)
{
	int ret, recvsize, me, csize;
	int proc, sendsize, sendc = 0;

	if (sendcounts != NULL)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event :  SCATTERV_EV                 value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		if (sendcounts != NULL)
			for (proc = 0; proc < csize; proc++)
				sendc += sendcounts[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTERV_EV, EVT_BEGIN, root, sendsize * sendc,
		  me, comm, recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCATTERV_EV, EVT_BEGIN, root, 0, me, comm,
		  recvcount * recvsize);
	}

	ret = PMPI_Scatterv (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);

	/*
	*   event : SCATTERV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_SCATTERV_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendc * sendsize);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize, 0);

	return ret;
}

/******************************************************************************
 ***  MPI_Reduce_Scatter_C_Wrapper
 ******************************************************************************/

int MPI_Reduce_Scatter_C_Wrapper (void *sendbuf, void *recvbuf,
	int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int me, size, ierror;
	int i;
	int sendcount = 0;
	int csize;

	ierror = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ierror, PMPI_Comm_rank);

	if (recvcounts != NULL)
	{
		ierror = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ierror, PMPI_Type_size);
	}

	ierror = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ierror, PMPI_Comm_size);

	if (recvcounts != NULL)
		for (i=0; i<csize; i++)
			sendcount += recvcounts[i];

	/*
	*   type : REDUCESCAT_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : Bytes sent per process in the reduce phase
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : Bytes received per process after the scatter phase
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REDUCESCAT_EV, EVT_BEGIN, op, sendcount * size, me, comm, recvcounts[me] * size);

	ierror = PMPI_Reduce_scatter (sendbuf, recvbuf, recvcounts, datatype,
	  op, comm);

	/*
	*   event : REDUCESCAT_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_REDUCESCAT_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == 0)
        updateStats_COLLECTIVE(global_mpi_stats, sendcount * size, sendcount * size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcounts[me] * size, sendcount * size);

	return ierror;
}


/******************************************************************************
 ***  MPI_Scan_C_Wrapper
 ******************************************************************************/

int MPI_Scan_C_Wrapper (void *sendbuf, void *recvbuf, int count,
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int me, ierror, size;
	int csize;

	ierror = MPI_Comm_rank (comm, &me);
	MPI_CHECK(ierror, MPI_Comm_rank);

	if (count != 0)
	{
		ierror = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ierror, PMPI_Type_size);
	}

	ierror = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ierror, PMPI_Comm_size);

	/*
	*   type : SCAN_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SCAN_EV, EVT_BEGIN, op, count * size, me, comm,
	  EMPTY);

	ierror = PMPI_Scan (sendbuf, recvbuf, count, datatype, op, comm);

	/*
	*   event : SCAN_EV                          value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_SCAN_EV, EVT_END, EMPTY, csize, EMPTY, comm, 
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */

	if (me != csize - 1)
        updateStats_COLLECTIVE(global_mpi_stats, 0, count * size);
	if (me != 0)
        updateStats_COLLECTIVE(global_mpi_stats, count * size, 0);

	return ierror;
}

#if defined(MPI3)

/******************************************************************************
 ***  MPI_Ireduce_C_Wrapper
 ******************************************************************************/

int MPI_Ireduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *req)
{
	int me, ret, size, csize;

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	size *= count;

	/*
	*   event : IREDUCE_EV                    value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send (non root) /received (root)
	*   tag : rank                           commid: communicator Id
	*   aux : root rank
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IREDUCE_EV, EVT_BEGIN, op, size, me, comm, root);

	ret = PMPI_Ireduce (sendbuf, recvbuf, count, datatype, op, root, comm, req);

	/*
	*   event : IREDUCE_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
		updateStats_COLLECTIVE(global_mpi_stats, size, 0);
	else
		updateStats_COLLECTIVE(global_mpi_stats, 0, size);

	return ret;
}


/******************************************************************************
 ***  MPI_Iallreduce_C_Wrapper
 ******************************************************************************/

int MPI_Iallreduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req)
{
	int me, ret, size, csize;

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	size *= count;

	/*
	*   event : IALLREDUCE_EV                 value : EVT_BEGIN
	*   target: reduce operation ident.      size : bytes send and received
	*   tag : rank                           commid: communicator Id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLREDUCE_EV, EVT_BEGIN, op, size, me, comm, EMPTY);

	ret = PMPI_Iallreduce (sendbuf, recvbuf, count, datatype, op, comm, req);

	/*
	*   event : IALLREDUCE_EV                 value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator Id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLREDUCE_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, size, size);

	return ret;
}


/******************************************************************************
 ***  MPI_Ibarrier_C_Wrapper
 ******************************************************************************/

int MPI_Ibarrier_C_Wrapper (MPI_Comm comm, MPI_Request *req)
{
  int me, ret, csize;

  ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

  ret = PMPI_Comm_size (comm, &csize);
        MPI_CHECK(ret, PMPI_Comm_size);

  /*
   *   event : IBARRIER_EV                    value : EVT_BEGIN
   *   target : ---                          size  : ---
   *   tag : rank                            commid: communicator identifier
   *   aux : ---
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, comm, EMPTY);
  }
#else
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBARRIER_EV, EVT_BEGIN, EMPTY, EMPTY, me, comm, EMPTY);
#endif

  ret = PMPI_Ibarrier (comm, req);

  /*
   *   event : IBARRIER_EV                   value : EVT_END
   *   target : ---                         size  : size of the communicator
   *   tag : ---                            commid: communicator identifier
   *   aux : global op counter
   */

#if defined(IS_BGL_MACHINE)
  if (!BGL_disable_barrier_inside)
  {
    TRACE_MPIEVENT (TIME, MPI_IBARRIER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());
  }
#else
  TRACE_MPIEVENT (TIME, MPI_IBARRIER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());
#endif

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, 0, 0);

  return ret;
}


/******************************************************************************
 ***  MPI_Ibcast_C_Wrapper
 ******************************************************************************/

int MPI_Ibcast_C_Wrapper (void *buffer, int count, MPI_Datatype datatype, int root,
	MPI_Comm comm, MPI_Request *req)
{
	int me, ret, size, csize;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
		
	size *= count;

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	/*
	*   event : IBCAST_EV                     value : EVT_BEGIN
	*   target : root_rank                   size  : message size
	*   tag : rank                           commid: communicator identifier
	*   aux : ---
	*/
#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 1;
#endif

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBCAST_EV, EVT_BEGIN, root, size, me, comm, EMPTY);

	ret = PMPI_Ibcast (buffer, count, datatype, root, comm, req);

	/*
	*   event : IBCAST_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                           commid : communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IBCAST_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, 0, size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, size, 0);

	return ret;
}



/******************************************************************************
 ***  MPI_Ialltoall_C_Wrapper
 ******************************************************************************/

int MPI_Ialltoall_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req)
{
	int me, ret, sendsize, recvsize, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : IALLTOALL_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLTOALL_EV, EVT_BEGIN, EMPTY,
	  sendcount * sendsize, me, comm, recvcount * recvsize * csize);

	ret = PMPI_Ialltoall (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, comm, req);

	/*
	*   event : IALLTOALL_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Ialltoallv_C_Wrapper
 ******************************************************************************/

int MPI_Ialltoallv_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
  MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls,
  MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req)
{
	int me, ret, sendsize, recvsize, csize;
	int proc, sendc = 0, recvc = 0;

	if (sendcounts != NULL)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcounts != NULL)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		if (sendcounts != NULL)
			sendc += sendcounts[proc];
		if (recvcounts != NULL)
			recvc += recvcounts[proc];
	}

	/*
	*   event : IALLTOALLV_EV                  value : EVT_BEGIN
	*   target : ---                         size  : sent size
	*   tag : rank                           commid: communicator id
	*   aux : received size
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLTOALLV_EV, EVT_BEGIN, EMPTY,
	  sendsize * sendc, me, comm, recvsize * recvc);

	ret = PMPI_Ialltoallv (sendbuf, sendcounts, sdispls, sendtype,
	  recvbuf, recvcounts, rdispls, recvtype, comm, req);

	/*
	*   event : IALLTOALLV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendc * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Iallgather_C_Wrapper
 ******************************************************************************/

int MPI_Iallgather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req)
{
	int ret, sendsize, recvsize, me, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : IALLGATHER_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLGATHER_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
	  me, comm, recvcount * recvsize * csize);

	ret = PMPI_Iallgather (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, recvtype, comm, req);

	/*
	*   event : IALLGATHER_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

	return ret;
}


/******************************************************************************
 ***  MPI_Iallgatherv_C_Wrapper
 ******************************************************************************/

int MPI_Iallgatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm,
	MPI_Request *req)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcounts != NULL)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	if (recvcounts != NULL)
		for (proc = 0; proc < csize; proc++)
			recvc += recvcounts[proc];

	/*
	*   event : IALLGATHERV_EV                    value : EVT_BEGIN
	*   target : ---                         size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLGATHERV_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
	  me, comm, recvsize * recvc);

	ret = PMPI_Iallgatherv (sendbuf, sendcount, sendtype,
	  recvbuf, recvcounts, displs, recvtype, comm, req);

	/*
	*   event : IALLGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Igather_C_Wrapper
 ******************************************************************************/

int MPI_Igather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
	MPI_Request *req)
{
	int ret, sendsize, recvsize, me, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : GATHER_EV                    value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHER_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, recvcount * recvsize * csize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHER_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, 0);
	}

	ret = PMPI_Igather (sendbuf, sendcount, sendtype,
	  recvbuf, recvcount, recvtype, root, comm, req);

	/*
	*   event : GATHER_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IGATHER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, 0);
	else
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Igatherv_C_Wrapper
 ******************************************************************************/

int MPI_Igatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root,
  MPI_Comm comm, MPI_Request *req)
{
	int ret, sendsize, me, csize;
	int proc, recvsize, recvc = 0;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcounts != NULL)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : IGATHERV_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		if (recvcounts != NULL)
			for (proc = 0; proc < csize; proc++)
				recvc += recvcounts[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHERV_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, recvsize * recvc);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_IGATHERV_EV, EVT_BEGIN, root, sendcount * sendsize,
		  me, comm, 0);
	}

	ret = PMPI_Igatherv (sendbuf, sendcount, sendtype,
	  recvbuf, recvcounts, displs, recvtype, root, comm, req);

	/*
	*   event : IGATHERV_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, 0);
	else
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendcount * sendsize);

	return ret;
}



/******************************************************************************
 ***  MPI_Iscatter_C_Wrapper
 ******************************************************************************/

int MPI_Iscatter_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
	MPI_Request *req)
{
	int ret, sendsize, recvsize, me, csize;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event : ISCATTER_EV                   value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTER_EV, EVT_BEGIN, root,
		  sendcount * sendsize * csize, me, comm, recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTER_EV, EVT_BEGIN, root, 0, me, comm,
		  recvcount * recvsize);
	}

	ret = PMPI_Iscatter (sendbuf, sendcount, sendtype, recvbuf, recvcount,
	  recvtype, root, comm, req);

	/*
	*   event : ISCATTER_EV                   value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ISCATTER_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendcount * sendsize * csize);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize, 0);

	return ret;
}



/******************************************************************************
 ***  MPI_Iscatterv_C_Wrapper
 ******************************************************************************/

int MPI_Iscatterv_C_Wrapper (void *sendbuf, int *sendcounts, int *displs,
  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
  int root, MPI_Comm comm, MPI_Request *req)
{
	int ret, recvsize, me, csize;
	int proc, sendsize, sendc = 0;

	if (sendcounts != NULL)
	{
		ret = PMPI_Type_size (sendtype, &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	/*
	*   event :  ISCATTERV_EV                 value : EVT_BEGIN
	*   target : root rank                   size  : bytes sent
	*   tag : rank                           commid: communicator identifier
	*   aux : bytes received
	*/
	if (me == root)
	{
		if (sendcounts != NULL)
			for (proc = 0; proc < csize; proc++)
				sendc += sendcounts[proc];

		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTERV_EV, EVT_BEGIN, root, sendsize * sendc,
		  me, comm, recvcount * recvsize);
	}
	else
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCATTERV_EV, EVT_BEGIN, root, 0, me, comm,
		  recvcount * recvsize);
	}

	ret = PMPI_Iscatterv (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, req);

	/*
	*   event : SCATTERV_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator identifier
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ISCATTERV_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == root)
        updateStats_COLLECTIVE(global_mpi_stats, 0, sendc * sendsize);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize, 0);

	return ret;
}

/******************************************************************************
 ***  MPI_Ireduce_Scatter_C_Wrapper
 ******************************************************************************/

int MPI_Ireduce_Scatter_C_Wrapper (void *sendbuf, void *recvbuf,
	int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
	MPI_Request *req)
{
	int me, size, ierror;
	int i;
	int sendcount = 0;
	int csize;

	ierror = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ierror, PMPI_Comm_rank);

	if (recvcounts != NULL)
	{
		ierror = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ierror, PMPI_Type_size);
	}

	ierror = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ierror, PMPI_Comm_size);

	if (recvcounts != NULL)
		for (i=0; i<csize; i++)
			sendcount += recvcounts[i];

	/*
	*   type : IREDUCESCAT_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : Bytes sent per process in the reduce phase
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : Bytes received per process after the scatter phase
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IREDUCESCAT_EV, EVT_BEGIN, op, sendcount * size, me, comm, recvcounts[me] * size);

	ierror = PMPI_Ireduce_scatter (sendbuf, recvbuf, recvcounts, datatype,
	  op, comm, req);

	/*
	*   event : IREDUCESCAT_EV                    value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IREDUCESCAT_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	if (me == 0)
        updateStats_COLLECTIVE(global_mpi_stats, sendcount * size, sendcount * size);
	else
        updateStats_COLLECTIVE(global_mpi_stats, recvcounts[me] * size, sendcount * size);

	return ierror;
}


/******************************************************************************
 ***  MPI_Scan_C_Wrapper
 ******************************************************************************/

int MPI_Iscan_C_Wrapper (void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req)
{
	int me, ierror, size;
	int csize;

	ierror = MPI_Comm_rank (comm, &me);
	MPI_CHECK(ierror, MPI_Comm_rank);

	if (count != 0)
	{
		ierror = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ierror, PMPI_Type_size);
	}

	ierror = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ierror, PMPI_Comm_size);

	/*
	*   type : ISCAN_EV                    value : EVT_BEGIN
	*   target : reduce operation ident.    size  : data size
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISCAN_EV, EVT_BEGIN, op, count * size, me, comm,
	  EMPTY);

	ierror = PMPI_Iscan (sendbuf, recvbuf, count, datatype, op, comm, req);

	/*
	*   event : ISCAN_EV                          value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ISCAN_EV, EVT_END, EMPTY, csize, EMPTY, comm, 
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */

	if (me != csize - 1)
        updateStats_COLLECTIVE(global_mpi_stats, 0, count * size);
	if (me != 0)
        updateStats_COLLECTIVE(global_mpi_stats, count * size, 0);

	return ierror;
}

/******************************************************************************
 ***  MPI_Reduce_Scatter_Block_C_Wrapper
 ******************************************************************************/

int MPI_Reduce_Scatter_Block_C_Wrapper (void *sendbuf, void *recvbuf,
	int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int me, size, ierror;
	int sendcount = 0;
	int csize;

	ierror = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ierror, PMPI_Comm_rank);

	if (recvcount != 0)
	{
		ierror = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ierror, PMPI_Type_size);
	}

	ierror = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ierror, PMPI_Comm_size);

	/*
	*   type : REDUCE_SCATTER_BLOCK_EV      value : EVT_BEGIN
	*   target : reduce operation ident.    size  : Bytes sent per process in the reduce phase
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : Bytes received per process after the scatter phase
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REDUCE_SCATTER_BLOCK_EV, EVT_BEGIN, op, sendcount * size, me, comm, recvcount * size);

	ierror = PMPI_Reduce_scatter_block (sendbuf, recvbuf, recvcount, datatype,
	  op, comm);

	/*
	*   event : REDUCE_SCATTER_BLOCK_EV      value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_REDUCE_SCATTER_BLOCK_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvcount * size, sendcount * size);

	return ierror;
}

/******************************************************************************
 ***  MPI_Ireduce_Scatter_Block_C_Wrapper
 ******************************************************************************/

int MPI_Ireduce_Scatter_Block_C_Wrapper (void *sendbuf, void *recvbuf,
	int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
	MPI_Request *req)
{
	int me, size, ierror;
	int sendcount = 0;
	int csize;

	ierror = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ierror, PMPI_Comm_rank);

	if (recvcount != 0)
	{
		ierror = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ierror, PMPI_Type_size);
	}

	ierror = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ierror, PMPI_Comm_size);

	/*
	*   type : IREDUCE_SCATTER_BLOCK_EV     value : EVT_BEGIN
	*   target : reduce operation ident.    size  : Bytes sent per process in the reduce phase
	*   tag : whoami (comm rank)            comm : communicator id
	*   aux : Bytes received per process after the scatter phase
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IREDUCE_SCATTER_BLOCK_EV, EVT_BEGIN, op, sendcount * size, me, comm, recvcount * size);

	ierror = PMPI_Ireduce_scatter_block (sendbuf, recvbuf, recvcount, datatype,
	  op, comm, req);

	/*
	*   event : IREDUCE_SCATTER_BLOCK_EV     value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            comm : communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IREDUCE_SCATTER_BLOCK_EV, EVT_END, EMPTY, csize, EMPTY, comm, Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
	updateStats_COLLECTIVE(global_mpi_stats, recvcount * size, sendcount * size);

	return ierror;
}

/******************************************************************************
 ***  MPI_Alltoallw_C_Wrapper
 ******************************************************************************/

int MPI_Alltoallw_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
	MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, int *rdispls,
    MPI_Datatype *recvtypes, MPI_Comm comm)
{
	int me, ret, csize;
	int proc, sendbytes = 0, recvbytes = 0;

	

	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		int sendsize;
		ret = PMPI_Type_size (sendtypes[proc], &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
		
		int recvsize;
		ret = PMPI_Type_size (recvtypes[proc], &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
						
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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ALLTOALLW_EV, EVT_BEGIN, EMPTY,
	  sendbytes, me, comm, recvbytes);

	ret = PMPI_Alltoallw (sendbuf, sendcounts, sdispls, sendtypes,
	  recvbuf, recvcounts, rdispls, recvtypes, comm);

	/*
	*   event : ALLTOALLW_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_ALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvbytes, sendbytes);

	return ret;
}

/******************************************************************************
 ***  MPI_Ialltoallw_C_Wrapper
 ******************************************************************************/

int MPI_Ialltoallw_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
	MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, int *rdispls,
    MPI_Datatype *recvtypes, MPI_Comm comm, MPI_Request *req)
{
	int me, ret, csize;
	int proc, sendbytes = 0, recvbytes = 0;


	ret = PMPI_Comm_size (comm, &csize);
	MPI_CHECK(ret, PMPI_Comm_size);

	ret = PMPI_Comm_rank (comm, &me);
	MPI_CHECK(ret, PMPI_Comm_rank);

	for (proc = 0; proc < csize; proc++)
	{
		int sendsize;
		ret = PMPI_Type_size (sendtypes[proc], &sendsize);
		MPI_CHECK(ret, PMPI_Type_size);
		
		int recvsize;
		ret = PMPI_Type_size (recvtypes[proc], &recvsize);
		MPI_CHECK(ret, PMPI_Type_size);
						
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
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IALLTOALLW_EV, EVT_BEGIN, EMPTY,
	  sendbytes, me, comm, recvbytes);

	ret = PMPI_Ialltoallw (sendbuf, sendcounts, sdispls, sendtypes,
	  recvbuf, recvcounts, rdispls, recvtypes, comm, req);

	/*
	*   event : IALLTOALLW_EV                  value : EVT_END
	*   target : ---                         size  : size of the communicator
	*   tag : ---                            commid: communicator id
	*   aux : global op counter
	*/
	TRACE_MPIEVENT (TIME, MPI_IALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, comm,
	  Extrae_MPI_getCurrentOpGlobal());

	/* MPI Stats */
    updateStats_COLLECTIVE(global_mpi_stats, recvbytes, sendbytes);

	return ret;
}

/**
 * xtr_MPI_Comm_neighbors_count
 * 
 * Returns the number of neighbors of the given process topology. If the communicator 'comm' was created with...
 * 1) MPI_Graph_create, both 'indegree' and 'outdegree' are set to the number of neighbors retrieved by MPI_Graph_neighbors_count
 * 2) MPI_Cart_create, both 'indegree' and 'outdegree' are set to (2 * ndims), where ndims is retrieved by MPI_Cartdim_get
 * 3) MPI_Dist_graph_create, 'indegree' and 'outdegree' are set to the corresponding return value of MPI_Dist_graph_neighbors_count
 * 4) In any other case, 'indegree' and 'outdegree' are set to 0
 *
 * @return the rank of the current task
 */
int xtr_MPI_Comm_neighbors_count(MPI_Comm comm, int *indegree, int *outdegree)
{
  int ret = 0, me = 0, status = 0, nneighbors = 0, ndims = 0;

  ret = PMPI_Comm_rank(comm, &me);
  MPI_CHECK(ret, PMPI_Comm_rank);

  ret = PMPI_Topo_test(comm, &status);
  MPI_CHECK(ret, PMPI_Topo_test);

  switch (status)
  {
    case MPI_GRAPH:
    {
      ret = PMPI_Graph_neighbors_count(comm, me, &nneighbors);
      MPI_CHECK(ret, PMPI_Graph_neighbors_count);
      if (indegree  != NULL) *indegree  = nneighbors;
      if (outdegree != NULL) *outdegree = nneighbors;
      break;
    }
    case MPI_CART:
    {
      ret = PMPI_Cartdim_get(comm, &ndims);
      MPI_CHECK(ret, PMPI_Cartdim_get);
      if (indegree  != NULL) *indegree  = (2 * ndims);
      if (outdegree != NULL) *outdegree = (2 * ndims);
      break;
    }
    case MPI_DIST_GRAPH:
    {
      int weighted; 
      ret = PMPI_Dist_graph_neighbors_count(comm, indegree, outdegree, &weighted);
      MPI_CHECK(ret, PMPI_Dist_graph_neighbors_count)
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
 ***  MPI_Graph_create_C_Wrapper
 ******************************************************************************/

int MPI_Graph_create_C_Wrapper (MPI_Comm comm_old, int nnodes, int *index, int *edges, int reorder, MPI_Comm *comm_graph)
{
  int ierror;

  TRACE_MPIEVENT (LAST_READ_TIME, MPI_GRAPH_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Graph_create (comm_old, nnodes, index, edges, reorder, comm_graph);

  if (*comm_graph != MPI_COMM_NULL && ierror == MPI_SUCCESS)
  {
    Trace_MPI_Communicator (*comm_graph, LAST_READ_TIME, FALSE);
  }

  TRACE_MPIEVENT (TIME, MPI_GRAPH_CREATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  updateStats_OTHER(global_mpi_stats);

  return ierror;
}

/******************************************************************************
 ***  MPI_Dist_graph_create_C_Wrapper
 ******************************************************************************/

int MPI_Dist_graph_create_C_Wrapper (MPI_Comm comm_old, int n, int *sources, int *degrees, int *destinations, int *weights, MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)
{
  int ierror;

  TRACE_MPIEVENT (LAST_READ_TIME, MPI_DIST_GRAPH_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Dist_graph_create (comm_old, n, sources, degrees, destinations, weights, info, reorder, comm_dist_graph);

  if (*comm_dist_graph != MPI_COMM_NULL && ierror == MPI_SUCCESS)
  {
    Trace_MPI_Communicator (*comm_dist_graph, LAST_READ_TIME, FALSE);
  }

  TRACE_MPIEVENT (TIME, MPI_DIST_GRAPH_CREATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  updateStats_OTHER(global_mpi_stats);

  return ierror;
}

/******************************************************************************
 ***  MPI_Neighbor_allgather_C_Wrapper
 ******************************************************************************/

int MPI_Neighbor_allgather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;

  if (sendcount != 0)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcount != 0)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, NULL);

  /*
   *   event  : NEIGHBOR_ALLGATHER_EV       value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_NEIGHBOR_ALLGATHER_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
    me, comm, recvcount * recvsize * indegree);

  ret = PMPI_Neighbor_allgather (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

  /*
   *   event  : NEIGHBOR_ALLGATHER_EV       value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_NEIGHBOR_ALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Ineighbor_allgather_C_Wrapper
 ******************************************************************************/

int MPI_Ineighbor_allgather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;

  if (sendcount != 0)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcount != 0)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, NULL);

  /*
   *   event  : INEIGHBOR_ALLGATHER_EV      value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_INEIGHBOR_ALLGATHER_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
    me, comm, recvcount * recvsize * indegree);

  ret = PMPI_Ineighbor_allgather (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

  /*
   *   event  : INEIGHBOR_ALLGATHER_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_INEIGHBOR_ALLGATHER_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Neighbor_allgatherv_C_Wrapper
 ******************************************************************************/

int MPI_Neighbor_allgatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm)
{
  int proc = 0, ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0, recvc = 0;

  if (sendcount != 0)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcounts != NULL)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, NULL);

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
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_NEIGHBOR_ALLGATHERV_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
    me, comm, recvsize * recvc);

  ret = PMPI_Neighbor_allgatherv (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);

  /*
   *   event  : NEIGHBOR_ALLGATHERV_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_NEIGHBOR_ALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendcount * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Ineighbor_allgatherv_C_Wrapper
 ******************************************************************************/

int MPI_Ineighbor_allgatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
  int proc = 0, ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0, recvc = 0;

  if (sendcount != 0)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcounts != NULL)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, NULL);

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
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_INEIGHBOR_ALLGATHERV_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
    me, comm, recvsize * recvc);

  ret = PMPI_Ineighbor_allgatherv (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);

  /*
   *   event  : INEIGHBOR_ALLGATHERV_EV     value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_INEIGHBOR_ALLGATHERV_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendcount * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Neighbor_alltoall_C_Wrapper
 ******************************************************************************/

int MPI_Neighbor_alltoall_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;

  if (sendcount != 0)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcount != 0)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, NULL);

  /*
   *   event  : NEIGHBOR_ALLTOALL_EV        value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_NEIGHBOR_ALLTOALL_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
    me, comm, recvcount * recvsize * indegree);

  ret = PMPI_Neighbor_alltoall (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

  /*
   *   event  : NEIGHBOR_ALLTOALL_EV        value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_NEIGHBOR_ALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Ineighbor_alltoall_C_Wrapper
 ******************************************************************************/

int MPI_Ineighbor_alltoall_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
  int ret = 0, me = 0, sendsize = 0, recvsize = 0, csize = 0, indegree = 0;

  if (sendcount != 0)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcount != 0)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, NULL);

  /*
   *   event  : INEIGHBOR_ALLTOALL_EV       value  : EVT_BEGIN
   *   target : ---                         size   : bytes sent
   *   tag    : rank                        commid : communicator identifier
   *   aux    : bytes received
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_INEIGHBOR_ALLTOALL_EV, EVT_BEGIN, EMPTY, sendcount * sendsize,
    me, comm, recvcount * recvsize * indegree);

  ret = PMPI_Ineighbor_alltoall (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

  /*
   *   event  : INEIGHBOR_ALLTOALL_EV        value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_INEIGHBOR_ALLTOALL_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvcount * recvsize * csize, sendcount * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Neighbor_alltoallv_C_Wrapper
 ******************************************************************************/

int MPI_Neighbor_alltoallv_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int proc = 0, ret = 0, me = 0, sendsize = 0, sendc = 0, recvsize = 0, recvc = 0, csize = 0, indegree = 0, outdegree = 0;

  if (sendcounts != NULL)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcounts != NULL)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, &outdegree);

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
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_NEIGHBOR_ALLTOALLV_EV, EVT_BEGIN, EMPTY, sendsize * sendc,
    me, comm, recvsize * recvc);

  ret = PMPI_Neighbor_alltoallv (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);

  /*
   *   event  : NEIGHBOR_ALLTOALLV_EV       value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_NEIGHBOR_ALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, comm,
  Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendc * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Ineighbor_alltoallv_C_Wrapper
 ******************************************************************************/

int MPI_Ineighbor_alltoallv_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
  int proc = 0, me = 0, ret = 0, sendsize = 0, sendc = 0, recvsize = 0, recvc = 0, csize = 0, indegree = 0, outdegree = 0;

  if (sendcounts != NULL)
  {
    ret = PMPI_Type_size (sendtype, &sendsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  if (recvcounts != NULL)
  {
    ret = PMPI_Type_size (recvtype, &recvsize);
    MPI_CHECK(ret, PMPI_Type_size);
  }

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, &outdegree);

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
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_INEIGHBOR_ALLTOALLV_EV, EVT_BEGIN, EMPTY, sendsize * sendc,
    me, comm, recvsize * recvc);

  ret = PMPI_Ineighbor_alltoallv (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);

  /*
   *   event  : INEIGHBOR_ALLTOALLV_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_INEIGHBOR_ALLTOALLV_EV, EVT_END, EMPTY, csize, EMPTY, comm,
  Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvc * recvsize, sendc * sendsize);

  return ret;
}

/******************************************************************************
 ***  MPI_Neighbor_alltoallw_C_Wrapper
 ******************************************************************************/

int MPI_Neighbor_alltoallw_C_Wrapper (void *sendbuf, int *sendcounts, MPI_Aint *sdispls, MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, MPI_Aint *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm)
{
  int proc = 0, ret = 0, me = 0, csize = 0, indegree = 0, outdegree = 0, sendbytes = 0, recvbytes = 0;

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, &outdegree);

  for (proc = 0; proc < outdegree; proc++)
  {
    int sendsize = 0;

    if (sendtypes != NULL)
    {
      ret = PMPI_Type_size (sendtypes[proc], &sendsize);
      MPI_CHECK(ret, PMPI_Type_size);

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
      ret = PMPI_Type_size (recvtypes[proc], &recvsize);
      MPI_CHECK(ret, PMPI_Type_size);

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
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_NEIGHBOR_ALLTOALLW_EV, EVT_BEGIN, EMPTY, sendbytes,
    me, comm, recvbytes);

  ret = PMPI_Neighbor_alltoallw (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);

  /*
   *   event  : NEIGHBOR_ALLTOALLW_EV       value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_NEIGHBOR_ALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvbytes, sendbytes);

  return ret;
}

/******************************************************************************
 ***  MPI_Ineighbor_alltoall_C_Wrapper
 ******************************************************************************/

int MPI_Ineighbor_alltoallw_C_Wrapper (void *sendbuf, int *sendcounts, MPI_Aint *sdispls, MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, MPI_Aint *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm, MPI_Request *request)
{
  int proc = 0, ret = 0, me = 0, csize = 0, indegree = 0, outdegree = 0, sendbytes = 0, recvbytes = 0;

  ret = PMPI_Comm_size (comm, &csize);
  MPI_CHECK(ret, PMPI_Comm_size);

  me = xtr_MPI_Comm_neighbors_count(comm, &indegree, &outdegree);

  for (proc = 0; proc < outdegree; proc++)
  {
    int sendsize = 0;

    if (sendtypes != NULL)
    {
      ret = PMPI_Type_size (sendtypes[proc], &sendsize);
      MPI_CHECK(ret, PMPI_Type_size);

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
      ret = PMPI_Type_size (recvtypes[proc], &recvsize);
      MPI_CHECK(ret, PMPI_Type_size);

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
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_INEIGHBOR_ALLTOALLW_EV, EVT_BEGIN, EMPTY, sendbytes,
    me, comm, recvbytes);

  ret = PMPI_Ineighbor_alltoallw (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request);

  /*
   *   event  : INEIGHBOR_ALLTOALLW_EV      value  : EVT_END
   *   target : ---                         size   : size of the communicator
   *   tag    : ---                         commid : communicator identifier
   *   aux    : global op counter
   */
  TRACE_MPIEVENT (TIME, MPI_INEIGHBOR_ALLTOALLW_EV, EVT_END, EMPTY, csize, EMPTY, comm,
    Extrae_MPI_getCurrentOpGlobal());

  /* MPI Stats */
  updateStats_COLLECTIVE(global_mpi_stats, recvbytes, sendbytes);

  return ret;
}

#endif /* MPI3 */

