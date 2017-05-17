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
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#include <mpi.h>
#include "wrapper.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"
#include "mpi_interface.h"
#include "dlb.h"

#if defined(C_SYMBOLS) && defined(FORTRAN_SYMBOLS)
# define COMBINED_SYMBOLS
#endif

#define ENTER	TRUE
#define LEAVE	FALSE

//#define DEBUG_MPITRACE

#if defined(DEBUG_MPITRACE)
#	define DEBUG_INTERFACE(enter) \
	{ fprintf (stderr, "Task %d %s %s\n", TASKID, (enter)?"enters":"leaves", __func__); }
#else
#	define DEBUG_INTERFACE(enter)
#endif

/*
	NAME_ROUTINE_C/F/C2F are macros to translate MPI interface names to 
	patches that will be hooked by the DynInst mutator.

	_C -> converts names for C MPI symbols
	_F -> converts names for Fortran MPI symbols (ignoring the number of underscores,
	      i.e does not honor _UNDERSCORES defines and CtoF77 macro)
	      This is convenient when using the attribute construction of the compiler to
	      provide all the names for the symbols.
	_C2F-> converts names for Fortran MPI symbols (honoring _UNDERSCORES and
	      CtoF77 macro)
*/

#if defined(DYNINST_MODULE)
# define NAME_ROUTINE_C(x) PATCH_P##x  /* MPI_Send is converted to PATCH_PMPI_Send */
#else
# define NAME_ROUTINE_C(x) x
#endif

#if defined(C_SYMBOLS)

/******************************************************************************
 ***  MPI_Reduce
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Reduce) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Reduce_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Reduce_C_Wrapper 
			(MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Reduce (sendbuf, recvbuf, count, datatype, op, root, comm);

	DLB(DLB_MPI_Reduce_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Reduce_scatter
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Reduce_scatter) (MPI3_CONST void *sendbuf, void *recvbuf,
	MPI3_CONST int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Reduce_scatter_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_C_INT_P_CAST recvcounts, datatype, op, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Reduce_Scatter_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_C_INT_P_CAST recvcounts, datatype,
			op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Reduce_scatter (sendbuf, recvbuf, recvcounts, datatype, op,
			comm);

	DLB(DLB_MPI_Reduce_scatter_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Allreduce
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Allreduce) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Allreduce_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Allreduce_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Allreduce (sendbuf, recvbuf, count, datatype, op, comm);

	DLB(DLB_MPI_Allreduce_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Barrier
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Barrier) (MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Barrier_enter, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Barrier_C_Wrapper (comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Barrier (comm);

	DLB(DLB_MPI_Barrier_leave);

	return res;
}

/******************************************************************************
 ***  MPI_BCast
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Bcast) (void *buffer, int count, MPI_Datatype datatype,
	int root, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Bcast_enter, buffer, count, datatype, root, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_BCast_C_Wrapper (buffer, count, datatype, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Bcast (buffer, count, datatype, root, comm);

	DLB(DLB_MPI_Bcast_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Alltoall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Alltoall) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Alltoall_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Alltoall_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Alltoall
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

	DLB(DLB_MPI_Alltoall_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Alltoallv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Alltoallv) (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *sdispls,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
	MPI_Datatype recvtype, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Alltoallv_enter, MPI3_VOID_P_CAST sendbuf, MPI3_VOID_P_CAST sendcounts, MPI3_VOID_P_CAST sdispls, sendtype, recvbuf, MPI3_VOID_P_CAST recvcounts, MPI3_VOID_P_CAST rdispls, recvtype, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Alltoallv_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, MPI3_VOID_P_CAST sendcounts, MPI3_VOID_P_CAST sdispls, sendtype, recvbuf, MPI3_VOID_P_CAST recvcounts,
		  MPI3_VOID_P_CAST rdispls, recvtype, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Alltoallv
		  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
		  rdispls, recvtype, comm);

	DLB(DLB_MPI_Alltoallv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Allgather
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Allgather) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Allgather_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Allgather_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
		  comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Allgather
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
		  comm);

	DLB(DLB_MPI_Allgather_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Allgatherv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Allgatherv) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Allgatherv_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_C_INT_P_CAST recvcounts,MPI3_C_INT_P_CAST  displs, recvtype, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Allgatherv_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_C_INT_P_CAST recvcounts, MPI3_C_INT_P_CAST displs,
		  recvtype, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Allgatherv
		  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
		  recvtype, comm);

	DLB(DLB_MPI_Allgatherv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Gather
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Gather) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Gather_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Gather_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Gather
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);

	DLB(DLB_MPI_Gather_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Gatherv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Gatherv) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, int root, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Gatherv_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcounts, MPI3_C_INT_P_CAST displs, recvtype, root, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Gatherv_C_Wrapper
            (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_C_INT_P_CAST recvcounts, MPI3_C_INT_P_CAST displs,
             recvtype, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Gatherv
		  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
		  recvtype, root, comm);

	DLB(DLB_MPI_Gatherv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Scatter
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Scatter) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Scatter_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Scatter_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Scatter
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);

	DLB(DLB_MPI_Scatter_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Scatterv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Scatterv) (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *displs, 
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Scatterv_enter, MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Scatterv_C_Wrapper
            (MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST displs, sendtype, recvbuf, recvcount,
             recvtype, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Scatterv
		  (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
		  recvtype, root, comm);

	DLB(DLB_MPI_Scatterv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Scan
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Scan) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Scan_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Scan_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Scan (sendbuf, recvbuf, count, datatype, op, comm);

	DLB(DLB_MPI_Scan_leave);

	return res;
}

#if defined(MPI3)
/******************************************************************************
 ***  MPI_Ireduce
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ireduce) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ireduce_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ireduce_C_Wrapper 
			(MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ireduce (sendbuf, recvbuf, count, datatype, op, root, comm, req);

	DLB(DLB_MPI_Ireduce_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ireduce_scatter
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ireduce_scatter) (MPI3_CONST void *sendbuf, void *recvbuf,
	MPI3_CONST int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
	MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ireduce_scatter_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_C_INT_P_CAST recvcounts, datatype, op, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ireduce_Scatter_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_C_INT_P_CAST recvcounts, datatype, op, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ireduce_scatter (sendbuf, recvbuf, recvcounts, datatype, op,
			comm, req);

	DLB(DLB_MPI_Ireduce_scatter_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Iallreduce
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iallreduce) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Iallreduce_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Iallreduce_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Iallreduce (sendbuf, recvbuf, count, datatype, op, comm, req);

	DLB(DLB_MPI_Iallreduce_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ibarrier
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ibarrier) (MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ibarrier_enter, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Ibarrier_C_Wrapper (comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ibarrier (comm, req);

	DLB(DLB_MPI_Ibarrier_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ibcast
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ibcast) (void *buffer, int count, MPI_Datatype datatype,
	int root, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ibcast_enter, buffer, count, datatype, root, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ibcast_C_Wrapper (buffer, count, datatype, root, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ibcast (buffer, count, datatype, root, comm, req);

	DLB(DLB_MPI_Ibcast_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ialltoall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ialltoall) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ialltoall_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ialltoall_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ialltoall
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req);

	DLB(DLB_MPI_Ialltoall_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Alltoallv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ialltoallv) (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *sdispls,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
	MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ialltoallv_enter, MPI3_VOID_P_CAST sendbuf, MPI3_VOID_P_CAST sendcounts, MPI3_VOID_P_CAST sdispls, sendtype, recvbuf, MPI3_VOID_P_CAST recvcounts, MPI3_VOID_P_CAST rdispls, recvtype, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ialltoallv_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, MPI3_VOID_P_CAST sendcounts, MPI3_VOID_P_CAST sdispls, sendtype, recvbuf, MPI3_VOID_P_CAST recvcounts,
		  MPI3_VOID_P_CAST rdispls, recvtype, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ialltoallv
		  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
		  rdispls, recvtype, comm, req);

	DLB(DLB_MPI_Alltoallv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Iallgather
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iallgather) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Iallgather_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Iallgather_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
		  comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Iallgather
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
		  comm, req);

	DLB(DLB_MPI_Iallgather_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Iallgatherv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iallgatherv) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Iallgatherv_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_C_INT_P_CAST recvcounts,MPI3_C_INT_P_CAST  displs, recvtype, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Iallgatherv_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_C_INT_P_CAST recvcounts, MPI3_C_INT_P_CAST displs,
		  recvtype, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Iallgatherv
		  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
		  recvtype, comm, req);

	DLB(DLB_MPI_Iallgatherv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Igather
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Igather) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Igather_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Igather_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Igather
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm, req);

	DLB(DLB_MPI_Igather_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Igatherv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Igatherv) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Igatherv_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcounts, MPI3_C_INT_P_CAST displs, recvtype, root, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Igatherv_C_Wrapper
            (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_C_INT_P_CAST recvcounts, MPI3_C_INT_P_CAST displs,
             recvtype, root, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Igatherv
		  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
		  recvtype, root, comm, req);

	DLB(DLB_MPI_Igatherv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Iscatter
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iscatter) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Iscatter_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Iscatter_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Iscatter
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm, req);

	DLB(DLB_MPI_Iscatter_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Iscatterv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iscatterv) (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *displs, 
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Iscatterv_enter, MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Iscatterv_C_Wrapper
            (MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Iscatterv
		  (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
		  recvtype, root, comm, req);

	DLB(DLB_MPI_Iscatterv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Iscan
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iscan) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Iscan_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Iscan_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Iscan (sendbuf, recvbuf, count, datatype, op, comm, req);

	DLB(DLB_MPI_Iscan_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Reduce_scatter_block
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Reduce_scatter_block) (MPI3_CONST void *sendbuf, void *recvbuf,
	int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Reduce_scatter_block_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype, op, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Reduce_Scatter_Block_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype,
			op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Reduce_scatter_block (sendbuf, recvbuf, recvcount, datatype, op,
			comm);

	DLB(DLB_MPI_Reduce_scatter_block_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ireduce_scatter_block
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ireduce_scatter_block) (MPI3_CONST void *sendbuf, void *recvbuf,
	int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
	MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ireduce_scatter_block_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype, op, comm, req);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ireduce_Scatter_Block_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype, op, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ireduce_scatter_block (sendbuf, recvbuf, recvcount, datatype, op,
			comm, req);

	DLB(DLB_MPI_Ireduce_scatter_block_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Alltoallw
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Alltoallw) (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *sdispls,
	MPI3_CONST MPI_Datatype *sendtypes, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
	MPI3_CONST MPI_Datatype *recvtypes, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Alltoallw_enter, MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST sdispls, MPI3_MPI_DATATYPE_P_CAST sendtypes, recvbuf, MPI3_C_INT_P_CAST recvcounts, MPI3_C_INT_P_CAST rdispls, MPI3_MPI_DATATYPE_P_CAST recvtypes, comm);

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Alltoallw_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST sdispls, MPI3_MPI_DATATYPE_P_CAST sendtypes, recvbuf, MPI3_C_INT_P_CAST recvcounts,
		  MPI3_C_INT_P_CAST rdispls, MPI3_MPI_DATATYPE_P_CAST recvtypes, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Alltoallw
		  (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
		  rdispls, recvtypes, comm);

	DLB(DLB_MPI_Alltoallw_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ialltoallw
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ialltoallw) (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *sdispls,
	MPI3_CONST MPI_Datatype *sendtypes, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
	MPI3_CONST MPI_Datatype *recvtypes, MPI_Comm comm, MPI_Request *req)
{
	int res;

	DLB(DLB_MPI_Ialltoallw_enter, MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST sdispls, MPI3_MPI_DATATYPE_P_CAST sendtypes, recvbuf, MPI3_C_INT_P_CAST recvcounts, MPI3_C_INT_P_CAST rdispls, MPI3_MPI_DATATYPE_P_CAST recvtypes, comm, req);

 

	Extrae_MPI_ProcessCollectiveCommunicator (comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ialltoallw_C_Wrapper
		  (MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST sdispls, MPI3_MPI_DATATYPE_P_CAST sendtypes, recvbuf, MPI3_C_INT_P_CAST recvcounts,
		  MPI3_C_INT_P_CAST rdispls, MPI3_MPI_DATATYPE_P_CAST recvtypes, comm, req);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ialltoallw
		  (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts,
		  rdispls, recvtypes, comm, req);

	DLB(DLB_MPI_Ialltoallw_leave);


	return res;
}

#endif /* MPI3 */

#endif /* C_SYMBOLS */
