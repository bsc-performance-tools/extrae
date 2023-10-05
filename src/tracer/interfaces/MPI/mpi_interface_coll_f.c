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
# define NAME_ROUTINE_F(x) patch_p##x  /* mpi_send is converted to patch_pmpi_send */
# define NAME_ROUTINE_FU(x) patch_P##x  /* mpi_send is converted to patch_Pmpi_send */
# define NAME_ROUTINE_C2F(x) CtoF77(patch_p##x)  /* mpi_send may be converted to patch_pmpi_send_ */
#else
# define NAME_ROUTINE_F(x) x
# define NAME_ROUTINE_C2F(x) CtoF77(x)
#endif

#if defined(FORTRAN_SYMBOLS)
# include "extrae_mpif.h"
#endif

#if defined(HAVE_ALIAS_ATTRIBUTE) 

/* This macro defines r1, r2 and r3 to be aliases to "orig" routine.
   params are the same parameters received by "orig" */

# if defined(DYNINST_MODULE)

/* MPI_F_SYMS define different Fortran synonymous using the __attribute__ 
	 compiler constructor. Use r3 in the UPPERCASE VERSION of the MPI call. */

#  define MPI_F_SYMS(r1,r2,r3,orig,params) \
    void NAME_ROUTINE_F(r1) params __attribute__ ((alias ("patch_p"#orig))); \
    void NAME_ROUTINE_F(r2) params __attribute__ ((alias ("patch_p"#orig))); \
    void NAME_ROUTINE_FU(r3) params __attribute__ ((alias ("patch_p"#orig)));
# else
#  define MPI_F_SYMS(r1,r2,r3,orig,params) \
    void r1 params __attribute__ ((alias (#orig))); \
    void r2 params __attribute__ ((alias (#orig))); \
    void r3 params __attribute__ ((alias (#orig)));

# endif
 
#endif

#if defined(FORTRAN_SYMBOLS)

/******************************************************************************
 ***  MPI_Reduce
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_reduce__,mpi_reduce_,MPI_REDUCE,mpi_reduce,(void *sendbuf, void *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_reduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_reduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Reduce_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Reduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                         ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_reduce) (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                          ierror);

	DLB(DLB_MPI_Reduce_F_leave);
}

/******************************************************************************
 ***  MPI_Reduce_scatter
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_reduce_scatter__,mpi_reduce_scatter_,MPI_REDUCE_SCATTER,mpi_reduce_scatter,(void *sendbuf, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_reduce_scatter) (void *sendbuf, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_reduce_scatter) (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Reduce_scatter_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Reduce_Scatter_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_reduce_scatter) (sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, ierror);

        DLB(DLB_MPI_Reduce_scatter_F_leave);
}

/******************************************************************************
 ***  MPI_AllReduce
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_allreduce__,mpi_allreduce_,MPI_ALLREDUCE,mpi_allreduce,(void *sendbuf, void *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_allreduce) (void *sendbuf, void *recvbuf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_allreduce) (void *sendbuf, void *recvbuf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Allreduce_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count,
		datatype, op, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_AllReduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf,
			count, datatype, op, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_allreduce) (sendbuf, recvbuf, count, datatype, op,
			comm, ierror);
	DLB(DLB_MPI_Allreduce_F_leave);
}

/******************************************************************************
 ***  MPI_Barrier
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_barrier__,mpi_barrier_,MPI_BARRIER,mpi_barrier,(MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_barrier) (MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_barrier) (MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Barrier_F_enter, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);
    
	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Barrier_Wrapper (comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_barrier) (comm, ierror);

	DLB(DLB_MPI_Barrier_F_leave);
}

/******************************************************************************
 ***  MPI_BCast
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_bcast__,mpi_bcast_,MPI_BCAST,mpi_bcast,(void *buffer, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_bcast) (void *buffer, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_bcast) (void *buffer, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Bcast_F_enter, buffer, count, datatype, root, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_BCast_Wrapper (buffer, count, datatype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_bcast) (buffer, count, datatype, root, comm, ierror);

	DLB(DLB_MPI_Bcast_F_leave);
}

/******************************************************************************
 ***  MPI_AllToAll
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_alltoall__,mpi_alltoall_,MPI_ALLTOALL,mpi_alltoall, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_alltoall) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_alltoall) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Alltoall_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_AllToAll_Wrapper (MPI3_VOID_P_CAST sendbuf,
			sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_alltoall) (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, ierror);
	DLB(DLB_MPI_Alltoall_F_leave);
}


/******************************************************************************
 ***  MPI_AllToAllV
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_alltoallv__,mpi_alltoallv_,MPI_ALLTOALLV,mpi_alltoallv, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_alltoallv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_alltoallv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Alltoallv_F_enter, MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_AllToAllV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf,
                            MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_alltoallv) (sendbuf, sendcount, sdispls, sendtype,
			recvbuf, recvcount, rdispls, recvtype, comm, ierror);
	DLB(DLB_MPI_Alltoallv_F_leave);
}


/******************************************************************************
 ***  MPI_Allgather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_allgather__,mpi_allgather_,MPI_ALLGATHER,mpi_allgather, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_allgather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_allgather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Allgather_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Allgather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_allgather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, comm, ierror);
	DLB(DLB_MPI_Allgather_F_leave);
}


/******************************************************************************
 ***  MPI_Allgatherv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_allgatherv__,mpi_allgatherv_,MPI_ALLGATHERV,mpi_allgatherv, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_allgatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_allgatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Allgatherv_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, MPI3_VOID_P_CAST recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Allgatherv_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, MPI3_VOID_P_CAST recvbuf, 
			MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs,
			recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_allgatherv) (sendbuf, sendcount, sendtype, recvbuf,
                              recvcount, displs, recvtype, comm, ierror);
	DLB(DLB_MPI_Allgatherv_F_leave);
}


/******************************************************************************
 ***  MPI_Gather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_gather__,mpi_gather_,MPI_GATHER,mpi_gather, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_gather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_gather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Gather_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Gather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_gather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, ierror);
	DLB(DLB_MPI_Gather_F_leave);
}

/******************************************************************************
 ***  MPI_GatherV
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_gatherv__,mpi_gatherv_,MPI_GATHERV,mpi_gatherv, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_gatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_gatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Gatherv_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, root, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_GatherV_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount,
			MPI3_F_INT_P_CAST displs, recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_gatherv) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, displs, recvtype, root, comm, ierror);
			
	DLB(DLB_MPI_Gatherv_F_leave);
}

/******************************************************************************
 ***  MPI_Scatter
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_scatter__,mpi_scatter_,MPI_SCATTER,mpi_scatter,(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_scatter) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_scatter) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Scatter_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Scatter_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_scatter) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, ierror);
	DLB(DLB_MPI_Scatter_F_leave);
}

/******************************************************************************
 ***  MPI_ScatterV
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_scatterv__,mpi_scatterv_,MPI_SCATTERV,mpi_scatterv,(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_scatterv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_scatterv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Scatterv_F_enter, MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_ScatterV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf,
                           recvcount, recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_scatterv) (sendbuf, sendcount, displs, sendtype,
			recvbuf, recvcount, recvtype, root, comm, ierror);
	DLB(DLB_MPI_Scatterv_F_leave);
}

/******************************************************************************
 ***  MPI_Scan
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_scan__,mpi_scan_,MPI_SCAN,mpi_scan, (void *sendbuf, void *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_scan) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_scan) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Scan_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Scan_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count,
			datatype, op, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_scan) (sendbuf, recvbuf, count, datatype, op, comm,
			ierror);
			
	DLB(DLB_MPI_Scan_F_leave);
}


#if defined(MPI3)
/******************************************************************************
 ***  MPI_Ireduce
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ireduce__,mpi_ireduce_,MPI_IREDUCE,mpi_ireduce,(void *sendbuf, void *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ireduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ireduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ireduce_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Ireduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                         req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ireduce) (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                          req, ierror);

	DLB(DLB_MPI_Ireduce_F_leave);
}

/******************************************************************************
 ***  MPI_Ireduce_scatter
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ireduce_scatter__,mpi_ireduce_scatter_,MPI_IREDUCE_SCATTER,mpi_ireduce_scatter,(void *sendbuf, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ireduce_scatter) (void *sendbuf, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ireduce_scatter) (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ireduce_scatter_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Ireduce_Scatter_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ireduce_scatter) (sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, req, ierror);

	DLB(DLB_MPI_Ireduce_scatter_F_leave);
}

/******************************************************************************
 ***  MPI_Iallreduce
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_iallreduce__,mpi_iallreduce_,MPI_IALLREDUCE,mpi_iallreduce,(void *sendbuf, void *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_iallreduce) (void *sendbuf, void *recvbuf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_iallreduce) (void *sendbuf, void *recvbuf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Iallreduce_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count,
		datatype, op, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_IallReduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf,
			count, datatype, op, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iallreduce) (sendbuf, recvbuf, count, datatype, op,
			comm, req, ierror);
			
	DLB(DLB_MPI_Iallreduce_F_leave);
}

/******************************************************************************
 ***  MPI_Ibarrier
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ibarrier__,mpi_ibarrier_,MPI_IBARRIER,mpi_ibarrier,(MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ibarrier) (MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ibarrier) (MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ibarrier_F_enter, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);
    
	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Ibarrier_Wrapper (comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ibarrier) (comm, req, ierror);

	DLB(DLB_MPI_Ibarrier_F_leave);
}

/******************************************************************************
 ***  MPI_Ibcast
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ibcast__,mpi_ibcast_,MPI_IBCAST,mpi_ibcast,(void *buffer, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ibcast) (void *buffer, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ibcast) (void *buffer, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ibcast_F_enter, buffer, count, datatype, root, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Ibcast_Wrapper (buffer, count, datatype, root, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ibcast) (buffer, count, datatype, root, comm, req, ierror);

	DLB(DLB_MPI_Ibcast_F_leave);
}

/******************************************************************************
 ***  MPI_Ialltoall
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ialltoall__,mpi_ialltoall_,MPI_IALLTOALL,mpi_ialltoall, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ialltoall) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ialltoall) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ialltoall_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_IallToAll_Wrapper (MPI3_VOID_P_CAST sendbuf,
			sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
			req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ialltoall) (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);

	DLB(DLB_MPI_Ialltoall_F_leave);
}


/******************************************************************************
 ***  MPI_Ialltoallv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ialltoallv__,mpi_ialltoallv_,MPI_IALLTOALLV,mpi_ialltoallv, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ialltoallv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ialltoallv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ialltoallv_F_enter, MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_IallToAllV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, req,  ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ialltoallv) (sendbuf, sendcount, sdispls, sendtype,
			recvbuf, recvcount, rdispls, recvtype, comm, req, ierror);

	DLB(DLB_MPI_Ialltoallv_F_leave);
}


/******************************************************************************
 ***  MPI_Iallgather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_iallgather__,mpi_iallgather_,MPI_IALLGATHER,mpi_iallgather, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_iallgather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_iallgather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Iallgather_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Iallgather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iallgather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, comm, req, ierror);

	DLB(DLB_MPI_Iallgather_F_leave);
}


/******************************************************************************
 ***  MPI_Allgatherv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_iallgatherv__,mpi_iallgatherv_,MPI_IALLGATHERV,mpi_iallgatherv, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_iallgatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_iallgatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Iallgatherv_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, MPI3_VOID_P_CAST recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Iallgatherv_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, MPI3_VOID_P_CAST recvbuf, 
			MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs,
			recvtype, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iallgatherv) (sendbuf, sendcount, sendtype, recvbuf,
                              recvcount, displs, recvtype, comm, req, ierror);

	DLB(DLB_MPI_Iallgatherv_F_leave);
}


/******************************************************************************
 ***  MPI_Igather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_igather__,mpi_igather_,MPI_IGATHER,mpi_igather, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_igather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_igather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Igather_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Igather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_igather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, req, ierror);

	DLB(DLB_MPI_Igather_F_leave);
}

/******************************************************************************
 ***  MPI_Igatherv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_igatherv__,mpi_igatherv_,MPI_IGATHERV,mpi_igatherv, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_igatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_igatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Igatherv_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, root, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_IgatherV_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount,
			MPI3_F_INT_P_CAST displs, recvtype, root, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_igatherv) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, displs, recvtype, root, comm, req, ierror);

	DLB(DLB_MPI_Igatherv_F_leave);
}

/******************************************************************************
 ***  MPI_Iscatter
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_iscatter__,mpi_iscatter_,MPI_ISCATTER,mpi_iscatter,(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_iscatter) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_iscatter) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Iscatter_F_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Iscatter_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iscatter) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, req, ierror);

	DLB(DLB_MPI_Iscatter_F_leave);
}

/******************************************************************************
 ***  MPI_Iscatterv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_iscatterv__,mpi_iscatterv_,MPI_ISCATTERV,mpi_iscatterv,(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_iscatterv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_iscatterv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Iscatterv_F_enter, MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_IscatterV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iscatterv) (sendbuf, sendcount, displs, sendtype,
			recvbuf, recvcount, recvtype, root, comm, req, ierror);

	DLB(DLB_MPI_Iscatterv_F_leave);
}

/******************************************************************************
 ***  MPI_Iscan
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_iscan__,mpi_iscan_,MPI_ISCAN,mpi_iscan, (void *sendbuf, void *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_iscan) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_iscan) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Iscan_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Iscan_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count,
			datatype, op, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iscan) (sendbuf, recvbuf, count, datatype, op, comm,
			req, ierror);

	DLB(DLB_MPI_Iscan_F_leave);
}

/******************************************************************************
 ***  MPI_Ireduce_scatter_block
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ireduce_scatter_block__,mpi_ireduce_scatter_block_,MPI_IREDUCE_SCATTER_BLOCK,mpi_ireduce_scatter_block,(void *sendbuf, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ireduce_scatter_block) (void *sendbuf, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ireduce_scatter_block) (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ireduce_scatter_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype, op, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Ireduce_Scatter_Block_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype, op, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ireduce_scatter_block) (sendbuf, recvbuf, recvcount, datatype, op, comm, req, ierror);

	DLB(DLB_MPI_Ireduce_scatter_block_F_leave);
}

/******************************************************************************
 ***  MPI_Reduce_scatter_block
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_reduce_scatter_block__,mpi_reduce_scatter_block_,MPI_REDUCE_SCATTER_BLOCK,mpi_reduce_scatter_block,(void *sendbuf, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_reduce_scatter_block) (void *sendbuf, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_reduce_scatter_block) (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Reduce_scatter_block_F_enter, MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype, op, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_Reduce_Scatter_Block_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, recvcount, datatype, op, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_reduce_scatter_block) (sendbuf, recvbuf, recvcount, datatype, op,
			comm, ierror);

	DLB(DLB_MPI_Reduce_scatter_block_F_leave);
}

/******************************************************************************
 ***  MPI_Alltoallw
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_alltoallw__,mpi_alltoallw_,MPI_ALLTOALLW,mpi_alltoallw, (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_alltoallw) (void *sendbuf, MPI_Fint *sendcounts,
	MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *rdispls, MPI_Fint *recvtypes,	MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_alltoallw) (void *sendbuf, MPI_Fint *sendcounts,
	MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *rdispls, MPI_Fint *recvtypes,	MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Alltoallw_F_enter, MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcounts, MPI3_F_INT_P_CAST sdispls, MPI3_F_INT_P_CAST sendtypes, recvbuf, MPI3_F_INT_P_CAST recvcounts, MPI3_F_INT_P_CAST rdispls, MPI3_F_INT_P_CAST recvtypes, comm, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_AllToAllW_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcounts, MPI3_F_INT_P_CAST sdispls, MPI3_F_INT_P_CAST sendtypes, recvbuf,
                            MPI3_F_INT_P_CAST recvcounts, MPI3_F_INT_P_CAST rdispls, MPI3_F_INT_P_CAST recvtypes, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_alltoallw) (sendbuf, sendcounts, sdispls, sendtypes,
			recvbuf, recvcounts, rdispls, recvtypes, comm, ierror);
	DLB(DLB_MPI_Alltoallw_F_leave);
}

/******************************************************************************
 ***  MPI_Ialltoallw
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ialltoallw__,mpi_ialltoallw_,MPI_IALLTOALLW,mpi_ialltoallw, (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ialltoallw) (void *sendbuf, MPI_Fint *sendcounts,
	MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *rdispls, MPI_Fint *recvtypes,	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ialltoallw) (void *sendbuf, MPI_Fint *sendcounts,
	MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *rdispls, MPI_Fint *recvtypes,	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ialltoallw_F_enter, MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcounts, MPI3_F_INT_P_CAST sdispls, MPI3_F_INT_P_CAST sendtypes, recvbuf, MPI3_F_INT_P_CAST recvcounts, MPI3_F_INT_P_CAST rdispls, MPI3_F_INT_P_CAST recvtypes, comm, req, ierror);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		PMPI_IallToAllW_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcounts, MPI3_F_INT_P_CAST sdispls, MPI3_F_INT_P_CAST sendtypes, recvbuf, MPI3_F_INT_P_CAST recvcounts, MPI3_F_INT_P_CAST rdispls, MPI3_F_INT_P_CAST recvtypes, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ialltoallw) (sendbuf, sendcounts, sdispls, sendtypes,
			recvbuf, recvcounts, rdispls, recvtypes, comm, req, ierror);
	DLB(DLB_MPI_Ialltoallw_F_leave);
}

/******************************************************************************
 ***  MPI_Graph_create
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_graph_create__,mpi_graph_create_,MPI_GRAPH_CREATE,mpi_graph_create, 
  (MPI_Fint *comm_old, MPI_Fint *nnodes, MPI_Fint *index, MPI_Fint *edges, MPI_Fint *reorder, MPI_Fint *comm_graph, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_graph_create) (MPI_Fint *comm_old, MPI_Fint *nnodes, MPI_Fint *index, MPI_Fint *edges, MPI_Fint *reorder, MPI_Fint *comm_graph, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_graph_create) (MPI_Fint *comm_old, MPI_Fint *nnodes, MPI_Fint *index, MPI_Fint *edges, MPI_Fint *reorder, MPI_Fint *comm_graph, MPI_Fint *ierr)
#endif
{
	DLB(DLB_MPI_Graph_create_F_enter, comm_old, nnodes, index, edges, reorder, comm_graph, ierr);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Graph_create_Wrapper (comm_old, nnodes, index, edges, reorder, comm_graph, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_graph_create) (comm_old, nnodes, index, edges, reorder, comm_graph, ierr);
	}

	DLB(DLB_MPI_Graph_create_F_leave);
}

/******************************************************************************
 ***  MPI_Dist_graph_create
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_dist_graph_create__,mpi_dist_graph_create_,MPI_DIST_GRAPH_CREATE,mpi_dist_graph_create,
  (MPI_Fint *comm_old, MPI_Fint *n, MPI_Fint *sources, MPI_Fint *degrees, MPI_Fint *destinations, MPI_Fint *weights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_dist_graph_create) (MPI_Fint *comm_old, MPI_Fint *n, MPI_Fint *sources, MPI_Fint *degrees, MPI_Fint *destinations, MPI_Fint *weights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_dist_graph_create) (MPI_Fint *comm_old, MPI_Fint *n, MPI_Fint *sources, MPI_Fint *degrees, MPI_Fint *destinations, MPI_Fint *weights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr)
#endif
{
	DLB(DLB_MPI_Dist_graph_create_F_enter, comm_old, n, sources, degrees, destinations, weights, info, reorder, comm_dist_graph, ierr);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Dist_graph_create_Wrapper (comm_old, n, sources, degrees, destinations, weights, info, reorder, comm_dist_graph, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_dist_graph_create) (comm_old, n, sources, degrees, destinations, weights, info, reorder, comm_dist_graph, ierr);
	}

	DLB(DLB_MPI_Dist_graph_create_F_leave);
}

/******************************************************************************
 ***  MPI_Dist_graph_create_adjacent
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_dist_graph_create_adjacent__,mpi_dist_graph_create_adjacent_,MPI_DIST_GRAPH_CREATE_ADJACENT,mpi_dist_graph_create_adjacent,
  (MPI_Fint *comm_old, MPI_Fint *indegree, MPI_Fint *sources, MPI_Fint *sourceweights, MPI_Fint *outdegree, MPI_Fint *destinations, MPI_Fint *destweights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_dist_graph_create_adjacent) (MPI_Fint *comm_old, MPI_Fint *indegree, MPI_Fint *sources, MPI_Fint *sourceweights, MPI_Fint *outdegree, MPI_Fint *destinations, MPI_Fint *destweights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_dist_graph_create_adjacent) (MPI_Fint *comm_old, MPI_Fint *indegree, MPI_Fint *sources, MPI_Fint *sourceweights, MPI_Fint *outdegree, MPI_Fint *destinations, MPI_Fint *destweights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr)
#endif
{
	DLB(DLB_MPI_Dist_graph_create_adjacent_F_enter, comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph, ierr);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Dist_graph_create_adjacent_Wrapper (comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_dist_graph_create_adjacent) (comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph, ierr);
	}

	DLB(DLB_MPI_Dist_graph_create_adjacent_F_leave);
}

/******************************************************************************
 ***  MPI_Neighbor_allgather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_neighbor_allgather__,mpi_neighbor_allgather_,MPI_NEIGHBOR_ALLGATHER,mpi_neighbor_allgather,
  (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_neighbor_allgather) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_neighbor_allgather) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Neighbor_allgather_F_enter, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Neighbor_allgather_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77 (pmpi_neighbor_allgather) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
	}

	DLB(DLB_MPI_Neighbor_allgather_F_leave);
}

/******************************************************************************
 ***  MPI_Ineighbor_allgather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ineighbor_allgather__,mpi_ineighbor_allgather_,MPI_INEIGHBOR_ALLGATHER,mpi_ineighbor_allgather,
  (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_ineighbor_allgather) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_ineighbor_allgather) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ineighbor_allgather_F_enter, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Ineighbor_allgather_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77 (pmpi_ineighbor_allgather) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
	}

	DLB(DLB_MPI_Ineighbor_allgather_F_leave);
}

/******************************************************************************
 ***  MPI_Neighbor_allgatherv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_neighbor_allgatherv__,mpi_neighbor_allgatherv_,MPI_NEIGHBOR_ALLGATHERV,mpi_neighbor_allgatherv,
  (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_neighbor_allgatherv) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_neighbor_allgatherv) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Neighbor_allgatherv_F_enter, sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Neighbor_allgatherv_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_neighbor_allgatherv) (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr);
	}

	DLB(DLB_MPI_Neighbor_allgatherv_F_leave);
}

/******************************************************************************
 ***  MPI_Ineighbor_allgatherv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ineighbor_allgatherv__,mpi_ineighbor_allgatherv_,MPI_INEIGHBOR_ALLGATHERV,mpi_ineighbor_allgatherv,
  (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_ineighbor_allgatherv) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_ineighbor_allgatherv) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ineighbor_allgatherv_F_enter, sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Ineighbor_allgatherv_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_ineighbor_allgatherv) (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request, ierr);
	}

	DLB(DLB_MPI_Ineighbor_allgatherv_F_leave);
}

/******************************************************************************
 ***  MPI_Neighbor_alltoall
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_neighbor_alltoall__,mpi_neighbor_alltoall_,MPI_NEIGHBOR_ALLTOALL,mpi_neighbor_alltoall,
  (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_neighbor_alltoall) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_neighbor_alltoall) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Neighbor_alltoall_F_enter, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Neighbor_alltoall_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_neighbor_alltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierr);
	}

	DLB(DLB_MPI_Neighbor_alltoall_F_leave);
}

/******************************************************************************
 ***  MPI_Ineighbor_alltoall
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ineighbor_alltoall__,mpi_ineighbor_alltoall_,MPI_INEIGHBOR_ALLTOALL,mpi_ineighbor_alltoall,
  (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_ineighbor_alltoall) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_ineighbor_alltoall) (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ineighbor_alltoall_F_enter, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Ineighbor_alltoall_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_ineighbor_alltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request, ierr);
	}

	DLB(DLB_MPI_Ineighbor_alltoall_F_leave);
}

/******************************************************************************
 ***  MPI_Neighbor_alltoallv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_neighbor_alltoallv__,mpi_neighbor_alltoallv_,MPI_NEIGHBOR_ALLTOALLV,mpi_neighbor_alltoallv,
  (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_neighbor_alltoallv) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_neighbor_alltoallv) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Neighbor_alltoallv_F_enter, sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Neighbor_alltoallv_Wrapper (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_neighbor_alltoallv) (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, ierr);
	}

	DLB(DLB_MPI_Neighbor_alltoallv_F_leave);
}

/******************************************************************************
 ***  MPI_Ineighbor_alltoallv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ineighbor_alltoallv__,mpi_ineighbor_alltoallv_,MPI_INEIGHBOR_ALLTOALLV,mpi_ineighbor_alltoallv,
  (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_ineighbor_alltoallv) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_ineighbor_alltoallv) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ineighbor_alltoallv_F_enter, sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Ineighbor_alltoallv_Wrapper (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_ineighbor_alltoallv) (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request, ierr);
	}

	DLB(DLB_MPI_Ineighbor_alltoallv_F_leave);
}

/******************************************************************************
 ***  MPI_Neighbor_alltoallw
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_neighbor_alltoallw__,mpi_neighbor_alltoallw_,MPI_NEIGHBOR_ALLTOALLW,mpi_neighbor_alltoallw,
  (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_neighbor_alltoallw) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_neighbor_alltoallw) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Neighbor_alltoallw_F_enter, sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Neighbor_alltoallw_Wrapper (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_neighbor_alltoallw) (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr);
	}

	DLB(DLB_MPI_Neighbor_alltoallw_F_leave);
}

/******************************************************************************
 ***  MPI_Ineighbor_alltoallw
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ineighbor_alltoallw__,mpi_ineighbor_alltoallw_,MPI_INEIGHBOR_ALLTOALLW,mpi_ineighbor_alltoallw,
  (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_ineighbor_alltoallw) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_ineighbor_alltoallw) (void *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
#endif
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

	DLB(DLB_MPI_Ineighbor_alltoallw_F_enter, sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER);
		Backend_Enter_Instrumentation ();
		PMPI_Ineighbor_alltoallw_Wrapper (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE);
	}
	else
	{
		CtoF77(pmpi_ineighbor_alltoallw) (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr);
	}

	DLB(DLB_MPI_Ineighbor_alltoallw_F_leave);
}

#endif /* MPI3 */

#endif /* defined(FORTRAN_SYMBOLS) */
