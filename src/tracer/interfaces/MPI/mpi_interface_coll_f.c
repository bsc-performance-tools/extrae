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

#if defined(MPI3)
#define MPI3_CONST const
#define MPI3_VOID_P_CAST (void *)
#define MPI3_CHAR_P_CAST (char *)
#define MPI3_F_INT_P_CAST (MPI_Fint *)
#define MPI3_C_INT_P_CAST (int *)
#define MPI3_MPI_INFO_P_CAST (MPI_Info *)
#else
#define MPI3_CONST
#define MPI3_VOID_P_CAST
#define MPI3_CHAR_P_CAST
#define MPI3_F_INT_P_CAST
#define MPI3_C_INT_P_CAST
#define MPI3_MPI_INFO_P_CAST
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
# if defined(FORTRAN_SYMBOLS)
#  include "MPI_interfaceF.h"
# endif
# if defined(C_SYMBOLS)
#  include "MPI_interface.h"
# endif
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Reduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                         ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_reduce) (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                          ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_scatter_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Reduce_Scatter_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_reduce_scatter) (sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_scatter_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allreduce_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count,
		datatype, op, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_AllReduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf,
			count, datatype, op, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_allreduce) (sendbuf, recvbuf, count, datatype, op,
			comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allreduce_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Barrier_F_enter (comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);
    
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Barrier_Wrapper (comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_barrier) (comm, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Barrier_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bcast_F_enter (buffer, count, datatype, root, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_BCast_Wrapper (buffer, count, datatype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_bcast) (buffer, count, datatype, root, comm, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bcast_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoall_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_AllToAll_Wrapper (MPI3_VOID_P_CAST sendbuf,
			sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_alltoall) (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoall_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoallv_F_enter (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_AllToAllV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf,
                            MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_alltoallv) (sendbuf, sendcount, sdispls, sendtype,
			recvbuf, recvcount, rdispls, recvtype, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoallv_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgather_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Allgather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_allgather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgather_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgatherv_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, MPI3_VOID_P_CAST recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgatherv_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gather_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Gather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_gather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gather_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gatherv_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, root, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_GatherV_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount,
			MPI3_F_INT_P_CAST displs, recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_gatherv) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, displs, recvtype, root, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gatherv_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatter_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Scatter_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_scatter) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatter_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatterv_F_enter (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_ScatterV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf,
                           recvcount, recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_scatterv) (sendbuf, sendcount, displs, sendtype,
			recvbuf, recvcount, recvtype, root, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatterv_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scan_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Scan_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count,
			datatype, op, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_scan) (sendbuf, recvbuf, count, datatype, op, comm,
			ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scan_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ireduce_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Ireduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                         req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ireduce) (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm,
                          req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ireduce_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ireduce_scatter_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Ireduce_Scatter_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ireduce_scatter) (sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op,
			comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ireduce_scatter_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iallreduce_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count,
		datatype, op, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IallReduce_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf,
			count, datatype, op, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iallreduce) (sendbuf, recvbuf, count, datatype, op,
			comm, req, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iallreduce_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibarrier_F_enter (comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);
    
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Ibarrier_Wrapper (comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ibarrier) (comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibarrier_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibcast_F_enter (buffer, count, datatype, root, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Ibcast_Wrapper (buffer, count, datatype, root, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ibcast) (buffer, count, datatype, root, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibcast_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_IAllToAll
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ialltoall_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IallToAll_Wrapper (MPI3_VOID_P_CAST sendbuf,
			sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
			req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ialltoall) (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ialltoall_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_IAllToAllV
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ialltoallv_F_enter (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IallToAllV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, req,  ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ialltoallv) (sendbuf, sendcount, sdispls, sendtype,
			recvbuf, recvcount, rdispls, recvtype, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ialltoallv_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_IAllgather
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iallgather_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Iallgather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iallgather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iallgather_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iallgatherv_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, MPI3_VOID_P_CAST recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iallgatherv_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Igather_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Igather_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_igather) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Igather_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_IGatherV
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Igatherv_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, root, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IgatherV_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount,
			MPI3_F_INT_P_CAST displs, recvtype, root, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_igatherv) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, displs, recvtype, root, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Igatherv_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iscatter_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Iscatter_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, recvbuf, recvcount, recvtype, root, comm,
			req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iscatter) (sendbuf, sendcount, sendtype, recvbuf,
			recvcount, recvtype, root, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iscatter_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_IScatterV
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iscatterv_F_enter (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IscatterV_Wrapper (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iscatterv) (sendbuf, sendcount, displs, sendtype,
			recvbuf, recvcount, recvtype, root, comm, req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iscatterv_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iscan_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, req, ierror);
#endif

	Extrae_MPI_ProcessCollectiveCommunicator (c);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Iscan_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count,
			datatype, op, comm, req, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iscan) (sendbuf, recvbuf, count, datatype, op, comm,
			req, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iscan_F_leave ();
#endif
}
#endif /* MPI3 */

#endif /* defined(FORTRAN_SYMBOLS) */
