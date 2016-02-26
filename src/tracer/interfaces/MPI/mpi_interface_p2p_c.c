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

#include "mpi_interface_coll_helper.h"

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
 ***  MPI_Bsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Bsend) (MPI3_CONST void* buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Bsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Bsend (buf, count, datatype, dest, tag, comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Ssend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ssend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ssend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ssend (buf, count, datatype, dest, tag, comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Rsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Rsend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Rsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Rsend (buf, count, datatype, dest, tag, comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Send
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Send) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Send_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Send (buf, count, datatype, dest, tag, comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Ibsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ibsend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibsend_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ibsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ibsend (buf, count, datatype, dest, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibsend_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Isend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Isend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Isend_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Isend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Isend_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Issend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Issend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Issend_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Issend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Issend (buf, count, datatype, dest, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Issend_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Irsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Irsend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irsend_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Irsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Irsend (buf, count, datatype, dest, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irsend_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Recv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Recv) (void* buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_enter (buf, count, datatype, source, tag, comm, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Recv_C_Wrapper (buf, count, datatype, source, tag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Recv (buf, count, datatype, source, tag, comm, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Irecv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Irecv) (void* buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irecv_enter (buf, count, datatype, source, tag, comm, request);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Irecv_C_Wrapper (buf, count, datatype, source, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Irecv (buf, count, datatype, source, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irecv_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Probe
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Probe) (int source, int tag, MPI_Comm comm,
	MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Probe_enter (source, tag, comm, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Probe_C_Wrapper (source, tag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Probe (source, tag, comm, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Probe_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Iprobe
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iprobe) (int source, int tag, MPI_Comm comm, int *flag,
	MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iprobe_enter (source, tag, comm, flag, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (4+Caller_Count[CALLER_MPI]);
		res = MPI_Iprobe_C_Wrapper (source, tag, comm, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		return PMPI_Iprobe (source, tag, comm, flag, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iprobe_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Test
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Test) (MPI_Request *request, int *flag, MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Test_enter (request, flag, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (5+Caller_Count[CALLER_MPI]);
		res = MPI_Test_C_Wrapper (request, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Test (request, flag, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Test_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Testall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Testall) (int count, MPI_Request *requests,
	int *flag, MPI_Status *statuses)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testall_enter (count, requests, flag, statuses);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+count+Caller_Count[CALLER_MPI]);
		res = MPI_Testall_C_Wrapper (count, requests, flag, statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Testall (count, requests, flag, statuses);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testall_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Testany
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Testany) (int count, MPI_Request *requests, int *index,
	int *flag, MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testany_enter (count, requests, index, flag, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		res = MPI_Testany_C_Wrapper (count, requests, index, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Testany (count, requests, index, flag, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testany_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Testsome
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Testsome) (int incount, MPI_Request * requests,
	int *outcount, int *indices, MPI_Status *statuses)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testsome_enter (incount, requests, outcount, indices, statuses);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+incount+Caller_Count[CALLER_MPI]);
		res = MPI_Testsome_C_Wrapper (incount, requests, outcount, indices, statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Testsome (incount, requests, outcount, indices, statuses);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testsome_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Wait
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Wait) (MPI_Request *request, MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Wait_enter (request, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		res = MPI_Wait_C_Wrapper (request, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Wait (request, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Wait_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Waitall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Waitall) (int count, MPI_Request *requests,
	MPI_Status *statuses)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitall_enter (count, requests, statuses);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+count+Caller_Count[CALLER_MPI]);
		res = MPI_Waitall_C_Wrapper (count, requests, statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Waitall (count, requests, statuses);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitall_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Waitany
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Waitany) (int count, MPI_Request *requests, int *index,
	MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitany_enter (count, requests, index, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		res = MPI_Waitany_C_Wrapper (count, requests, index, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = (PMPI_Waitany (count, requests, index, status));

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitany_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Waitsome
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Waitsome) (int incount, MPI_Request * requests,
	int *outcount, int *indices, MPI_Status *statuses)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitsome_enter (incount, requests, outcount, indices, statuses);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+incount+Caller_Count[CALLER_MPI]);
		res = MPI_Waitsome_C_Wrapper (incount,requests, outcount, indices,
			statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Waitsome (incount, requests, outcount, indices, statuses);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitsome_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Recv_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Recv_init) (void *buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_init_enter (buf, count, datatype, source, tag, comm,
		request);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Recv_init_C_Wrapper
		  (buf, count, datatype, source, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res =  PMPI_Recv_init
		  (buf, count, datatype, source, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_init_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Send_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Send_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_init_enter (buf, count, datatype, dest, tag, comm,
		request);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Send_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Send_init (buf, count, datatype, dest, tag, comm,
			request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_init_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Bsend_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Bsend_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_init_enter (buf, count, datatype, dest, tag, comm,
		request);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Bsend_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Bsend_init (buf, count, datatype, dest, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_init_leave ();
#endif
	return res;
}


/******************************************************************************
 ***  MPI_Rsend_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Rsend_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_init_enter (buf, count, datatype, dest, tag, comm,
		request);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Rsend_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Rsend_init (buf, count, datatype, dest, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_init_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Ssend_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ssend_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_init_enter (buf, count, datatype, dest, tag, comm,
		request);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Ssend_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ssend_init (buf, count, datatype, dest, tag, comm, request);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_init_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Sendrecv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Sendrecv) (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount,
	MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
	MPI_Status * status) 
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, dest, sendtag,
		recvbuf, recvcount, recvtype, source, recvtag, comm, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Sendrecv_C_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, dest, sendtag,
		  recvbuf, recvcount, recvtype, source, recvtag, comm, status); 
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Sendrecv (sendbuf, sendcount, sendtype, dest, sendtag,
		  recvbuf, recvcount, recvtype, source, recvtag, comm, status); 

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Sendrecv_replace
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Sendrecv_replace) (void *buf, int count, MPI_Datatype type,
	int dest, int sendtag, int source, int recvtag, MPI_Comm comm,
	MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_replace_enter (buf, count, type, dest, sendtag, source,
		recvtag, comm, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Sendrecv_replace_C_Wrapper (buf, count, type, dest, sendtag,
		  source, recvtag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Sendrecv_replace (buf, count, type, dest, sendtag, source,
		  recvtag, comm, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_replace_leave ();
#endif

	return res;
}

#endif /* defined(C_SYMBOLS) */

