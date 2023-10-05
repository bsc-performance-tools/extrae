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
 ***  MPI_Bsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Bsend) (MPI3_CONST void* buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Bsend_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Bsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Bsend (buf, count, datatype, dest, tag, comm);

	DLB(DLB_MPI_Bsend_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ssend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ssend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Ssend_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Ssend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ssend (buf, count, datatype, dest, tag, comm);

	DLB(DLB_MPI_Ssend_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Rsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Rsend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Rsend_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Rsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Rsend (buf, count, datatype, dest, tag, comm);

	DLB(DLB_MPI_Rsend_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Send
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Send) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)
{
	int res;

	DLB(DLB_MPI_Send_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Send_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Send (buf, count, datatype, dest, tag, comm);

	DLB(DLB_MPI_Send_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Ibsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ibsend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Ibsend_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Ibsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ibsend (buf, count, datatype, dest, tag, comm, request);

	DLB(DLB_MPI_Ibsend_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Isend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Isend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Isend_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Isend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);

	DLB(DLB_MPI_Isend_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Issend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Issend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Issend_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Issend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Issend (buf, count, datatype, dest, tag, comm, request);

	DLB(DLB_MPI_Issend_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Irsend
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Irsend) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Irsend_enter, MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Irsend_C_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Irsend (buf, count, datatype, dest, tag, comm, request);

	DLB(DLB_MPI_Irsend_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Recv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Recv) (void* buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_Recv_enter, buf, count, datatype, source, tag, comm, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Recv_C_Wrapper (buf, count, datatype, source, tag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Recv (buf, count, datatype, source, tag, comm, status);

	DLB(DLB_MPI_Recv_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Irecv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Irecv) (void* buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Irecv_enter, buf, count, datatype, source, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Irecv_C_Wrapper (buf, count, datatype, source, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Irecv (buf, count, datatype, source, tag, comm, request);

	DLB(DLB_MPI_Irecv_leave);

	return res;
}

#if defined(MPI3)

/******************************************************************************
 ***  MPI_Mrecv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Mrecv) (void* buf, int count, MPI_Datatype datatype,
        MPI_Message *message, MPI_Status *status)
{
        int res;

        DLB(DLB_MPI_Mrecv_enter, buf, count, datatype, message, status);

        if (INSTRUMENT_THIS_MPI)
        {
                DEBUG_INTERFACE(ENTER)
                Backend_Enter_Instrumentation ();
                res = MPI_Mrecv_C_Wrapper (buf, count, datatype, message, status);
                Backend_Leave_Instrumentation ();
                DEBUG_INTERFACE(LEAVE)
        }
        else
                res = PMPI_Mrecv (buf, count, datatype, message, status);

        DLB(DLB_MPI_Mrecv_leave);

        return res;
}

/******************************************************************************
 ***  MPI_Imrecv
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Imrecv) (void* buf, int count, MPI_Datatype datatype,
        MPI_Message *message, MPI_Request *request)
{
        int res;

        DLB(DLB_MPI_Imrecv_enter, buf, count, datatype, message, request);

        if (INSTRUMENT_THIS_MPI)
        {
                DEBUG_INTERFACE(ENTER)
                Backend_Enter_Instrumentation ();
                res = MPI_Imrecv_C_Wrapper (buf, count, datatype, message, request);
                Backend_Leave_Instrumentation ();
                DEBUG_INTERFACE(LEAVE)
        }
        else
                res = PMPI_Imrecv (buf, count, datatype, message, request);

        DLB(DLB_MPI_Imrecv_leave);

        return res;
}

#endif /* MPI3 */

/******************************************************************************
 ***  MPI_Probe
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Probe) (int source, int tag, MPI_Comm comm,
	MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_Probe_enter, source, tag, comm, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Probe_C_Wrapper (source, tag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Probe (source, tag, comm, status);

	DLB(DLB_MPI_Probe_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Iprobe
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Iprobe) (int source, int tag, MPI_Comm comm, int *flag,
	MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_Iprobe_enter, source, tag, comm, flag, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Iprobe_C_Wrapper (source, tag, comm, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		return PMPI_Iprobe (source, tag, comm, flag, status);

	DLB(DLB_MPI_Iprobe_leave);
	
	return res;
}

#if defined(MPI3)

/******************************************************************************
 ***  MPI_Mprobe
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Mprobe) (int source, int tag, MPI_Comm comm,
        MPI_Message *message, MPI_Status *status)
{
        int res;

        DLB(DLB_MPI_Mprobe_enter, source, tag, comm, message, status);

        if (INSTRUMENT_THIS_MPI)
        {
                DEBUG_INTERFACE(ENTER)
                Backend_Enter_Instrumentation ();
                res = MPI_Mprobe_C_Wrapper (source, tag, comm, message, status);
                Backend_Leave_Instrumentation ();
                DEBUG_INTERFACE(LEAVE)
        }
        else
                res = PMPI_Mprobe (source, tag, comm, message, status);

        DLB(DLB_MPI_Mprobe_leave);

        return res;
}

/******************************************************************************
 ***  MPI_Improbe
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Improbe) (int source, int tag, MPI_Comm comm, int *flag,
        MPI_Message *message, MPI_Status *status)
{
        int res;

        DLB(DLB_MPI_Improbe_enter, source, tag, comm, flag, message, status);

        if (INSTRUMENT_THIS_MPI)
        {
                DEBUG_INTERFACE(ENTER)
                Backend_Enter_Instrumentation ();
                res = MPI_Improbe_C_Wrapper (source, tag, comm, flag, message, status);
                Backend_Leave_Instrumentation ();
                DEBUG_INTERFACE(LEAVE)
        }
        else
                return PMPI_Improbe (source, tag, comm, flag, message, status);

        DLB(DLB_MPI_Improbe_leave);

        return res;
}

#endif /* MPI3 */

/******************************************************************************
 ***  MPI_Test
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Test) (MPI_Request *request, int *flag, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_Test_enter, request, flag, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Test_C_Wrapper (request, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Test (request, flag, status);

	DLB(DLB_MPI_Test_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_Testall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Testall) (int count, MPI_Request *requests,
	int *flag, MPI_Status *statuses)
{
	int res;

	DLB(DLB_MPI_Testall_enter, count, requests, flag, statuses);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Testall_C_Wrapper (count, requests, flag, statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Testall (count, requests, flag, statuses);

	DLB(DLB_MPI_Testall_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Testany
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Testany) (int count, MPI_Request *requests, int *index,
	int *flag, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_Testany_enter, count, requests, index, flag, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Testany_C_Wrapper (count, requests, index, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Testany (count, requests, index, flag, status);

	DLB(DLB_MPI_Testany_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Testsome
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Testsome) (int incount, MPI_Request * requests,
	int *outcount, int *indices, MPI_Status *statuses)
{
	int res;

	DLB(DLB_MPI_Testsome_enter, incount, requests, outcount, indices, statuses);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Testsome_C_Wrapper (incount, requests, outcount, indices, statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Testsome (incount, requests, outcount, indices, statuses);

	DLB(DLB_MPI_Testsome_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Wait
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Wait) (MPI_Request *request, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_Wait_enter, request, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Wait_C_Wrapper (request, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Wait (request, status);

	DLB(DLB_MPI_Wait_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Waitall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Waitall) (int count, MPI_Request *requests,
	MPI_Status *statuses)
{
	int res;

	DLB(DLB_MPI_Waitall_enter, count, requests, statuses);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Waitall_C_Wrapper (count, requests, statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Waitall (count, requests, statuses);

	DLB(DLB_MPI_Waitall_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Waitany
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Waitany) (int count, MPI_Request *requests, int *index,
	MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_Waitany_enter, count, requests, index, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Waitany_C_Wrapper (count, requests, index, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = (PMPI_Waitany (count, requests, index, status));

	DLB(DLB_MPI_Waitany_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Waitsome
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Waitsome) (int incount, MPI_Request * requests,
	int *outcount, int *indices, MPI_Status *statuses)
{
	int res;

	DLB(DLB_MPI_Waitsome_enter, incount, requests, outcount, indices, statuses);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Waitsome_C_Wrapper (incount,requests, outcount, indices,
			statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Waitsome (incount, requests, outcount, indices, statuses);

	DLB(DLB_MPI_Waitsome_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_Recv_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Recv_init) (void *buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Recv_init_enter, buf, count, datatype, source, tag, comm,
		request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Recv_init_C_Wrapper
		  (buf, count, datatype, source, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res =  PMPI_Recv_init
		  (buf, count, datatype, source, tag, comm, request);

	DLB(DLB_MPI_Recv_init_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_Send_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Send_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Send_init_enter, buf, count, datatype, dest, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Send_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Send_init (buf, count, datatype, dest, tag, comm,
			request);

	DLB(DLB_MPI_Send_init_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Bsend_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Bsend_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Bsend_init_enter, buf, count, datatype, dest, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Bsend_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Bsend_init (buf, count, datatype, dest, tag, comm, request);

	DLB(DLB_MPI_Bsend_init_leave);
	
	return res;
}


/******************************************************************************
 ***  MPI_Rsend_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Rsend_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Rsend_init_enter, buf, count, datatype, dest, tag, comm, request);
	
	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Rsend_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Rsend_init (buf, count, datatype, dest, tag, comm, request);

	DLB(DLB_MPI_Rsend_init_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_Ssend_init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Ssend_init) (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
	int res;

	DLB(DLB_MPI_Ssend_init_enter, buf, count, datatype, dest, tag, comm, request);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Ssend_init_C_Wrapper
		  (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Ssend_init (buf, count, datatype, dest, tag, comm, request);

	DLB(DLB_MPI_Ssend_init_leave);

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

	DLB(DLB_MPI_Sendrecv_enter, MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, dest, sendtag,
		recvbuf, recvcount, recvtype, source, recvtag, comm, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Sendrecv_C_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, dest, sendtag,
		  recvbuf, recvcount, recvtype, source, recvtag, comm, status); 
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Sendrecv (sendbuf, sendcount, sendtype, dest, sendtag,
		  recvbuf, recvcount, recvtype, source, recvtag, comm, status); 

	DLB(DLB_MPI_Sendrecv_leave);

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

	DLB(DLB_MPI_Sendrecv_replace_enter, buf, count, type, dest, sendtag, source,
		recvtag, comm, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_Sendrecv_replace_C_Wrapper (buf, count, type, dest, sendtag,
		  source, recvtag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Sendrecv_replace (buf, count, type, dest, sendtag, source,
		  recvtag, comm, status);

	DLB(DLB_MPI_Sendrecv_replace_leave);

	return res;
}

#endif /* defined(C_SYMBOLS) */

