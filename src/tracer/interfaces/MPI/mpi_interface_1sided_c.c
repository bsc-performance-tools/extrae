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

#if MPI_SUPPORTS_MPI_1SIDED

int MPI_Win_create (void *base, MPI_Aint size, int disp_unit, MPI_Info info,
	MPI_Comm comm, MPI_Win *win)
{
	int res;

	DLB(DLB_MPI_Win_create_enter, base, size, disp_unit, info, comm, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_create_C_Wrapper (base, size, disp_unit, info, comm, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_create (base, size, disp_unit, info, comm, win);

	DLB(DLB_MPI_Win_create_leave);

	return res;
}

int MPI_Win_fence (int assert, MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Win_fence_enter, assert, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_fence_C_Wrapper (assert, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_fence (assert, win);

	DLB(DLB_MPI_Win_fence_leave);

	return res;
}

int MPI_Win_start (MPI_Group group, int assert, MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Win_start_enter, group, assert, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_start_C_Wrapper (group, assert, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_start (group, assert, win);

	DLB(DLB_MPI_Win_start_leave);

	return res;
}

int MPI_Win_free (MPI_Win *win)
{
	int res;

	DLB(DLB_MPI_Win_free_enter, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_free_C_Wrapper (win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_free (win);

	DLB(DLB_MPI_Win_free_leave);

	return res;
}

int MPI_Win_complete (MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Win_complete_enter, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_complete_C_Wrapper (win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_complete (win);

	DLB(DLB_MPI_Win_complete_leave);

	return res;
}

int MPI_Win_wait (MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Win_wait_enter, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_wait_C_Wrapper (win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_wait (win);

	DLB(DLB_MPI_Win_wait_leave);

	return res;
}

int MPI_Win_post (MPI_Group group, int assert, MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Win_post_enter, group, assert, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_post_C_Wrapper (group, assert, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_post (group, assert, win);

	DLB(DLB_MPI_Win_post_leave);

	return res;
}

int MPI_Get (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	int target_rank, MPI_Aint target_disp, int target_count,
	MPI_Datatype target_datatype, MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Get_enter, origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Get_C_Wrapper (origin_addr, origin_count, origin_datatype,
			target_rank, target_disp, target_count, target_datatype, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Get (origin_addr, origin_count, origin_datatype,
			target_rank, target_disp, target_count, target_datatype,
			win);

	DLB(DLB_MPI_Get_leave);

	return res;
}

int MPI_Put (MPI3_CONST void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	int target_rank, MPI_Aint target_disp, int target_count,
	MPI_Datatype target_datatype, MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Put_enter, origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Put_C_Wrapper (MPI3_VOID_P_CAST origin_addr, origin_count, origin_datatype,
			target_rank, target_disp, target_count, target_datatype, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Put (origin_addr, origin_count, origin_datatype,
			target_rank, target_disp, target_count, target_datatype,
			win);

	DLB(DLB_MPI_Put_leave);
	
	return res;
}

int MPI_Win_lock (int lock_type, int rank, int assert, MPI_Win win)
{

	int res;

	DLB(DLB_MPI_Win_lock_enter, lock_type, rank, assert, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_lock_C_Wrapper (lock_type, rank, assert, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_lock (lock_type, rank, assert, win);

	DLB(DLB_MPI_Win_lock_leave);
	return res;

}


int MPI_Win_unlock (int rank, MPI_Win win)
{

	int res;

	DLB(DLB_MPI_Win_unlock_enter, rank, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Win_unlock_C_Wrapper (rank, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Win_unlock (rank, win);

	DLB(DLB_MPI_Win_unlock_leave);
	return res;

}


int MPI_Get_accumulate (MPI3_CONST void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
        void *result_addr, int result_count, MPI_Datatype result_datatype,
	int target_rank, MPI_Aint target_disp, int target_count,
	MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
{
	int res;

	DLB(DLB_MPI_Get_accumulate_enter, MPI3_VOID_P_CAST origin_addr, origin_count, origin_datatype, result_addr,
		result_count, result_datatype, target_rank, target_disp, target_count,
                target_datatype, op, win);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Get_accumulate_C_Wrapper (MPI3_VOID_P_CAST origin_addr, origin_count, origin_datatype,
			result_addr, result_count, result_datatype,
			target_rank, target_disp, target_count, target_datatype, op, win);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Get_accumulate (origin_addr, origin_count, origin_datatype,
			result_addr, result_count, result_datatype,
			target_rank, target_disp, target_count, target_datatype, op, win);

	DLB(DLB_MPI_Get_accumulate_leave);

	return res;
}


#endif /* MPI_SUPPORTS_MPI_1SIDED */

#endif /* C_SYMBOLS */
