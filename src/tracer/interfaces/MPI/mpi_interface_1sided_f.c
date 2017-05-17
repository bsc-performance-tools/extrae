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

#if defined(DYNINST_MODULE)

/* MPI_F_SYMS define different Fortran synonymous using the __attribute__ 
	 compiler constructor. Use r3 in the UPPERCASE VERSION of the MPI call. */

#define MPI_F_SYMS(r1,r2,r3,orig,params) \
    void NAME_ROUTINE_F(r1) params __attribute__ ((alias ("patch_p"#orig))); \
    void NAME_ROUTINE_F(r2) params __attribute__ ((alias ("patch_p"#orig))); \
    void NAME_ROUTINE_FU(r3) params __attribute__ ((alias ("patch_p"#orig)));
#else
#define MPI_F_SYMS(r1,r2,r3,orig,params) \
    void r1 params __attribute__ ((alias (#orig))); \
    void r2 params __attribute__ ((alias (#orig))); \
    void r3 params __attribute__ ((alias (#orig)));

#endif
 
#endif

#if defined(FORTRAN_SYMBOLS)

#if MPI_SUPPORTS_MPI_1SIDED

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_create__,mpi_win_create_,MPI_WIN_CREATE,mpi_win_create,(void *base, void *size, MPI_Fint *disp_unit, void *info, MPI_Fint *comm, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_create)(void *base, void *size, MPI_Fint *disp_unit, void *info, MPI_Fint *comm, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_create)(void *base, void *size, MPI_Fint *disp_unit, void *info, MPI_Fint *comm, void *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Win_create_F_enter, base, size, disp_unit, info, comm, win, ierror);
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_create_Fortran_Wrapper (base, size, disp_unit, info, comm, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_create)(base, size, disp_unit, info, comm, win, ierror);
	
	DLB(DLB_MPI_Win_create_F_leave);
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_fence__,mpi_win_fence_,MPI_WIN_FENCE,mpi_win_fence,(MPI_Fint *assert, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_fence)(MPI_Fint *assert, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_fence)(MPI_Fint *assert, void *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Win_fence_F_enter, assert, win, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_fence_Fortran_Wrapper (assert, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_fence)(assert, win, ierror);
		
	DLB(DLB_MPI_Win_fence_F_leave);
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_start__,mpi_win_start_,MPI_WIN_START,mpi_win_start,(void *group, void *assert, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_start)(void *group, void *assert, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_start)(void *group, void *assert, void *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Win_start_F_enter, group, assert, win, ierror);
	
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_start_Fortran_Wrapper (group, assert, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_start)(group, assert, win, ierror);
		
	DLB(DLB_MPI_Win_start_F_leave);
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_free__,mpi_win_free_,MPI_WIN_FREE,mpi_win_free,(void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_free)(void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_free)(void *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Win_free_F_enter, win, ierror);
	
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_free_Fortran_Wrapper (win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_free)(win, ierror);
		
	DLB(DLB_MPI_Win_free_F_leave);

}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_complete__,mpi_win_complete_,MPI_WIN_COMPLETE,mpi_win_complete,(void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_complete)(void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_complete)(void *win, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Win_complete_F_enter, win, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_complete_Fortran_Wrapper (win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_complete)(win, ierror);

	DLB(DLB_MPI_Win_complete_F_leave);

}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_wait__,mpi_win_wait_,MPI_WIN_WAIT,mpi_win_wait,(void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_wait)(void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_wait)(void *win, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Win_wait_F_enter, win, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_wait_Fortran_Wrapper (win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_wait)(win, ierror);

	DLB(DLB_MPI_Win_wait_F_leave);

}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_post__,mpi_win_post_,MPI_WIN_POST,mpi_win_post,(void *group, void *assert, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_post)(void *group, void *assert, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_post)(void *group, void *assert, void *win, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Win_post_F_enter, group, assert, win, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_post_Fortran_Wrapper (group, assert, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_post)(group, assert, win, ierror);

	DLB(DLB_MPI_Win_post_F_leave);

}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_get__,mpi_get_,MPI_GET,mpi_get,(void *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_get)(void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_get)(void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Get_F_enter, origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win,
		ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Get_Fortran_Wrapper (origin_addr, origin_count, origin_datatype,
			target_rank, target_disp, target_count, target_datatype, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_get)(origin_addr, origin_count, origin_datatype, target_rank,
			target_disp, target_count, target_datatype, win, ierror);

	DLB(DLB_MPI_Get_F_leave);

}

#if defined(HAVE_ALIAS_ATTRIBUTE)
/* FIXME: origin_addr is defined as void * in MPI3 and previous versions, it must be reviewed!!! */
MPI_F_SYMS(mpi_put__,mpi_put_,MPI_PUT,mpi_put,(void *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_put)(void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_put)(void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Put_F_enter, origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win,
		ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Put_Fortran_Wrapper (MPI3_VOID_P_CAST origin_addr, origin_count, origin_datatype,
			target_rank, target_disp, target_count, target_datatype, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_put)(origin_addr, origin_count, origin_datatype, target_rank,
			target_disp, target_count, target_datatype, win, ierror);

	DLB(DLB_MPI_Put_F_leave);

}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_lock__,mpi_win_lock_,MPI_WIN_LOCK,mpi_win_lock,(MPI_Fint *lock_type, MPI_Fint *rank, MPI_Fint *assert, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_lock)(MPI_Fint *lock_type, MPI_Fint *rank, MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_lock)(MPI_Fint *lock_type,MPI_Fint *rank, MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Win_lock_F_enter, lock_type, rank, assert, win, ierror);
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_lock_Fortran_Wrapper (lock_type, rank, assert, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_lock)(lock_type,rank, assert, win, ierror);
	DLB(DLB_MPI_Win_lock_F_leave);
}


#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_unlock__,mpi_win_unlock_,MPI_WIN_UNLOCK,mpi_win_unlock,(MPI_Fint *rank, MPI_Fint *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_unlock)(MPI_Fint *rank, MPI_Fint *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_unlock)(MPI_Fint *rank, MPI_Fint *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Win_unlock_F_enter, rank, win, ierror);
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Win_unlock_Fortran_Wrapper (rank, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_win_unlock)(rank, win, ierror);
	DLB(DLB_MPI_Win_unlock_F_leave);
}


#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_get_accumulate__,mpi_get_accumulate_,MPI_GET_ACCUMULATE,mpi_get_accumulate,(void *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, void *result_addr, MPI_Fint *result_count, MPI_Fint *result_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_get_accumulate)(void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, void *result_addr, MPI_Fint *result_count, MPI_Fint *result_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_get_accumulate)(void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, void *result_addr, MPI_Fint *result_count, MPI_Fint *result_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Get_accumulate_F_enter, origin_addr, origin_count, origin_datatype,
                result_addr, result_count, result_datatype, target_rank, target_disp, 
                target_count, target_datatype, op, win, ierror);
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Get_accumulate_Fortran_Wrapper (origin_addr, origin_count, origin_datatype,
                        result_addr, result_count, result_datatype, target_rank, target_disp,
 			target_count, target_datatype, op, win, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_get_accumulate)(origin_addr, origin_count, origin_datatype,
			result_addr, result_count, result_datatype, target_rank,
			target_disp, target_count, target_datatype, op, win, ierror);
	DLB(DLB_MPI_Get_accumulate_F_leave);
}


#endif /* MPI_SUPPORTS_MPI_1SIDED */


#endif /* defined(FORTRAN_SYMBOLS) */
