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
# define NAME_ROUTINE_F(x) patch_p##x  /* mpi_send is converted to patch_pmpi_send */
# define NAME_ROUTINE_FU(x) patch_P##x  /* mpi_send is converted to patch_Pmpi_send */
# define NAME_ROUTINE_C2F(x) CtoF77(patch_p##x)  /* mpi_send may be converted to patch_pmpi_send_ */
#else
# define NAME_ROUTINE_C(x) x
# define NAME_ROUTINE_F(x) x
# define NAME_ROUTINE_C2F(x) CtoF77(x)
#endif

/*
  MPICH 1.2.6/7 (not 1.2.7p1) contains a silly bug where
  MPI_Comm_create/split/dup also invoke MPI_Allreduce directly (not
  PMPI_Allreduce) and gets instrumentend when it shouldn't. The following code
  is to circumvent the problem
*/
#if defined(MPI_VERSION) && defined(MPI_SUBVERSION) && defined(MPICH_NAME)
# if MPI_VERSION == 1 && MPI_SUBVERSION == 2 && MPICH_NAME == 1
#  define MPICH_1_2_Comm_Allreduce_bugfix /* we can control the subsubversion */
static int Extrae_MPICH12_COMM_inside = FALSE;
# endif
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

/* Some C libraries do not contain the mpi_init symbol (fortran)
	 When compiling the combined (C+Fortran) dyninst module, the resulting
	 module CANNOT be loaded if mpi_init is not found. The top #if def..
	 is a workaround for this situation

   NOTE: Some C libraries (mpich 1.2.x) use the C initialization and do not
   offer mpi_init (fortran).
*/

#if defined(FORTRAN_SYMBOLS)

/*
HSG: I think that MPI_C_CONTAINS_FORTRAN_MPI_INIT is not the proper check to do here
#if (defined(COMBINED_SYMBOLS) && !defined(MPI_C_CONTAINS_FORTRAN_MPI_INIT) || \
     !defined(COMBINED_SYMBOLS))
*/

/******************************************************************************
 ***  MPI_Init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_init__,mpi_init_,MPI_INIT,mpi_init,(MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_init) (MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_init) (MPI_Fint *ierror)
#endif
{
	DLB(DLB_MPI_Init_F_enter, ierror);

	/* En qualsevol cas, cal cridar al Wrapper que inicialitzara tot el que cal */
	DEBUG_INTERFACE(ENTER)
	PMPI_Init_Wrapper (ierror);
	DEBUG_INTERFACE(LEAVE)

	DLB(DLB_MPI_Init_F_leave);
}

#if defined(MPI_HAS_INIT_THREAD_F)
/******************************************************************************
 ***  MPI_Init_thread
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_init_thread__,mpi_init_thread_,MPI_INIT_THREAD,mpi_init_thread,(MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_init_thread) (MPI_Fint *required, MPI_Fint *provided,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_init_thread) (MPI_Fint *required, MPI_Fint *provided,
	MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Init_thread_F_enter, required, provided, ierror);

	/* En qualsevol cas, cal cridar al Wrapper que inicialitzara tot el que cal */
	DEBUG_INTERFACE(ENTER)
	PMPI_Init_thread_Wrapper (required, provided, ierror);
	DEBUG_INTERFACE(LEAVE)


	DLB(DLB_MPI_Init_thread_F_leave);

}
#endif /* MPI_HAS_INIT_THREAD_F */

/* 
//#endif
     (defined(COMBINED_SYMBOLS) && !defined(MPI_C_CONTAINS_FORTRAN_MPI_INIT) || \
     !defined(COMBINED_SYMBOLS))
     */

/******************************************************************************
 ***  MPI_Finalize
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_finalize__,mpi_finalize_,MPI_FINALIZE,mpi_finalize, (MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_finalize) (MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_finalize) (MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Finalize_F_enter, ierror);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Finalize_Wrapper (ierror);
		DEBUG_INTERFACE(LEAVE)
	}
	else if (!mpitrace_on && Extrae_getCheckControlFile())
	{
		/* This case happens when the tracing isn't activated due to the inexistance
			of the control file. Just remove the temporal files! */
		DEBUG_INTERFACE(ENTER)
		remove_temporal_files();
		MPI_remove_file_list (FALSE);
		DEBUG_INTERFACE(LEAVE)
		CtoF77 (pmpi_finalize) (ierror);
	}
	else
		CtoF77 (pmpi_finalize) (ierror);


	DLB(DLB_MPI_Finalize_F_leave);

}

/******************************************************************************
 ***  MPI_Request_get_status
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_request_get_status__,mpi_request_get_status_,MPI_REQUEST_GET_STATUS,mpi_request_get_status,(MPI_Fint *request, int *flag, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_request_get_status) (MPI_Fint *request, int *flag, MPI_Fint *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_request_get_status) (MPI_Fint *request, int *flag, MPI_Fint *status, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Request_get_status_F_enter, request, flag, status, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (4+Caller_Count[CALLER_MPI]);
		PMPI_Request_get_status_Wrapper (request, flag, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_request_get_status) (request, flag, status, ierror);

	DLB(DLB_MPI_Request_get_status_F_leave);

}

/******************************************************************************
 ***  MPI_Cancel
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_cancel__,mpi_cancel_,MPI_CANCEL,mpi_cancel,(MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_cancel) (MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_cancel) (MPI_Fint *request, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Cancel_F_enter, request, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Cancel_Wrapper (request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_cancel) (request, ierror);

	DLB(DLB_MPI_Cancel_F_leave);

}

/******************************************************************************
 ***  MPI_Comm_Rank
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_rank__,mpi_comm_rank_,MPI_COMM_RANK,mpi_comm_rank,(MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_comm_rank) (MPI_Fint *comm, MPI_Fint *rank,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_rank) (MPI_Fint *comm, MPI_Fint *rank,
	MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Comm_rank_F_enter, comm, rank, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Comm_Rank_Wrapper (comm, rank, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_comm_rank) (comm, rank, ierror);

	DLB(DLB_MPI_Comm_rank_F_leave);
}

/******************************************************************************
 ***  MPI_Comm_Size
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_size__,mpi_comm_size_,MPI_COMM_SIZE,mpi_comm_size,(MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_comm_size) (MPI_Fint *comm, MPI_Fint *size,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_size) (MPI_Fint *comm, MPI_Fint *size,
	MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Comm_size_F_enter, comm, size, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Comm_Size_Wrapper (comm, size, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_comm_size) (comm, size, ierror);

	DLB(DLB_MPI_Comm_size_F_leave);

}

/******************************************************************************
 ***  MPI_Comm_Create
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_create__,mpi_comm_create_,MPI_COMM_CREATE,mpi_comm_create,(MPI_Fint *comm, MPI_Fint *group, MPI_Fint *newcomm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_comm_create) (MPI_Fint *comm, MPI_Fint *group,
	MPI_Fint *newcomm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_create) (MPI_Fint *comm, MPI_Fint *group,
	MPI_Fint *newcomm, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Comm_create_F_enter, comm, group, newcomm, ierror);

	if (mpitrace_on)
	{
#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = TRUE;
#endif

		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Comm_Create_Wrapper (comm, group, newcomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)

#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = FALSE;
#endif
	}
	else
		CtoF77 (pmpi_comm_create) (comm, group, newcomm, ierror);

	DLB(DLB_MPI_Comm_create_F_leave);

}

/******************************************************************************
 ***  MPI_Comm_Free
 ***  NOTE We cannot let freeing communicators
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_free__,mpi_comm_free_,MPI_COMM_FREE,mpi_comm_free,(MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_comm_free) (MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_free) (MPI_Fint *comm, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Comm_free_F_enter, comm, ierror);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Comm_Free_Wrapper (comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)

	}
	else
		*ierror = MPI_SUCCESS;

	DLB(DLB_MPI_Comm_free_F_leave);

}

/******************************************************************************
 ***  MPI_Comm_Dup
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_dup__,mpi_comm_dup_,MPI_COMM_DUP,mpi_comm_dup,(MPI_Fint *comm, MPI_Fint *newcomm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_comm_dup) (MPI_Fint *comm, MPI_Fint *newcomm,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_dup) (MPI_Fint *comm, MPI_Fint *newcomm,
	MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Comm_dup_F_enter, comm, newcomm, ierror);

	if (mpitrace_on)
	{
#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = TRUE;
#endif

		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Comm_Dup_Wrapper (comm, newcomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)

#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = FALSE;
#endif
	}
	else
		CtoF77 (pmpi_comm_dup) (comm, newcomm, ierror);

	DLB(DLB_MPI_Comm_dup_F_leave);

}


/******************************************************************************
 ***  MPI_Comm_Split
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_split__,mpi_comm_split_,MPI_COMM_SPLIT,mpi_comm_split,(MPI_Fint *comm, MPI_Fint *color, MPI_Fint *key, MPI_Fint *newcomm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_comm_split) (MPI_Fint *comm, MPI_Fint *color,
	MPI_Fint *key, MPI_Fint *newcomm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_split) (MPI_Fint *comm, MPI_Fint *color,
	MPI_Fint *key, MPI_Fint *newcomm, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Comm_split_F_enter, comm, color, key, newcomm, ierror);

	if (mpitrace_on)
	{
#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = TRUE;
#endif

		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Comm_Split_Wrapper (comm, color, key, newcomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)

#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = FALSE;
#endif
	}
	else
		CtoF77 (pmpi_comm_split) (comm, color, key, newcomm, ierror);

	DLB(DLB_MPI_Comm_split_F_leave);

}


#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/******************************************************************************
 ***  MPI_Comm_spawn
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_spawn__,mpi_comm_spawn_,MPI_COMM_SPAWN,mpi_comm_spawn,(char *command, char *argv, MPI_Fint *maxprocs, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror ))

void NAME_ROUTINE_F(mpi_comm_spawn) (char *command, char *argv, MPI_Fint *maxprocs, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_spawn) (char *command, char *argv, MPI_Fint *maxprocs, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Comm_spawn_F_enter, command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (5 + (*maxprocs) + Caller_Count[CALLER_MPI]);
		PMPI_Comm_Spawn_Wrapper (MPI3_CHAR_P_CAST command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_comm_spawn) (command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierror);

	DLB(DLB_MPI_Comm_spawn_F_leave);

}
#endif


#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/******************************************************************************
 ***  MPI_Comm_spawn_multiple
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_comm_spawn_multiple__,mpi_comm_spawn_multiple_,MPI_COMM_SPAWN_MULTIPLE,mpi_comm_spawn_multiple,( MPI_Fint *count, char *array_of_commands, char *array_of_argv, MPI_Fint *array_of_maxprocs, MPI_Fint *array_of_info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror ))

void NAME_ROUTINE_F(mpi_comm_spawn_multiple)   (MPI_Fint *count, char *array_of_commands, char *array_of_argv, MPI_Fint *array_of_maxprocs, MPI_Fint *array_of_info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_comm_spawn_multiple) (MPI_Fint *count, char *array_of_commands, char *array_of_argv, MPI_Fint *array_of_maxprocs, MPI_Fint *array_of_info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)
#endif
{
	int i, n_events = 0;


	DLB(DLB_MPI_Comm_spawn_multiple_F_enter, count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		for (i=0; i<(*count); i++) 
		{
			n_events += 5 + array_of_maxprocs[i] + Caller_Count[CALLER_MPI];
		}
		Backend_Enter_Instrumentation (n_events);
		PMPI_Comm_Spawn_Multiple_Wrapper (count, array_of_commands, array_of_argv, MPI3_F_INT_P_CAST array_of_maxprocs, MPI3_F_INT_P_CAST array_of_info, root, comm, intercomm, array_of_errcodes, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_comm_spawn_multiple) (count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes, ierror);

	DLB(DLB_MPI_Comm_spawn_multiple_F_leave);

}
#endif


/******************************************************************************
 *** MPI_Cart_create
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_cart_create__,mpi_cart_create_,MPI_CART_CREATE,mpi_cart_create, (MPI_Fint *comm_old, MPI_Fint *ndims, MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_cart_create) (MPI_Fint *comm_old, MPI_Fint *ndims,
	MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_cart_create) (MPI_Fint *comm_old, MPI_Fint *ndims,
	MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart,
	MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Cart_create_F_enter, comm_old, ndims, dims, periods, reorder,
		comm_cart, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Cart_create_Wrapper (comm_old, ndims, MPI3_F_INT_P_CAST dims, MPI3_F_INT_P_CAST periods, reorder,
                              comm_cart, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_cart_create) (comm_old, ndims, dims, periods,
			reorder, comm_cart, ierror);

	DLB(DLB_MPI_Cart_create_F_leave);

}

/******************************************************************************
 *** MPI_Cart_sub
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_cart_sub__,mpi_cart_sub_,MPI_CART_SUB,mpi_cart_sub, (MPI_Fint *comm, MPI_Fint *remain_dims, MPI_Fint *comm_new, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_cart_sub) (MPI_Fint *comm, MPI_Fint *remain_dims,
	MPI_Fint *comm_new, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_cart_sub) (MPI_Fint *comm, MPI_Fint *remain_dims,
	MPI_Fint *comm_new, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Cart_sub_F_enter, comm, remain_dims, comm_new, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Cart_sub_Wrapper (comm, MPI3_F_INT_P_CAST remain_dims, comm_new, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_cart_sub) (comm, remain_dims, comm_new, ierror);

	DLB(DLB_MPI_Cart_sub_F_leave);

}


/******************************************************************************
 *** MPI_Intercomm_create
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_intercomm_create__,mpi_intercomm_create_,MPI_INTERCOMM_CREATE,mpi_intercomm_create, (MPI_Fint * local_comm, MPI_Fint *local_leader, MPI_Fint *peer_comm, MPI_Fint *remote_leader, MPI_Fint *tag, MPI_Fint *new_intercomm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_intercomm_create) (MPI_Fint * local_comm,
	MPI_Fint *local_leader, MPI_Fint *peer_comm, MPI_Fint *remote_leader,
	MPI_Fint *tag, MPI_Fint *new_intercomm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_intercomm_create) (MPI_Fint *local_comm,
	MPI_Fint *local_leader, MPI_Fint *peer_comm, MPI_Fint *remote_leader,
	MPI_Fint *tag, MPI_Fint *new_intercomm, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Intercomm_create_F_enter, local_comm, local_leader, peer_comm,
	  remote_leader, tag, new_intercomm, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Intercomm_create_F_Wrapper (local_comm, local_leader, peer_comm,
		  remote_leader, tag, new_intercomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_intercomm_create) (local_comm, local_leader, peer_comm, 
		  remote_leader, tag, new_intercomm, ierror);

	DLB(DLB_MPI_Intercomm_create_F_leave);

}

/******************************************************************************
 *** MPI_Intercomm_merge
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_intercomm_merge__,mpi_intercomm_merge_,MPI_INTERCOMM_MERGE,mpi_intercomm_merge, (MPI_Fint *intercomm, MPI_Fint *high, MPI_Fint *newintracomm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_intercomm_merge) (MPI_Fint *intercomm, MPI_Fint *high,
	MPI_Fint *newintracomm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_intercomm_merge) (MPI_Fint *intercomm, MPI_Fint *high,
	MPI_Fint *newintracomm, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Intercomm_merge_F_enter, intercomm, high, newintracomm, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		PMPI_Intercomm_merge_F_Wrapper (intercomm, high, newintracomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (mpi_intercomm_merge) (intercomm, high, newintracomm, ierror);

	DLB(DLB_MPI_Intercomm_merge_F_leave);

}


/******************************************************************************
 ***  MPI_Start
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_start__,mpi_start_,MPI_START,mpi_start, (MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_start) (MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_start) (MPI_Fint *request, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Start_F_enter, request, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		PMPI_Start_Wrapper (request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_start) (request, ierror);

	DLB(DLB_MPI_Start_F_leave);

}

/******************************************************************************
 ***  MPI_Startall
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_startall__,mpi_startall_,MPI_STARTALL,mpi_startall, (MPI_Fint *count, MPI_Fint array_of_requests[], MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_startall) (MPI_Fint *count,
	MPI_Fint array_of_requests[], MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_startall) (MPI_Fint *count,
	MPI_Fint array_of_requests[], MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Startall_F_enter, count, array_of_requests, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*count+Caller_Count[CALLER_MPI]);
		PMPI_Startall_Wrapper (count, array_of_requests, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_startall) (count, array_of_requests, ierror);

	DLB(DLB_MPI_Startall_F_leave);

}

/******************************************************************************
 ***  MPI_Request_free
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_request_free__,mpi_request_free_,MPI_REQUEST_FREE,mpi_request_free, (MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_request_free) (MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_request_free) (MPI_Fint *request, MPI_Fint *ierror)
#endif
{

	DLB(DLB_MPI_Request_free_F_enter, request, ierror);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Request_free_Wrapper (request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_request_free) (request, ierror);

	DLB(DLB_MPI_Request_free_F_leave);

}

#endif /* defined(FORTRAN_SYMBOLS) */

#if defined(C_SYMBOLS)

/******************************************************************************
 ***  MPI_Init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Init) (int *argc, char ***argv)
{
	int res;


	DLB(DLB_MPI_Init_enter, argc, argv);


	/* This should be called always, whenever the tracing takes place or not */
	DEBUG_INTERFACE(ENTER)
	res = MPI_Init_C_Wrapper (argc, argv);
	DEBUG_INTERFACE(LEAVE)


	DLB(DLB_MPI_Init_leave);


	return res;
}

#if defined(MPI_HAS_INIT_THREAD_C)
/******************************************************************************
 ***  MPI_Init_thread
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Init_thread) (int *argc, char ***argv, int required, int *provided)
{
	int res;


	DLB(DLB_MPI_Init_thread_enter, argc, argv, required, provided);


	/* This should be called always, whenever the tracing takes place or not */
	DEBUG_INTERFACE(ENTER)
	res = MPI_Init_thread_C_Wrapper (argc, argv, required, provided);
	DEBUG_INTERFACE(LEAVE)


	DLB(DLB_MPI_Init_thread_leave);


	return res;
}
#endif /* MPI_HAS_INIT_THREAD_C */

/******************************************************************************
 ***  MPI_Finalize
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Finalize) (void)
{
	int res;


	DLB(DLB_MPI_Finalize_enter);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Finalize_C_Wrapper ();
		DEBUG_INTERFACE(LEAVE)
	}
	else if (!mpitrace_on && Extrae_getCheckControlFile())
	{
		/* This case happens when the tracing isn't activated due to the inexistance
			of the control file. Just remove the temporal files! */
		DEBUG_INTERFACE(ENTER)
		remove_temporal_files();
		MPI_remove_file_list (FALSE);
		DEBUG_INTERFACE(LEAVE)
		res = PMPI_Finalize ();
	}
	else
		res = PMPI_Finalize ();


	DLB(DLB_MPI_Finalize_leave);


	return res;
}

/******************************************************************************
 *** MPI_Request_get_status 
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Request_get_status) (MPI_Request request, int *flag,
	MPI_Status *status)
{
	int res;


	DLB(DLB_MPI_Request_get_status_enter, request, flag, status);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (4+Caller_Count[CALLER_MPI]);
		res = MPI_Request_get_status_C_Wrapper (request, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Request_get_status(request, flag, status);


	DLB(DLB_MPI_Request_get_status_leave);


	return res;
}

/******************************************************************************
 ***  MPI_Cancel
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Cancel) (MPI_Request *request)
{
	int res;


	DLB(DLB_MPI_Cancel_enter, request);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Cancel_C_Wrapper (request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Cancel (request);


	DLB(DLB_MPI_Cancel_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Comm_rank
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_rank) (MPI_Comm comm, int *rank)
{
	int res;


	DLB(DLB_MPI_Comm_rank_enter, comm, rank);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Comm_rank_C_Wrapper (comm, rank);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Comm_rank (comm, rank);


	DLB(DLB_MPI_Comm_rank_leave);


	return res;
}

/******************************************************************************
 ***  MPI_Comm_size
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_size) (MPI_Comm comm, int *size)
{
	int res;


	DLB(DLB_MPI_Comm_size_enter, comm, size);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Comm_size_C_Wrapper (comm, size);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Comm_size (comm, size);


	DLB(DLB_MPI_Comm_size_leave);


	return res;
}

/******************************************************************************
 ***  MPI_Comm_create
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_create) (MPI_Comm comm, MPI_Group group,
	MPI_Comm *newcomm)
{
	int res;


	DLB(DLB_MPI_Comm_create_enter, comm, group, newcomm);


	if (mpitrace_on)
	{
#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = TRUE;
#endif

		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = MPI_Comm_create_C_Wrapper (comm, group, newcomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)

#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = FALSE;
#endif
	}
	else
    		res = PMPI_Comm_create (comm, group, newcomm);


	DLB(DLB_MPI_Comm_create_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Comm_free
 ***  NOTE we cannot let freeing communicators!
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_free) (MPI_Comm *comm)
{
	int res;

	DLB(DLB_MPI_Comm_free_enter, comm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = MPI_Comm_free_C_Wrapper (comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
    		res = MPI_SUCCESS;


	DLB(DLB_MPI_Comm_free_leave);


	return res;
}

/******************************************************************************
 ***  MPI_Comm_dup
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_dup) (MPI_Comm comm, MPI_Comm *newcomm)
{
	int res;


	DLB(DLB_MPI_Comm_dup_enter, comm, newcomm);

	if (mpitrace_on)
	{
#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = TRUE;
#endif

		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = MPI_Comm_dup_C_Wrapper (comm, newcomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)

#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = FALSE;
#endif
	}
	else
    		res = PMPI_Comm_dup (comm, newcomm);

	DLB(DLB_MPI_Comm_dup_leave);


	return res;
}

/******************************************************************************
 ***  MPI_Comm_split
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_split) (MPI_Comm comm, int color, int key,
	MPI_Comm *newcomm)
{
	int res;


	DLB(DLB_MPI_Comm_split_enter, comm, color, key, newcomm);


	if (mpitrace_on)
	{
#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = TRUE;
#endif

		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = MPI_Comm_split_C_Wrapper (comm, color, key, newcomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)

#ifdef MPICH_1_2_Comm_Allreduce_bugfix
		Extrae_MPICH12_COMM_inside = FALSE;
#endif
	}
	else
		res = PMPI_Comm_split (comm, color, key, newcomm);


	DLB(DLB_MPI_Comm_split_leave);


	return res;
}

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/******************************************************************************
 ***  MPI_Comm_spawn
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_spawn) (
  MPI3_CONST char *command,
  char           **argv,
  int              maxprocs,
  MPI_Info         info,
  int              root,
  MPI_Comm         comm,
  MPI_Comm        *intercomm,
  int             *array_of_errcodes)
{
	int res;


	DLB(DLB_MPI_Comm_spawn_enter, command, argv, maxprocs, info, root, comm,
		intercomm, array_of_errcodes);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (5 + maxprocs + Caller_Count[CALLER_MPI]);
		res = MPI_Comm_spawn_C_Wrapper (MPI3_CHAR_P_CAST command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
	{
		res = PMPI_Comm_spawn (command, argv, maxprocs, info, root,
			comm, intercomm, array_of_errcodes);
	}


	DLB(DLB_MPI_Comm_spawn_leave);


	return res;
}
#endif


#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/******************************************************************************
 ***  MPI_Comm_spawn_multiple
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_spawn_multiple) (
  int                 count,
  char               *array_of_commands[],
  char              **array_of_argv[],
  MPI3_CONST int      array_of_maxprocs[],
  MPI3_CONST MPI_Info array_of_info[],
  int                 root,
  MPI_Comm            comm,
  MPI_Comm           *intercomm,
  int                 array_of_errcodes[])
{
	int i, n_events = 0, res;


	DLB(DLB_MPI_Comm_spawn_multiple_enter, count, array_of_commands,
		array_of_argv, MPI3_C_INT_P_CAST array_of_maxprocs,
		MPI3_MPI_INFO_P_CAST array_of_info, root, comm, intercomm,
		array_of_errcodes);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		for (i=0; i<count; i++)
		{
			n_events += 5 + array_of_maxprocs[i] + Caller_Count[CALLER_MPI];
		}
		Backend_Enter_Instrumentation (n_events);
		res = MPI_Comm_spawn_multiple_C_Wrapper (count, array_of_commands, array_of_argv, MPI3_C_INT_P_CAST array_of_maxprocs, MPI3_MPI_INFO_P_CAST array_of_info, root, comm, intercomm, array_of_errcodes);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
	{
		res = PMPI_Comm_spawn_multiple (count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes);
	}


	DLB(DLB_MPI_Comm_spawn_multiple_leave);


	return res;
}
#endif

/******************************************************************************
 *** MPI_Cart_create
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Cart_create) (MPI_Comm comm_old, int ndims, MPI3_CONST int *dims,
	MPI3_CONST int *periods, int reorder, MPI_Comm *comm_cart)
{
	int res;


	DLB(DLB_MPI_Cart_create_enter, comm_old, ndims, MPI3_C_INT_P_CAST dims,
		MPI3_C_INT_P_CAST periods, reorder, comm_cart);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = MPI_Cart_create_C_Wrapper (comm_old, ndims, MPI3_C_INT_P_CAST dims, MPI3_C_INT_P_CAST periods, reorder,
                                      comm_cart);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Cart_create (comm_old, ndims, dims, periods, reorder,
                             comm_cart);

	DLB(DLB_MPI_Cart_create_leave);

	return res;
}

/******************************************************************************
 *** MPI_Cart_sub
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Cart_sub) (MPI_Comm comm, MPI3_CONST int *remain_dims,
	MPI_Comm *comm_new)
{
	int res;


	DLB(DLB_MPI_Cart_sub_enter, comm, remain_dims, comm_new);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res =  MPI_Cart_sub_C_Wrapper (comm, MPI3_C_INT_P_CAST remain_dims, comm_new);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Cart_sub (comm, remain_dims, comm_new);


	DLB(DLB_MPI_Cart_sub_leave);

	return res;
}

/******************************************************************************
 *** MPI_Intercom_create
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Intercomm_create) (MPI_Comm local_comm, int local_leader,
	MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm)
{
	int res;


	DLB(DLB_MPI_Intercomm_create_enter, local_comm, local_leader, peer_comm,
	  remote_leader, tag, newintercomm);

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = MPI_Intercomm_create_C_Wrapper (local_comm, local_leader, peer_comm,
		  remote_leader, tag, newintercomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res =  PMPI_Intercomm_create (local_comm, local_leader, peer_comm,
		  remote_leader, tag, newintercomm);


	DLB(DLB_MPI_Intercomm_create_leave);

	return res;
}

/******************************************************************************
 *** MPI_Intercom_merge
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Intercomm_merge) (MPI_Comm intercomm, int high,
	MPI_Comm *newintracomm)
{
	int res;


	DLB(DLB_MPI_Intercomm_merge_enter, intercomm, high, newintracomm);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = MPI_Intercomm_merge_C_Wrapper (intercomm, high, newintracomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res =  PMPI_Intercomm_merge (intercomm, high, newintracomm);


	DLB(DLB_MPI_Intercomm_merge_leave);

	return res;
}

/******************************************************************************
 ***  MPI_Start
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Start) (MPI_Request *request)
{
	int res;


	DLB(DLB_MPI_Start_enter, request);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		res =  MPI_Start_C_Wrapper (request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Start (request);


	DLB(DLB_MPI_Start_leave);


	return res;
}

/******************************************************************************
 ***  MPI_Startall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Startall) (int count, MPI_Request *requests)
{
	int res;


	DLB(DLB_MPI_Startall_enter, count, requests);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+count+Caller_Count[CALLER_MPI]);
		res = MPI_Startall_C_Wrapper (count, requests);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Startall (count, requests);


	DLB(DLB_MPI_Startall_leave);


	return res;
}

/******************************************************************************
 ***  MPI_Request_free
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Request_free) (MPI_Request *request)
{
	int res;


	DLB(DLB_MPI_Request_free_enter, request);


	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_Request_free_C_Wrapper (request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Request_free (request);


	DLB(DLB_MPI_Request_free_leave);


	return res;
}

#endif /* defined(C_SYMBOLS) */

/**************************************************************************
 **
 ** Interfaces to gather network routes and counters!
 **
 **************************************************************************/

#include "misc_interface.h"

#if defined(C_SYMBOLS)

# if defined(HAVE_ALIAS_ATTRIBUTE)

INTERFACE_ALIASES_C(_network_counters, Extrae_network_counters,(void),void)
void Extrae_network_counters (void)
{
	if (mpitrace_on)
		Extrae_network_counters_Wrapper ();
}

INTERFACE_ALIASES_C(_network_routes, Extrae_network_routes,(int mpi_rank),void)
void Extrae_network_routes (int mpi_rank)
{
	if (mpitrace_on)
		Extrae_network_routes_Wrapper (mpi_rank);
}

INTERFACE_ALIASES_C(_set_tracing_tasks, Extrae_set_tracing_tasks,(unsigned from, unsigned to),void)
void Extrae_set_tracing_tasks (unsigned from, unsigned to)
{
	if (mpitrace_on)
		Extrae_tracing_tasks_Wrapper (from, to);
}

# else /* HAVE_ALIAS_ATTRIBUTE */

/*** FORTRAN BINDINGS + non alias routine duplication ****/
 
# define apiTRACE_NETWORK_ROUTES(x) \
    void x##_network_routes (int mpi_rank) \
   { \
    if (mpitrace_on) \
        Extrae_network_routes_Wrapper (mpi_rank); \
   }
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NETWORK_ROUTES);

#define apiTRACE_SETTRACINGTASKS(x) \
	void x##_set_tracing_tasks (unsigned from, unsigned to) \
   { \
   	if (mpitrace_on) \
      	Extrae_tracing_tasks_Wrapper (from, to); \
   }
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SETTRACINGTASKS);

# endif /* HAVE_ALIAS_ATTRIBUTE */

#endif /* defined(C_SYMBOLS) */


#if defined(FORTRAN_SYMBOLS)

# if defined(HAVE_ALIAS_ATTRIBUTE)

INTERFACE_ALIASES_F(_network_counters,_NETWORK_COUNTERS,extrae_network_counters,(void),void)
void extrae_network_counters (void)
{
	if (mpitrace_on)
		Extrae_network_counters_Wrapper ();
}

INTERFACE_ALIASES_F(_network_routes,_NETWORK_ROUTES,extrae_network_routes,(int *mpi_rank),void)
void extrae_network_routes (int *mpi_rank)
{
	if (mpitrace_on)
		Extrae_network_routes_Wrapper (*mpi_rank);
}

INTERFACE_ALIASES_F(_set_tracing_tasks,_SET_TRACING_TASKS,extrae_set_tracing_tasks,(unsigned *from, unsigned *to),void)
void extrae_set_tracing_tasks (unsigned *from, unsigned *to)
{
	if (mpitrace_on)
		Extrae_tracing_tasks_Wrapper (*from, *to);
}

# else /* HAVE_ALIAS_ATTRIBUTE */

#  define apifTRACE_NETWORK_COUNTERS(x) \
	void CtoF77(x##_network_counters) () \
	{ \
		if (mpitrace_on) \
			Extrae_network_counters_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NETWORK_COUNTERS);

#  define apifTRACE_NETWORK_ROUTES(x) \
	void CtoF77(x##_network_routes) (int *mpi_rank) \
	{ \
		if (mpitrace_on) \
			Extrae_network_routes_Wrapper (*mpi_rank); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NETWORK_ROUTES);

#define apifTRACE_SETTRACINGTASKS(x) \
	void CtoF77(x##_set_tracing_tasks) (unsigned *from, unsigned *to) \
	{ \
		if (mpitrace_on) \
			Extrae_tracing_tasks_Wrapper (*from, *to); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SETTRACINGTASKS)

# endif /* HAVE_ALIAS_ATTRIBUTE */

#endif /* defined(FORTRAN_SYMBOLS) */

