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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

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

unsigned int MPI_NumOpsGlobals = 0;
unsigned int MPI_CurrentOpGlobal = 0;

unsigned int get_MPI_NumOpsGlobals()
{
	return MPI_NumOpsGlobals;
}

#if defined(FORTRAN_SYMBOLS)
# include "mpif.h"
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_F_enter (ierror);
#endif

	/* En qualsevol cas, cal cridar al Wrapper que inicialitzara tot el que cal */
	DEBUG_INTERFACE(ENTER)
	PMPI_Init_Wrapper (ierror);
	DEBUG_INTERFACE(LEAVE)

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_thread_F_enter (required, provided, ierror);
#endif

	/* En qualsevol cas, cal cridar al Wrapper que inicialitzara tot el que cal */
	DEBUG_INTERFACE(ENTER)
	PMPI_Init_thread_Wrapper (required, provided, ierror);
	DEBUG_INTERFACE(LEAVE)

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_thread_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Finalize_F_enter (ierror);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Finalize_Wrapper (ierror);
		DEBUG_INTERFACE(LEAVE)
	}
	else if (!mpitrace_on && CheckForControlFile)
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Finalize_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_BSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_bsend__,mpi_bsend_,MPI_BSEND,mpi_bsend,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_bsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror) 
#else
void NAME_ROUTINE_C2F(mpi_bsend) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_BSend_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_bsend) (buf, count, datatype, dest, tag, comm, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_SSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ssend__,mpi_ssend_,MPI_SSEND,mpi_ssend,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ssend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ssend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_SSend_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ssend) (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_RSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_rsend__,mpi_rsend_,MPI_RSEND,mpi_rsend,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_rsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_rsend) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_RSend_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
 		CtoF77 (pmpi_rsend) (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Send
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_send__,mpi_send_,MPI_SEND,mpi_send,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_send) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_send) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Send_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_send) (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_IBSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ibsend__,mpi_ibsend_,MPI_IBSEND,mpi_ibsend,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ibsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ibsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibsend_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IBSend_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
                         ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ibsend) (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
                          ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ibsend_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_ISend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_isend__,mpi_isend_,MPI_ISEND,mpi_isend,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_isend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_isend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Isend_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_ISend_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_isend) (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
			ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Isend_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_ISSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_issend__,mpi_issend_,MPI_ISSEND,mpi_issend,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_issend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_issend) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Issend_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_ISSend_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_issend) (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
			ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Issend_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_IRSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_irsend__,mpi_irsend_,MPI_IRSEND,mpi_irsend,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_irsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_irsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irsend_F_enter (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IRSend_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_irsend) (MPI3_VOID_P_CAST buf, count, datatype, dest, tag, comm, request,
			ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irsend_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Recv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_recv__,mpi_recv_,MPI_RECV,mpi_recv,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_recv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_recv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, 
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_F_enter (buf, count, datatype, source, tag, comm, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Recv_Wrapper (buf, count, datatype, source, tag, comm, status,
                       ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_recv) (buf, count, datatype, source, tag, comm, status,
                        ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_IRecv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_irecv__,mpi_irecv_,MPI_IRECV,mpi_irecv,(void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_irecv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_irecv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irecv_F_enter (buf, count, datatype, source, tag, comm, request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_IRecv_Wrapper (buf, count, datatype, source, tag, comm, request,
                        ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_irecv) (buf, count, datatype, source, tag, comm, request,
                         ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Irecv_F_leave ();
#endif
}


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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_scatter_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_F_INT_P_CAST recvcounts, datatype, op, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int trace_it = mpitrace_on;
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allreduce_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count,
		datatype, op, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

#if defined(MPICH_1_2_Comm_Allreduce_bugfix)
	trace_it = trace_it && !Extrae_MPICH12_COMM_inside;
#endif

	if (trace_it)
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
 ***  MPI_Probe
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_probe__,mpi_probe_,MPI_PROBE,mpi_probe,(MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_probe) (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_probe) (MPI_Fint *source, MPI_Fint *tag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Probe_F_enter(source, tag, comm, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Probe_Wrapper (source, tag, comm, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_probe) (source, tag, comm, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Probe_F_leave();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_get_status_F_enter (request, flag, status, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_get_status_F_leave ();
#endif
}



/******************************************************************************
 ***  MPI_IProbe
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_iprobe__,mpi_iprobe_,MPI_IPROBE,mpi_iprobe,(MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_iprobe) (MPI_Fint *source, MPI_Fint *tag,
	MPI_Fint *comm, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_iprobe) (MPI_Fint *source, MPI_Fint *tag,
	MPI_Fint *comm, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iprobe_F_enter (source, tag, comm, flag, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (4+Caller_Count[CALLER_MPI]);
		PMPI_IProbe_Wrapper (source, tag, comm, flag, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_iprobe) (source, tag, comm, flag, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Iprobe_F_leave();
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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Barrier_F_enter (comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;
    
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
 ***  MPI_Cancel
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_cancel__,mpi_cancel_,MPI_CANCEL,mpi_cancel,(MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_cancel) (MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_cancel) (MPI_Fint *request, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cancel_F_enter (request, ierror);
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cancel_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Test
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_test__,mpi_test_,MPI_TEST,mpi_test,(MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_test) (MPI_Fint *request, MPI_Fint *flag,
	MPI_Fint *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_test) (MPI_Fint *request, MPI_Fint *flag,
	MPI_Fint *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Test_F_enter (request, flag, status, ierror);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (5+Caller_Count[CALLER_MPI]);
		PMPI_Test_Wrapper (request, flag, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_test) (request, flag, status, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Test_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_TestAll
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_testall__,mpi_testall_,MPI_TESTALL,mpi_testall,(MPI_Fint * count, MPI_Fint array_of_requests[], MPI_Fint *flag, MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror))

void NAME_ROUTINE_F(mpi_testall) (MPI_Fint * count,
	MPI_Fint array_of_requests[], MPI_Fint *flag,
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS],
	MPI_Fint * ierror)
#else
void NAME_ROUTINE_C2F(mpi_testall) (MPI_Fint * count,
	MPI_Fint array_of_requests[], MPI_Fint *flag,
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS],
	MPI_Fint * ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testall_F_enter (count, array_of_requests, flag, (MPI_Fint*) array_of_statuses, ierror);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*count+Caller_Count[CALLER_MPI]);
		PMPI_TestAll_Wrapper (count, array_of_requests, flag, array_of_statuses, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
	CtoF77 (pmpi_testall) (count, array_of_requests, flag,
		array_of_statuses, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testall_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_TestAny
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_testany__,mpi_testany_,MPI_TESTANY,mpi_testany,(MPI_Fint *count, MPI_Fint array_of_requests[],MPI_Fint *index, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_testany) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_testany) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testany_F_enter (count, array_of_requests, index, flag, status, ierror);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		PMPI_TestAny_Wrapper (count, array_of_requests, index, flag, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_testany) (count, array_of_requests, index, flag, status, ierror);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testany_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_TestSome
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_testsome__,mpi_testsome_,MPI_TESTSOME,mpi_testsome, (MPI_Fint *incount, MPI_Fint *array_of_requests, MPI_Fint *outcount, MPI_Fint *array_of_indices, MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_testsome) (MPI_Fint *incount,
	MPI_Fint array_of_requests[], MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_testsome) (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testsome_F_enter (incount, array_of_requests, outcount,
		array_of_indices, (MPI_Fint*) array_of_statuses, ierror);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*incount+Caller_Count[CALLER_MPI]);
		PMPI_TestSome_Wrapper (incount, array_of_requests, outcount,
                           array_of_indices, array_of_statuses, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_testsome) (incount, array_of_requests, outcount,
                            array_of_indices, array_of_statuses, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Testsome_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_Wait
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_wait__,mpi_wait_,MPI_WAIT,mpi_wait,(MPI_Fint *request, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_wait) (MPI_Fint *request, MPI_Fint *status,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_wait) (MPI_Fint *request, MPI_Fint *status,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Wait_F_enter (request, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		PMPI_Wait_Wrapper (request, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_wait) (request, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Wait_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_WaitAll
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_waitall__,mpi_waitall_,MPI_WAITALL,mpi_waitall,(MPI_Fint * count, MPI_Fint array_of_requests[], MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror))

void NAME_ROUTINE_F(mpi_waitall) (MPI_Fint * count,
	MPI_Fint array_of_requests[], MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS],
	MPI_Fint * ierror)
#else
void NAME_ROUTINE_C2F(mpi_waitall) (MPI_Fint * count,
	MPI_Fint array_of_requests[], MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS],
	MPI_Fint * ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitall_F_enter (count, array_of_requests, (MPI_Fint*) array_of_statuses, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*count+Caller_Count[CALLER_MPI]);
		PMPI_WaitAll_Wrapper (count, array_of_requests, array_of_statuses,
		  ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_waitall) (count, array_of_requests, array_of_statuses,
		  ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitall_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_WaitAny
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_waitany__,mpi_waitany_,MPI_WAITANY,mpi_waitany, (MPI_Fint *count, MPI_Fint array_of_requests[],MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_waitany) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_waitany) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitany_F_enter (count, array_of_requests, index, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		PMPI_WaitAny_Wrapper (count, array_of_requests, index, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
	    CtoF77 (pmpi_waitany) (count, array_of_requests, index, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitany_F_leave ();
#endif
}


/******************************************************************************
 ***  MPI_WaitSome
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_waitsome__,mpi_waitsome_,MPI_WAITSOME,mpi_waitsome, (MPI_Fint *incount, MPI_Fint array_of_requests[], MPI_Fint *outcount, MPI_Fint array_of_indices[], MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_waitsome) (MPI_Fint *incount,
	MPI_Fint array_of_requests[], MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_waitsome) (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitsome_F_enter (incount, array_of_requests, outcount,
		array_of_indices, (int*)array_of_statuses, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*incount+Caller_Count[CALLER_MPI]);
		PMPI_WaitSome_Wrapper (incount, array_of_requests, outcount,
		  array_of_indices, array_of_statuses, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_waitsome) (incount, array_of_requests, outcount,
		  array_of_indices, array_of_statuses, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Waitsome_F_leave ();
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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bcast_F_enter (buffer, count, datatype, root, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoall_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoallv_F_enter (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST sdispls, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST rdispls, recvtype, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgather_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgatherv_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, MPI3_VOID_P_CAST recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gather_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gatherv_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_F_INT_P_CAST recvcount, MPI3_F_INT_P_CAST displs, recvtype, root, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatter_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatterv_F_enter (MPI3_VOID_P_CAST sendbuf, MPI3_F_INT_P_CAST sendcount, MPI3_F_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_rank_F_enter (comm, rank, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_rank_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_size_F_enter (comm, size, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_size_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_create_F_enter (comm, group, newcomm, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_create_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_free_F_enter (comm, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_free_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_dup_F_enter (comm, newcomm, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_dup_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_split_F_enter (comm, color, key, newcomm, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_split_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_F_enter (command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_F_leave ();
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_multiple_F_enter (count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_multiple_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_create_F_enter (comm_old, ndims, dims, periods, reorder,
		comm_cart, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_create_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_sub_F_enter (comm, remain_dims, comm_new, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_sub_F_leave ();
#endif
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
void NAME_ROUTINE_C2F(mpi_intercomm_create) (MPI_Fint *local_comm
	MPI_Fint *local_leader, MPI_Fint *peer_comm, MPI_Fint *remote_leader,
	MPI_Fint *tag, MPI_Fint *new_intercomm, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_create_F_enter (local_comm, local_leader, peer_comm,
	  remote_leader, tag, new_intercomm, ierror);
#endif
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
		CtoF77 (mpi_intercomm_create) (local_comm, local_leader, peer_comm, 
		  remote_leader, tag, new_intercomm, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_create_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_merge_F_enter (intercomm, high, newintracomm, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_merge_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Start_F_enter (request, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Start_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Startall_F_enter (count, array_of_requests, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Startall_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_free_F_enter (request, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_free_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Recv_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_recv_init__,mpi_recv_init_,MPI_RECV_INIT,mpi_recv_init, (void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_recv_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_recv_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_init_F_enter (buf, count, datatype, source, tag, comm,
	  request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Recv_init_Wrapper (buf, count, datatype, source, tag,
                            comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_recv_init) (buf, count, datatype, source, tag,
                             comm, request, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Recv_init_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Send_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_send_init__,mpi_send_init_,MPI_SEND_INIT,mpi_send_init, (void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_send_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_send_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_init_F_enter (buf, count, datatype, dest, tag, comm,
	  request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Send_init_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag,
                            comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_send_init) (buf, count, datatype, dest, tag, comm,
			request, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Send_init_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Bsend_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_bsend_init__,mpi_bsend_init_,MPI_BSEND_INIT,mpi_bsend_init, (void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_bsend_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_bsend_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_init_F_enter (buf, count, datatype, dest, tag, comm,
	  request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Bsend_init_Wrapper (MPI3_VOID_P_CAST buf, count, datatype,
			dest, tag, comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_bsend_init) (buf, count, datatype, dest, tag,
                              comm, request, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bsend_init_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Rsend_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_rsend_init__,mpi_rsend_init_,MPI_RSEND_INIT,mpi_rsend_init, (void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_rsend_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_rsend_init) (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_init_F_enter (buf, count, datatype, dest, tag, comm,
	  request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Rsend_init_Wrapper (MPI3_VOID_P_CAST buf, count, datatype, dest, tag,
                             comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_rsend_init) (buf, count, datatype, dest, tag, comm,
			request, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Rsend_init_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_Ssend_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_ssend_init__,mpi_ssend_init_,MPI_SSEND_INIT,mpi_ssend_init, (void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_ssend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_ssend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_init_F_enter (buf, count, datatype, dest, tag, comm,
	  request, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_Ssend_init_Wrapper (MPI3_VOID_P_CAST buf, count, datatype,
			dest, tag, comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_ssend_init) (buf, count, datatype, dest, tag, comm,
			request, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Ssend_init_F_leave ();
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
	int result;
	MPI_Comm c = MPI_Comm_f2c(*comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scan_F_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm, ierror);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &result);
	if (result == MPI_IDENT || result == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

/******************************************************************************
 ***  MPI_Sendrecv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_sendrecv__,mpi_sendrecv_,MPI_SENDRECV,mpi_sendrecv, (void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_sendrecv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_sendrecv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_F_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, dest, sendtag,
		recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Sendrecv_Fortran_Wrapper (MPI3_VOID_P_CAST sendbuf, sendcount,
			sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
			source, recvtag, comm, status, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
  		CtoF77(pmpi_sendrecv) (sendbuf, sendcount, sendtype, dest,
			sendtag, recvbuf, recvcount, recvtype, source, recvtag,
			comm, status, ierr);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_F_leave();
#endif
}

/******************************************************************************
 ***  MPI_Sendrecv_replace
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_sendrecv_replace__,mpi_sendrecv_replace_,MPI_SENDRECV_REPLACE,mpi_sendrecv_replace, (void *buf, MPI_Fint *count, MPI_Fint *type, MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source, MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr))

void NAME_ROUTINE_F(mpi_sendrecv_replace) (void *buf, MPI_Fint *count,
	MPI_Fint *type, MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source,
	MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr)
#else
void NAME_ROUTINE_C2F(mpi_sendrecv_replace) (void *buf, MPI_Fint *count,
	MPI_Fint *type, MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source,
	MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_replace_F_enter (buf, count, type, dest, sendtag, source,
		recvtag, comm, status, ierr);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		MPI_Sendrecv_replace_Fortran_Wrapper (buf, count, type, dest,
			sendtag, source, recvtag, comm, status, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77(pmpi_sendrecv_replace) (buf, count, type, dest, sendtag,
			source, recvtag, comm, status, ierr);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Sendrecv_replace_F_leave ();
#endif
}

/*************************************************************
 **********************      MPIIO      **********************
 *************************************************************/

#if MPI_SUPPORTS_MPI_IO

/******************************************************************************
 ***  MPI_File_open
 ******************************************************************************/
#if 0 
/* Instrumentation of mpi_file_open is buggy because conversion from Fortran/string
into C/string is non-direct. ATM, this routine is not instrumented */
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_open__,mpi_file_open_,MPI_FILE_OPEN,mpi_file_open, (MPI_Fint *comm, char *filename, MPI_Fint *amode, MPI_Fint *info, MPI_File *fh, MPI_Fint *len))

void NAME_ROUTINE_F(mpi_file_open) (MPI_Fint *comm, char *filename,
	MPI_Fint *amode, MPI_Fint *info, MPI_File *fh, MPI_Fint *len)
#else
void NAME_ROUTINE_C2F(mpi_file_open) (MPI_Fint *comm, char *filename,
	MPI_Fint *amode, MPI_Fint *info, MPI_File *fh, MPI_Fint *len)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_open_Fortran_Wrapper (comm, filename, amode, info, fh, len);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_open) (comm, filename, amode, info, fh, len);
}
#endif /* Buggy mpi_file_open */

/******************************************************************************
 ***  MPI_File_close
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_close__,mpi_file_close_,MPI_FILE_CLOSE,mpi_file_close, (MPI_File *fh, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_close) (MPI_File *fh, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_close) (MPI_File *fh, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_close_F_enter (fh, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_close_Fortran_Wrapper (fh, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_close) (fh, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_close_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_File_read
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_read__,mpi_file_read_,MPI_FILE_READ,mpi_file_read, (MPI_File *fh, void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_read) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_read) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#endif
{ 
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_F_enter (fh, buf, count, datatype, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_read_Fortran_Wrapper (fh, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_read) (fh, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_File_read_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_read_all__,mpi_file_read_all_,MPI_FILE_READ_ALL,mpi_file_read_all, (MPI_File *fh, void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_read_all) (MPI_File *fh, void *buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_read_all) (MPI_File *fh, void *buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_all_F_enter (fh, buf, count, datatype, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_read_all_Fortran_Wrapper (fh, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_read_all) (fh, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_all_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_File_write
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_write__,mpi_file_write_,MPI_FILE_WRITE,mpi_file_write, (MPI_File *fh, void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_write) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_write) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_F_enter (fh, buf, count, datatype, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_write_Fortran_Wrapper (fh, MPI3_VOID_P_CAST buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_write) (fh, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_File_write_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_write_all__,mpi_file_write_all_,MPI_FILE_WRITE_ALL,mpi_file_write_all, (MPI_File *fh, void *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_write_all) (MPI_File *fh, void *buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_write_all) (MPI_File *fh, void *buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_all_F_enter (fh, buf, count, datatype, status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_write_all_Fortran_Wrapper (fh, MPI3_VOID_P_CAST buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_write_all) (fh, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_all_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_File_read_at
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_read_at__,mpi_file_read_at_,MPI_FILE_READ_AT,mpi_file_read_at, (MPI_File *fh, MPI_Offset *offset, void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_read_at) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_read_at) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_F_enter (fh, offset, buf, count, datatype,
	  status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_read_at_Fortran_Wrapper (fh, offset, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_read_at) (fh, offset, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_File_read_at_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_read_at_all__,mpi_file_read_at_all_,MPI_FILE_READ_AT_ALL,mpi_file_read_at_all, (MPI_File *fh, MPI_Offset *offset, void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_read_at_all) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_read_at_all) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_all_F_enter (fh, offset, buf, count, datatype,
	  status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_read_at_all_Fortran_Wrapper (fh, offset, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_read_at_all) (fh, offset, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_all_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_file_write_at
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_write_at__,mpi_file_write_at_,MPI_FILE_WRITE_AT,mpi_file_write_at, (MPI_File *fh, MPI_Offset *offset, void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_write_at) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_write_at) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_F_enter (fh, offset, buf, count, datatype,
	  status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_write_at_Fortran_Wrapper (fh, offset, MPI3_VOID_P_CAST buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_write_at) (fh, offset, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_F_leave ();
#endif
}

/******************************************************************************
 ***  MPI_File_write_at_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_file_write_at_all__,mpi_file_write_at_all_,MPI_FILE_WRITE_AT_ALL,mpi_file_write_at_all,(MPI_File *fh, MPI_Offset *offset, void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_file_write_at_all) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_file_write_at_all) (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_all_F_enter (fh, offset, buf, count, datatype,
	  status, ierror);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PMPI_File_write_at_all_Fortran_Wrapper (fh, offset, MPI3_VOID_P_CAST buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (pmpi_file_write_at_all) (fh, offset, buf, count, datatype, status, ierror);
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_all_F_leave ();
#endif
}

#endif /* MPI_SUPPORTS_MPI_IO */

#if MPI_SUPPORTS_MPI_1SIDED

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_create__,mpi_win_create_,MPI_WIN_CREATE,mpi_win_create,(void *base, void *size, MPI_Fint *disp_unit, void *info, void *comm, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_create)(void *base, void *size, MPI_Fint *disp_unit, void *info, void *comm, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_create)(void *base, void *size, MPI_Fint *disp_unit, void *info, void *comm, void *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_create_F_enter (base, size, disp_unit, info, comm, win, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_create_F_leave ();
#endif
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_fence__,mpi_win_fence_,MPI_WIN_FENCE,mpi_win_fence,(MPI_Fint *assert, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_fence)(MPI_Fint *assert, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_fence)(MPI_Fint *assert, void *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_fence_F_enter (assert, win, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_fence_F_leave ();
#endif
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_start__,mpi_win_start_,MPI_WIN_START,mpi_win_start,(void *group, void *assert, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_start)(void *group, void *assert, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_start)(void *group, void *assert, void *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_start_F_enter (group, assert, win, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_start_F_leave ();
#endif
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_free__,mpi_win_free_,MPI_WIN_FREE,mpi_win_free,(void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_free)(void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_free)(void *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_free_F_enter (win, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_free_F_leave ();
#endif
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_complete__,mpi_win_complete_,MPI_WIN_COMPLETE,mpi_win_complete,(void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_complete)(void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_complete)(void *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_complete_F_enter (win, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_complete_F_leave ();
#endif
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_wait__,mpi_win_wait_,MPI_WIN_WAIT,mpi_win_wait,(void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_wait)(void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_wait)(void *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_wait_F_enter (win, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_wait_F_leave ();
#endif
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_win_post__,mpi_win_post_,MPI_WIN_POST,mpi_win_post,(void *group, void *assert, void *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_win_post)(void *group, void *assert, void *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_win_post)(void *group, void *assert, void *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_post_F_enter (group, assert, win, ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_post_F_leave ();
#endif
}

#if defined(HAVE_ALIAS_ATTRIBUTE)
MPI_F_SYMS(mpi_get__,mpi_get_,MPI_GET,mpi_get,(MPI_Fint *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror))

void NAME_ROUTINE_F(mpi_get)(MPI_Fint *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)
#else
void NAME_ROUTINE_C2F(mpi_get)(MPI_Fint *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)
#endif
{
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Get_F_enter (origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win,
		ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Get_F_leave ();
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Put_F_enter (origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win,
		ierror);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Put_F_leave ();
#endif
}

#endif /* MPI_SUPPORTS_MPI_1SIDED */

#endif /* defined(FORTRAN_SYMBOLS) */

#if defined(C_SYMBOLS)

/******************************************************************************
 ***  MPI_Init
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Init) (int *argc, char ***argv)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_enter (argc, argv);
#endif

	/* This should be called always, whenever the tracing takes place or not */
	DEBUG_INTERFACE(ENTER)
	res = MPI_Init_C_Wrapper (argc, argv);
	DEBUG_INTERFACE(LEAVE)

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_leave ();
#endif

	return res;
}

#if defined(MPI_HAS_INIT_THREAD_C)
/******************************************************************************
 ***  MPI_Init_thread
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Init_thread) (int *argc, char ***argv, int required, int *provided)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_thread_enter (argc, argv, required, provided);
#endif

	/* This should be called always, whenever the tracing takes place or not */
	DEBUG_INTERFACE(ENTER)
	res = MPI_Init_thread_C_Wrapper (argc, argv, required, provided);
	DEBUG_INTERFACE(LEAVE)

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Init_thread_leave ();
#endif

	return res;
}
#endif /* MPI_HAS_INIT_THREAD_C */

/******************************************************************************
 ***  MPI_Finalize
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Finalize) (void)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Finalize_enter ();
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Finalize_C_Wrapper ();
		DEBUG_INTERFACE(LEAVE)
	}
	else if (!mpitrace_on && CheckForControlFile)
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Finalize_leave ();
#endif

	return res;
}


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
 ***  MPI_Reduce
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Reduce) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, root, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = (++MPI_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Reduce_scatter
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Reduce_scatter) (MPI3_CONST void *sendbuf, void *recvbuf,
	MPI3_CONST int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_scatter_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, MPI3_C_INT_P_CAST recvcounts, datatype, op, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Reduce_scatter_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Allreduce
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Allreduce) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int trace_it = mpitrace_on;
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allreduce_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

#if defined(MPICH_1_2_Comm_Allreduce_bugfix)
	trace_it = trace_it && !Extrae_MPICH12_COMM_inside;
#endif

	if (trace_it)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_Allreduce_C_Wrapper (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_Allreduce (sendbuf, recvbuf, count, datatype, op, comm);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allreduce_leave ();
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
 *** MPI_Request_get_status 
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Request_get_status) (MPI_Request request, int *flag,
	MPI_Status *status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_get_status_enter (request, flag, status);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_get_status_leave ();
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
 ***  MPI_Barrier
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Barrier) (MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Barrier_enter (comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Barrier_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Cancel
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Cancel) (MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cancel_enter (request);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cancel_leave ();
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
 ***  MPI_BCast
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Bcast) (void *buffer, int count, MPI_Datatype datatype,
	int root, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bcast_enter (buffer, count, datatype, root, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Bcast_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoall_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoall_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoallv_enter (MPI3_VOID_P_CAST sendbuf, MPI3_VOID_P_CAST sendcounts, MPI3_VOID_P_CAST sdispls, sendtype, recvbuf, MPI3_VOID_P_CAST recvcounts, MPI3_VOID_P_CAST rdispls, recvtype, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = (++MPI_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Alltoallv_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgather_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgather_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgatherv_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, MPI3_C_INT_P_CAST recvcounts,MPI3_C_INT_P_CAST  displs, recvtype, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Allgatherv_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gather_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gather_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gatherv_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcounts, MPI3_C_INT_P_CAST displs, recvtype, root, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Gatherv_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatter_enter (MPI3_VOID_P_CAST sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatter_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatterv_enter (MPI3_VOID_P_CAST sendbuf, MPI3_C_INT_P_CAST sendcounts, MPI3_C_INT_P_CAST displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = (++MPI_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scatterv_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Comm_rank
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_rank) (MPI_Comm comm, int *rank)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_rank_enter (comm, rank);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_rank_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Comm_size
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_size) (MPI_Comm comm, int *size)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_size_enter (comm, size);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_size_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Comm_create
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_create) (MPI_Comm comm, MPI_Group group,
	MPI_Comm *newcomm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_create_enter (comm, group, newcomm);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_create_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Comm_free
 ***  NOTE we cannot let freeing communicators!
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_free) (MPI_Comm *comm)
{
	int res;
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_free_enter (comm);
#endif
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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_free_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Comm_dup
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_dup) (MPI_Comm comm, MPI_Comm *newcomm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_dup_enter (comm, newcomm);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_dup_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Comm_split
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Comm_split) (MPI_Comm comm, int color, int key,
	MPI_Comm *newcomm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_split_enter (comm, color, key, newcomm);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_split_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_enter (command, argv, maxprocs, info, root, comm,
		intercomm, array_of_errcodes);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_multiple_enter (count, array_of_commands,
		array_of_argv, MPI3_C_INT_P_CAST array_of_maxprocs,
		MPI3_MPI_INFO_P_CAST array_of_info, root, comm, intercomm,
		array_of_errcodes);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Comm_spawn_multiple_leave ();
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_create_enter (comm_old, ndims, MPI3_C_INT_P_CAST dims,
		MPI3_C_INT_P_CAST periods, reorder, comm_cart);
#endif
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
#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_create_leave ();
#endif
	return res;
}

/******************************************************************************
 *** MPI_Cart_sub
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Cart_sub) (MPI_Comm comm, MPI3_CONST int *remain_dims,
	MPI_Comm *comm_new)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_sub_enter (comm, remain_dims, comm_new);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Cart_sub_leave ();
#endif
	return res;
}

/******************************************************************************
 *** MPI_Intercom_create
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Intercomm_create) (MPI_Comm local_comm, int local_leader,
	MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_create_enter (local_comm, local_leader, peer_comm,
	  remote_leader, tag, newintercomm);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_create_leave ();
#endif
	return res;
}

/******************************************************************************
 *** MPI_Intercom_merge
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Intercomm_merge) (MPI_Comm intercomm, int high,
	MPI_Comm *newintracomm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_merge_enter (intercomm, high, newintracomm);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Intercomm_merge_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_Start
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Start) (MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Start_enter (request);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Start_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Startall
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Startall) (int count, MPI_Request *requests)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Startall_enter (count, requests);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Startall_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_Request_free
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Request_free) (MPI_Request *request)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_free_enter (request);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Request_free_leave ();
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
 ***  MPI_Scan
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_Scan) (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scan_enter (MPI3_VOID_P_CAST sendbuf, recvbuf, count, datatype, op, comm);
#endif

	PMPI_Comm_compare (MPI_COMM_WORLD, comm, &res);
	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else MPI_CurrentOpGlobal = 0;

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Scan_leave ();
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

#if MPI_SUPPORTS_MPI_IO

/******************************************************************************
 ***  MPI_File_open
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_open) (MPI_Comm comm, MPI3_CONST char * filename, int amode,
	MPI_Info info, MPI_File *fh)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_open_enter (comm, filename, amode, info, fh);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_open_C_Wrapper (comm, (char *)filename, amode, info, fh);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_open (comm, filename, amode, info, fh);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_open_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_File_close
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_close) (MPI_File* fh)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_close_enter (fh);
#endif
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_close_C_Wrapper (fh);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_close (fh);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_close_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_File_read
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_read) (MPI_File fh, void* buf, int count,
	MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_enter (fh, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  MPI_File_read_C_Wrapper (fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read (fh, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_File_read_all
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_read_all) (MPI_File fh, void* buf, int count,
	MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_all_enter (fh, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_read_all_C_Wrapper (fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_all (fh, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_all_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_File_write
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_write) (MPI_File fh, MPI3_CONST void * buf, int count,
	MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_enter (fh, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_write_C_Wrapper (fh, MPI3_VOID_P_CAST buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write (fh, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_File_write_all
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_write_all) (MPI_File fh, MPI3_CONST void* buf, int count, 
	MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_all_enter (fh, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_write_all_C_Wrapper (fh, MPI3_VOID_P_CAST buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_all (fh, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_all_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_File_read_at
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_read_at) (MPI_File fh, MPI_Offset offset, void* buf,
	int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_enter (fh, offset, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_read_at_C_Wrapper (fh, offset, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_at (fh, offset, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_leave ();
#endif

	return res;
}

/******************************************************************************
 ***  MPI_File_read_at_all
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_read_at_all) (MPI_File fh, MPI_Offset offset,
	void* buf, int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_all_enter (fh, offset, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_read_at_all_C_Wrapper (fh, offset, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_at_all (fh, offset, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_read_at_all_leave ();
#endif
	return res;
}

/******************************************************************************
 ***  MPI_File_write_at
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_write_at) (MPI_File fh, MPI_Offset offset, MPI3_CONST void * buf,
	int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_enter (fh, offset, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_write_at_C_Wrapper (fh, offset, MPI3_VOID_P_CAST buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_at (fh, offset, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_leave ();
#endif
	return res;
}


/******************************************************************************
 ***  MPI_File_write_at_all
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_write_at_all) (MPI_File fh, MPI_Offset offset,
	MPI3_CONST void* buf, int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_all_enter (fh, offset, buf, count, datatype, status);
#endif

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = MPI_File_write_at_all_C_Wrapper (fh, offset, MPI3_VOID_P_CAST buf, count, datatype, status);	
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_at_all (fh, offset, buf, count, datatype, status);

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_File_write_at_all_leave ();
#endif

	return res;
}

#endif /* MPI_SUPPORTS_MPI_IO */

#if MPI_SUPPORTS_MPI_1SIDED

int MPI_Win_create (void *base, MPI_Aint size, int disp_unit, MPI_Info info,
	MPI_Comm comm, MPI_Win *win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_create_enter (base, size, disp_unit, info, comm, win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_create_leave ();
#endif

	return res;
}

int MPI_Win_fence (int assert, MPI_Win win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_fence_enter (assert, win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_fence_leave ();
#endif

	return res;
}

int MPI_Win_start (MPI_Group group, int assert, MPI_Win win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_start_enter (group, assert, win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_start_leave ();
#endif

	return res;
}

int MPI_Win_free (MPI_Win *win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_free_enter (win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_free_leave ();
#endif

	return res;
}

int MPI_Win_complete (MPI_Win win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_complete_enter (win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_complete_leave ();
#endif

	return res;
}

int MPI_Win_wait (MPI_Win win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_wait_enter (win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_wait_leave ();
#endif

	return res;
}

int MPI_Win_post (MPI_Group group, int assert, MPI_Win win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_post_enter (group, assert, win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Win_post_leave ();
#endif

	return res;
}

int MPI_Get (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	int target_rank, MPI_Aint target_disp, int target_count,
	MPI_Datatype target_datatype, MPI_Win win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Get_enter (origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Get_leave ();
#endif

	return res;
}

int MPI_Put (MPI3_CONST void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	int target_rank, MPI_Aint target_disp, int target_count,
	MPI_Datatype target_datatype, MPI_Win win)
{
	int res;

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Put_enter (origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count, target_datatype, win);
#endif

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

#if defined(ENABLE_LOAD_BALANCING)
	DLB_MPI_Put_leave ();
#endif
	return res;
}

#endif /* MPI_SUPPORTS_MPI_1SIDED */

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

