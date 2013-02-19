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
#include <pacx.h>
#include "pacx_interface.h"
#include "pacx_wrapper.h"
#include "wrapper.h"

#if defined(C_SYMBOLS) && defined(FORTRAN_SYMBOLS)
# define COMBINED_SYMBOLS
#endif

#define ENTER	TRUE
#define LEAVE	FALSE

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

#define NAME_ROUTINE_C(x) x
#define NAME_ROUTINE_F(x) x
#define NAME_ROUTINE_C2F(x) CtoF77(x)

unsigned int PACX_NumOpsGlobals = 0;
unsigned int PACX_CurrentOpGlobal = 0;

unsigned int get_PACX_NumOpsGlobals()
{
	return PACX_NumOpsGlobals;
}

/*
#if defined(FORTRAN_SYMBOLS)
# include "mpif.h"
#endif
*/

#if defined(HAVE_ALIAS_ATTRIBUTE) 

/* This macro defines r1, r2 and r3 to be aliases to "orig" routine.
   params are the same parameters received by "orig" */

# define PACX_F_SYMS(r1,r2,r3,orig,params) \
    void r1 params __attribute__ ((alias (#orig))); \
    void r2 params __attribute__ ((alias (#orig))); \
    void r3 params __attribute__ ((alias (#orig)));
#endif

/* Some C libraries do not contain the pacx_init symbol (fortran)
	 When compiling the combined (C+Fortran) dyninst module, the resulting
	 module CANNOT be loaded if pacx_init is not found. The top #if def..
	 is a workaround for this situation
*/

#if defined(FORTRAN_SYMBOLS)

/*#if (defined(COMBINED_SYMBOLS) && defined(PACX_C_CONTAINS_FORTRAN_PACX_INIT) || \
     !defined(COMBINED_SYMBOLS) && defined(FORTRAN_SYMBOLS))
*/
/******************************************************************************
 ***  PACX_Init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_init__,pacx_init_,PACX_INIT,pacx_init,(PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_init) (PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_init) (PACX_Fint *ierror)
#endif
{
	/* En qualsevol cas, cal cridar al Wrapper que inicialitzara tot el que cal */
	DEBUG_INTERFACE(ENTER)
	PPACX_Init_Wrapper (ierror);
	DEBUG_INTERFACE(LEAVE)
}

/*
#endif  
*/
/*
#if (defined(COMBINED_SYMBOLS) && defined(PACX_C_CONTAINS_FORTRAN_PACX_INIT) || \
     !defined(COMBINED_SYMBOLS) && defined(FORTRAN_SYMBOLS))
*/

#if defined(PACX_HAS_INIT_THREAD)
/******************************************************************************
 ***  PACX_Init_thread
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_init_thread__,pacx_init_thread_,PACX_INIT_THREAD,pacx_init_thread,(PACX_Fint *required, PACX_Fint *provided, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_init_thread) (PACX_Fint *required, PACX_Fint *provided,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_init_thread) (PACX_Fint *required, PACX_Fint *provided,
	PACX_Fint *ierror)
#endif
{
	/* En qualsevol cas, cal cridar al Wrapper que inicialitzara tot el que cal */
	DEBUG_INTERFACE(ENTER)
	PPACX_Init_thread_Wrapper (required, provided, ierror);
	DEBUG_INTERFACE(LEAVE)
}
#endif /* PACX_HAS_INIT_THREAD */


/******************************************************************************
 ***  PACX_Finalize
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_finalize__,pacx_finalize_,PACX_FINALIZE,pacx_finalize, (PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_finalize) (PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_finalize) (PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_Finalize_Wrapper (ierror);
		DEBUG_INTERFACE(LEAVE)
	}
	else if (!mpitrace_on && CheckForControlFile)
	{
		/* This case happens when the tracing isn't activated due to the inexistance
			of the control file. Just remove the temporal files! */
		DEBUG_INTERFACE(ENTER)
		remove_temporal_files();
		remove_file_list();
		DEBUG_INTERFACE(LEAVE)
		CtoF77 (ppacx_finalize) (ierror);
	}
	else
		CtoF77 (ppacx_finalize) (ierror);
}


/******************************************************************************
 ***  PACX_BSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_bsend__,pacx_bsend_,PACX_BSEND,pacx_bsend,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_bsend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror) 
#else
void NAME_ROUTINE_C2F(pacx_bsend) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_BSend_Wrapper (buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_bsend) (buf, count, datatype, dest, tag, comm, ierror);
}

/******************************************************************************
 ***  PACX_SSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_ssend__,pacx_ssend_,PACX_SSEND,pacx_ssend,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_ssend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_ssend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_SSend_Wrapper (buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_ssend) (buf, count, datatype, dest, tag, comm, ierror);
}


/******************************************************************************
 ***  PACX_RSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_rsend__,pacx_rsend_,PACX_RSEND,pacx_rsend,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_rsend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_rsend) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_RSend_Wrapper (buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
 		CtoF77 (ppacx_rsend) (buf, count, datatype, dest, tag, comm, ierror);
}

/******************************************************************************
 ***  PACX_Send
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_send__,pacx_send_,PACX_SEND,pacx_send,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_send) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_send) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_Send_Wrapper (buf, count, datatype, dest, tag, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_send) (buf, count, datatype, dest, tag, comm, ierror);
}


/******************************************************************************
 ***  PACX_IBSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_ibsend__,pacx_ibsend_,PACX_IBSEND,pacx_ibsend,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_ibsend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_ibsend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_IBSend_Wrapper (buf, count, datatype, dest, tag, comm, request,
                         ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_ibsend) (buf, count, datatype, dest, tag, comm, request,
                          ierror);
}

/******************************************************************************
 ***  PACX_ISend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_isend__,pacx_isend_,PACX_ISEND,pacx_isend,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_isend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_isend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_ISend_Wrapper (buf, count, datatype, dest, tag, comm, request,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_isend) (buf, count, datatype, dest, tag, comm, request,
			ierror);
}

/******************************************************************************
 ***  PACX_ISSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_issend__,pacx_issend_,PACX_ISSEND,pacx_issend,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_issend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_issend) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_ISSend_Wrapper (buf, count, datatype, dest, tag, comm, request,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_issend) (buf, count, datatype, dest, tag, comm, request,
			ierror);
}

/******************************************************************************
 ***  PACX_IRSend
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_irsend__,pacx_irsend_,PACX_IRSEND,pacx_irsend,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_irsend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_irsend) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_IRSend_Wrapper (buf, count, datatype, dest, tag, comm, request,
			ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_irsend) (buf, count, datatype, dest, tag, comm, request,
			ierror);
}

/******************************************************************************
 ***  PACX_Recv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_recv__,pacx_recv_,PACX_RECV,pacx_recv,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_recv) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *status,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_recv) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *status, 
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_Recv_Wrapper (buf, count, datatype, source, tag, comm, status,
                       ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_recv) (buf, count, datatype, source, tag, comm, status,
                        ierror);
}

/******************************************************************************
 ***  PACX_IRecv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_irecv__,pacx_irecv_,PACX_IRECV,pacx_irecv,(void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_irecv) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_irecv) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_IRecv_Wrapper (buf, count, datatype, source, tag, comm, request,
                        ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_irecv) (buf, count, datatype, source, tag, comm, request,
                         ierror);
}


/******************************************************************************
 ***  PACX_Reduce
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_reduce__,pacx_reduce_,PACX_REDUCE,pacx_reduce,(void *sendbuf, void *recvbuf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_reduce) (void *sendbuf, void *recvbuf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *root, PACX_Fint *comm,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_reduce) (void *sendbuf, void *recvbuf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *root, PACX_Fint *comm,
	PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
		PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_Reduce_Wrapper (sendbuf, recvbuf, count, datatype, op, root, comm,
                         ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_reduce) (sendbuf, recvbuf, count, datatype, op, root, comm,
                          ierror);
}

/******************************************************************************
 ***  PACX_Reduce_scatter
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_reduce_scatter__,pacx_reduce_scatter_,PACX_REDUCE_SCATTER,pacx_reduce_scatter,(void *sendbuf, void *recvbuf, PACX_Fint *recvcounts, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_reduce_scatter) (void *sendbuf, void *recvbuf, PACX_Fint *recvcounts,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_reduce_scatter) (void *sendbuf, void *recvbuf,
	PACX_Fint *recvcounts, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm,
	PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
		PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_Reduce_Scatter_Wrapper (sendbuf, recvbuf, recvcounts, datatype, op,
			comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_reduce_scatter) (sendbuf, recvbuf, recvcounts, datatype, op,
			comm, ierror);
}

/******************************************************************************
 ***  PACX_AllReduce
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_allreduce__,pacx_allreduce_,PACX_ALLREDUCE,pacx_allreduce,(void *sendbuf, void *recvbuf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_allreduce) (void *sendbuf, void *recvbuf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_allreduce) (void *sendbuf, void *recvbuf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm,
	PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_AllReduce_Wrapper (sendbuf, recvbuf, count, datatype, op, comm,
                            ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_allreduce) (sendbuf, recvbuf, count, datatype, op, comm,
                             ierror);
}

/******************************************************************************
 ***  PACX_Probe
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_probe__,pacx_probe_,PACX_PROBE,pacx_probe,(PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_probe) (PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_probe) (PACX_Fint *source, PACX_Fint *tag,
	PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Probe_Wrapper (source, tag, comm, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_probe) (source, tag, comm, status, ierror);
}


/******************************************************************************
 ***  PACX_IProbe
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_iprobe__,pacx_iprobe_,PACX_IPROBE,pacx_iprobe,(PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *flag, PACX_Fint *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_iprobe) (PACX_Fint *source, PACX_Fint *tag,
	PACX_Fint *comm, PACX_Fint *flag, PACX_Fint *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_iprobe) (PACX_Fint *source, PACX_Fint *tag,
	PACX_Fint *comm, PACX_Fint *flag, PACX_Fint *status, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (4+Caller_Count[CALLER_MPI]);
    PPACX_IProbe_Wrapper (source, tag, comm, flag, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_iprobe) (source, tag, comm, flag, status, ierror);
}


/******************************************************************************
 ***  PACX_Barrier
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_barrier__,pacx_barrier_,PACX_BARRIER,pacx_barrier,(PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_barrier) (PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_barrier) (PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;
    
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Barrier_Wrapper (comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_barrier) (comm, ierror);
}


/******************************************************************************
 ***  PACX_Cancel
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_cancel__,pacx_cancel_,PACX_CANCEL,pacx_cancel,(PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_cancel) (PACX_Fint *request, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_cancel) (PACX_Fint *request, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Cancel_Wrapper (request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_cancel) (request, ierror);
}

/******************************************************************************
 ***  PACX_Test
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_test__,pacx_test_,PACX_TEST,pacx_test,(PACX_Fint *request, PACX_Fint *flag, PACX_Fint *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_test) (PACX_Fint *request, PACX_Fint *flag,
	PACX_Fint *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_test) (PACX_Fint *request, PACX_Fint *flag,
	PACX_Fint *status, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (5+Caller_Count[CALLER_MPI]);
    PPACX_Test_Wrapper (request, flag, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_test) (request, flag, status, ierror);
}

/******************************************************************************
 ***  PACX_Wait
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_wait__,pacx_wait_,PACX_WAIT,pacx_wait,(PACX_Fint *request, PACX_Fint *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_wait) (PACX_Fint *request, PACX_Fint *status,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_wait) (PACX_Fint *request, PACX_Fint *status,
	PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
    PPACX_Wait_Wrapper (request, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_wait) (request, status, ierror);
}

/******************************************************************************
 ***  PACX_WaitAll
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_waitall__,pacx_waitall_,PACX_WAITALL,pacx_waitall,(PACX_Fint * count, PACX_Fint array_of_requests[], PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS], PACX_Fint * ierror))

void NAME_ROUTINE_F(pacx_waitall) (PACX_Fint * count,
	PACX_Fint array_of_requests[], PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS],
	PACX_Fint * ierror)
#else
void NAME_ROUTINE_C2F(pacx_waitall) (PACX_Fint * count,
	PACX_Fint array_of_requests[], PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS],
	PACX_Fint * ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*count+Caller_Count[CALLER_MPI]);
    PPACX_WaitAll_Wrapper (count, array_of_requests, array_of_statuses,
                          ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_waitall) (count, array_of_requests, array_of_statuses,
                           ierror);
}


/******************************************************************************
 ***  PACX_WaitAny
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_waitany__,pacx_waitany_,PACX_WAITANY,pacx_waitany, (PACX_Fint *count, PACX_Fint array_of_requests[],PACX_Fint *index, PACX_Fint *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_waitany) (PACX_Fint *count, PACX_Fint array_of_requests[],
	PACX_Fint *index, PACX_Fint *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_waitany) (PACX_Fint *count, PACX_Fint array_of_requests[],
	PACX_Fint *index, PACX_Fint *status, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
    PPACX_WaitAny_Wrapper (count, array_of_requests, index, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_waitany) (count, array_of_requests, index, status, ierror);
}


/******************************************************************************
 ***  PACX_WaitSome
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_waitsome__,pacx_waitsome_,PACX_WAITSOME,pacx_waitsome, (PACX_Fint *incount, PACX_Fint array_of_requests[], PACX_Fint *outcount, PACX_Fint array_of_indices[], PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS], PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_waitsome) (PACX_Fint *incount,
	PACX_Fint array_of_requests[], PACX_Fint *outcount, PACX_Fint array_of_indices[],
	PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS], PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_waitsome) (PACX_Fint *incount, PACX_Fint array_of_requests[],
	PACX_Fint *outcount, PACX_Fint array_of_indices[],
	PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS], PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*incount+Caller_Count[CALLER_MPI]);
		PPACX_WaitSome_Wrapper (incount, array_of_requests, outcount,
                           array_of_indices, array_of_statuses, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_waitsome) (incount, array_of_requests, outcount,
                            array_of_indices, array_of_statuses, ierror);
}

/******************************************************************************
 ***  PACX_BCast
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_bcast__,pacx_bcast_,PACX_BCAST,pacx_bcast,(void *buffer, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_bcast) (void *buffer, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_bcast) (void *buffer, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_BCast_Wrapper (buffer, count, datatype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_bcast) (buffer, count, datatype, root, comm, ierror);

}

/******************************************************************************
 ***  PACX_AllToAll
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_alltoall__,pacx_alltoall_,PACX_ALLTOALL,pacx_alltoall, (void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_alltoall) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_alltoall) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_AllToAll_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                           recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_alltoall) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, comm, ierror);
}


/******************************************************************************
 ***  PACX_AllToAllV
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_alltoallv__,pacx_alltoallv_,PACX_ALLTOALLV,pacx_alltoallv, (void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sdispls, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *rdispls, PACX_Fint *recvtype, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_alltoallv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sdispls, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount,
	PACX_Fint *rdispls, PACX_Fint *recvtype,	PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_alltoallv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sdispls, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount,
	PACX_Fint *rdispls, PACX_Fint *recvtype,	PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_AllToAllV_Wrapper (sendbuf, sendcount, sdispls, sendtype, recvbuf,
                            recvcount, rdispls, recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_alltoallv) (sendbuf, sendcount, sdispls, sendtype, recvbuf,
                             recvcount, rdispls, recvtype, comm, ierror);
}


/******************************************************************************
 ***  PACX_Allgather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_allgather__,pacx_allgather_,PACX_ALLGATHER,pacx_allgather, (void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_allgather) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_allgather) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Allgather_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_allgather) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                             recvtype, comm, ierror);
}


/******************************************************************************
 ***  PACX_Allgatherv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_allgatherv__,pacx_allgatherv_,PACX_ALLGATHERV,pacx_allgatherv, (void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs, PACX_Fint *recvtype, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_allgatherv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs,
	PACX_Fint *recvtype, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_allgatherv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs,
	PACX_Fint *recvtype, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Allgatherv_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                             displs, recvtype, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_allgatherv) (sendbuf, sendcount, sendtype, recvbuf,
                              recvcount, displs, recvtype, comm, ierror);
}


/******************************************************************************
 ***  PACX_Gather
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_gather__,pacx_gather_,PACX_GATHER,pacx_gather, (void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_gather) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_gather) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Gather_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                         recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_gather) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, root, comm, ierror);
}

/******************************************************************************
 ***  PACX_GatherV
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_gatherv__,pacx_gatherv_,PACX_GATHERV,pacx_gatherv, (void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs, PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_gatherv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs,
	PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_gatherv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs,
	PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_GatherV_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          displs, recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_gatherv) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                           displs, recvtype, root, comm, ierror);
}

/******************************************************************************
 ***  PACX_Scatter
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_scatter__,pacx_scatter_,PACX_SCATTER,pacx_scatter,(void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_scatter) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_scatter) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Scatter_Wrapper (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_scatter) (sendbuf, sendcount, sendtype, recvbuf, recvcount,
                           recvtype, root, comm, ierror);
}

/******************************************************************************
 ***  PACX_ScatterV
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_scatterv__,pacx_scatterv_,PACX_SCATTERV,pacx_scatterv,(void *sendbuf, PACX_Fint *sendcount, PACX_Fint *displs, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_scatterv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *displs, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount,
	PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_scatterv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *displs, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount,
	PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_ScatterV_Wrapper (sendbuf, sendcount, displs, sendtype, recvbuf,
                           recvcount, recvtype, root, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_scatterv) (sendbuf, sendcount, displs, sendtype, recvbuf,
                            recvcount, recvtype, root, comm, ierror);
}

/******************************************************************************
 ***  PACX_Comm_Rank
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_comm_rank__,pacx_comm_rank_,PACX_COMM_RANK,pacx_comm_rank,(PACX_Fint *comm, PACX_Fint *rank, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_comm_rank) (PACX_Fint *comm, PACX_Fint *rank,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_comm_rank) (PACX_Fint *comm, PACX_Fint *rank,
	PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Comm_Rank_Wrapper (comm, rank, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_comm_rank) (comm, rank, ierror);
}

/******************************************************************************
 ***  PACX_Comm_Size
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_comm_size__,pacx_comm_size_,PACX_COMM_SIZE,pacx_comm_size,(PACX_Fint *comm, PACX_Fint *size, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_comm_size) (PACX_Fint *comm, PACX_Fint *size,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_comm_size) (PACX_Fint *comm, PACX_Fint *size,
	PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Comm_Size_Wrapper (comm, size, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_comm_size) (comm, size, ierror);
}

/******************************************************************************
 ***  PACX_Comm_Create
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_comm_create__,pacx_comm_create_,PACX_COMM_CREATE,pacx_comm_create,(PACX_Fint *comm, PACX_Fint *group, PACX_Fint *newcomm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_comm_create) (PACX_Fint *comm, PACX_Fint *group,
	PACX_Fint *newcomm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_comm_create) (PACX_Fint *comm, PACX_Fint *group,
	PACX_Fint *newcomm, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
    PPACX_Comm_Create_Wrapper (comm, group, newcomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_comm_create) (comm, group, newcomm, ierror);
}

/******************************************************************************
 ***  PACX_Comm_Dup
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_comm_dup__,pacx_comm_dup_,PACX_COMM_DUP,pacx_comm_dup,(PACX_Fint *comm, PACX_Fint *newcomm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_comm_dup) (PACX_Fint *comm, PACX_Fint *newcomm,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_comm_dup) (PACX_Fint *comm, PACX_Fint *newcomm,
	PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
    PPACX_Comm_Dup_Wrapper (comm, newcomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_comm_dup) (comm, newcomm, ierror);
}


/******************************************************************************
 ***  PACX_Comm_Split
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_comm_split__,pacx_comm_split_,PACX_COMM_SPLIT,pacx_comm_split,(PACX_Fint *comm, PACX_Fint *color, PACX_Fint *key, PACX_Fint *newcomm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_comm_split) (PACX_Fint *comm, PACX_Fint *color,
	PACX_Fint *key, PACX_Fint *newcomm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_comm_split) (PACX_Fint *comm, PACX_Fint *color,
	PACX_Fint *key, PACX_Fint *newcomm, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
    PPACX_Comm_Split_Wrapper (comm, color, key, newcomm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_comm_split) (comm, color, key, newcomm, ierror);
}

/******************************************************************************
 *** PACX_Cart_create
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_cart_create__,pacx_cart_create_,PACX_CART_CREATE,pacx_cart_create, (PACX_Fint *comm_old, PACX_Fint *ndims, PACX_Fint *dims, PACX_Fint *periods, PACX_Fint *reorder, PACX_Fint *comm_cart, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_cart_create) (PACX_Fint *comm_old, PACX_Fint *ndims,
	PACX_Fint *dims, PACX_Fint *periods, PACX_Fint *reorder, PACX_Fint *comm_cart,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_cart_create) (PACX_Fint *comm_old, PACX_Fint *ndims,
	PACX_Fint *dims, PACX_Fint *periods, PACX_Fint *reorder, PACX_Fint *comm_cart,
	PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
    PPACX_Cart_create_Wrapper (comm_old, ndims, dims, periods, reorder,
                              comm_cart, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_cart_create) (comm_old, ndims, dims, periods, reorder,
                               comm_cart, ierror);
}

/******************************************************************************
 *** PACX_Cart_sub
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_cart_sub__,pacx_cart_sub_,PACX_CART_SUB,pacx_cart_sub, (PACX_Fint *comm, PACX_Fint *remain_dims, PACX_Fint *comm_new, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_cart_sub) (PACX_Fint *comm, PACX_Fint *remain_dims,
	PACX_Fint *comm_new, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_cart_sub) (PACX_Fint *comm, PACX_Fint *remain_dims,
	PACX_Fint *comm_new, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
    PPACX_Cart_sub_Wrapper (comm, remain_dims, comm_new, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_cart_sub) (comm, remain_dims, comm_new, ierror);
}

/******************************************************************************
 ***  PACX_Start
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_start__,pacx_start_,PACX_START,pacx_start, (PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_start) (PACX_Fint *request, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_start) (PACX_Fint *request, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
    PPACX_Start_Wrapper (request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_start) (request, ierror);
}

/******************************************************************************
 ***  PACX_Startall
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_startall__,pacx_startall_,PACX_STARTALL,pacx_startall, (PACX_Fint *count, PACX_Fint array_of_requests[], PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_startall) (PACX_Fint *count,
	PACX_Fint array_of_requests[], PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_startall) (PACX_Fint *count,
	PACX_Fint array_of_requests[], PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+*count+Caller_Count[CALLER_MPI]);
    PPACX_Startall_Wrapper (count, array_of_requests, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_startall) (count, array_of_requests, ierror);
}

/******************************************************************************
 ***  PACX_Request_free
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_request_free__,pacx_request_free_,PACX_REQUEST_FREE,pacx_request_free, (PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_request_free) (PACX_Fint *request, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_request_free) (PACX_Fint *request, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Request_free_Wrapper (request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_request_free) (request, ierror);
}

/******************************************************************************
 ***  PACX_Recv_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_recv_init__,pacx_recv_init_,PACX_RECV_INIT,pacx_recv_init, (void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_recv_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_recv_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Recv_init_Wrapper (buf, count, datatype, source, tag,
                            comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_recv_init) (buf, count, datatype, source, tag,
                             comm, request, ierror);
}

/******************************************************************************
 ***  PACX_Send_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_send_init__,pacx_send_init_,PACX_SEND_INIT,pacx_send_init, (void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_send_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_send_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Send_init_Wrapper (buf, count, datatype, dest, tag,
                            comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_send_init) (buf, count, datatype, dest, tag,
                             comm, request, ierror);
}

/******************************************************************************
 ***  PACX_Bsend_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_bsend_init__,pacx_bsend_init_,PACX_BSEND_INIT,pacx_bsend_init, (void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_bsend_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_bsend_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Bsend_init_Wrapper (buf, count, datatype, dest, tag,
                             comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_bsend_init) (buf, count, datatype, dest, tag,
                              comm, request, ierror);
}

/******************************************************************************
 ***  PACX_Rsend_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_rsend_init__,pacx_rsend_init_,PACX_RSEND_INIT,pacx_rsend_init, (void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_rsend_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_rsend_init) (void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *request, PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Rsend_init_Wrapper (buf, count, datatype, dest, tag,
                             comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_rsend_init) (buf, count, datatype, dest, tag,
                              comm, request, ierror);
}

/******************************************************************************
 ***  PACX_Ssend_init
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_ssend_init__,pacx_ssend_init_,PACX_SSEND_INIT,pacx_ssend_init, (void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_ssend_init) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_ssend_init) (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror)
#endif
{
  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Ssend_init_Wrapper (buf, count, datatype, dest, tag,
                             comm, request, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_ssend_init) (buf, count, datatype, dest, tag,
                              comm, request, ierror);
}

/******************************************************************************
 ***  PACX_Scan
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_scan__,pacx_scan_,PACX_SCAN,pacx_scan, (void *sendbuf, void *recvbuf, PACX_Fint *count, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_scan) (void *sendbuf, void *recvbuf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_scan) (void *sendbuf, void *recvbuf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror)
#endif
{
	int sizeofcomm;
	PACX_Comm c = PACX_Comm_f2c(*comm);

	PPACX_Comm_size (c, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

  if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
    PPACX_Scan_Wrapper (sendbuf, recvbuf, count, datatype, op, comm, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
    CtoF77 (ppacx_scan) (sendbuf, recvbuf, count, datatype, op, comm, ierror);
}

/******************************************************************************
 ***  PACX_Sendrecv
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_sendrecv__,pacx_sendrecv_,PACX_SENDRECV,pacx_sendrecv, (void *sendbuf, PACX_Fint *sendcount, PACX_Fint *sendtype, PACX_Fint *dest, PACX_Fint *sendtag, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *source, PACX_Fint *recvtag, PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr))

void NAME_ROUTINE_F(pacx_sendrecv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, PACX_Fint *dest, PACX_Fint *sendtag, void *recvbuf,
	PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *source, PACX_Fint *recvtag,
	PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr)
#else
void NAME_ROUTINE_C2F(pacx_sendrecv) (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, PACX_Fint *dest, PACX_Fint *sendtag, void *recvbuf,
	PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *source, PACX_Fint *recvtag,
	PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr)
#endif
{
	if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
  	PACX_Sendrecv_Fortran_Wrapper (sendbuf, sendcount, sendtype, dest, sendtag,
    	recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
  	CtoF77(ppacx_sendrecv) (sendbuf, sendcount, sendtype, dest, sendtag,
    	recvbuf, recvcount, recvtype, source, recvtag, comm, status, ierr);
}

/******************************************************************************
 ***  PACX_Sendrecv_replace
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_sendrecv_replace__,pacx_sendrecv_replace_,PACX_SENDRECV_REPLACE,pacx_sendrecv_replace, (void *buf, PACX_Fint *count, PACX_Fint *type, PACX_Fint *dest, PACX_Fint *sendtag, PACX_Fint *source, PACX_Fint *recvtag, PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr))

void NAME_ROUTINE_F(pacx_sendrecv_replace) (void *buf, PACX_Fint *count,
	PACX_Fint *type, PACX_Fint *dest, PACX_Fint *sendtag, PACX_Fint *source,
	PACX_Fint *recvtag, PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr)
#else
void NAME_ROUTINE_C2F(pacx_sendrecv_replace) (void *buf, PACX_Fint *count,
	PACX_Fint *type, PACX_Fint *dest, PACX_Fint *sendtag, PACX_Fint *source,
	PACX_Fint *recvtag, PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr)
#endif
{
	if (mpitrace_on)
  {
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
  	PACX_Sendrecv_replace_Fortran_Wrapper (buf, count, type, dest, sendtag, source, recvtag, comm, status, ierr);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
  }
  else
  	CtoF77(ppacx_sendrecv_replace) (buf, count, type, dest, sendtag, source, recvtag, comm, status, ierr);
}

/*************************************************************
 **********************    PACX-IO      **********************
 *************************************************************/

#if 0 /* defined(PACX_SUPPORTS_PACX_IO) */

/******************************************************************************
 ***  PACX_File_open
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_open__,pacx_file_open_,PACX_FILE_OPEN,pacx_file_open, (PACX_Fint *comm, char *filename, PACX_Fint *amode, PACX_Fint *info, PACX_File *fh, PACX_Fint *len))

void NAME_ROUTINE_F(pacx_file_open) (PACX_Fint *comm, char *filename,
	PACX_Fint *amode, PACX_Fint *info, PACX_File *fh, PACX_Fint *len)
#else
void NAME_ROUTINE_C2F(pacx_file_open) (PACX_Fint *comm, char *filename,
	PACX_Fint *amode, PACX_Fint *info, PACX_File *fh, PACX_Fint *len)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_open_Fortran_Wrapper (comm, filename, amode, info, fh, len);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_open) (comm, filename, amode, info, fh, len);
}

/******************************************************************************
 ***  PACX_File_close
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_close__,pacx_file_close_,PACX_FILE_CLOSE,pacx_file_close, (PACX_File *fh, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_close) (PACX_File *fh, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_close) (PACX_File *fh, PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_close_Fortran_Wrapper (fh, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_close) (fh, ierror);
}

/******************************************************************************
 ***  PACX_File_read
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_read__,pacx_file_read_,PACX_FILE_READ,pacx_file_read, (PACX_File *fh, void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_read) (PACX_File *fh, void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_read) (PACX_File *fh, void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#endif
{ 
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_read_Fortran_Wrapper (fh, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_read) (fh, buf, count, datatype, status, ierror);
}

/******************************************************************************
 ***  PACX_File_read_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_read_all__,pacx_file_read_all_,PACX_FILE_READ_ALL,pacx_file_read_all, (PACX_File *fh, void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_read_all) (PACX_File *fh, void *buf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_read_all) (PACX_File *fh, void *buf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_read_all_Fortran_Wrapper (fh, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_read_all) (fh, buf, count, datatype, status, ierror);
}

/******************************************************************************
 ***  PACX_File_write
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_write__,pacx_file_write_,PACX_FILE_WRITE,pacx_file_write, (PACX_File *fh, void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_write) (PACX_File *fh, void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_write) (PACX_File *fh, void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_write_Fortran_Wrapper (fh, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_write) (fh, buf, count, datatype, status, ierror);
}

/******************************************************************************
 ***  PACX_File_write_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_write_all__,pacx_file_write_all_,PACX_FILE_WRITE_ALL,pacx_file_write_all, (PACX_File *fh, void *buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_write_all) (PACX_File *fh, void *buf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_write_all) (PACX_File *fh, void *buf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_write_all_Fortran_Wrapper (fh, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_write_all) (fh, buf, count, datatype, status, ierror);
}

/******************************************************************************
 ***  PACX_File_read_at
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_read_at__,pacx_file_read_at_,PACX_FILE_READ_AT,pacx_file_read_at, (PACX_File *fh, PACX_Offset *offset, void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_read_at) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_read_at) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_read_at_Fortran_Wrapper (fh, offset, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_read_at) (fh, offset, buf, count, datatype, status, ierror);
}

/******************************************************************************
 ***  PACX_File_read_at_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_read_at_all__,pacx_file_read_at_all_,PACX_FILE_READ_AT_ALL,pacx_file_read_at_all, (PACX_File *fh, PACX_Offset *offset, void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_read_at_all) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_read_at_all) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_read_at_all_Fortran_Wrapper (fh, offset, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_read_at_all) (fh, offset, buf, count, datatype, status, ierror);
}

/******************************************************************************
 ***  PACX_file_write_at
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_write_at__,pacx_file_write_at_,PACX_FILE_WRITE_AT,pacx_file_write_at, (PACX_File *fh, PACX_Offset *offset, void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_write_at) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_write_at) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_write_at_Fortran_Wrapper (fh, offset, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_write_at) (fh, offset, buf, count, datatype, status, ierror);
}

/******************************************************************************
 ***  PACX_File_write_at_all
 ******************************************************************************/
#if defined(HAVE_ALIAS_ATTRIBUTE)
PACX_F_SYMS(pacx_file_write_at_all__,pacx_file_write_at_all_,PACX_FILE_WRITE_AT_ALL,pacx_file_write_at_all,(PACX_File *fh, PACX_Offset *offset, void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror))

void NAME_ROUTINE_F(pacx_file_write_at_all) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#else
void NAME_ROUTINE_C2F(pacx_file_write_at_all) (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror)
#endif
{
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		PPACX_File_write_at_all_Fortran_Wrapper (fh, offset, buf, count, datatype, status, ierror);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		CtoF77 (ppacx_file_write_at_all) (fh, offset, buf, count, datatype, status, ierror);
}

#endif /* PACX_SUPPORTS_PACX_IO */

#endif /* defined(FORTRAN_SYMBOLS) */

#if defined(C_SYMBOLS)

/******************************************************************************
 ***  PACX_Init
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Init) (int *argc, char ***argv)
{
	int res;

	/* This should be called always, whenever the tracing takes place or not */
	DEBUG_INTERFACE(ENTER)
	res = PACX_Init_C_Wrapper (argc, argv);
	DEBUG_INTERFACE(LEAVE)

	return res;
}

#if defined(PACX_HAS_INIT_THREAD)
/******************************************************************************
 ***  PACX_Init_thread
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Init_thread) (int *argc, char ***argv, int required, int *provided)
{
	int res;

	/* This should be called always, whenever the tracing takes place or not */
	DEBUG_INTERFACE(ENTER)
	res = PACX_Init_thread_C_Wrapper (argc, argv, required, provided);
	DEBUG_INTERFACE(LEAVE)

	return res;
}
#endif /* PACX_HAS_INIT_THREAD */

/******************************************************************************
 ***  PACX_Finalize
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Finalize) (void)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Finalize_C_Wrapper ();
		DEBUG_INTERFACE(LEAVE)
	}
	else if (!mpitrace_on && CheckForControlFile)
	{
		/* This case happens when the tracing isn't activated due to the inexistance
			of the control file. Just remove the temporal files! */
		DEBUG_INTERFACE(ENTER)
		remove_temporal_files();
		remove_file_list();
		DEBUG_INTERFACE(LEAVE)
		res = PPACX_Finalize ();
	}
  else
		res = PPACX_Finalize ();

	return res;
}


/******************************************************************************
 ***  PACX_Bsend
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Bsend) (void* buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Bsend_C_Wrapper (buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Bsend (buf, count, datatype, dest, tag, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Ssend
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Ssend) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Ssend_C_Wrapper (buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Ssend (buf, count, datatype, dest, tag, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Rsend
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Rsend) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Rsend_C_Wrapper (buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Rsend (buf, count, datatype, dest, tag, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Send
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Send) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Send_C_Wrapper (buf, count, datatype, dest, tag, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Send (buf, count, datatype, dest, tag, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Ibsend
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Ibsend) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Ibsend_C_Wrapper (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Ibsend (buf, count, datatype, dest, tag, comm, request);

	return res;
}

/******************************************************************************
 ***  PACX_Isend
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Isend) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Isend_C_Wrapper (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Isend (buf, count, datatype, dest, tag, comm, request);

	return res;
}

/******************************************************************************
 ***  PACX_Issend
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Issend) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Issend_C_Wrapper (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Issend (buf, count, datatype, dest, tag, comm, request);

	return res;
}

/******************************************************************************
 ***  PACX_Irsend
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Irsend) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Irsend_C_Wrapper (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Irsend (buf, count, datatype, dest, tag, comm, request);

	return res;
}

/******************************************************************************
 ***  PACX_Recv
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Recv) (void* buf, int count, PACX_Datatype datatype,
	int source, int tag, PACX_Comm comm, PACX_Status *status)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Recv_C_Wrapper (buf, count, datatype, source, tag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Recv (buf, count, datatype, source, tag, comm, status);

	return res;
}

/******************************************************************************
 ***  PACX_Irecv
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Irecv) (void* buf, int count, PACX_Datatype datatype,
	int source, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Irecv_C_Wrapper (buf, count, datatype, source, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Irecv (buf, count, datatype, source, tag, comm, request);

	return res;
}

/******************************************************************************
 ***  PACX_Reduce
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Reduce) (void *sendbuf, void *recvbuf, int count,
	PACX_Datatype datatype, PACX_Op op, int root, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Reduce_C_Wrapper 
			(sendbuf, recvbuf, count, datatype, op, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
  else
    res = PPACX_Reduce (sendbuf, recvbuf, count, datatype, op, root, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Reduce_scatter
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Reduce_scatter) (void *sendbuf, void *recvbuf,
	int *recvcounts, PACX_Datatype datatype, PACX_Op op, PACX_Comm comm)
{
	int sizeofcomm;
	int res;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Reduce_Scatter_C_Wrapper (sendbuf, recvbuf, recvcounts, datatype,
			op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Reduce_scatter (sendbuf, recvbuf, recvcounts, datatype, op,
			comm);

	return res;
}

/******************************************************************************
 ***  PACX_Allreduce
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Allreduce) (void *sendbuf, void *recvbuf, int count,
	PACX_Datatype datatype, PACX_Op op, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Allreduce_C_Wrapper (sendbuf, recvbuf, count, datatype, op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Allreduce (sendbuf, recvbuf, count, datatype, op, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Probe
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Probe) (int source, int tag, PACX_Comm comm,
	PACX_Status *status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Probe_C_Wrapper (source, tag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Probe (source, tag, comm, status);
}

/******************************************************************************
 ***  PACX_Iprobe
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Iprobe) (int source, int tag, PACX_Comm comm, int *flag,
	PACX_Status *status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (4+Caller_Count[CALLER_MPI]);
		res = PACX_Iprobe_C_Wrapper (source, tag, comm, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Iprobe (source, tag, comm, flag, status);
}

/******************************************************************************
 ***  PACX_Barrier
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Barrier) (PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  PACX_Barrier_C_Wrapper (comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Barrier (comm);

	return res;
}

/******************************************************************************
 ***  PACX_Cancel
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Cancel) (PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Cancel_C_Wrapper (request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Cancel (request);
}

/******************************************************************************
 ***  PACX_Test
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Test) (PACX_Request *request, int *flag, PACX_Status *status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (5+Caller_Count[CALLER_MPI]);
		res = PACX_Test_C_Wrapper (request, flag, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Test (request, flag, status);
}

/******************************************************************************
 ***  PACX_Wait
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Wait) (PACX_Request *request, PACX_Status *status)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		res = PACX_Wait_C_Wrapper (request, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Wait (request, status);

	return res;
}

/******************************************************************************
 ***  PACX_Waitall
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Waitall) (int count, PACX_Request *requests,
	PACX_Status *statuses)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+count+Caller_Count[CALLER_MPI]);
		res = PACX_Waitall_C_Wrapper (count, requests, statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Waitall (count, requests, statuses);

	return res;
}

/******************************************************************************
 ***  PACX_Waitany
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Waitany) (int count, PACX_Request *requests, int *index,
	PACX_Status *status)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		res = PACX_Waitany_C_Wrapper (count, requests, index, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = (PPACX_Waitany (count, requests, index, status));

	return res;
}

/******************************************************************************
 ***  PACX_Waitsome
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Waitsome) (int incount, PACX_Request * requests,
	int *outcount, int *indices, PACX_Status *statuses)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+incount+Caller_Count[CALLER_MPI]);
		res = PACX_Waitsome_C_Wrapper (incount,requests, outcount, indices,
			statuses);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Waitsome (incount, requests, outcount, indices, statuses);

	return res;
}

/******************************************************************************
 ***  PACX_BCast
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Bcast) (void *buffer, int count, PACX_Datatype datatype,
	int root, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_BCast_C_Wrapper (buffer, count, datatype, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Bcast (buffer, count, datatype, root, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Alltoall
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Alltoall) (void *sendbuf, int sendcount,
	PACX_Datatype sendtype, void *recvbuf, int recvcount, PACX_Datatype recvtype,
	PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Alltoall_C_Wrapper
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Alltoall
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Alltoallv
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Alltoallv) (void *sendbuf, int *sendcounts, int *sdispls,
	PACX_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls,
	PACX_Datatype recvtype, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Alltoallv_C_Wrapper
		  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
		  rdispls, recvtype, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Alltoallv
		  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
		  rdispls, recvtype, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Allgather
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Allgather) (void *sendbuf, int sendcount,
	PACX_Datatype sendtype, void *recvbuf, int recvcount, PACX_Datatype recvtype,
	PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Allgather_C_Wrapper
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
		  comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Allgather
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
		  comm);

	return res;
}

/******************************************************************************
 ***  PACX_Allgatherv
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Allgatherv) (void *sendbuf, int sendcount,
	PACX_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs,
	PACX_Datatype recvtype, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  PACX_Allgatherv_C_Wrapper
		  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
		  recvtype, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Allgatherv
		  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
		  recvtype, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Gather
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Gather) (void *sendbuf, int sendcount,
	PACX_Datatype sendtype, void *recvbuf, int recvcount, PACX_Datatype recvtype,
	int root, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  PACX_Gather_C_Wrapper
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Gather
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);

	return res;
}

/******************************************************************************
 ***  PACX_Gatherv
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Gatherv) (void *sendbuf, int sendcount,
	PACX_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs,
	PACX_Datatype recvtype, int root, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  PACX_Gatherv_C_Wrapper
            (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
             recvtype, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Gatherv
		  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
		  recvtype, root, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Scatter
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Scatter) (void *sendbuf, int sendcount,
	PACX_Datatype sendtype, void *recvbuf, int recvcount, PACX_Datatype recvtype,
	int root, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  PACX_Scatter_C_Wrapper
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Scatter
		  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
		  comm);

	return res;
}

/******************************************************************************
 ***  PACX_Scatterv
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Scatterv) (void *sendbuf, int *sendcounts, int *displs, 
	PACX_Datatype sendtype, void *recvbuf, int recvcount, PACX_Datatype recvtype,
	int root, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Scatterv_C_Wrapper
            (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
             recvtype, root, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Scatterv
		  (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount,
		  recvtype, root, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Comm_rank
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Comm_rank) (PACX_Comm comm, int *rank)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Comm_rank_C_Wrapper (comm, rank);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Comm_rank (comm, rank);
}

/******************************************************************************
 ***  PACX_Comm_size
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Comm_size) (PACX_Comm comm, int *size)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Comm_size_C_Wrapper (comm, size);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Comm_size (comm, size);
}

/******************************************************************************
 ***  PACX_Comm_create
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Comm_create) (PACX_Comm comm, PACX_Group group,
	PACX_Comm *newcomm)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = PACX_Comm_create_C_Wrapper (comm, group, newcomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
    		return PPACX_Comm_create (comm, group, newcomm);
}

/******************************************************************************
 ***  PACX_Comm_dup
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Comm_dup) (PACX_Comm comm, PACX_Comm *newcomm)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
    res = PACX_Comm_dup_C_Wrapper (comm, newcomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
  else
    return PPACX_Comm_dup (comm, newcomm);
}

/******************************************************************************
 ***  PACX_Comm_split
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Comm_split) (PACX_Comm comm, int color, int key,
	PACX_Comm *newcomm)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = PACX_Comm_split_C_Wrapper (comm, color, key, newcomm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Comm_split (comm, color, key, newcomm);
}


/******************************************************************************
 *** PACX_Cart_create
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Cart_create) (PACX_Comm comm_old, int ndims, int *dims,
	int *periods, int reorder, PACX_Comm *comm_cart)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res = PACX_Cart_create_C_Wrapper (comm_old, ndims, dims, periods, reorder,
                                      comm_cart);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Cart_create (comm_old, ndims, dims, periods, reorder,
                             comm_cart);
}

/******************************************************************************
 *** PACX_Cart_sub
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Cart_sub) (PACX_Comm comm, int *remain_dims,
	PACX_Comm *comm_new)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Extrae_get_num_tasks()+Caller_Count[CALLER_MPI]);
		res =  PACX_Cart_sub_C_Wrapper (comm, remain_dims, comm_new);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Cart_sub (comm, remain_dims, comm_new);
}

/******************************************************************************
 ***  PACX_Start
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Start) (PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+1+Caller_Count[CALLER_MPI]);
		res =  PACX_Start_C_Wrapper (request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Start (request);
}

/******************************************************************************
 ***  PACX_Startall
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Startall) (int count, PACX_Request *requests)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+count+Caller_Count[CALLER_MPI]);
    res = PACX_Startall_C_Wrapper (count, requests);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
  else
    return PPACX_Startall (count, requests);
}

/******************************************************************************
 ***  PACX_Request_free
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Request_free) (PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  PACX_Request_free_C_Wrapper (request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Request_free (request);
}

/******************************************************************************
 ***  PACX_Recv_init
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Recv_init) (void *buf, int count, PACX_Datatype datatype,
	int source, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Recv_init_C_Wrapper
		  (buf, count, datatype, source, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Recv_init
		  (buf, count, datatype, source, tag, comm, request);
}

/******************************************************************************
 ***  PACX_Send_init
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Send_init) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Send_init_C_Wrapper
		  (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Send_init (buf, count, datatype, dest, tag, comm, request);
}

/******************************************************************************
 ***  PACX_Bsend_init
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Bsend_init) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Bsend_init_C_Wrapper
		  (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Bsend_init (buf, count, datatype, dest, tag, comm, request);
}


/******************************************************************************
 ***  PACX_Rsend_init
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Rsend_init) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Rsend_init_C_Wrapper
		  (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Rsend_init (buf, count, datatype, dest, tag, comm, request);
}

/******************************************************************************
 ***  PACX_Ssend_init
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Ssend_init) (void *buf, int count, PACX_Datatype datatype,
	int dest, int tag, PACX_Comm comm, PACX_Request *request)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Ssend_init_C_Wrapper
		  (buf, count, datatype, dest, tag, comm, request);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_Ssend_init (buf, count, datatype, dest, tag, comm, request);
}

/******************************************************************************
 ***  PACX_Scan
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Scan) (void *sendbuf, void *recvbuf, int count,
	PACX_Datatype datatype, PACX_Op op, PACX_Comm comm)
{
	int res;
	int sizeofcomm;

	PPACX_Comm_size (comm, &sizeofcomm);
	if (sizeofcomm == Extrae_get_num_tasks())
	{
        PACX_CurrentOpGlobal = (++PACX_NumOpsGlobals);

		if (CheckForControlFile)
			CheckControlFile();
		if (CheckForGlobalOpsTracingIntervals)
			CheckGlobalOpsTracingIntervals();
	}
	else PACX_CurrentOpGlobal = 0;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Scan_C_Wrapper (sendbuf, recvbuf, count, datatype, op, comm);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Scan (sendbuf, recvbuf, count, datatype, op, comm);

	return res;
}

/******************************************************************************
 ***  PACX_Sendrecv
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Sendrecv) (void *sendbuf, int sendcount,
	PACX_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount,
	PACX_Datatype recvtype, int source, int recvtag, PACX_Comm comm,
	PACX_Status * status) 
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Sendrecv_C_Wrapper (sendbuf, sendcount, sendtype, dest, sendtag,
		  recvbuf, recvcount, recvtype, source, recvtag, comm, status); 
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Sendrecv (sendbuf, sendcount, sendtype, dest, sendtag,
		  recvbuf, recvcount, recvtype, source, recvtag, comm, status); 

	return res;
}

/******************************************************************************
 ***  PACX_Sendrecv_replace
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_Sendrecv_replace) (void *buf, int count, PACX_Datatype type,
	int dest, int sendtag, int source, int recvtag, PACX_Comm comm,
	PACX_Status* status)
{
	int res;

	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_Sendrecv_replace_C_Wrapper (buf, count, type, dest, sendtag,
		  source, recvtag, comm, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PPACX_Sendrecv_replace (buf, count, type, dest, sendtag, source,
		  recvtag, comm, status);

	return res;
}

#if defined(PACX_SUPPORTS_PACX_IO)

/******************************************************************************
 ***  PACX_File_open
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_open) (PACX_Comm comm, char * filename, int amode,
	PACX_Info info, PACX_File *fh)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_open_C_Wrapper (comm, filename, amode, info, fh);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_open (comm, filename, amode, info, fh);
}

/******************************************************************************
 ***  PACX_File_read_all
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_close) (PACX_File* fh)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_close_C_Wrapper (fh);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_close (fh);
}

/******************************************************************************
 ***  PACX_File_read_all
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_read) (PACX_File fh, void* buf, int count,
	PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res =  PACX_File_read_C_Wrapper (fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_read (fh, buf, count, datatype, status);
}

/******************************************************************************
 ***  PACX_File_read_all
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_read_all) (PACX_File fh, void* buf, int count,
	PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_read_all_C_Wrapper (fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_read_all (fh, buf, count, datatype, status);
}

/******************************************************************************
 ***  PACX_File_write
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_write) (PACX_File fh, void * buf, int count,
	PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_write_C_Wrapper (fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_write (fh, buf, count, datatype, status);
}

/******************************************************************************
 ***  PACX_File_write_all
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_write_all) (PACX_File fh, void* buf, int count, 
	PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_write_all_C_Wrapper (fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_write_all (fh, buf, count, datatype, status);
}

/******************************************************************************
 ***  PACX_File_read_at
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_read_at) (PACX_File fh, PACX_Offset offset, void* buf,
	int count, PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_read_at_C_Wrapper (fh, offset, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_read_at (fh, offset, buf, count, datatype, status);
}

/******************************************************************************
 ***  PACX_File_read_at_all
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_read_at_all) (PACX_File fh, PACX_Offset offset,
	void* buf, int count, PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_read_at_all_C_Wrapper (fh, offset, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_read_at_all (fh, offset, buf, count, datatype, status);
}

/******************************************************************************
 ***  PACX_File_write_at
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_write_at) (PACX_File fh, PACX_Offset offset, void * buf,
	int count, PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_write_at_C_Wrapper (fh, offset, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_write_at (fh, offset, buf, count, datatype, status);
}


/******************************************************************************
 ***  PACX_File_write_at_all
 ******************************************************************************/
int NAME_ROUTINE_C(PACX_File_write_at_all) (PACX_File fh, PACX_Offset offset,
	void* buf, int count, PACX_Datatype datatype, PACX_Status* status)
{
	int res;
	if (mpitrace_on)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation (2+Caller_Count[CALLER_MPI]);
		res = PACX_File_write_at_all_C_Wrapper (fh, offset, buf, count, datatype, status);	
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
		return res;
	}
	else
		return PPACX_File_write_at_all (fh, offset, buf, count, datatype, status);
}

#endif /* PACX_SUPPORTS_PACX_IO */

#endif /* defined(C_SYMBOLS) */
