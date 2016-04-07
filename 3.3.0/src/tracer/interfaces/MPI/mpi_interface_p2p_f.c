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

#endif /* defined(FORTRAN_SYMBOLS) */

