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
	DLB(DLB_MPI_File_close_F_enter, fh, ierror);

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

	DLB(DLB_MPI_File_close_F_leave);

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
	DLB(DLB_MPI_File_read_F_enter, fh, buf, count, datatype, status, ierror);
	
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

	DLB(DLB_MPI_File_read_F_leave);

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
	
	DLB(DLB_MPI_File_read_all_F_enter, fh, buf, count, datatype, status, ierror);

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
		
	DLB(DLB_MPI_File_read_all_F_leave);

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

	DLB(DLB_MPI_File_write_F_enter, fh, buf, count, datatype, status, ierror);

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

	DLB(DLB_MPI_File_write_F_leave);

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

	DLB(DLB_MPI_File_write_all_F_enter, fh, buf, count, datatype, status, ierror);

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

	DLB(DLB_MPI_File_write_all_F_leave);

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

	DLB(DLB_MPI_File_read_at_F_enter, fh, offset, buf, count, datatype,
	  status, ierror);

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

	DLB(DLB_MPI_File_read_at_F_leave);

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

	DLB(DLB_MPI_File_read_at_all_F_enter, fh, offset, buf, count, datatype,
	  status, ierror);

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

	DLB(DLB_MPI_File_read_at_all_F_leave);

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

	DLB(DLB_MPI_File_write_at_F_enter, fh, offset, buf, count, datatype,
	  status, ierror);

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

	DLB(DLB_MPI_File_write_at_F_leave);

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

	DLB(DLB_MPI_File_write_at_all_F_enter, fh, offset, buf, count, datatype,
	  status, ierror);

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

	DLB(DLB_MPI_File_write_at_all_F_leave);

}

#endif /* MPI_SUPPORTS_MPI_IO */

#endif /* defined(FORTRAN_SYMBOLS) */
