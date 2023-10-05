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
	{ fprintf (stderr, "[DEBUG] Task %d %s %s\n", TASKID, (enter)?"enters":"leaves", __func__); }
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

#if MPI_SUPPORTS_MPI_IO

/******************************************************************************
 ***  MPI_File_open
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_open) (MPI_Comm comm, MPI3_CONST char * filename, int amode,
	MPI_Info info, MPI_File *fh)
{
	int res;

/*  Do not call DLB as Fortran does not call it, agreed with Victor Oct13th,2015
	DLB(DLB_MPI_File_open_enter, comm, filename, amode, info, fh);
*/

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_File_open_C_Wrapper (comm, (char *)filename, amode, info, fh);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_open (comm, filename, amode, info, fh);

/*  Do not call DLB as Fortran does not call it, agreed with Victor Oct13th,2015
	DLB(DLB_MPI_File_open_leave);
*/

	return res;
}

/******************************************************************************
 ***  MPI_File_close
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_close) (MPI_File* fh)
{
	int res;

	DLB(DLB_MPI_File_close_enter, fh);


	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_File_close_C_Wrapper (fh);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_close (fh);

	DLB(DLB_MPI_File_close_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_read
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_read) (MPI_File fh, void* buf, int count,
	MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_read_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res =  MPI_File_read_C_Wrapper (fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read (fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_read_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_read_all
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_all) (MPI_File fh, void* buf, int count,
    MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_read_all_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_all_C_Wrapper(fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_all(fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_read_all_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_read_all_begin
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_all_begin)(MPI_File fh, void* buf, int count,
    MPI_Datatype datatype)
{
	int res;

	DLB(DLB_MPI_File_read_all_begin_enter, fh, buf, count, datatype);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_all_begin_C_Wrapper(fh, buf, count, datatype);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_all_begin(fh, buf, count, datatype);

	DLB(DLB_MPI_File_read_all_begin_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_read_all_end
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_all_end)(MPI_File fh, void* buf, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_read_all_end_enter, fh, buf, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_all_end_C_Wrapper(fh, buf, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_all_end(fh, buf, status);

	DLB(DLB_MPI_File_read_all_end_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_read_at
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_read_at) (MPI_File fh, MPI_Offset offset, void* buf,
	int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_read_at_enter, fh, offset, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_File_read_at_C_Wrapper (fh, offset, buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_at (fh, offset, buf, count, datatype, status);

	DLB(DLB_MPI_File_read_at_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_read_at_all
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_at_all) (MPI_File fh, MPI_Offset offset, void* buf,
    int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_read_at_all_enter, fh, offset, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_at_all_C_Wrapper
		    (fh, offset, buf, count, datatype, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_at_all(fh, offset, buf, count, datatype, status);

	DLB(DLB_MPI_File_read_at_all_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_read_at_all_begin
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_at_all_begin) (MPI_File fh, MPI_Offset offset,
    void* buf, int count, MPI_Datatype datatype)
{
	int res;

	DLB(DLB_MPI_File_read_at_all_begin_enter, fh, offset, buf, count, datatype);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_at_all_begin_C_Wrapper
		    (fh, offset, buf, count, datatype);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_at_all_begin(fh, offset, buf, count, datatype);

	DLB(DLB_MPI_File_read_at_all_begin_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_read_at_all_end
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_at_all_end) (MPI_File fh, void* buf,
    MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_read_at_all_end_enter, fh, buf, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_at_all_end_C_Wrapper(fh, buf, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_at_all_end(fh, buf, status);

	DLB(DLB_MPI_File_read_at_all_end_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_read_ordered
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_ordered) (MPI_File fh, void* buf, int count,
    MPI_Datatype datatype, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_read_ordered_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_ordered_C_Wrapper(fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_ordered(fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_read_ordered_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_read_ordered_begin
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_ordered_begin) (MPI_File fh, void* buf, int count,
    MPI_Datatype datatype)
{
	int res;

	DLB(DLB_MPI_File_read_ordered_begin_enter, fh, buf, count, datatype);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_ordered_begin_C_Wrapper(fh, buf, count, datatype);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_ordered_begin(fh, buf, count, datatype);

	DLB(DLB_MPI_File_read_ordered_begin_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_read_ordered_end
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_ordered_end) (MPI_File fh, void* buf,
    MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_read_ordered_end_enter, fh, buf, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_ordered_end_C_Wrapper(fh, buf, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_ordered_end(fh, buf, status);

	DLB(DLB_MPI_File_read_ordered_end_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_read_shared
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_read_shared) (MPI_File fh, void* buf, int count,
    MPI_Datatype datatype, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_read_shared_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_read_shared_C_Wrapper(fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_read_shared(fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_read_shared_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_write
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_write) (MPI_File fh, MPI3_CONST void * buf, int count,
	MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_write_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_File_write_C_Wrapper (fh, MPI3_VOID_P_CAST buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write (fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_write_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_all
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_all) (MPI_File fh, MPI3_CONST void* buf, int count, 
    MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_write_all_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_all_C_Wrapper
		    (fh, MPI3_VOID_P_CAST buf, count, datatype, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_all(fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_write_all_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_all_begin
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_all_begin)(MPI_File fh, MPI3_CONST void* buf,
    int count, MPI_Datatype datatype)
{
	int res;

	DLB(DLB_MPI_File_write_all_begin_enter, fh, buf, count, datatype);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_all_begin_C_Wrapper
		    (fh, MPI3_VOID_P_CAST buf, count, datatype);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_all_begin(fh, buf, count, datatype);

	DLB(DLB_MPI_File_write_all_begin_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_all_end
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_all_end)(MPI_File fh, MPI3_CONST void* buf,
    MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_write_all_end_enter, fh, buf, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_all_end_C_Wrapper
		    (fh, MPI3_VOID_P_CAST buf, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_all_end(fh, buf, status);

	DLB(DLB_MPI_File_write_all_end_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_at
 ******************************************************************************/
int NAME_ROUTINE_C(MPI_File_write_at) (MPI_File fh, MPI_Offset offset, MPI3_CONST void * buf,
	int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_write_at_enter, fh, offset, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation ();
		res = MPI_File_write_at_C_Wrapper (fh, offset, MPI3_VOID_P_CAST buf, count, datatype, status);
		Backend_Leave_Instrumentation ();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_at (fh, offset, buf, count, datatype, status);

	DLB(DLB_MPI_File_write_at_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_at_all
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_at_all) (MPI_File fh, MPI_Offset offset,
    MPI3_CONST void* buf, int count, MPI_Datatype datatype, MPI_Status* status)
{
	int res;

	DLB(DLB_MPI_File_write_at_all_enter, fh, offset, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_at_all_C_Wrapper
		    (fh, offset, MPI3_VOID_P_CAST buf, count, datatype, status);	
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_at_all(fh, offset, buf, count, datatype, status);

	DLB(DLB_MPI_File_write_at_all_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_at_all_begin
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_at_all_begin) (MPI_File fh, MPI_Offset offset,
    MPI3_CONST void* buf, int count, MPI_Datatype datatype)
{
	int res;

	DLB(DLB_MPI_File_write_at_all_begin_enter, fh, offset, buf, count, datatype);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_at_all_begin_C_Wrapper
		    (fh, offset, MPI3_VOID_P_CAST buf, count, datatype);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_at_all_begin(fh, offset, buf, count, datatype);

	DLB(DLB_MPI_File_write_at_all_begin_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_at_all_end
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_at_all_end) (MPI_File fh, MPI3_CONST void* buf,
    MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_write_at_all_end_enter, fh, buf, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_at_all_end_C_Wrapper
		    (fh, MPI3_VOID_P_CAST buf, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_at_all_end(fh, buf, status);

	DLB(DLB_MPI_File_write_at_all_end_leave);

	return res;
}

/******************************************************************************
 ***  MPI_File_write_ordered
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_ordered) (MPI_File fh, MPI3_CONST void* buf,
    int count, MPI_Datatype datatype, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_write_ordered_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_ordered_C_Wrapper(fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_ordered(fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_write_ordered_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_write_ordered_begin
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_ordered_begin) (MPI_File fh, MPI3_CONST void* buf,
    int count, MPI_Datatype datatype)
{
	int res;

	DLB(DLB_MPI_File_write_ordered_begin_enter, fh, buf, count, datatype);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_ordered_begin_C_Wrapper(fh, buf, count, datatype);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_ordered_begin(fh, buf, count, datatype);

	DLB(DLB_MPI_File_write_ordered_begin_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_write_ordered_end
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_ordered_end) (MPI_File fh, MPI3_CONST void* buf,
    MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_write_ordered_end_enter, fh, buf, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_ordered_end_C_Wrapper(fh, buf, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_ordered_end(fh, buf, status);

	DLB(DLB_MPI_File_write_ordered_end_leave);
	
	return res;
}

/******************************************************************************
 ***  MPI_File_write_shared
 ******************************************************************************/
int
NAME_ROUTINE_C(MPI_File_write_shared) (MPI_File fh, MPI3_CONST void* buf,
    int count, MPI_Datatype datatype, MPI_Status *status)
{
	int res;

	DLB(DLB_MPI_File_write_shared_enter, fh, buf, count, datatype, status);

	if (INSTRUMENT_THIS_MPI)
	{
		DEBUG_INTERFACE(ENTER)
		Backend_Enter_Instrumentation();
		res = MPI_File_write_shared_C_Wrapper(fh, buf, count, datatype, status);
		Backend_Leave_Instrumentation();
		DEBUG_INTERFACE(LEAVE)
	}
	else
		res = PMPI_File_write_shared(fh, buf, count, datatype, status);

	DLB(DLB_MPI_File_write_shared_leave);
	
	return res;
}
#endif /* MPI_SUPPORTS_MPI_IO */

#endif /* C_SYMBOLS */
