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
#ifdef HAVE_SYS_FILE_H
# include <sys/file.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include <mpi.h>
#include "extrae_mpif.h"
#include "utils.h"
#include "utils_mpi.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"
#include "mpi_wrapper.h"
#include "wrapper.h"
#include "clock.h"
#include "signals.h"
#include "misc_wrapper.h"
#include "mpi_interface.h"
#include "mode.h"
#include "threadinfo.h"

#include "hash_table.h"

#if defined(C_SYMBOLS) && defined(FORTRAN_SYMBOLS)
# define COMBINED_SYMBOLS
#endif

#if defined(HAVE_MRNET)
# include "mrnet_be.h"
#endif

#include "misc_wrapper.h"

#define MPI_CHECK(mpi_error, routine) \
	if (mpi_error != MPI_SUCCESS) \
	{ \
		fprintf (stderr, "Error in MPI call %s (file %s, line %d, routine %s) returned %d\n", \
			#routine, __FILE__, __LINE__, __func__, mpi_error); \
		fflush (stderr); \
		exit (1); \
	}

#if defined(FORTRAN_SYMBOLS)

#if MPI_SUPPORTS_MPI_1SIDED

void MPI_Win_create_Fortran_Wrapper (void *base, void* size, void* disp_unit,
	void* info, void* comm, void *win, void *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	CtoF77 (pmpi_win_create)(base, size, disp_unit, info, comm, win, ierror);
	TRACE_MPIEVENT(TIME, MPI_WIN_CREATE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Win_fence_Fortran_Wrapper (MPI_Fint* assert, void* win, void *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_FENCE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	CtoF77 (pmpi_win_fence)(assert, win, ierror);
	TRACE_MPIEVENT(TIME, MPI_WIN_FENCE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Win_start_Fortran_Wrapper (void* group, MPI_Fint* assert, void *win, void *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_START_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	CtoF77 (pmpi_win_start)(group, assert, win, ierror);
	TRACE_MPIEVENT(TIME, MPI_WIN_START_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Win_post_Fortran_Wrapper (void* group, MPI_Fint* assert, void *win, void *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_POST_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	CtoF77 (pmpi_win_post)(group, assert, win, ierror);
	TRACE_MPIEVENT(TIME, MPI_WIN_POST_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Win_free_Fortran_Wrapper (void *win, MPI_Fint *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_FREE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	CtoF77 (pmpi_win_free)(win, ierror);
	TRACE_MPIEVENT(TIME, MPI_WIN_FREE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Win_complete_Fortran_Wrapper (void *win, MPI_Fint *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_COMPLETE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	CtoF77 (pmpi_win_complete)(win, ierror);
	TRACE_MPIEVENT(TIME, MPI_WIN_COMPLETE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Win_wait_Fortran_Wrapper (void *win, MPI_Fint *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_WAIT_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	CtoF77 (pmpi_win_wait)(win, ierror);
	TRACE_MPIEVENT(TIME, MPI_WIN_WAIT_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Get_Fortran_Wrapper (void *origin_addr, MPI_Fint* origin_count, MPI_Fint* origin_datatype,
  MPI_Fint* target_rank, MPI_Fint* target_disp, MPI_Fint* target_count, MPI_Fint* target_datatype,
	MPI_Fint* win, MPI_Fint* ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_GET_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	CtoF77(pmpi_get) (origin_addr, origin_count, origin_datatype, target_rank,
		target_disp, target_count, target_datatype, win, ierror);
	TRACE_MPIEVENT(TIME, MPI_GET_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Put_Fortran_Wrapper (void *origin_addr, MPI_Fint* origin_count, MPI_Fint* origin_datatype,
  MPI_Fint* target_rank, MPI_Fint* target_disp, MPI_Fint* target_count, MPI_Fint* target_datatype,
	MPI_Fint* win, MPI_Fint *ierror)
{
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_PUT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	CtoF77(pmpi_put) (origin_addr, origin_count, origin_datatype, target_rank,
		target_disp, target_count, target_datatype, win, ierror);
	TRACE_MPIEVENT(TIME, MPI_PUT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

#endif /* MPI_SUPPORTS_MPI_1SIDED */


#endif /* defined(FORTRAN_SYMBOLS) */

