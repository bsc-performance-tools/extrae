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
#include "utils.h"
#include "utils_mpi.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"
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
#if defined(C_SYMBOLS)

#if MPI_SUPPORTS_MPI_1SIDED

int MPI_Win_create_C_Wrapper (void *base, MPI_Aint size, int disp_unit,
	MPI_Info info, MPI_Comm comm, MPI_Win *win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_create (base, size, disp_unit, info, comm, win);	
	TRACE_MPIEVENT(TIME, MPI_WIN_CREATE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	
	return res;
}

int MPI_Win_fence_C_Wrapper (int assert, MPI_Win win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_FENCE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_fence (assert, win);
	TRACE_MPIEVENT(TIME, MPI_WIN_FENCE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Win_start_C_Wrapper (MPI_Group group, int assert, MPI_Win win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_START_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_start (group, assert, win);
	TRACE_MPIEVENT(TIME, MPI_WIN_START_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Win_post_C_Wrapper (MPI_Group group, int assert, MPI_Win win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_POST_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_post (group, assert, win);
	TRACE_MPIEVENT(TIME, MPI_WIN_POST_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Win_free_C_Wrapper (MPI_Win *win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_FREE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_free (win);
	TRACE_MPIEVENT(TIME, MPI_WIN_FREE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Win_complete_C_Wrapper (MPI_Win win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_COMPLETE_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_complete (win);
	TRACE_MPIEVENT(TIME, MPI_WIN_COMPLETE_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Win_wait_C_Wrapper (MPI_Win win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_WAIT_EV, EVT_BEGIN, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_wait (win);
	TRACE_MPIEVENT(TIME, MPI_WIN_WAIT_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Get_C_Wrapper (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
  int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
	MPI_Win win)
{
	int ierror;
	int origin_datatype_size, target_datatype_size;

	ierror = PMPI_Type_size(origin_datatype, &origin_datatype_size);
	MPI_CHECK(ierror, PMPI_Type_size);
	
	ierror = PMPI_Type_size(target_datatype, &target_datatype_size);
	MPI_CHECK(ierror, PMPI_Type_size);

TRACE_MPIEVENT(LAST_READ_TIME, MPI_GET_EV, EVT_BEGIN, target_rank, origin_datatype_size * origin_count, EMPTY, target_disp * target_datatype_size, origin_addr);
	ierror = PMPI_Get (origin_addr, origin_count, origin_datatype, target_rank,
		target_disp, target_count, target_datatype, win);
	TRACE_MPIEVENT(TIME, MPI_GET_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}

int MPI_Put_C_Wrapper (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
  int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
	MPI_Win win)
{
	int ierror;

	int origin_datatype_size, target_datatype_size;

	ierror = PMPI_Type_size(origin_datatype, &origin_datatype_size);
	MPI_CHECK(ierror, PMPI_Type_size);
	
	ierror = PMPI_Type_size(target_datatype, &target_datatype_size);
	MPI_CHECK(ierror, PMPI_Type_size);

	TRACE_MPIEVENT(LAST_READ_TIME, MPI_PUT_EV, EVT_BEGIN, target_rank, target_datatype_size * target_count, EMPTY, target_disp * target_datatype_size, origin_addr);
	ierror = PMPI_Put (origin_addr, origin_count, origin_datatype, target_rank,
		target_disp, target_count, target_datatype, win);
	TRACE_MPIEVENT(TIME, MPI_PUT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}

int MPI_Win_lock_C_Wrapper (int lock_type, int rank, int assert, MPI_Win win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_LOCK_EV, EVT_BEGIN, rank, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_lock (lock_type, rank, assert, win);
	TRACE_MPIEVENT(TIME, MPI_WIN_LOCK_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Win_unlock_C_Wrapper (int rank, MPI_Win win)
{
	int res;
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_WIN_UNLOCK_EV, EVT_BEGIN, rank, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	res = PMPI_Win_unlock (rank, win);
	TRACE_MPIEVENT(TIME, MPI_WIN_UNLOCK_EV, EVT_END, EMPTY, EMPTY,
	  EMPTY, EMPTY, EMPTY);
	updateStats_OTHER(global_mpi_stats);
	return res;
}

int MPI_Get_accumulate_C_Wrapper (void *origin_addr, int origin_count, MPI_Datatype origin_datatype, 	   void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, 
	MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op,
	MPI_Win win)
{
	int ierror;
	int origin_datatype_size, result_datatype_size, target_datatype_size;

	ierror = PMPI_Type_size(origin_datatype, &origin_datatype_size);
	MPI_CHECK(ierror, PMPI_Type_size);
	
	ierror = PMPI_Type_size(result_datatype, &result_datatype_size);
	MPI_CHECK(ierror, PMPI_Type_size);
	
	ierror = PMPI_Type_size(target_datatype, &target_datatype_size);
	MPI_CHECK(ierror, PMPI_Type_size);
	
	TRACE_MPIEVENT(LAST_READ_TIME, MPI_GET_ACCUMULATE_EV, EVT_BEGIN, target_rank, ((origin_datatype_size * origin_count) + (target_datatype_size * target_count)), EMPTY, target_datatype_size * target_disp, origin_addr);
	ierror = PMPI_Get_accumulate (origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
	TRACE_MPIEVENT(TIME, MPI_GET_ACCUMULATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}
#endif /* MPI_SUPPORTS_MPI_1SIDED */


#endif /* defined(C_SYMBOLS) */
