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

#ifndef MPI_WRAPPER_1SIDED_C_DEFINED
#define MPI_WRAPPER_1SIDED_C_DEFINED

#if !defined(MPI_SUPPORT)
# error "This should not be included"
#endif

#include <config.h>

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif

/* C Wrappers */
#if defined(C_SYMBOLS)

#if MPI_SUPPORTS_MPI_1SIDED

int MPI_Win_create_C_Wrapper (void *base, MPI_Aint size, int disp_unit,
	MPI_Info info, MPI_Comm comm, MPI_Win *win);

int MPI_Win_fence_C_Wrapper (int assert, MPI_Win win);

int MPI_Win_start_C_Wrapper (MPI_Group group, int assert, MPI_Win win);

int MPI_Win_post_C_Wrapper (MPI_Group group, int assert, MPI_Win win);

int MPI_Win_free_C_Wrapper (MPI_Win *win);

int MPI_Win_complete_C_Wrapper (MPI_Win win);

int MPI_Win_wait_C_Wrapper (MPI_Win win);

int MPI_Get_C_Wrapper (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
  int target_rank, MPI_Aint target_disp, int target_count,
  MPI_Datatype target_datatype, MPI_Win win);

int MPI_Put_C_Wrapper (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
  int target_rank, MPI_Aint target_disp, int target_count,
  MPI_Datatype target_datatype, MPI_Win win);

int MPI_Win_lock_C_Wrapper (int lock_type, int rank, int assert, MPI_Win win);

int MPI_Win_unlock_C_Wrapper (int rank, MPI_Win win);

int MPI_Get_accumulate_C_Wrapper (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
  void *result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win);


#endif /* MPI_SUPPORTS_MPI_1SIDED */

#endif /* C_SYMBOLS */

#endif /* MPI_WRAPPER_DEFINED */

