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

#ifndef MPI_WRAPPER_1SIDED_F_DEFINED
#define MPI_WRAPPER_1SIDED_F_DEFINED

#if !defined(MPI_SUPPORT)
# error "This should not be included"
#endif

#include <config.h>

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif
#include "extrae_mpif.h"

/* Fortran Wrappers */

#if defined(FORTRAN_SYMBOLS)

#if MPI_SUPPORTS_MPI_1SIDED

void MPI_Win_create_Fortran_Wrapper (void *base, void* size, void* disp_unit,
	void* info, void* comm, void *win, void *ierror);

void MPI_Win_fence_Fortran_Wrapper (MPI_Fint* assert, void* win, void *ierror);

void MPI_Win_start_Fortran_Wrapper (void* group, MPI_Fint* assert, void *win, void *ierror);

void MPI_Win_post_Fortran_Wrapper (void* group, MPI_Fint* assert, void *win, void *ierror);

void MPI_Win_free_Fortran_Wrapper (void *win, MPI_Fint *ierror);

void MPI_Win_complete_Fortran_Wrapper (void *win, MPI_Fint *ierror);

void MPI_Win_wait_Fortran_Wrapper (void *win, MPI_Fint *ierror);

void MPI_Get_Fortran_Wrapper (void *origin_addr, MPI_Fint* origin_count, MPI_Fint* origin_datatype,
  MPI_Fint* target_rank, MPI_Fint* target_disp, MPI_Fint* target_count, MPI_Fint* target_datatype,
	MPI_Fint* win, MPI_Fint* ierror);

void MPI_Put_Fortran_Wrapper (void *origin_addr, MPI_Fint* origin_count, MPI_Fint* origin_datatype,
  MPI_Fint* target_rank, MPI_Fint* target_disp, MPI_Fint* target_count, MPI_Fint* target_datatype,
	MPI_Fint* win, MPI_Fint* ierror);

void MPI_Win_lock_Fortran_Wrapper (MPI_Fint* lock_type, MPI_Fint* rank, MPI_Fint* assert, void *win, void *ierror);
void MPI_Win_unlock_Fortran_Wrapper (MPI_Fint* rank, void *win, void *ierror);

void MPI_Get_accumulate_Fortran_Wrapper (void *origin_addr, MPI_Fint* origin_count, MPI_Fint* origin_datatype, void *result_addr, MPI_Fint* result_count, MPI_Fint* result_datatype, MPI_Fint* target_rank, MPI_Fint* target_disp, MPI_Fint* target_count, MPI_Fint* target_datatype, MPI_Fint* op, MPI_Fint* win, MPI_Fint* ierror);


#endif /* MPI_SUPPORTS_MPI_1SIDED */

#endif /* defined(FORTRAN_SYMBOLS) */

#endif /* MPI_WRAPPER_1SIDED_F_DEFINED */


