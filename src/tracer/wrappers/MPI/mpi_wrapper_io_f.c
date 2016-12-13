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

#if MPI_SUPPORTS_MPI_IO
/*************************************************************
 **********************      MPIIO      **********************
 *************************************************************/
void PMPI_File_open_Fortran_Wrapper (MPI_Fint *comm, char *filename, MPI_Fint *amode,
	MPI_Fint *info, MPI_File *fh, MPI_Fint *len)
{
    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_OPEN_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_open) (comm, filename, amode, info, fh, len);
    TRACE_MPIEVENT (TIME, MPI_FILE_OPEN_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_close_Fortran_Wrapper (MPI_File *fh, MPI_Fint *ierror)
{
    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_CLOSE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_close) (fh, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_CLOSE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_read_Fortran_Wrapper (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_READ_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_read) (fh, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_READ_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_read_all_Fortran_Wrapper (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_READ_ALL_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_read_all) (fh, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_READ_ALL_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_write_Fortran_Wrapper (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_WRITE_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_write) (fh, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_WRITE_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_write_all_Fortran_Wrapper (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_WRITE_ALL_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_write_all) (fh, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_WRITE_ALL_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_read_at_Fortran_Wrapper (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_READ_AT_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_read_at) (fh, offset, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_READ_AT_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_read_at_all_Fortran_Wrapper (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_READ_AT_ALL_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_read_at_all) (fh, offset, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_READ_AT_ALL_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_write_at_Fortran_Wrapper (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_WRITE_AT_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_write_at) (fh, offset, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_WRITE_AT_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_File_write_at_all_Fortran_Wrapper (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)
{
    int size, ret;

    CtoF77 (pmpi_type_size) (datatype, &size, &ret);
    MPI_CHECK(ret, pmpi_type_size);

    TRACE_MPIEVENT (LAST_READ_TIME, MPI_FILE_WRITE_AT_ALL_EV, EVT_BEGIN, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);
    CtoF77 (pmpi_file_write_at_all) (fh, offset, buf, count, datatype, status, ierror);
    TRACE_MPIEVENT (TIME, MPI_FILE_WRITE_AT_ALL_EV, EVT_END, EMPTY, *count * size, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

#endif /* MPI_SUPPORTS_MPI_IO */


#endif /* defined(FORTRAN_SYMBOLS) */

