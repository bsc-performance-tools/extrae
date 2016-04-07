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

#ifndef MPI_WRAPPER_IO_C_DEFINED
#define MPI_WRAPPER_IO_C_DEFINED

#if !defined(MPI_SUPPORT)
# error "This should not be included"
#endif

#include <config.h>

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif

/* C Wrappers */

#if defined(C_SYMBOLS)

#if MPI_SUPPORTS_MPI_IO

int MPI_File_open_C_Wrapper (MPI_Comm comm, char *filename, int amode,
  MPI_Info info, MPI_File *fh);

int MPI_File_close_C_Wrapper (MPI_File *fh);

int MPI_File_read_C_Wrapper (MPI_File fh, void *buf, int count,
  MPI_Datatype datatype, MPI_Status *status);

int MPI_File_read_all_C_Wrapper (MPI_File fh, void *buf, int count,
  MPI_Datatype datatype, MPI_Status *status);

int MPI_File_write_C_Wrapper (MPI_File fh, void *buf, int count,
	MPI_Datatype datatype, MPI_Status *status);

int MPI_File_write_all_C_Wrapper (MPI_File fh, void *buf, int count,
  MPI_Datatype datatype, MPI_Status *status);

int MPI_File_read_at_C_Wrapper (MPI_File fh, MPI_Offset offset, void *buf, 
  int count, MPI_Datatype datatype, MPI_Status *status);

int MPI_File_read_at_all_C_Wrapper (MPI_File fh, MPI_Offset offset, void *buf,
  int count, MPI_Datatype datatype, MPI_Status *status);

int MPI_File_write_at_C_Wrapper (MPI_File fh, MPI_Offset offset, void *buf,
  int count, MPI_Datatype datatype, MPI_Status *status);

int MPI_File_write_at_all_C_Wrapper (MPI_File fh, MPI_Offset offset, void *buf,
  int count, MPI_Datatype datatype, MPI_Status *status);

#endif /* MPI_SUPPORTS_MPI_IO */

#endif /* C_SYMBOLS */

#endif /* MPI_WRAPPER_DEFINED */

