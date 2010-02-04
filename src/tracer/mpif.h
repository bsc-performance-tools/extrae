/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

#ifndef _MPI_FORTRAN_H_INCLUDED_
#define _MPI_FORTRAN_H_INCLUDED_

#if defined(FORTRAN_SYMBOLS)

#include <config.h>

#include "defines.h"

/* MPI Fortran interface */

void CtoF77 (mpi_init) (MPI_Fint *ierror);

void CtoF77 (mpi_finalize) (MPI_Fint *ierror);

void CtoF77 (mpi_get_processor_name) (char *, int *, int *);

void CtoF77 (mpi_type_size) (MPI_Fint *datatype, MPI_Fint *size,
	MPI_Fint *ret);

void CtoF77 (mpi_get_count) (MPI_Fint *status, MPI_Fint *datatype, 
	MPI_Fint *recved_count, MPI_Fint *ret);

void CtoF77 (mpi_test_cancelled) (MPI_Fint *status, MPI_Fint *cancelled,
	MPI_Fint *ret);

void CtoF77 (mpi_bsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_ssend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_rsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_send) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_ibsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_isend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_issend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_irsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_recv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, 
   MPI_Fint *ierror);

void CtoF77 (mpi_irecv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_reduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *ierror);

void CtoF77 (mpi_reduce_scatter) (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror);

void CtoF77 (mpi_allreduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror);
   
void CtoF77 (mpi_probe) (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *status, MPI_Fint *ierror);
   
void CtoF77 (mpi_iprobe) (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror);

void CtoF77 (mpi_barrier) (MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_cancel) (MPI_Fint *request, MPI_Fint *ierror);

void CtoF77 (mpi_test) (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
	MPI_Fint *ierror);

void CtoF77 (mpi_wait) (MPI_Fint *request, MPI_Fint *status, MPI_Fint *ierror);

void CtoF77 (mpi_waitall) (MPI_Fint * count, MPI_Fint array_of_requests[],
  MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror);

void CtoF77 (mpi_waitany) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror);
   
void CtoF77 (mpi_waitsome) (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror);

void CtoF77 (mpi_bcast) (void *buffer, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_alltoall) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_alltoallv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
   MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_allgather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_allgatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_gather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_gatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_scatter) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_scatterv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_comm_rank) (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *ierror);

void CtoF77 (mpi_comm_size) (MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierror);

void CtoF77 (mpi_comm_create) (MPI_Fint *comm, MPI_Fint *group,
	MPI_Fint *newcomm, MPI_Fint *ierror);

void CtoF77 (mpi_comm_dup) (MPI_Fint *comm, MPI_Fint *newcomm,
	MPI_Fint *ierror);

void CtoF77 (mpi_comm_split) (MPI_Fint *comm, MPI_Fint *color, MPI_Fint *key,
	MPI_Fint *newcomm, MPI_Fint *ierror);

void CtoF77 (mpi_comm_group) (MPI_Fint *com, MPI_Fint *grup, MPI_Fint *ierror);

void CtoF77 (mpi_comm_test_inter) (MPI_Fint *com, MPI_Fint *inter,
	MPI_Fint *ret);

void CtoF77 (mpi_comm_remote_group) (MPI_Fint *comm, MPI_Fint *group,
	MPI_Fint *ret);

void CtoF77 (mpi_group_size) (MPI_Fint *group, MPI_Fint *num_tasks,
	MPI_Fint *ierror);

void CtoF77 (mpi_group_free) (MPI_Fint *group, MPI_Fint *ret);

void CtoF77 (mpi_group_translate_ranks) (MPI_Fint *group, MPI_Fint *cnt,
	MPI_Fint *dest, MPI_Fint *other_group, MPI_Fint *receiver, MPI_Fint *ret);

void CtoF77 (mpi_cart_create) (MPI_Fint *comm_old, MPI_Fint *ndims,
	MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart,
	MPI_Fint *ierror);

void CtoF77 (mpi_cart_sub) (MPI_Fint *comm, MPI_Fint *remain_dims,
	MPI_Fint *comm_new, MPI_Fint *ierror);

void CtoF77 (mpi_start) (MPI_Fint *request, MPI_Fint *ierror);

void CtoF77 (mpi_startall) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *ierror);

void CtoF77 (mpi_request_free) (MPI_Fint *request, MPI_Fint *ierror);

void CtoF77 (mpi_recv_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);
   
void CtoF77 (mpi_send_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_bsend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_rsend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_ssend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (mpi_scan) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (mpi_sendrecv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr);
   
void CtoF77 (mpi_sendrecv_replace) (void *buf, MPI_Fint *count, MPI_Fint *type,
	MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr);

void CtoF77 (mpi_file_open) (MPI_Fint *comm, char *filename, MPI_Fint *amode,
	MPI_Fint *info, MPI_File *fh, MPI_Fint *len);

void CtoF77 (mpi_file_close) (MPI_File *fh, MPI_Fint *ierror);

void CtoF77 (mpi_file_read) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);
   
void CtoF77 (mpi_file_read_all) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (mpi_file_write) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (mpi_file_write_all) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (mpi_file_read_at) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (mpi_file_read_at_all) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (mpi_file_write_at) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (mpi_file_write_at_all) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

/* PMPI Fortran interface */
void CtoF77 (pmpi_init) (MPI_Fint *ierror);

void CtoF77 (pmpi_finalize) (MPI_Fint *ierror);

void CtoF77 (pmpi_get_processor_name) (char *, int *, int *);

void CtoF77 (pmpi_type_size) (MPI_Fint *datatype, MPI_Fint *size,
	MPI_Fint *ret);

void CtoF77 (pmpi_get_count) (MPI_Fint *status, MPI_Fint *datatype, 
	MPI_Fint *recved_count, MPI_Fint *ret);

void CtoF77 (pmpi_test_cancelled) (MPI_Fint *status, MPI_Fint *cancelled,
	MPI_Fint *ret);

void CtoF77 (pmpi_bsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_ssend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_rsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_send) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_ibsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_isend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_issend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_irsend) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_recv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, 
   MPI_Fint *ierror);

void CtoF77 (pmpi_irecv) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_reduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *ierror);

void CtoF77 (pmpi_reduce_scatter) (void *sendbuf, void *recvbuf,
	MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror);

void CtoF77 (pmpi_allreduce) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror);
   
void CtoF77 (pmpi_probe) (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *status, MPI_Fint *ierror);
   
void CtoF77 (pmpi_iprobe) (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror);

void CtoF77 (pmpi_barrier) (MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_cancel) (MPI_Fint *request, MPI_Fint *ierror);

void CtoF77 (pmpi_test) (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
	MPI_Fint *ierror);

void CtoF77 (pmpi_wait) (MPI_Fint *request, MPI_Fint *status, MPI_Fint *ierror);

void CtoF77 (pmpi_waitall) (MPI_Fint * count, MPI_Fint array_of_requests[],
  MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror);

void CtoF77 (pmpi_waitany) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror);
   
void CtoF77 (pmpi_waitsome) (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror);

void CtoF77 (pmpi_bcast) (void *buffer, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_alltoall) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_alltoallv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
   MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_allgather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_allgatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_gather) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_gatherv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_scatter) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_scatterv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_comm_rank) (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *ierror);

void CtoF77 (pmpi_comm_size) (MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierror);

void CtoF77 (pmpi_comm_create) (MPI_Fint *comm, MPI_Fint *group,
	MPI_Fint *newcomm, MPI_Fint *ierror);

void CtoF77 (pmpi_comm_dup) (MPI_Fint *comm, MPI_Fint *newcomm,
	MPI_Fint *ierror);

void CtoF77 (pmpi_comm_split) (MPI_Fint *comm, MPI_Fint *color, MPI_Fint *key,
	MPI_Fint *newcomm, MPI_Fint *ierror);

void CtoF77 (pmpi_comm_group) (MPI_Fint *com, MPI_Fint *grup, MPI_Fint *ierror);

void CtoF77 (pmpi_comm_test_inter) (MPI_Fint *com, MPI_Fint *inter,
	MPI_Fint *ret);

void CtoF77 (pmpi_comm_remote_group) (MPI_Fint *comm, MPI_Fint *group,
	MPI_Fint *ret);

void CtoF77 (pmpi_group_size) (MPI_Fint *group, MPI_Fint *num_tasks,
	MPI_Fint *ierror);

void CtoF77 (pmpi_group_free) (MPI_Fint *group, MPI_Fint *ret);

void CtoF77 (pmpi_group_translate_ranks) (MPI_Fint *group, MPI_Fint *cnt,
	MPI_Fint *dest, MPI_Fint *other_group, MPI_Fint *receiver, MPI_Fint *ret);

void CtoF77 (pmpi_cart_create) (MPI_Fint *comm_old, MPI_Fint *ndims,
	MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart,
	MPI_Fint *ierror);

void CtoF77 (pmpi_cart_sub) (MPI_Fint *comm, MPI_Fint *remain_dims,
	MPI_Fint *comm_new, MPI_Fint *ierror);

void CtoF77 (pmpi_start) (MPI_Fint *request, MPI_Fint *ierror);

void CtoF77 (pmpi_startall) (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *ierror);

void CtoF77 (pmpi_request_free) (MPI_Fint *request, MPI_Fint *ierror);

void CtoF77 (pmpi_recv_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);
   
void CtoF77 (pmpi_send_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_bsend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_rsend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_ssend_init) (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void CtoF77 (pmpi_scan) (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror);

void CtoF77 (pmpi_sendrecv) (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr);
   
void CtoF77 (pmpi_sendrecv_replace) (void *buf, MPI_Fint *count, MPI_Fint *type,
	MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr);

void CtoF77 (pmpi_file_open) (MPI_Fint *comm, char *filename, MPI_Fint *amode,
	MPI_Fint *info, MPI_File *fh, MPI_Fint *len);

void CtoF77 (pmpi_file_close) (MPI_File *fh, MPI_Fint *ierror);

void CtoF77 (pmpi_file_read) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);
   
void CtoF77 (pmpi_file_read_all) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (pmpi_file_write) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (pmpi_file_write_all) (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (pmpi_file_read_at) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (pmpi_file_read_at_all) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (pmpi_file_write_at) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

void CtoF77 (pmpi_file_write_at_all) (MPI_File *fh, MPI_Offset *offset, void* buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror);

#endif /* defined(FORTRAN_SYMBOLS) */

#endif
