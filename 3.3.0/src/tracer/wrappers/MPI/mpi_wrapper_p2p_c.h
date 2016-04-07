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

#ifndef MPI_WRAPPER_P2P_C_DEFINED
#define MPI_WRAPPER_P2P_C_DEFINED

#if !defined(MPI_SUPPORT)
# error "This should not be included"
#endif

#include <config.h>

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif

#if defined(C_SYMBOLS)

/* C Wrappers */

int MPI_Bsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
  int tag, MPI_Comm comm);

int MPI_Ssend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
  int tag, MPI_Comm comm);

int MPI_Rsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
  int tag, MPI_Comm comm);

int MPI_Send_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
  int tag, MPI_Comm comm);

int MPI_Ibsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
  int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Isend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Issend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
  int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Irsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int dest, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Recv_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int source, int tag, MPI_Comm comm, MPI_Status *status);

int MPI_Irecv_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int source, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Probe_C_Wrapper (int source, int tag, MPI_Comm comm, MPI_Status *status);

int MPI_Iprobe_C_Wrapper (int source, int tag, MPI_Comm comm, int *flag,
  MPI_Status *status);

int MPI_Test_C_Wrapper (MPI_Request * request, int *flag, MPI_Status *status);

int MPI_Testall_C_Wrapper (int count, MPI_Request* requests, int *flag, MPI_Status *statuses);

int MPI_Testany_C_Wrapper (int count, MPI_Request* requests, int *index,
  int *flag, MPI_Status *status);

int MPI_Testsome_C_Wrapper (int incount, MPI_Request* requests, int *outcount,
  int *indices, MPI_Status *statuses);

int MPI_Wait_C_Wrapper (MPI_Request * request, MPI_Status *status);

int MPI_Waitall_C_Wrapper (int count, MPI_Request* requests, MPI_Status *statuses);

int MPI_Waitany_C_Wrapper (int count, MPI_Request* requests, int *index,
  MPI_Status *status);

int MPI_Waitsome_C_Wrapper (int incount, MPI_Request* requests, int *outcount,
  int *indices, MPI_Status *statuses);

int MPI_Recv_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int source, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Send_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int dest, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Bsend_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int dest, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Rsend_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int dest, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Ssend_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
  int dest, int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Sendrecv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
  int source, int recvtag, MPI_Comm comm, MPI_Status * status);

int MPI_Sendrecv_replace_C_Wrapper (void *buf, int count, MPI_Datatype type,
  int dest, int sendtag, int source, int recvtag, MPI_Comm comm,
  MPI_Status * status);

#endif /* C_SYMBOLS */

#endif /* MPI_WRAPPER_P2P_C_DEFINED */

