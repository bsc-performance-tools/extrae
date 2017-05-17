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

#ifndef MPI_WRAPPER_COLLECTIVES_C_DEFINED
#define MPI_WRAPPER_COLLECTIVES_C_DEFINED

#if !defined(MPI_SUPPORT)
# error "This should not be included"
#endif

#include <config.h>

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif

/* C Wrappers */

#if defined(C_SYMBOLS)

int MPI_Reduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

int MPI_Allreduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int MPI_Barrier_C_Wrapper (MPI_Comm comm);

int MPI_BCast_C_Wrapper (void *buffer, int count, MPI_Datatype datatype,
  int root, MPI_Comm comm);

int MPI_Alltoall_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
  MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Allgather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Allgatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Gather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

int MPI_Gatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);

int MPI_Scatter_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

int MPI_Scatterv_C_Wrapper (void *sendbuf, int *sendcounts, int *displs,
  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

int MPI_Reduce_Scatter_C_Wrapper (void *sendbuf, void *recvbuf, int *recvcounts,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int MPI_Scan_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
  
int MPI_Reduce_Scatter_Block_C_Wrapper (void *sendbuf, void *recvbuf, int recvcount,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
  
int MPI_Alltoallw_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
	MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, int *rdispls,
    MPI_Datatype *recvtypes, MPI_Comm comm);



#if defined(MPI3)

int MPI_Ireduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *req);

int MPI_Iallreduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req);

int MPI_Ibarrier_C_Wrapper (MPI_Comm comm, MPI_Request *req);

int MPI_Ibcast_C_Wrapper (void *buffer, int count, MPI_Datatype datatype,
  int root, MPI_Comm comm, MPI_Request *req);

int MPI_Ialltoall_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req);

int MPI_Ialltoallv_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
  MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req);

int MPI_Iallgather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req);

int MPI_Iallgatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req);

int MPI_Igather_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *req);

int MPI_Igatherv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *req);

int MPI_Iscatter_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *req);

int MPI_Iscatterv_C_Wrapper (void *sendbuf, int *sendcounts, int *displs,
  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *req);

int MPI_Ireduce_Scatter_C_Wrapper (void *sendbuf, void *recvbuf, int *recvcounts,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req);

int MPI_Iscan_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req);
  
int MPI_Ireduce_Scatter_Block_C_Wrapper (void *sendbuf, void *recvbuf, int recvcount, 
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req);

int MPI_Ialltoallw_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
	MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, int *rdispls,
    MPI_Datatype *recvtypes, MPI_Comm comm, MPI_Request *req);

#endif /* MPI3 */

#endif /* C_SYMBOLS */

#endif /* MPI_WRAPPER_COLLECTIVES_C_DEFINED */

