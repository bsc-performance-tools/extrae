/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/mpitrace/fusion/trunk/src/tracer/wrappers/MPI/pacx_wrapper.h $
 | 
 | @last_commit: $Date: 2009-10-29 13:06:27 +0100 (dj, 29 oct 2009) $
 | @version:     $Revision: 15 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef PACX_WRAPPER_DEFINED
#define PACX_WRAPPER_DEFINED

#if !defined(PACX_SUPPORT)
# error "This should not be included"
#endif

# include <config.h>

#ifdef HAVE_PACX_H
# include <pacx.h>
#endif
/* #include "mpif.h" */
#include "defines.h"

#include "wrapper.h"

void gettopology (void);
void configure_PACX_vars (void);
unsigned long long CalculateNumOpsForPeriod (unsigned long long wannaPeriod, unsigned long long NumOfGlobals, unsigned long long runnedPeriod);
void CheckControlFile (void);
void CheckGlobalOpsTracingIntervals (void);
void CtoF77 (mptrace_set_mlp_rank) (int *rank, int *size);
void remove_file_list (void);
extern int mpit_gathering_enabled;

void OMPItrace_network_counters_Wrapper (void);
void OMPItrace_network_routes_Wrapper (int pacx_rank);
void OMPItrace_tracing_tasks_Wrapper (int from, int to);

/* Fortran Wrappers */

#if defined(FORTRAN_SYMBOLS)

#if defined(DYNINST_MODULE) && \
    defined(PACX_C_CONTAINS_FORTRAN_PACX_INIT) && \
    defined(C_SYMBOLS) && defined(FORTRAN_SYMBOLS)
void PPACX_Init_Wrapper (PACX_Fint *ierror);
#endif

void PPACX_Init_thread_Wrapper (PACX_Fint *required, PACX_Fint *provided, PACX_Fint *ierror);

void PPACX_Finalize_Wrapper (PACX_Fint *ierror);

void PPACX_BSend_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_SSend_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_RSend_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Send_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_IBSend_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_ISend_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_ISSend_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_IRSend_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_Recv_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *status, 
   PACX_Fint *ierror);

void PPACX_IRecv_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_Reduce_Wrapper (void *sendbuf, void *recvbuf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *root, PACX_Fint *comm,
	PACX_Fint *ierror);

void PPACX_AllReduce_Wrapper (void *sendbuf, void *recvbuf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Probe_Wrapper (PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *status, PACX_Fint *ierror);

void PPACX_IProbe_Wrapper (PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm,
	PACX_Fint *flag, PACX_Fint *status, PACX_Fint *ierror);

void PPACX_Barrier_Wrapper (PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Cancel_Wrapper (PACX_Fint *request, PACX_Fint *ierror);

void PPACX_Test_Wrapper (PACX_Fint *request, PACX_Fint *flag, PACX_Fint *status,
	PACX_Fint *ierror);

void PPACX_Wait_Wrapper (PACX_Fint *request, PACX_Fint *status, PACX_Fint *ierror);

void PPACX_WaitAll_Wrapper (PACX_Fint * count, PACX_Fint array_of_requests[],
	PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS], PACX_Fint * ierror);

void PPACX_WaitAny_Wrapper (PACX_Fint *count, PACX_Fint array_of_requests[],
	PACX_Fint *index, PACX_Fint *status, PACX_Fint *ierror);

void PPACX_WaitSome_Wrapper (PACX_Fint *incount, PACX_Fint array_of_requests[],
	PACX_Fint *outcount, PACX_Fint array_of_indices[],
	PACX_Fint array_of_statuses[][SIZEOF_PACX_STATUS], PACX_Fint *ierror);

void PPACX_BCast_Wrapper (void *buffer, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_AllToAll_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_AllToAllV_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sdispls, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount,
   PACX_Fint *rdispls, PACX_Fint *recvtype,	PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Allgather_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Allgatherv_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs,
	PACX_Fint *recvtype, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Gather_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_GatherV_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *displs,
	PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Scatter_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount, PACX_Fint *recvtype,
	PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_ScatterV_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *displs, PACX_Fint *sendtype, void *recvbuf, PACX_Fint *recvcount,
	PACX_Fint *recvtype, PACX_Fint *root, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Comm_Rank_Wrapper (PACX_Fint *comm, PACX_Fint *rank, PACX_Fint *ierror);

void PPACX_Comm_Size_Wrapper (PACX_Fint *comm, PACX_Fint *size, PACX_Fint *ierror);

void PPACX_Comm_Create_Wrapper (PACX_Fint *comm, PACX_Fint *group,
	PACX_Fint *newcomm, PACX_Fint *ierror);

void PPACX_Comm_Dup_Wrapper (PACX_Fint *comm, PACX_Fint *newcomm,
	PACX_Fint *ierror);

void PPACX_Comm_Split_Wrapper (PACX_Fint *comm, PACX_Fint *color, PACX_Fint *key,
	PACX_Fint *newcomm, PACX_Fint *ierror);

void PPACX_Reduce_Scatter_Wrapper (void *sendbuf, void *recvbuf,
	PACX_Fint *recvcounts, PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm,
	PACX_Fint *ierror);

void PPACX_Scan_Wrapper (void *sendbuf, void *recvbuf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Fint *op, PACX_Fint *comm, PACX_Fint *ierror);

void PPACX_Start_Wrapper (PACX_Fint *request, PACX_Fint *ierror);

void PPACX_Startall_Wrapper (PACX_Fint *count, PACX_Fint array_of_requests[],
	PACX_Fint *ierror);

void PPACX_Request_free_Wrapper (PACX_Fint *request, PACX_Fint *ierror);

void PPACX_Recv_init_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *source, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_Send_init_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_Bsend_init_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_Rsend_init_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_Ssend_init_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *datatype,
	PACX_Fint *dest, PACX_Fint *tag, PACX_Fint *comm, PACX_Fint *request,
	PACX_Fint *ierror);

void PPACX_Cart_sub_Wrapper (PACX_Fint *comm, PACX_Fint *remain_dims,
	PACX_Fint *comm_new, PACX_Fint *ierror);

void PPACX_Cart_create_Wrapper (PACX_Fint *comm_old, PACX_Fint *ndims,
	PACX_Fint *dims, PACX_Fint *periods, PACX_Fint *reorder, PACX_Fint *comm_cart,
	PACX_Fint *ierror);

void PACX_Sendrecv_Fortran_Wrapper (void *sendbuf, PACX_Fint *sendcount,
	PACX_Fint *sendtype, PACX_Fint *dest, PACX_Fint *sendtag, void *recvbuf,
	PACX_Fint *recvcount, PACX_Fint *recvtype, PACX_Fint *source, PACX_Fint *recvtag,
	PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr);

void PACX_Sendrecv_replace_Fortran_Wrapper (void *buf, PACX_Fint *count, PACX_Fint *type,
	PACX_Fint *dest, PACX_Fint *sendtag, PACX_Fint *source, PACX_Fint *recvtag,
	PACX_Fint *comm, PACX_Fint *status, PACX_Fint *ierr);

#if defined(PACX_SUPPORTS_PACX_IO)

void PPACX_File_open_Fortran_Wrapper (PACX_Fint *comm, char *filename,
	PACX_Fint *amode, PACX_Fint *info, PACX_File *fh, PACX_Fint *len);

void PPACX_File_close_Fortran_Wrapper (PACX_File *fh, PACX_Fint *ierror);

void PPACX_File_read_Fortran_Wrapper (PACX_File *fh, void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror);

void PPACX_File_read_all_Fortran_Wrapper (PACX_File *fh, void *buf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror);

void PPACX_File_write_Fortran_Wrapper (PACX_File *fh, void *buf, PACX_Fint *count,
	PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror);

void PPACX_File_write_all_Fortran_Wrapper (PACX_File *fh, void *buf,
	PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status, PACX_Fint *ierror);

void PPACX_File_read_at_Fortran_Wrapper (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror);

void PPACX_File_read_at_all_Fortran_Wrapper (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
   PACX_Fint *ierror);

void PPACX_File_write_at_Fortran_Wrapper (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror);

void PPACX_File_write_at_all_Fortran_Wrapper (PACX_File *fh, PACX_Offset *offset,
	void* buf, PACX_Fint *count, PACX_Fint *datatype, PACX_Status *status,
	PACX_Fint *ierror);

#endif /* PACX_SUPPORTS_PACX_IO */

#endif /* defined(FORTRAN_SYMBOLS) */

/* C Wrappers */

#if defined(C_SYMBOLS)

int PACX_Init_C_Wrapper (int *argc, char ***argv);

int PACX_Init_thread_C_Wrapper (int *argc, char ***argv, int required, int *provided);

int PACX_Finalize_C_Wrapper (void);

int PACX_Bsend_C_Wrapper (void *buf, int count, PACX_Datatype datatype, int dest,
  int tag, PACX_Comm comm);

int PACX_Ssend_C_Wrapper (void *buf, int count, PACX_Datatype datatype, int dest,
  int tag, PACX_Comm comm);

int PACX_Rsend_C_Wrapper (void *buf, int count, PACX_Datatype datatype, int dest,
  int tag, PACX_Comm comm);

int PACX_Send_C_Wrapper (void *buf, int count, PACX_Datatype datatype, int dest,
  int tag, PACX_Comm comm);

int PACX_Ibsend_C_Wrapper (void *buf, int count, PACX_Datatype datatype, int dest,
  int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Isend_C_Wrapper (void *buf, int count, PACX_Datatype datatype, int dest, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Issend_C_Wrapper (void *buf, int count, PACX_Datatype datatype, int dest,
  int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Irsend_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int dest, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Recv_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int source, int tag, PACX_Comm comm, PACX_Status *status);

int PACX_Irecv_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int source, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Reduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  PACX_Datatype datatype, PACX_Op op, int root, PACX_Comm comm);

int PACX_Allreduce_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  PACX_Datatype datatype, PACX_Op op, PACX_Comm comm);

int PACX_Probe_C_Wrapper (int source, int tag, PACX_Comm comm, PACX_Status *status);

int PACX_Iprobe_C_Wrapper (int source, int tag, PACX_Comm comm, int *flag,
  PACX_Status *status);

int PACX_Iprobe_C_Wrapper (int source, int tag, PACX_Comm comm, int *flag,
  PACX_Status *status);

int PACX_Barrier_C_Wrapper (PACX_Comm comm);

int PACX_Cancel_C_Wrapper (PACX_Request * request);

int PACX_Test_C_Wrapper (PACX_Request * request, int *flag, PACX_Status *status);

int PACX_Wait_C_Wrapper (PACX_Request * request, PACX_Status *status);

int PACX_Waitall_C_Wrapper (int count, PACX_Request* requests, PACX_Status *statuses);

int PACX_Waitany_C_Wrapper (int count, PACX_Request* requests, int *index,
  PACX_Status *status);

int PACX_Waitsome_C_Wrapper (int incount, PACX_Request* requests, int *outcount,
  int *indices, PACX_Status *statuses);

int PACX_BCast_C_Wrapper (void *buffer, int count, PACX_Datatype datatype,
  int root, PACX_Comm comm);

int PACX_Alltoall_C_Wrapper (void *sendbuf, int sendcount, PACX_Datatype sendtype,
  void *recvbuf, int recvcount, PACX_Datatype recvtype, PACX_Comm comm);

int PACX_Alltoallv_C_Wrapper (void *sendbuf, int *sendcounts, int *sdispls,
  PACX_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, PACX_Datatype recvtype, PACX_Comm comm);

int PACX_Allgather_C_Wrapper (void *sendbuf, int sendcount, PACX_Datatype sendtype,
  void *recvbuf, int recvcount, PACX_Datatype recvtype, PACX_Comm comm);

int PACX_Allgatherv_C_Wrapper (void *sendbuf, int sendcount, PACX_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, PACX_Datatype recvtype, PACX_Comm comm);

int PACX_Gather_C_Wrapper (void *sendbuf, int sendcount, PACX_Datatype sendtype,
  void *recvbuf, int recvcount, PACX_Datatype recvtype, int root, PACX_Comm comm);

int PACX_Gatherv_C_Wrapper (void *sendbuf, int sendcount, PACX_Datatype sendtype,
  void *recvbuf, int *recvcounts, int *displs, PACX_Datatype recvtype, int root, PACX_Comm comm);

int PACX_Scatter_C_Wrapper (void *sendbuf, int sendcount, PACX_Datatype sendtype,
  void *recvbuf, int recvcount, PACX_Datatype recvtype, int root, PACX_Comm comm);

int PACX_Scatterv_C_Wrapper (void *sendbuf, int *sendcounts, int *displs,
  PACX_Datatype sendtype, void *recvbuf, int recvcount, PACX_Datatype recvtype, int root, PACX_Comm comm);

int PACX_Comm_rank_C_Wrapper (PACX_Comm comm, int *rank);

int PACX_Comm_size_C_Wrapper (PACX_Comm comm, int *size);

int PACX_Comm_create_C_Wrapper (PACX_Comm comm, PACX_Group group, PACX_Comm *newcomm);

int PACX_Comm_dup_C_Wrapper (PACX_Comm comm, PACX_Comm *newcomm);

int PACX_Comm_split_C_Wrapper (PACX_Comm comm, int color, int key, PACX_Comm *newcomm);

int PACX_Reduce_Scatter_C_Wrapper (void *sendbuf, void *recvbuf, int *recvcounts,
  PACX_Datatype datatype, PACX_Op op, PACX_Comm comm);

int PACX_Scan_C_Wrapper (void *sendbuf, void *recvbuf, int count,
  PACX_Datatype datatype, PACX_Op op, PACX_Comm comm);

int PACX_Cart_create_C_Wrapper (PACX_Comm comm_old, int ndims, int *dims,
  int *periods, int reorder, PACX_Comm *comm_cart);

int PACX_Cart_sub_C_Wrapper (PACX_Comm comm, int *remain_dims, PACX_Comm *comm_new);

int PACX_Start_C_Wrapper (PACX_Request* request);

int PACX_Startall_C_Wrapper (int count, PACX_Request* requests);

int PACX_Request_free_C_Wrapper (PACX_Request * request);

int PACX_Recv_init_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int source, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Send_init_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int dest, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Bsend_init_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int dest, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Rsend_init_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int dest, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Ssend_init_C_Wrapper (void *buf, int count, PACX_Datatype datatype,
  int dest, int tag, PACX_Comm comm, PACX_Request * request);

int PACX_Sendrecv_C_Wrapper (void *sendbuf, int sendcount, PACX_Datatype sendtype,
  int dest, int sendtag, void *recvbuf, int recvcount, PACX_Datatype recvtype,
  int source, int recvtag, PACX_Comm comm, PACX_Status * status);

int PACX_Sendrecv_replace_C_Wrapper (void *buf, int count, PACX_Datatype type,
  int dest, int sendtag, int source, int recvtag, PACX_Comm comm,
  PACX_Status * status);

#if 0 /* defined(PACX_SUPPORTS_PACX_IO) */

int PACX_File_open_C_Wrapper (PACX_Comm comm, char *filename, int amode,
  PACX_Info info, PACX_File *fh);

int PACX_File_close_C_Wrapper (PACX_File *fh);

int PACX_File_read_C_Wrapper (PACX_File fh, void *buf, int count,
  PACX_Datatype datatype, PACX_Status *status);

int PACX_File_read_all_C_Wrapper (PACX_File fh, void *buf, int count,
  PACX_Datatype datatype, PACX_Status *status);

int PACX_File_write_C_Wrapper (PACX_File fh, void *buf, int count,
	PACX_Datatype datatype, PACX_Status *status);

int PACX_File_write_all_C_Wrapper (PACX_File fh, void *buf, int count,
  PACX_Datatype datatype, PACX_Status *status);

int PACX_File_read_at_C_Wrapper (PACX_File fh, PACX_Offset offset, void *buf, 
  int count, PACX_Datatype datatype, PACX_Status *status);

int PACX_File_read_at_all_C_Wrapper (PACX_File fh, PACX_Offset offset, void *buf,
  int count, PACX_Datatype datatype, PACX_Status *status);

int PACX_File_write_at_C_Wrapper (PACX_File fh, PACX_Offset offset, void *buf,
  int count, PACX_Datatype datatype, PACX_Status *status);

int PACX_File_write_at_all_C_Wrapper (PACX_File fh, PACX_Offset offset, void *buf,
  int count, PACX_Datatype datatype, PACX_Status *status);

#endif /* PACX_SUPPORTS_PACX_IO */

int PACX_Type_size (PACX_Datatype datatype, int *size);

#endif /* defined(C_SYMBOLS) */

#endif /* PACX_WRAPPER_DEFINED */

