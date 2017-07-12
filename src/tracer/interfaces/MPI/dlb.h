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


#if !defined _DLB_H
#define _DLB_H

#define DLB(func, args...) if ( func ) func( args );


void DLB_MPI_Init_F_enter (MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Init_F_leave (void)__attribute__((weak));


void DLB_MPI_Init_thread_F_enter (MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Init_thread_F_leave (void)__attribute__((weak));


void DLB_MPI_Finalize_F_enter (MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Finalize_F_leave (void)__attribute__((weak));


void DLB_MPI_Request_get_status_F_enter (MPI_Fint *request, int *flag, MPI_Fint *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Request_get_status_F_leave (void)__attribute__((weak));


void DLB_MPI_Cancel_F_enter (MPI_Fint *request, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Cancel_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_rank_F_enter (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_rank_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_size_F_enter (MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_size_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_create_F_enter (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *newcomm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_create_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_free_F_enter (MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_free_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_dup_F_enter (MPI_Fint *comm, MPI_Fint *newcomm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_dup_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_split_F_enter (MPI_Fint *comm, MPI_Fint *color, MPI_Fint *key, MPI_Fint *newcomm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_split_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_spawn_F_enter (char *command, char *argv, MPI_Fint *maxprocs, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_spawn_F_leave (void)__attribute__((weak));


void DLB_MPI_Comm_spawn_multiple_F_enter (MPI_Fint *count, char *array_of_commands, char *array_of_argv, MPI_Fint *array_of_maxprocs, MPI_Fint *array_of_info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Comm_spawn_multiple_F_leave (void)__attribute__((weak));


void DLB_MPI_Cart_create_F_enter (MPI_Fint *comm_old, MPI_Fint *ndims, MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Cart_create_F_leave (void)__attribute__((weak));


void DLB_MPI_Cart_sub_F_enter (MPI_Fint *comm, MPI_Fint *remain_dims, MPI_Fint *comm_new, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Cart_sub_F_leave (void)__attribute__((weak));


void DLB_MPI_Intercomm_create_F_enter (MPI_Fint * local_comm, MPI_Fint *local_leader, MPI_Fint *peer_comm, MPI_Fint *remote_leader, MPI_Fint *tag, MPI_Fint *new_intercomm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Intercomm_create_F_leave (void)__attribute__((weak));


void DLB_MPI_Intercomm_merge_F_enter (MPI_Fint *intercomm, MPI_Fint *high, MPI_Fint *newintracomm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Intercomm_merge_F_leave (void)__attribute__((weak));


void DLB_MPI_Start_F_enter (MPI_Fint *request, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Start_F_leave (void)__attribute__((weak));


void DLB_MPI_Startall_F_enter (MPI_Fint *count, MPI_Fint *array_of_requests, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Startall_F_leave (void)__attribute__((weak));


void DLB_MPI_Request_free_F_enter (MPI_Fint *request, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Request_free_F_leave (void)__attribute__((weak));


void DLB_MPI_Init_enter (int *argc, char ***argv)__attribute__((weak));
void DLB_MPI_Init_leave (void)__attribute__((weak));


void DLB_MPI_Init_thread_enter (int *argc, char ***argv, int required, int *provided)__attribute__((weak));
void DLB_MPI_Init_thread_leave (void)__attribute__((weak));


void DLB_MPI_Finalize_enter (void)__attribute__((weak));
void DLB_MPI_Finalize_leave (void)__attribute__((weak));


void DLB_MPI_Request_get_status_enter (MPI_Request request, int *flag, MPI_Status *status)__attribute__((weak));
void DLB_MPI_Request_get_status_leave (void)__attribute__((weak));


void DLB_MPI_Cancel_enter (MPI_Request *request)__attribute__((weak));
void DLB_MPI_Cancel_leave (void)__attribute__((weak));


void DLB_MPI_Comm_rank_enter (MPI_Comm comm, int *rank)__attribute__((weak));
void DLB_MPI_Comm_rank_leave (void)__attribute__((weak));


void DLB_MPI_Comm_size_enter (MPI_Comm comm, int *size)__attribute__((weak));
void DLB_MPI_Comm_size_leave (void)__attribute__((weak));


void DLB_MPI_Comm_create_enter (MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)__attribute__((weak));
void DLB_MPI_Comm_create_leave (void)__attribute__((weak));


void DLB_MPI_Comm_free_enter (MPI_Comm *comm)__attribute__((weak));
void DLB_MPI_Comm_free_leave (void)__attribute__((weak));


void DLB_MPI_Comm_dup_enter (MPI_Comm comm, MPI_Comm *newcomm)__attribute__((weak));
void DLB_MPI_Comm_dup_leave (void)__attribute__((weak));


void DLB_MPI_Comm_split_enter (MPI_Comm comm, int color, int key, MPI_Comm *newcomm)__attribute__((weak));
void DLB_MPI_Comm_split_leave (void)__attribute__((weak));


void DLB_MPI_Comm_spawn_enter (
  MPI3_CONST char *command,
  char           **argv,
  int              maxprocs,
  MPI_Info         info,
  int              root,
  MPI_Comm         comm,
  MPI_Comm        *intercomm,
  int             *array_of_errcodes)__attribute__((weak));
void DLB_MPI_Comm_spawn_leave (void)__attribute__((weak));
 
 
void DLB_MPI_Comm_spawn_multiple_enter (
  int                 count,
  char               *array_of_commands[],
  char              **array_of_argv[],
  MPI3_CONST int      array_of_maxprocs[],
  MPI3_CONST MPI_Info array_of_info[],
  int                 root,
  MPI_Comm            comm,
  MPI_Comm           *intercomm,
  int                 array_of_errcodes[])__attribute__((weak));
void DLB_MPI_Comm_spawn_multiple_leave (void)__attribute__((weak));
 
 
void DLB_MPI_Cart_create_enter (MPI_Comm comm_old, int ndims, MPI3_CONST int *dims,
	MPI3_CONST int *periods, int reorder, MPI_Comm *comm_cart)__attribute__((weak));
void DLB_MPI_Cart_create_leave (void)__attribute__((weak));


void DLB_MPI_Cart_sub_enter (MPI_Comm comm, MPI3_CONST int *remain_dims,
	MPI_Comm *comm_new)__attribute__((weak));
void DLB_MPI_Cart_sub_leave (void)__attribute__((weak));


void DLB_MPI_Intercomm_create_enter (MPI_Comm local_comm, int local_leader,
	MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm)__attribute__((weak));
void DLB_MPI_Intercomm_create_leave (void)__attribute__((weak));


void DLB_MPI_Intercomm_merge_enter (MPI_Comm intercomm, int high,
	MPI_Comm *newintracomm)__attribute__((weak));
void DLB_MPI_Intercomm_merge_leave (void)__attribute__((weak));


void DLB_MPI_Start_enter (MPI_Request *request)__attribute__((weak));
void DLB_MPI_Start_leave (void)__attribute__((weak));


void DLB_MPI_Startall_enter (int count, MPI_Request *requests)__attribute__((weak));
void DLB_MPI_Startall_leave (void)__attribute__((weak));


void DLB_MPI_Request_free_enter (MPI_Request *request)__attribute__((weak));
void DLB_MPI_Request_free_leave (void)__attribute__((weak));


/******************************************************************************
 *** P2P
 ******************************************************************************/

/***  Fortran  ***/

void DLB_MPI_Bsend_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Bsend_F_leave (void)__attribute__((weak));


void DLB_MPI_Ssend_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ssend_F_leave (void)__attribute__((weak));


void DLB_MPI_Rsend_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Rsend_F_leave (void)__attribute__((weak));


void DLB_MPI_Send_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Send_F_leave (void)__attribute__((weak));


void DLB_MPI_Ibsend_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ibsend_F_leave (void)__attribute__((weak));


void DLB_MPI_Isend_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Isend_F_leave (void)__attribute__((weak));


void DLB_MPI_Issend_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Issend_F_leave (void)__attribute__((weak));


void DLB_MPI_Irsend_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Irsend_F_leave (void)__attribute__((weak));


void DLB_MPI_Recv_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Recv_F_leave (void)__attribute__((weak));


void DLB_MPI_Irecv_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Irecv_F_leave (void)__attribute__((weak));


void DLB_MPI_Probe_F_enter (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Probe_F_leave (void)__attribute__((weak));


void DLB_MPI_Iprobe_F_enter (MPI_Fint *source, MPI_Fint *tag,
	MPI_Fint *comm, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Iprobe_F_leave (void)__attribute__((weak));


void DLB_MPI_Test_F_enter (MPI_Fint *request, MPI_Fint *flag,
	MPI_Fint *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Test_F_leave (void)__attribute__((weak));


void DLB_MPI_Testall_F_enter (MPI_Fint *count, MPI_Fint *array_of_requests,
	MPI_Fint *flag, MPI_Fint *array_of_statuses,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Testall_F_leave (void)__attribute__((weak));


void DLB_MPI_Testany_F_enter (MPI_Fint *count, MPI_Fint *array_of_requests,
	MPI_Fint *index, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Testany_F_leave (void)__attribute__((weak));


void DLB_MPI_Testsome_F_enter (MPI_Fint *incount,
	MPI_Fint *array_of_requests, MPI_Fint *outcount, MPI_Fint *array_of_indices,
	MPI_Fint *array_of_statuses, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Testsome_F_leave (void)__attribute__((weak));


void DLB_MPI_Wait_F_enter (MPI_Fint *request, MPI_Fint *status,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Wait_F_leave (void)__attribute__((weak));


void DLB_MPI_Waitall_F_enter (MPI_Fint *count,
	MPI_Fint *array_of_requests, MPI_Fint *array_of_statuses,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Waitall_F_leave (void)__attribute__((weak));


void DLB_MPI_Waitany_F_enter (MPI_Fint *count, MPI_Fint *array_of_requests,
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Waitany_F_leave (void)__attribute__((weak));


void DLB_MPI_Waitsome_F_enter (MPI_Fint *incount,
	MPI_Fint *array_of_requests, MPI_Fint *outcount, MPI_Fint *array_of_indices,
	MPI_Fint *array_of_statuses, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Waitsome_F_leave (void)__attribute__((weak));


void DLB_MPI_Recv_init_F_enter (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Recv_init_F_leave (void)__attribute__((weak));


void DLB_MPI_Send_init_F_enter (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Send_init_F_leave (void)__attribute__((weak));


void DLB_MPI_Bsend_init_F_enter (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Bsend_init_F_leave (void)__attribute__((weak));


void DLB_MPI_Rsend_init_F_enter (void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *request, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Rsend_init_F_leave (void)__attribute__((weak));


void DLB_MPI_Ssend_init_F_enter (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ssend_init_F_leave (void)__attribute__((weak));


void DLB_MPI_Sendrecv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr)__attribute__((weak));
void DLB_MPI_Sendrecv_F_leave (void)__attribute__((weak));


void DLB_MPI_Sendrecv_replace_F_enter (void *buf, MPI_Fint *count,
	MPI_Fint *type, MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source,
	MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr)__attribute__((weak));
void DLB_MPI_Sendrecv_replace_F_leave (void)__attribute__((weak));


/***  C  ***/

void DLB_MPI_Bsend_enter (MPI3_CONST void* buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Bsend_leave (void)__attribute__((weak));


void DLB_MPI_Ssend_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Ssend_leave (void)__attribute__((weak));


void DLB_MPI_Rsend_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Rsend_leave (void)__attribute__((weak));


void DLB_MPI_Send_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Send_leave (void)__attribute__((weak));


void DLB_MPI_Ibsend_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Ibsend_leave (void)__attribute__((weak));


void DLB_MPI_Isend_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Isend_leave (void)__attribute__((weak));


void DLB_MPI_Issend_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Issend_leave (void)__attribute__((weak));


void DLB_MPI_Irsend_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Irsend_leave (void)__attribute__((weak));


void DLB_MPI_Recv_enter (void* buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Status *status)__attribute__((weak));
void DLB_MPI_Recv_leave (void)__attribute__((weak));


void DLB_MPI_Irecv_enter (void* buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Irecv_leave (void)__attribute__((weak));


void DLB_MPI_Probe_enter (int source, int tag, MPI_Comm comm, MPI_Status *status)__attribute__((weak));
void DLB_MPI_Probe_leave (void)__attribute__((weak));


void DLB_MPI_Iprobe_enter (int source, int tag, MPI_Comm comm, int *flag,
	MPI_Status *status)__attribute__((weak));
void DLB_MPI_Iprobe_leave (void)__attribute__((weak));


void DLB_MPI_Test_enter (MPI_Request *request, int *flag, MPI_Status *status)__attribute__((weak));
void DLB_MPI_Test_leave (void)__attribute__((weak));


void DLB_MPI_Testall_enter (int count, MPI_Request *requests,
	int *flag, MPI_Status *statuses)__attribute__((weak));
void DLB_MPI_Testall_leave (void)__attribute__((weak));


void DLB_MPI_Testany_enter (int count, MPI_Request *requests, int *index,
	int *flag, MPI_Status *status)__attribute__((weak));
void DLB_MPI_Testany_leave (void)__attribute__((weak));


void DLB_MPI_Testsome_enter (int incount, MPI_Request * requests,
	int *outcount, int *indices, MPI_Status *statuses)__attribute__((weak));
void DLB_MPI_Testsome_leave (void)__attribute__((weak));


void DLB_MPI_Wait_enter (MPI_Request *request, MPI_Status *status)__attribute__((weak));
void DLB_MPI_Wait_leave (void)__attribute__((weak));


void DLB_MPI_Waitall_enter (int count, MPI_Request *requests, MPI_Status *statuses)__attribute__((weak));
void DLB_MPI_Waitall_leave (void)__attribute__((weak));


void DLB_MPI_Waitany_enter (int count, MPI_Request *requests, int *index,
	MPI_Status *status)__attribute__((weak));
void DLB_MPI_Waitany_leave (void)__attribute__((weak));


void DLB_MPI_Waitsome_enter (int incount, MPI_Request * requests,
	int *outcount, int *indices, MPI_Status *statuses)__attribute__((weak));
void DLB_MPI_Waitsome_leave (void)__attribute__((weak));


void DLB_MPI_Recv_init_enter (void *buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Recv_init_leave (void)__attribute__((weak));


void DLB_MPI_Send_init_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Send_init_leave (void)__attribute__((weak));


void DLB_MPI_Bsend_init_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Bsend_init_leave (void)__attribute__((weak));


void DLB_MPI_Rsend_init_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Rsend_init_leave (void)__attribute__((weak));


void DLB_MPI_Ssend_init_enter (MPI3_CONST void *buf, int count, MPI_Datatype datatype,
	int dest, int tag, MPI_Comm comm, MPI_Request *request)__attribute__((weak));
void DLB_MPI_Ssend_init_leave (void)__attribute__((weak));


void DLB_MPI_Sendrecv_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount,
	MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
	MPI_Status * status)__attribute__((weak));
void DLB_MPI_Sendrecv_leave (void)__attribute__((weak));


void DLB_MPI_Sendrecv_replace_enter (void *buf, int count, MPI_Datatype type,
	int dest, int sendtag, int source, int recvtag, MPI_Comm comm,
	MPI_Status* status)__attribute__((weak));
void DLB_MPI_Sendrecv_replace_leave (void)__attribute__((weak));

/******************************************************************************
 *** I/O
 ******************************************************************************/

/***  Fortran  ***/

void DLB_MPI_File_close_F_enter (MPI_File *fh, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_close_F_leave (void)__attribute__((weak));


void DLB_MPI_File_read_F_enter (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_read_F_leave (void)__attribute__((weak));


void DLB_MPI_File_read_all_F_enter(MPI_File *fh, void *buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_read_all_F_leave (void)__attribute__((weak));


void DLB_MPI_File_write_F_enter (MPI_File *fh, void *buf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_write_F_leave (void)__attribute__((weak));


void DLB_MPI_File_write_all_F_enter (MPI_File *fh, void *buf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_write_all_F_leave (void)__attribute__((weak));


void DLB_MPI_File_read_at_F_enter (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_read_at_F_leave (void)__attribute__((weak));


void DLB_MPI_File_read_at_all_F_enter (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_read_at_all_F_leave (void)__attribute__((weak));


void DLB_MPI_File_write_at_F_enter (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_write_at_F_leave (void)__attribute__((weak));


void DLB_MPI_File_write_at_all_F_enter (MPI_File *fh, MPI_Offset *offset,
	void* buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Status *status,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_File_write_at_all_F_leave (void)__attribute__((weak));
	
/***  C  ***/

void DLB_MPI_File_open_enter(MPI_Comm comm, MPI3_CONST char * filename, int amode,
	MPI_Info info, MPI_File *fh)__attribute__((weak));
void DLB_MPI_File_open_leave (void)__attribute__((weak));

void DLB_MPI_File_close_enter (MPI_File* fh)__attribute__((weak));
void DLB_MPI_File_close_leave (void)__attribute__((weak));


void DLB_MPI_File_read_enter (MPI_File fh, void* buf, int count,
	MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_read_leave (void)__attribute__((weak));


void DLB_MPI_File_read_all_enter (MPI_File fh, void* buf, int count,
	MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_read_all_leave (void)__attribute__((weak));


void DLB_MPI_File_write_enter (MPI_File fh, MPI3_CONST void * buf, int count,
	MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_write_leave (void)__attribute__((weak));


void DLB_MPI_File_write_all_enter (MPI_File fh, MPI3_CONST void* buf, int count, 
	MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_write_all_leave (void)__attribute__((weak));


void DLB_MPI_File_read_at_enter (MPI_File fh, MPI_Offset offset, void* buf,
	int count, MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_read_at_leave (void)__attribute__((weak));


void DLB_MPI_File_read_at_all_enter (MPI_File fh, MPI_Offset offset,
	void* buf, int count, MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_read_at_all_leave (void)__attribute__((weak));


void DLB_MPI_File_write_at_enter (MPI_File fh, MPI_Offset offset, MPI3_CONST void * buf,
	int count, MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_write_at_leave (void)__attribute__((weak));


void DLB_MPI_File_write_at_all_enter (MPI_File fh, MPI_Offset offset,
	MPI3_CONST void* buf, int count, MPI_Datatype datatype, MPI_Status* status)__attribute__((weak));
void DLB_MPI_File_write_at_all_leave (void)__attribute__((weak));

/******************************************************************************
 *** 1sided
 ******************************************************************************/
 
/***  Fortran  ***/

void DLB_MPI_Win_create_F_enter (void *base, void *size, MPI_Fint *disp_unit, void *info,
 MPI_Fint *comm, void *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_create_F_leave (void)__attribute__((weak));


void DLB_MPI_Win_fence_F_enter (MPI_Fint *assert, void *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_fence_F_leave (void)__attribute__((weak));


void DLB_MPI_Win_start_F_enter (void *group, void *assert, void *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_start_F_leave (void)__attribute__((weak));


void DLB_MPI_Win_free_F_enter (void *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_free_F_leave (void)__attribute__((weak));


void DLB_MPI_Win_complete_F_enter (void *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_complete_F_leave (void)__attribute__((weak));


void DLB_MPI_Win_wait_F_enter (void *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_wait_F_leave (void)__attribute__((weak));


void DLB_MPI_Win_post_F_enter (void *group, void *assert, void *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_post_F_leave (void)__attribute__((weak));


void DLB_MPI_Get_F_enter (void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Get_F_leave (void)__attribute__((weak));


void DLB_MPI_Put_F_enter (void *origin_addr, MPI_Fint *origin_count,
	MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Fint *target_disp,
	MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Put_F_leave (void)__attribute__((weak));


void DLB_MPI_Win_lock_F_enter (MPI_Fint *lock_type, MPI_Fint *rank, MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_lock_F_leave (void)__attribute__((weak));

void DLB_MPI_Win_unlock_F_enter (MPI_Fint *rank, MPI_Fint *win, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Win_unlock_F_leave (void)__attribute__((weak));

void DLB_MPI_Get_accumulate_F_enter (void *origin_addr, MPI_Fint *origin_count,
	                             MPI_Fint *origin_datatype, void *result_addr,
				     MPI_Fint *result_count, MPI_Fint *result_datatype,
				     MPI_Fint *target_rank, MPI_Fint *target_disp,
				     MPI_Fint *target_count, MPI_Fint *target_datatype,
				     MPI_Fint *op, MPI_Fint *win,
				     MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Get_accumulate_F_leave (void)__attribute__((weak));


/***  C  ***/

void DLB_MPI_Win_create_enter (void *base, MPI_Aint size, int disp_unit, MPI_Info info,
	MPI_Comm comm, MPI_Win *win)__attribute__((weak));
void DLB_MPI_Win_create_leave (void)__attribute__((weak));


void DLB_MPI_Win_fence_enter (int assert, MPI_Win win)__attribute__((weak));
void DLB_MPI_Win_fence_leave (void)__attribute__((weak));


void DLB_MPI_Win_start_enter (MPI_Group group, int assert, MPI_Win win)__attribute__((weak));
void DLB_MPI_Win_start_leave (void)__attribute__((weak));


void DLB_MPI_Win_free_enter (MPI_Win *win)__attribute__((weak));
void DLB_MPI_Win_free_leave (void)__attribute__((weak));


void DLB_MPI_Win_complete_enter (MPI_Win win)__attribute__((weak));
void DLB_MPI_Win_complete_leave (void)__attribute__((weak));


void DLB_MPI_Win_wait_enter (MPI_Win win)__attribute__((weak));
void DLB_MPI_Win_wait_leave (void)__attribute__((weak));


void DLB_MPI_Win_post_enter (MPI_Group group, int assert, MPI_Win win)__attribute__((weak));
void DLB_MPI_Win_post_leave (void)__attribute__((weak));


void DLB_MPI_Get_enter (void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	int target_rank, MPI_Aint target_disp, int target_count,
	MPI_Datatype target_datatype, MPI_Win win)__attribute__((weak));
void DLB_MPI_Get_leave (void)__attribute__((weak));


void DLB_MPI_Put_enter (MPI3_CONST void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	int target_rank, MPI_Aint target_disp, int target_count,
	MPI_Datatype target_datatype, MPI_Win win)__attribute__((weak));
void DLB_MPI_Put_leave (void)__attribute__((weak));


void DLB_MPI_Win_lock_enter (int lock_type, int rank, int assert, MPI_Win win)__attribute__((weak));
void DLB_MPI_Win_lock_leave (void)__attribute__((weak));


void DLB_MPI_Win_unlock_enter (int rank, MPI_Win win)__attribute__((weak));
void DLB_MPI_Win_unlock_leave (void)__attribute__((weak));


void DLB_MPI_Get_accumulate_enter (MPI3_CONST void *origin_addr, int origin_count,
                                   MPI_Datatype origin_datatype, void *result_addr,
			           int result_count, MPI_Datatype result_datatype,
	                           int target_rank, MPI_Aint target_disp,
			           int target_count, MPI_Datatype target_datatype,
			           MPI_Op op, MPI_Win win)__attribute__((weak));
void DLB_MPI_Get_accumulate_leave (void)__attribute__((weak));


/******************************************************************************
 *** Collectives
 ******************************************************************************/

/***  Fortran  ***/
 
void DLB_MPI_Reduce_F_enter (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Reduce_F_leave (void)__attribute__((weak));


void DLB_MPI_Reduce_scatter_F_enter (void *sendbuf, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Reduce_scatter_F_leave (void)__attribute__((weak));


void DLB_MPI_Allreduce_F_enter (void *sendbuf, void *recvbuf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Allreduce_F_leave (void)__attribute__((weak));


void DLB_MPI_Barrier_F_enter (MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Barrier_F_leave (void)__attribute__((weak));


void DLB_MPI_Bcast_F_enter (void *buffer, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Bcast_F_leave ()__attribute__((weak));


void DLB_MPI_Alltoall_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Alltoall_F_leave (void)__attribute__((weak));


void DLB_MPI_Alltoallv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Alltoallv_F_leave (void)__attribute__((weak));


void DLB_MPI_Allgather_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Allgather_F_leave (void)__attribute__((weak));


void DLB_MPI_Allgatherv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Allgatherv_F_leave (void)__attribute__((weak));


void DLB_MPI_Gather_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Gather_F_leave (void)__attribute__((weak));


void DLB_MPI_Gatherv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Gatherv_F_leave (void)__attribute__((weak));


void DLB_MPI_Scatter_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Scatter_F_leave (void)__attribute__((weak));


void DLB_MPI_Scatterv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Scatterv_F_leave (void)__attribute__((weak));


void DLB_MPI_Scan_F_enter (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Scan_F_leave (void)__attribute__((weak));


void DLB_MPI_Ireduce_F_enter (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ireduce_F_leave (void)__attribute__((weak));


void DLB_MPI_Ireduce_scatter_F_enter (void *sendbuf, void *recvbuf, MPI_Fint *recvcounts,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ireduce_scatter_F_leave (void)__attribute__((weak));


void DLB_MPI_Iallreduce_F_enter (void *sendbuf, void *recvbuf,
	MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
	MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Iallreduce_F_leave (void)__attribute__((weak));


void DLB_MPI_Ibarrier_F_enter (MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ibarrier_F_leave (void)__attribute__((weak));


void DLB_MPI_Ibcast_F_enter (void *buffer, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ibcast_F_leave (void)__attribute__((weak));


void DLB_MPI_Ialltoall_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ialltoall_F_leave (void)__attribute__((weak));


void DLB_MPI_Ialltoallv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sdispls, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *rdispls, MPI_Fint *recvtype,	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ialltoallv_F_leave (void)__attribute__((weak));


void DLB_MPI_Iallgather_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Iallgather_F_leave (void)__attribute__((weak));


void DLB_MPI_Iallgatherv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Iallgatherv_F_leave (void)__attribute__((weak));


void DLB_MPI_Igather_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Igather_F_leave (void)__attribute__((weak));


void DLB_MPI_Igatherv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *displs,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Igatherv_F_leave (void)__attribute__((weak));


void DLB_MPI_Iscatter_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
	MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Iscatter_F_leave (void)__attribute__((weak));


void DLB_MPI_Iscatterv_F_enter (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *displs, MPI_Fint *sendtype, void *recvbuf, MPI_Fint *recvcount,
	MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Iscatterv_F_leave (void)__attribute__((weak));


void DLB_MPI_Iscan_F_enter (void *sendbuf, void *recvbuf, MPI_Fint *count,
	MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Iscan_F_leave (void)__attribute__((weak));


void DLB_MPI_Reduce_scatter_block_F_enter (void *sendbuf, void *recvbuf,
    MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
    MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Reduce_scatter_block_F_leave (void)__attribute__((weak));


void DLB_MPI_Ireduce_scatter_block_F_enter (void *sendbuf, void *recvbuf,
    MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
    MPI_Fint *req, MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ireduce_scatter_block_F_leave (void)__attribute__((weak));


void DLB_MPI_Alltoallw_F_enter (void *sendbuf, MPI_Fint *sendcounts,
    MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
    MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierror)
    __attribute__((weak));
void DLB_MPI_Alltoallw_F_leave (void)__attribute__((weak));


void DLB_MPI_Ialltoallw_F_enter (void *sendbuf, MPI_Fint *sendcounts,
    MPI_Fint *sdispls, MPI_Fint *sendtypes, void *recvbuf, MPI_Fint *recvcounts,
    MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *req,
    MPI_Fint *ierror)__attribute__((weak));
void DLB_MPI_Ialltoallw_F_leave (void)__attribute__((weak));


/***  C  ***/

void DLB_MPI_Reduce_enter (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Reduce_leave(void)__attribute__((weak));


void DLB_MPI_Reduce_scatter_enter (MPI3_CONST void *sendbuf, void *recvbuf, MPI3_CONST int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Reduce_scatter_leave(void)__attribute__((weak));


void DLB_MPI_Allreduce_enter (MPI3_CONST void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Allreduce_leave(void)__attribute__((weak));


void DLB_MPI_Barrier_enter (MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Barrier_leave(void)__attribute__((weak));


void DLB_MPI_Bcast_enter (void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Bcast_leave (void)__attribute__((weak));


void DLB_MPI_Alltoall_enter (MPI3_CONST void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Alltoall_leave(void)__attribute__((weak));


void DLB_MPI_Alltoallv_enter (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *sdispls,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
	MPI_Datatype recvtype, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Alltoallv_leave(void)__attribute__((weak));


void DLB_MPI_Allgather_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Allgather_leave (void)__attribute__((weak));


void DLB_MPI_Allgatherv_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Allgatherv_leave (void)__attribute__((weak));


void DLB_MPI_Gather_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Gather_leave (void)__attribute__((weak));


void DLB_MPI_Gatherv_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, int root, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Gatherv_leave (void)__attribute__((weak));


void DLB_MPI_Scatter_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Scatter_leave (void)__attribute__((weak));


void DLB_MPI_Scatterv_enter (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *displs, 
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Scatterv_leave (void)__attribute__((weak));


void DLB_MPI_Scan_enter (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Scan_leave (void)__attribute__((weak));


void DLB_MPI_Ireduce_enter (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Ireduce_leave (void)__attribute__((weak));


void DLB_MPI_Ireduce_scatter_enter (MPI3_CONST void *sendbuf, void *recvbuf,
	MPI3_CONST int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
	MPI_Request *req)__attribute__((weak));
void DLB_MPI_Ireduce_scatter_leave (void)__attribute__((weak));


void DLB_MPI_Iallreduce_enter (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Iallreduce_leave (void)__attribute__((weak));


void DLB_MPI_Ibarrier_enter (MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Ibarrier_leave (void)__attribute__((weak));


void DLB_MPI_Ibcast_enter (void *buffer, int count, MPI_Datatype datatype,
	int root, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Ibcast_leave (void)__attribute__((weak));


void DLB_MPI_Ialltoall_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Ialltoall_leave (void)__attribute__((weak));


void DLB_MPI_Ialltoallv_enter (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *sdispls,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
	MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Alltoallv_leave (void)__attribute__((weak));


void DLB_MPI_Iallgather_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Iallgather_leave (void)__attribute__((weak));


void DLB_MPI_Iallgatherv_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Iallgatherv_leave (void)__attribute__((weak));


void DLB_MPI_Igather_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Igather_leave (void)__attribute__((weak));


void DLB_MPI_Igatherv_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, MPI3_CONST int *recvcounts, MPI3_CONST int *displs,
	MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Igatherv_leave (void)__attribute__((weak));


void DLB_MPI_Iscatter_enter (MPI3_CONST void *sendbuf, int sendcount,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Iscatter_leave (void)__attribute__((weak));


void DLB_MPI_Iscatterv_enter (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts, MPI3_CONST int *displs, 
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Iscatterv_leave (void)__attribute__((weak));


void DLB_MPI_Iscan_enter (MPI3_CONST void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *req)__attribute__((weak));
void DLB_MPI_Iscan_leave (void)__attribute__((weak));


void DLB_MPI_Reduce_scatter_block_enter (MPI3_CONST void *sendbuf, void *recvbuf,
    int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Reduce_scatter_block_leave (void)__attribute__((weak));


void DLB_MPI_Ireduce_scatter_block_enter (MPI3_CONST void *sendbuf, void *recvbuf,
    int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
    MPI_Request *req)__attribute__((weak));
void DLB_MPI_Ireduce_scatter_block_leave (void)__attribute__((weak));


void DLB_MPI_Alltoallw_enter (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts,
    MPI3_CONST int *sdispls, MPI3_CONST MPI_Datatype *sendtypes, void *recvbuf,
    MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
    MPI3_CONST MPI_Datatype *recvtypes, MPI_Comm comm)__attribute__((weak));
void DLB_MPI_Alltoallw_leave (void)__attribute__((weak));


void DLB_MPI_Ialltoallw_enter (MPI3_CONST void *sendbuf, MPI3_CONST int *sendcounts,
    MPI3_CONST int *sdispls, MPI3_CONST MPI_Datatype *sendtypes, void *recvbuf,
    MPI3_CONST int *recvcounts, MPI3_CONST int *rdispls,
    MPI3_CONST MPI_Datatype *recvtypes, MPI_Comm comm,
    MPI_Request *req)__attribute__((weak));
void DLB_MPI_Ialltoallw_leave (void)__attribute__((weak));


#endif
