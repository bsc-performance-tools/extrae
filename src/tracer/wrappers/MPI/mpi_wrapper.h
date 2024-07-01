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

#ifndef MPI_WRAPPER_DEFINED
#define MPI_WRAPPER_DEFINED

#if !defined(MPI_SUPPORT)
#error "This should not be included"
#endif

#include <config.h>

#ifdef HAVE_MPI_H
#include <mpi.h>
#endif
#include "extrae_mpif.h"
#include "defines.h"

#include "wrapper.h"
#include "hash_table.h"

#if !defined(MPI_HAS_MPI_F_STATUS_SIZE)
#warning MPI_F_STATUS_SIZE definition not found in mpi.h. Assuming our configure check for MPI_STATUS_INTEGER_FIELDS_COUNT.
#define MPI_F_STATUS_SIZE MPI_STATUS_INTEGER_FIELDS_COUNT
#endif

/*
 * MAX_MPI_HANDLES determines size of local arrays to store MPI_Request and MPI_Status objects.
 * We need to copy MPI_Request because they are invalidated after the PMPI call (see copyRequests_*).
 * We need to substitute MPI_Status because it may be set to MPI_STATUS_IGNORE (see substituteStatusIgnore_*).
 * If the number of objects is greater than MAX_MPI_HANDLES, we allocate memory dynamically.
 */
#define MAX_MPI_HANDLES 128

enum
{
	OP_TYPE_RECV = 0,
	OP_TYPE_SEND = 1
};

// Helper functions to process MPI wrapper's arguments
void getCommunicatorGroup(MPI_Comm comm, MPI_Group *group);
int  getMsgSizeFromCountAndDatatype(int count, MPI_Datatype datatype);
void getCommInfoFromStatus_C(MPI_Status *status, MPI_Datatype datatype, MPI_Comm comm, MPI_Group group, int *size, int *tag, int *source_global);
void getCommInfoFromStatus_F(MPI_Fint *f_status, MPI_Datatype datatype, MPI_Comm comm, MPI_Group group, int *size, int *tag, int *source_global);
void makeProxies_C(int count_requests, MPI_Request *user_requests, MPI_Request *local_requests, MPI_Request **proxy_requests, int count_statuses, MPI_Status *user_statuses, MPI_Status *local_statuses, MPI_Status **proxy_statuses);
void makeProxies_F(int count_requests, MPI_Fint *user_requests, MPI_Request *local_requests, MPI_Request **proxy_requests, int count_statuses, MPI_Fint *user_statuses, MPI_Fint *local_statuses, MPI_Fint **proxy_statuses);
void freeProxies(void *proxy_request, void *local_request, void *proxy_status, void *user_status, void *local_status);
void savePersistentRequest_C(MPI_Request p_request, MPI_Datatype datatype, MPI_Comm comm, int type, int count, int partner, int tag);
void savePersistentRequest_F(MPI_Fint *p_request, MPI_Fint *datatype, MPI_Fint *comm, int type, int count, int partner, int tag);
void saveRequest(MPI_Request request, MPI_Comm comm);
void processRequest_C(iotimer_t ts, MPI_Request request, MPI_Status *status);
void processRequest_F(iotimer_t ts, MPI_Request request, MPI_Fint *status);
void cancelRequest(MPI_Request request);




#if defined(MPI3)
void saveMessage(MPI_Message message, MPI_Comm comm);
MPI_Comm processMessage(MPI_Message message, MPI_Request *request);
#endif /* MPI3 */

void Trace_MPI_Communicator(MPI_Comm newcomm, UINT64 time, int trace);
void Trace_MPI_InterCommunicator(MPI_Comm newcomm, MPI_Comm local_comm,
								 int local_leader, MPI_Comm remote_comm, int remote_leader, UINT64 time,
								 int trace);

void gettopology(void);
void configure_MPI_vars(void);
unsigned long long CalculateNumOpsForPeriod(unsigned long long wannaPeriod,
											unsigned long long NumOfGlobals, unsigned long long runnedPeriod);
void CheckControlFile(void);
void CheckGlobalOpsTracingIntervals(void);
void MPI_remove_file_list(int all);

void Extrae_network_counters_Wrapper(void);
void Extrae_network_routes_Wrapper(int mpi_rank);
void Extrae_tracing_tasks_Wrapper(unsigned from, unsigned to);
char *Extrae_core_get_mpits_file_name(void);
void Extrae_MPI_prepareDirectoryStructures(int me, int world_size);

void translateLocalToGlobalRank(MPI_Comm comm, MPI_Group group, int dest, int *receiver);

int MPI_Generate_Task_File_List();

#define MPI_REQUEST_TO_HASH_KEY(r) ((uintptr_t)r)
#define MPI_MESSAGE_TO_HASH_KEY(m) ((uintptr_t)m)

/**
 * xtr_hash_data_persistent_request_t
 *
 * This structure holds extra data that needs to be stored when hashing persistent MPI_Request data.
 */
typedef struct xtr_hash_data_persistent_request_t
{
	MPI_Datatype datatype; /* Type of an element */
	MPI_Comm comm;		   /* Communicator identifier */
	int type;			   /* OP type: ISEND/IBSEND/ISSEND/IRSEND/IRECV */
	int count;			   /* Num of elements in the transmission */
	int task;			   /* Source/destination */
	int tag;			   /* Tag identifier */
} xtr_hash_data_persistent_request_t;

/**
 * xtr_hash_data_request_t
 *
 * This structure holds extra data that needs to be stored when hashing MPI_Request data.
 */
typedef struct xtr_hash_data_request_t
{
	MPI_Comm commid;
	MPI_Group group;
} xtr_hash_data_request_t;

/**
 * xtr_hash_data_message_t
 *
 * This structure holds extra data that needs to be stored when hashing MPI_Message data.
 */
typedef struct xtr_hash_data_message_t
{
	MPI_Comm commid;
	MPI_Group group;
} xtr_hash_data_message_t;

// MPI_Request stored in a hash to do the communication matching
extern xtr_hash *hash_persistent_requests;
extern xtr_hash *hash_requests;

// MPI_Message stored in a hash to do the communication matching
extern xtr_hash *hash_messages;

/* Fortran Wrappers */

#if defined(FORTRAN_SYMBOLS)

#include "mpi_wrapper_p2p_f.h"
#include "mpi_wrapper_coll_f.h"
#include "mpi_wrapper_1sided_f.h"
#include "mpi_wrapper_io_f.h"

#if (defined(COMBINED_SYMBOLS) && defined(MPI_C_CONTAINS_FORTRAN_MPI_INIT) || \
	 !defined(COMBINED_SYMBOLS))
void PMPI_Init_Wrapper(MPI_Fint *ierror);
#endif

void PMPI_Init_thread_Wrapper(MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierror);

void PMPI_Finalize_Wrapper(MPI_Fint *ierror);

void PMPI_Request_get_status_Wrapper(MPI_Fint *request, int *flag,
									 MPI_Fint *status, MPI_Fint *ierror);

void PMPI_Cancel_Wrapper(MPI_Fint *request, MPI_Fint *ierror);

void PMPI_Comm_Rank_Wrapper(MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *ierror);

void PMPI_Comm_Size_Wrapper(MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierror);

void PMPI_Comm_Create_Wrapper(MPI_Fint *comm, MPI_Fint *group, MPI_Fint *newcomm, MPI_Fint *ierror);

void PMPI_Comm_Create_Group_Wrapper(MPI_Fint *comm, MPI_Fint *group,
  MPI_Fint *tag, MPI_Fint *newcomm, MPI_Fint *ierror);

void PMPI_Comm_Free_Wrapper(MPI_Fint *comm, MPI_Fint *ierror);

void PMPI_Comm_Dup_Wrapper(MPI_Fint *comm, MPI_Fint *newcomm,
						   MPI_Fint *ierror);

void PMPI_Comm_Dup_With_Info_Wrapper(MPI_Fint *comm, MPI_Fint *info,
									 MPI_Fint *newcomm, MPI_Fint *ierror);

void PMPI_Comm_Split_Wrapper(MPI_Fint *comm, MPI_Fint *color, MPI_Fint *key,
							 MPI_Fint *newcomm, MPI_Fint *ierror);

void PMPI_Comm_Split_Type_Wrapper(MPI_Fint *comm, MPI_Fint *split_type, MPI_Fint *key,
								  MPI_Fint *info, MPI_Fint *newcomm, MPI_Fint *ierror);

void PMPI_Comm_Spawn_Wrapper(char *command, char *argv, MPI_Fint *maxprocs, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror);

void PMPI_Comm_Spawn_Multiple_Wrapper(MPI_Fint *count, char *array_of_commands, char *array_of_argv, MPI_Fint *array_of_maxprocs, MPI_Fint *array_of_info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror);

void PMPI_Start_Wrapper(MPI_Fint *request, MPI_Fint *ierror);

void PMPI_Startall_Wrapper(MPI_Fint *count, MPI_Fint *array_of_requests, MPI_Fint *ierror);

void PMPI_Request_free_Wrapper(MPI_Fint *request, MPI_Fint *ierror);

void PMPI_Cart_create_Wrapper(MPI_Fint *comm_old, MPI_Fint *ndims,
							  MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart,
							  MPI_Fint *ierror);

void PMPI_Cart_sub_Wrapper(MPI_Fint *comm, MPI_Fint *remain_dims,
						   MPI_Fint *comm_new, MPI_Fint *ierror);

void PMPI_Intercomm_create_F_Wrapper(MPI_Fint *local_comm, MPI_Fint *local_leader,
									 MPI_Fint *peer_comm, MPI_Fint *remote_leader, MPI_Fint *tag,
									 MPI_Fint *newintercomm, MPI_Fint *ierror);

void PMPI_Intercomm_merge_F_Wrapper(MPI_Fint *intercomm, MPI_Fint *high,
									MPI_Fint *newintracomm, MPI_Fint *ierror);

void PMPI_Graph_create_Wrapper(MPI_Fint *comm_old, MPI_Fint *nnodes, MPI_Fint *index, MPI_Fint *edges, MPI_Fint *reorder, MPI_Fint *comm_graph, MPI_Fint *ierr);

void PMPI_Dist_graph_create_Wrapper(MPI_Fint *comm_old, MPI_Fint *n, MPI_Fint *sources, MPI_Fint *degrees, MPI_Fint *destinations, MPI_Fint *weights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr);

void PMPI_Dist_graph_create_adjacent_Wrapper(MPI_Fint *comm_old, MPI_Fint *indegree, MPI_Fint *sources, MPI_Fint *sourceweights, MPI_Fint *outdegree, MPI_Fint *destinations, MPI_Fint *destweights, MPI_Fint *info, MPI_Fint *reorder, MPI_Fint *comm_dist_graph, MPI_Fint *ierr);

#endif /* defined(FORTRAN_SYMBOLS) */

/* C Wrappers */

#include "mpi_wrapper_p2p_c.h"
#include "mpi_wrapper_coll_c.h"
#include "mpi_wrapper_1sided_c.h"
#include "mpi_wrapper_io_c.h"

int MPI_Init_C_Wrapper(int *argc, char ***argv);

int MPI_Init_thread_C_Wrapper(int *argc, char ***argv, int required, int *provided);

int MPI_Finalize_C_Wrapper(void);

int MPI_Request_get_status_C_Wrapper(MPI_Request request, int *flag, MPI_Status *status);

int MPI_Iprobe_C_Wrapper(int source, int tag, MPI_Comm comm, int *flag,
						 MPI_Status *status);

int MPI_Cancel_C_Wrapper(MPI_Request *request);

int MPI_Comm_rank_C_Wrapper(MPI_Comm comm, int *rank);

int MPI_Comm_size_C_Wrapper(MPI_Comm comm, int *size);

int MPI_Comm_create_C_Wrapper(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);

int MPI_Comm_create_group_C_Wrapper(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm);

int MPI_Comm_free_C_Wrapper(MPI_Comm *comm);

int MPI_Comm_dup_C_Wrapper(MPI_Comm comm, MPI_Comm *newcomm);

int MPI_Comm_dup_with_info_C_Wrapper(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm);

int MPI_Comm_split_C_Wrapper(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);

int MPI_Comm_split_type_C_Wrapper(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm);

int MPI_Comm_spawn_C_Wrapper(char *command, char **argv, int maxprocs, MPI_Info info,
							 int root, MPI_Comm comm, MPI_Comm *intercomm, int *array_of_errcodes);

int MPI_Comm_spawn_multiple_C_Wrapper(int count, char *array_of_commands[], char **array_of_argv[],
									  int array_of_maxprocs[], MPI_Info array_of_info[], int root, MPI_Comm comm,
									  MPI_Comm *intercomm, int array_of_errcodes[]);

int MPI_Cart_create_C_Wrapper(MPI_Comm comm_old, int ndims, int *dims,
							  int *periods, int reorder, MPI_Comm *comm_cart);

int MPI_Cart_sub_C_Wrapper(MPI_Comm comm, int *remain_dims, MPI_Comm *comm_new);

int MPI_Intercomm_create_C_Wrapper(MPI_Comm local_comm, int local_leader,
								   MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm);

int MPI_Intercomm_merge_C_Wrapper(MPI_Comm intercomm, int high,
								  MPI_Comm *newintracomm);

int MPI_Start_C_Wrapper(MPI_Request *request);

int MPI_Startall_C_Wrapper(int count, MPI_Request *requests);

int MPI_Request_free_C_Wrapper(MPI_Request *request);

int MPI_Graph_create_C_Wrapper(MPI_Comm comm_old, int nnodes, int *index, int *edges, int reorder, MPI_Comm *comm_graph);

int MPI_Dist_graph_create_C_Wrapper(MPI_Comm comm_old, int n, int *sources, int *degrees, int *destinations, int *weights, MPI_Info info, int reorder, MPI_Comm *comm_dist_graph);

int MPI_Dist_graph_create_adjacent_C_Wrapper(MPI_Comm comm_old, int indegree, const int sources[],
											 const int sourceweights[], int outdegree, const int destinations[], const int destweights[],
											 MPI_Info info, int reorder, MPI_Comm *comm_dist_graph);

#endif /* MPI_WRAPPER_DEFINED */
