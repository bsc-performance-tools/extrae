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

#ifndef _MPI_EVENTENCODING_H
#define _MPI_EVENTENCODING_H

/* ==========================================================================
   ==== MPI Event Types
   ========================================================================== */

#define MPITYPE_PTOP               50000001
#define MPITYPE_COLLECTIVE         50000002

#define MPITYPE_OTHER              50000003
#define MPITYPE_RMA                50000004
#define MPITYPE_RMA_SIZE           50001000
#define MPITYPE_COMM               MPITYPE_OTHER
#define MPITYPE_GROUP              MPITYPE_OTHER
#define MPITYPE_TOPOLOGIES         MPITYPE_OTHER
#define MPITYPE_TYPE               MPITYPE_OTHER
#define MPITYPE_IO                 50000005

#define MPITYPE_PTOP_LABEL         "MPI Point-to-point"
#define MPITYPE_COLLECTIVE_LABEL   "MPI Collective Comm"

#define MPITYPE_OTHER_LABEL        "MPI Other"
#define MPITYPE_RMA_LABEL          "MPI One-sided"
#define MPITYPE_RMA_SIZE_LABEL     "MPI One-sided size"
#define MPITYPE_COMM_LABEL         MPITYPE_OTHER_LABEL
#define MPITYPE_GROUP_LABEL        MPITYPE_OTHER_LABEL
#define MPITYPE_TOPOLOGIES_LABEL   MPITYPE_OTHER_LABEL
#define MPITYPE_TYPE_LABEL         MPITYPE_OTHER_LABEL
#define MPITYPE_IO_LABEL           "MPI I/O"

/* ==========================================================================
   ==== MPI Event Values
   ========================================================================== */
typedef enum
{
  MPI_END_VAL, /* 0 */
  MPI_SEND_VAL,
  MPI_RECV_VAL,
  MPI_ISEND_VAL,
  MPI_IRECV_VAL,
  MPI_WAIT_VAL,
  MPI_WAITALL_VAL,
  MPI_BCAST_VAL,
  MPI_BARRIER_VAL,
  MPI_REDUCE_VAL,
  MPI_ALLREDUCE_VAL, /* 10 */
  MPI_ALLTOALL_VAL,
  MPI_ALLTOALLV_VAL,
  MPI_GATHER_VAL,
  MPI_GATHERV_VAL,
  MPI_SCATTER_VAL,
  MPI_SCATTERV_VAL,
  MPI_ALLGATHER_VAL,
  MPI_ALLGATHERV_VAL,
  MPI_COMM_RANK_VAL,
  MPI_COMM_SIZE_VAL, /* 20 */
  MPI_COMM_CREATE_VAL,
  MPI_COMM_DUP_VAL,
  MPI_COMM_SPLIT_VAL,
  MPI_COMM_GROUP_VAL,
  MPI_COMM_FREE_VAL,
  MPI_COMM_REMOTE_GROUP_VAL,
  MPI_COMM_REMOTE_SIZE_VAL,
  MPI_COMM_TEST_INTER_VAL,
  MPI_COMM_COMPARE_VAL,
  MPI_SCAN_VAL, /* 30 */
  MPI_INIT_VAL,
  MPI_FINALIZE_VAL,
  MPI_BSEND_VAL,
  MPI_SSEND_VAL,
  MPI_RSEND_VAL,
  MPI_IBSEND_VAL,
  MPI_ISSEND_VAL,
  MPI_IRSEND_VAL,
  MPI_TEST_VAL,
  MPI_CANCEL_VAL, /* 40 */
  MPI_SENDRECV_VAL,
  MPI_SENDRECV_REPLACE_VAL,
  MPI_CART_CREATE_VAL,
  MPI_CART_SHIFT_VAL,
  MPI_CART_COORDS_VAL,
  MPI_CART_GET_VAL,
  MPI_CART_MAP_VAL,
  MPI_CART_RANK_VAL,
  MPI_CART_SUB_VAL,
  MPI_CARTDIM_GET_VAL, /* 50 */
  MPI_DIMS_CREATE_VAL,
  MPI_GRAPH_GET_VAL,
  MPI_GRAPH_MAP_VAL,
  MPI_GRAPH_CREATE_VAL,
  MPI_GRAPH_NEIGHBORS_VAL,
  MPI_GRAPHDIMS_GET_VAL,
  MPI_GRAPH_NEIGHBORS_COUNT_VAL,
  MPI_TOPO_TEST_VAL,
  MPI_WAITANY_VAL,
  MPI_WAITSOME_VAL, /* 60 */
  MPI_PROBE_VAL,
  MPI_IPROBE_VAL,
  MPI_WIN_CREATE_VAL,
  MPI_WIN_FREE_VAL,
  MPI_PUT_VAL,
  MPI_GET_VAL,
  MPI_ACCUMULATE_VAL,
  MPI_WIN_FENCE_VAL,
  MPI_WIN_START_VAL,
  MPI_WIN_COMPLETE_VAL, /* 70 */
  MPI_WIN_POST_VAL,
  MPI_WIN_WAIT_VAL,
  MPI_WIN_TEST_VAL,
  MPI_WIN_LOCK_VAL,
  MPI_WIN_UNLOCK_VAL,
  MPI_PACK_VAL,
  MPI_UNPACK_VAL,
  MPI_OP_CREATE_VAL,
  MPI_OP_FREE_VAL,
  MPI_REDUCE_SCATTER_VAL, /* 80 */
  MPI_ATTR_DELETE_VAL,
  MPI_ATTR_GET_VAL,
  MPI_ATTR_PUT_VAL,
  MPI_GROUP_DIFFERENCE_VAL,
  MPI_GROUP_EXCL_VAL,
  MPI_GROUP_FREE_VAL,
  MPI_GROUP_INCL_VAL,
  MPI_GROUP_INTERSECTION_VAL,
  MPI_GROUP_RANK_VAL,
  MPI_GROUP_RANGE_EXCL_VAL, /* 90 */
  MPI_GROUP_RANGE_INCL_VAL,
  MPI_GROUP_SIZE_VAL,
  MPI_GROUP_TRANSLATE_RANKS_VAL,
  MPI_GROUP_UNION_VAL,
  MPI_GROUP_COMPARE_VAL,
  MPI_INTERCOMM_CREATE_VAL,
  MPI_INTERCOMM_MERGE_VAL,
  MPI_KEYVAL_FREE_VAL,
  MPI_KEYVAL_CREATE_VAL,
  MPI_ABORT_VAL, /* 100 */
  MPI_ERROR_CLASS_VAL,
  MPI_ERRHANDLER_CREATE_VAL,
  MPI_ERRHANDLER_FREE_VAL,
  MPI_ERRHANDLER_GET_VAL,
  MPI_ERROR_STRING_VAL,
  MPI_ERRHANDLER_SET_VAL,
  MPI_GET_PROCESSOR_NAME_VAL,
  MPI_INITIALIZED_VAL,
  MPI_WTICK_VAL,
  MPI_WTIME_VAL, /* 110 */
  MPI_ADDRESS_VAL,
  MPI_BSEND_INIT_VAL,
  MPI_BUFFER_ATTACH_VAL,
  MPI_BUFFER_DETACH_VAL,
  MPI_REQUEST_FREE_VAL,
  MPI_RECV_INIT_VAL,
  MPI_SEND_INIT_VAL,
  MPI_GET_COUNT_VAL,
  MPI_GET_ELEMENTS_VAL,
  MPI_PACK_SIZE_VAL, /* 120 */
  MPI_RSEND_INIT_VAL,
  MPI_SSEND_INIT_VAL,
  MPI_START_VAL,
  MPI_STARTALL_VAL,
  MPI_TESTALL_VAL,
  MPI_TESTANY_VAL,
  MPI_TEST_CANCELLED_VAL,
  MPI_TESTSOME_VAL,
  MPI_TYPE_COMMIT_VAL,
  MPI_TYPE_CONTIGUOUS_VAL, /* 130 */
  MPI_TYPE_EXTENT_VAL,
  MPI_TYPE_FREE_VAL,
  MPI_TYPE_HINDEXED_VAL,
  MPI_TYPE_HVECTOR_VAL,
  MPI_TYPE_INDEXED_VAL,
  MPI_TYPE_LB_VAL,
  MPI_TYPE_SIZE_VAL,
  MPI_TYPE_STRUCT_VAL,
  MPI_TYPE_UB_VAL,
  MPI_TYPE_VECTOR_VAL, /* 140 */
  MPI_FILE_OPEN_VAL,
  MPI_FILE_CLOSE_VAL,
  MPI_FILE_READ_VAL,
  MPI_FILE_READ_ALL_VAL,
  MPI_FILE_WRITE_VAL,
  MPI_FILE_WRITE_ALL_VAL,
  MPI_FILE_READ_AT_VAL,
  MPI_FILE_READ_AT_ALL_VAL,
  MPI_FILE_WRITE_AT_VAL,
  MPI_FILE_WRITE_AT_ALL_VAL, /* 150 */
  MPI_COMM_SPAWN_VAL,
  MPI_COMM_SPAWN_MULTIPLE_VAL,
  MPI_REQUEST_GET_STATUS_VAL,
  MPI_IREDUCE_VAL,
  MPI_IALLREDUCE_VAL, /* 155 */
  MPI_IBARRIER_VAL,
  MPI_IBCAST_VAL,
  MPI_IALLTOALL_VAL,
  MPI_IALLTOALLV_VAL,
  MPI_IALLGATHER_VAL, /* 160 */
  MPI_IALLGATHERV_VAL,
  MPI_IGATHER_VAL,
  MPI_IGATHERV_VAL,
  MPI_ISCATTER_VAL,
  MPI_ISCATTERV_VAL, /* 165 */
  MPI_IREDUCESCAT_VAL,
  MPI_ISCAN_VAL
}
MPIVal;

/* ==========================================================================
   ==== MPI Event Labels
   ========================================================================== */

#define  MPIEND_LABEL                      "End"
#define  MPI_SEND_LABEL                    "MPI_Send"
#define  MPI_RECV_LABEL                    "MPI_Recv"
#define  MPI_ISEND_LABEL                   "MPI_Isend"
#define  MPI_IRECV_LABEL                   "MPI_Irecv"
#define  MPI_WAIT_LABEL                    "MPI_Wait"
#define  MPI_WAITALL_LABEL                 "MPI_Waitall"

#define  MPI_REDUCE_LABEL                  "MPI_Reduce"
#define  MPI_ALLREDUCE_LABEL               "MPI_Allreduce"
#define  MPI_BARRIER_LABEL                 "MPI_Barrier"
#define  MPI_BCAST_LABEL                   "MPI_Bcast"
#define  MPI_ALLTOALL_LABEL                "MPI_Alltoall"
#define  MPI_ALLTOALLV_LABEL               "MPI_Alltoallv"
#define  MPI_ALLGATHER_LABEL               "MPI_Allgather"
#define  MPI_ALLGATHERV_LABEL              "MPI_Allgatherv"
#define  MPI_GATHER_LABEL                  "MPI_Gather"
#define  MPI_GATHERV_LABEL                 "MPI_Gatherv"
#define  MPI_SCATTER_LABEL                 "MPI_Scatter"
#define  MPI_SCATTERV_LABEL                "MPI_Scatterv"
#define  MPI_REDUCE_SCATTER_LABEL          "MPI_Reduce_scatter"
#define  MPI_SCAN_LABEL                    "MPI_Scan"

#define  MPI_IREDUCE_LABEL                 "MPI_Ireduce"
#define  MPI_IALLREDUCE_LABEL              "MPI_Iallreduce"
#define  MPI_IBARRIER_LABEL                "MPI_Ibarrier"
#define  MPI_IBCAST_LABEL                  "MPI_Ibcast"
#define  MPI_IALLTOALL_LABEL               "MPI_Ialltoall"
#define  MPI_IALLTOALLV_LABEL              "MPI_Ialltoallv"
#define  MPI_IALLGATHER_LABEL              "MPI_Iallgather"
#define  MPI_IALLGATHERV_LABEL             "MPI_Iallgatherv"
#define  MPI_IGATHER_LABEL                 "MPI_Igather"
#define  MPI_IGATHERV_LABEL                "MPI_Igatherv"
#define  MPI_ISCATTER_LABEL                "MPI_Iscatter"
#define  MPI_ISCATTERV_LABEL               "MPI_Iscatterv"
#define  MPI_IREDUCESCAT_LABEL             "MPI_Ireduce_scatter"
#define  MPI_ISCAN_LABEL                   "MPI_Iscan"

#define  MPI_INIT_LABEL                    "MPI_Init"
#define  MPI_FINALIZE_LABEL                "MPI_Finalize"
#define  MPI_BSEND_LABEL                   "MPI_Bsend"
#define  MPI_SSEND_LABEL                   "MPI_Ssend"
#define  MPI_RSEND_LABEL                   "MPI_Rsend"
#define  MPI_IBSEND_LABEL                  "MPI_Ibsend"
#define  MPI_ISSEND_LABEL                  "MPI_Issend"
#define  MPI_IRSEND_LABEL                  "MPI_Irsend"
#define  MPI_TEST_LABEL                    "MPI_Test"
#define  MPI_CANCEL_LABEL                  "MPI_Cancel"
#define  MPI_SENDRECV_LABEL                "MPI_Sendrecv"
#define  MPI_SENDRECV_REPLACE_LABEL        "MPI_Sendrecv_replace"
#define  MPI_CART_CREATE_LABEL             "MPI_Cart_create"
#define  MPI_CART_SHIFT_LABEL              "MPI_Cart_shift"
#define  MPI_CART_COORDS_LABEL             "MPI_Cart_coords"
#define  MPI_CART_GET_LABEL                "MPI_Cart_get"
#define  MPI_CART_MAP_LABEL                "MPI_Cart_map"
#define  MPI_CART_RANK_LABEL               "MPI_Cart_rank"
#define  MPI_CART_SUB_LABEL                "MPI_Cart_sub"
#define  MPI_CARTDIM_GET_LABEL             "MPI_Cartdim_get"
#define  MPI_DIMS_CREATE_LABEL             "MPI_Dims_create"
#define  MPI_GRAPH_GET_LABEL               "MPI_Graph_get"
#define  MPI_GRAPH_MAP_LABEL               "MPI_Graph_map"
#define  MPI_GRAPH_CREATE_LABEL            "MPI_Graph_create"
#define  MPI_GRAPH_NEIGHBORS_LABEL         "MPI_Graph_neighbors"
#define  MPI_GRAPHDIMS_GET_LABEL           "MPI_Graphdims_get"
#define  MPI_GRAPH_NEIGHBORS_COUNT_LABEL   "MPI_Graph_neighbors_count"
#define  MPI_WAITANY_LABEL                 "MPI_Waitany"
#define  MPI_TOPO_TEST_LABEL               "MPI_Topo_test"
#define  MPI_WAITSOME_LABEL                "MPI_Waitsome"
#define  MPI_PROBE_LABEL                   "MPI_Probe"
#define  MPI_IPROBE_LABEL                  "MPI_Iprobe"

#define  MPI_WIN_CREATE_LABEL              "MPI_Win_create"
#define  MPI_WIN_FREE_LABEL                "MPI_Win_free"
#define  MPI_PUT_LABEL                     "MPI_Put"
#define  MPI_GET_LABEL                     "MPI_Get"
#define  MPI_ACCUMULATE_LABEL              "MPI_Accumulate"
#define  MPI_WIN_FENCE_LABEL               "MPI_Win_fence"
#define  MPI_WIN_START_LABEL               "MPI_Win_complete"
#define  MPI_WIN_COMPLETE_LABEL            "MPI_Win_start"
#define  MPI_WIN_POST_LABEL                "MPI_Win_post"
#define  MPI_WIN_WAIT_LABEL                "MPI_Win_wait"
#define  MPI_WIN_TEST_LABEL                "MPI_Win_test"
#define  MPI_WIN_LOCK_LABEL                "MPI_Win_lock"
#define  MPI_WIN_UNLOCK_LABEL              "MPI_Win_unlock"

#define  MPI_PACK_LABEL                    "MPI_Pack"
#define  MPI_UNPACK_LABEL                  "MPI_Unpack"

#define  MPI_OP_CREATE_LABEL               "MPI_Op_create"
#define  MPI_OP_FREE_LABEL                 "MPI_Op_free"

#define  MPI_ATTR_DELETE_LABEL             "MPI_Attr_delete"
#define  MPI_ATTR_GET_LABEL                "MPI_Attr_get"
#define  MPI_ATTR_PUT_LABEL                "MPI_Attr_put"

#define  MPI_COMM_RANK_LABEL               "MPI_Comm_rank"
#define  MPI_COMM_SIZE_LABEL               "MPI_Comm_size"
#define  MPI_COMM_CREATE_LABEL             "MPI_Comm_create"
#define  MPI_COMM_DUP_LABEL                "MPI_Comm_dup"
#define  MPI_COMM_SPLIT_LABEL              "MPI_Comm_split"
#define  MPI_COMM_SPAWN_LABEL              "MPI_Comm_spawn"
#define  MPI_COMM_SPAWN_MULTIPLE_LABEL     "MPI_Comm_spawn_multiple"
#define  MPI_COMM_GROUP_LABEL              "MPI_Comm_group"
#define  MPI_COMM_FREE_LABEL               "MPI_Comm_free"
#define  MPI_COMM_REMOTE_GROUP_LABEL       "MPI_Comm_remote_group"
#define  MPI_COMM_REMOTE_SIZE_LABEL        "MPI_Comm_remote_size"
#define  MPI_COMM_TEST_INTER_LABEL         "MPI_Comm_test_inter"
#define  MPI_COMM_COMPARE_LABEL            "MPI_Comm_compare"

#define  MPI_GROUP_DIFFERENCE_LABEL        "MPI_Group_difference"
#define  MPI_GROUP_EXCL_LABEL              "MPI_Group_excl"
#define  MPI_GROUP_FREE_LABEL              "MPI_Group_free"
#define  MPI_GROUP_INCL_LABEL              "MPI_Group_incl"
#define  MPI_GROUP_INTERSECTION_LABEL      "MPI_Group_intersection"
#define  MPI_GROUP_RANK_LABEL              "MPI_Group_rank"
#define  MPI_GROUP_RANGE_EXCL_LABEL        "MPI_Group_range_excl"
#define  MPI_GROUP_RANGE_INCL_LABEL        "MPI_Group_range_incl"
#define  MPI_GROUP_SIZE_LABEL              "MPI_Group_size"
#define  MPI_GROUP_TRANSLATE_RANKS_LABEL   "MPI_Group_translate_ranks"
#define  MPI_GROUP_UNION_LABEL             "MPI_Group_union"
#define  MPI_GROUP_COMPARE_LABEL           "MPI_Group_compare"

#define  MPI_INTERCOMM_CREATE_LABEL        "MPI_Intercomm_create"
#define  MPI_INTERCOMM_MERGE_LABEL         "MPI_Intercomm_merge"
#define  MPI_KEYVAL_FREE_LABEL             "MPI_Keyval_free"
#define  MPI_KEYVAL_CREATE_LABEL           "MPI_Keyval_create"
#define  MPI_ABORT_LABEL                   "MPI_Abort"
#define  MPI_ERROR_CLASS_LABEL             "MPI_Error_class"
#define  MPI_ERRHANDLER_CREATE_LABEL       "MPI_Errhandler_create"
#define  MPI_ERRHANDLER_FREE_LABEL         "MPI_Errhandler_free"
#define  MPI_ERRHANDLER_GET_LABEL          "MPI_Errhandler_get"
#define  MPI_ERROR_STRING_LABEL            "MPI_Error_string"
#define  MPI_ERRHANDLER_SET_LABEL          "MPI_Errhandler_set"
#define  MPI_GET_PROCESSOR_NAME_LABEL      "MPI_Get_processor_name"
#define  MPI_INITIALIZED_LABEL             "MPI_Initialized"
#define  MPI_WTICK_LABEL                   "MPI_Wtick"
#define  MPI_WTIME_LABEL                   "MPI_Wtime"
#define  MPI_ADDRESS_LABEL                 "MPI_Address"
#define  MPI_BSEND_INIT_LABEL              "MPI_Bsend_init"
#define  MPI_BUFFER_ATTACH_LABEL           "MPI_Buffer_attach"
#define  MPI_BUFFER_DETACH_LABEL           "MPI_Buffer_detach"
#define  MPI_REQUEST_FREE_LABEL            "MPI_Request_free"
#define  MPI_RECV_INIT_LABEL               "MPI_Recv_init"
#define  MPI_SEND_INIT_LABEL               "MPI_Send_init"
#define  MPI_GET_COUNT_LABEL               "MPI_Get_count"
#define  MPI_GET_ELEMENTS_LABEL            "MPI_Get_elements"
#define  MPI_PACK_SIZE_LABEL               "MPI_Pack_size"
#define  MPI_RSEND_INIT_LABEL              "MPI_Rsend_init"
#define  MPI_SSEND_INIT_LABEL              "MPI_Ssend_init"
#define  MPI_START_LABEL                   "MPI_Start"
#define  MPI_STARTALL_LABEL                "MPI_Startall"
#define  MPI_TESTALL_LABEL                 "MPI_Testall"
#define  MPI_TESTANY_LABEL                 "MPI_Testany"
#define  MPI_TEST_CANCELLED_LABEL          "MPI_Test_cancelled"
#define  MPI_TESTSOME_LABEL                "MPI_Testsome"
#define  MPI_TYPE_COMMIT_LABEL             "MPI_Type_commit"
#define  MPI_TYPE_CONTIGUOUS_LABEL         "MPI_Type_contiguous"
#define  MPI_TYPE_EXTENT_LABEL             "MPI_Type_extent"
#define  MPI_TYPE_FREE_LABEL               "MPI_Type_free"
#define  MPI_TYPE_HINDEXED_LABEL           "MPI_Type_hindexed"
#define  MPI_TYPE_HVECTOR_LABEL            "MPI_Type_hvector"
#define  MPI_TYPE_INDEXED_LABEL            "MPI_Type_indexed"
#define  MPI_TYPE_LB_LABEL                 "MPI_Type_lb"
#define  MPI_TYPE_SIZE_LABEL               "MPI_Type_size"
#define  MPI_TYPE_STRUCT_LABEL             "MPI_Type_struct"
#define  MPI_TYPE_UB_LABEL                 "MPI_Type_ub"
#define  MPI_TYPE_VECTOR_LABEL             "MPI_Type_vector"
#define  MPI_FILE_OPEN_LABEL               "MPI_File_open"
#define  MPI_FILE_CLOSE_LABEL              "MPI_File_close"
#define  MPI_FILE_READ_LABEL               "MPI_File_read"
#define  MPI_FILE_READ_ALL_LABEL           "MPI_File_read_all"
#define  MPI_FILE_WRITE_LABEL              "MPI_File_write"
#define  MPI_FILE_WRITE_ALL_LABEL          "MPI_File_write_all"
#define  MPI_FILE_READ_AT_LABEL            "MPI_File_read_at"
#define  MPI_FILE_READ_AT_ALL_LABEL        "MPI_File_read_at_all"
#define  MPI_FILE_WRITE_AT_LABEL           "MPI_File_write_at"
#define  MPI_FILE_WRITE_AT_ALL_LABEL       "MPI_File_write_at_all"
#define  MPI_REQUEST_GET_STATUS_LABEL      "MPI_Request_get_status"


#endif /* _MPI_EVENTENCODING_H */
