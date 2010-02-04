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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/mpitrace/fusion/trunk/src/tracer/xml-parse.c $
 | @last_commit: $Date: 2009-10-29 13:06:27 +0100 (dj, 29 oct 2009) $
 | @version:     $Revision: 15 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _PACX_EVENTENCODING_H
#define _PACX_EVENTENCODING_H

/* ==========================================================================
   ==== MPI Event Types
   ========================================================================== */

#define PACXTYPE_PTOP               51000001
#define PACXTYPE_COLLECTIVE         51000002

#define PACXTYPE_OTHER              51000003
#define PACXTYPE_RMA                51000004
#define PACXTYPE_COMM               PACXTYPE_OTHER
#define PACXTYPE_GROUP              PACXTYPE_OTHER
#define PACXTYPE_TOPOLOGIES         PACXTYPE_OTHER
#define PACXTYPE_TYPE               PACXTYPE_OTHER
#define PACXTYPE_IO                 51000005

#define PACXTYPE_PTOP_LABEL         "PACX Point-to-point"
#define PACXTYPE_COLLECTIVE_LABEL   "PACX Collective Comm"

#define PACXTYPE_OTHER_LABEL        "PACX Other"
#define PACXTYPE_RMA_LABEL          "PACX One-sided"
#define PACXTYPE_COMM_LABEL         PACXTYPE_OTHER_LABEL
#define PACXTYPE_GROUP_LABEL        PACXTYPE_OTHER_LABEL
#define PACXTYPE_TOPOLOGIES_LABEL   PACXTYPE_OTHER_LABEL
#define PACXTYPE_TYPE_LABEL         PACXTYPE_OTHER_LABEL
#define PACXTYPE_IO_LABEL           "PACX I/O"

/* ==========================================================================
   ==== MPI Event Values
   ========================================================================== */
typedef enum
{
  PACX_END_VAL, /* 0 */
  PACX_SEND_VAL,
  PACX_RECV_VAL,
  PACX_ISEND_VAL,
  PACX_IRECV_VAL,
  PACX_WAIT_VAL,
  PACX_WAITALL_VAL,
  PACX_BCAST_VAL,
  PACX_BARRIER_VAL,
  PACX_REDUCE_VAL,
  PACX_ALLREDUCE_VAL, /* 10 */
  PACX_ALLTOALL_VAL,
  PACX_ALLTOALLV_VAL,
  PACX_GATHER_VAL,
  PACX_GATHERV_VAL,
  PACX_SCATTER_VAL,
  PACX_SCATTERV_VAL,
  PACX_ALLGATHER_VAL,
  PACX_ALLGATHERV_VAL,
  PACX_COMM_RANK_VAL,
  PACX_COMM_SIZE_VAL, /* 20 */
  PACX_COMM_CREATE_VAL,
  PACX_COMM_DUP_VAL,
  PACX_COMM_SPLIT_VAL,
  PACX_COMM_GROUP_VAL,
  PACX_COMM_FREE_VAL,
  PACX_COMM_REMOTE_GROUP_VAL,
  PACX_COMM_REMOTE_SIZE_VAL,
  PACX_COMM_TEST_INTER_VAL,
  PACX_COMM_COMPARE_VAL,
  PACX_SCAN_VAL, /* 30 */
  PACX_INIT_VAL,
  PACX_FINALIZE_VAL,
  PACX_BSEND_VAL,
  PACX_SSEND_VAL,
  PACX_RSEND_VAL,
  PACX_IBSEND_VAL,
  PACX_ISSEND_VAL,
  PACX_IRSEND_VAL,
  PACX_TEST_VAL,
  PACX_CANCEL_VAL, /* 40 */
  PACX_SENDRECV_VAL,
  PACX_SENDRECV_REPLACE_VAL,
  PACX_CART_CREATE_VAL,
  PACX_CART_SHIFT_VAL,
  PACX_CART_COORDS_VAL,
  PACX_CART_GET_VAL,
  PACX_CART_MAP_VAL,
  PACX_CART_RANK_VAL,
  PACX_CART_SUB_VAL,
  PACX_CARTDIM_GET_VAL, /* 50 */
  PACX_DIMS_CREATE_VAL,
  PACX_GRAPH_GET_VAL,
  PACX_GRAPH_MAP_VAL,
  PACX_GRAPH_CREATE_VAL,
  PACX_GRAPH_NEIGHBORS_VAL,
  PACX_GRAPHDIMS_GET_VAL,
  PACX_GRAPH_NEIGHBORS_COUNT_VAL,
  PACX_TOPO_TEST_VAL,
  PACX_WAITANY_VAL,
  PACX_WAITSOME_VAL, /* 60 */
  PACX_PROBE_VAL,
  PACX_IPROBE_VAL,
  PACX_WIN_CREATE_VAL,
  PACX_WIN_FREE_VAL,
  PACX_PUT_VAL,
  PACX_GET_VAL,
  PACX_ACCUMULATE_VAL,
  PACX_WIN_FENCE_VAL,
  PACX_WIN_START_VAL,
  PACX_WIN_COMPLETE_VAL, /* 70 */
  PACX_WIN_POST_VAL,
  PACX_WIN_WAIT_VAL,
  PACX_WIN_TEST_VAL,
  PACX_WIN_LOCK_VAL,
  PACX_WIN_UNLOCK_VAL,
  PACX_PACK_VAL,
  PACX_UNPACK_VAL,
  PACX_OP_CREATE_VAL,
  PACX_OP_FREE_VAL,
  PACX_REDUCE_SCATTER_VAL, /* 80 */
  PACX_ATTR_DELETE_VAL,
  PACX_ATTR_GET_VAL,
  PACX_ATTR_PUT_VAL,
  PACX_GROUP_DIFFERENCE_VAL,
  PACX_GROUP_EXCL_VAL,
  PACX_GROUP_FREE_VAL,
  PACX_GROUP_INCL_VAL,
  PACX_GROUP_INTERSECTION_VAL,
  PACX_GROUP_RANK_VAL,
  PACX_GROUP_RANGE_EXCL_VAL, /* 90 */
  PACX_GROUP_RANGE_INCL_VAL,
  PACX_GROUP_SIZE_VAL,
  PACX_GROUP_TRANSLATE_RANKS_VAL,
  PACX_GROUP_UNION_VAL,
  PACX_GROUP_COMPARE_VAL,
  PACX_INTERCOMM_CREATE_VAL,
  PACX_INTERCOMM_MERGE_VAL,
  PACX_KEYVAL_FREE_VAL,
  PACX_KEYVAL_CREATE_VAL,
  PACX_ABORT_VAL, /* 100 */
  PACX_ERROR_CLASS_VAL,
  PACX_ERRHANDLER_CREATE_VAL,
  PACX_ERRHANDLER_FREE_VAL,
  PACX_ERRHANDLER_GET_VAL,
  PACX_ERROR_STRING_VAL,
  PACX_ERRHANDLER_SET_VAL,
  PACX_GET_PROCESSOR_NAME_VAL,
  PACX_INITIALIZED_VAL,
  PACX_WTICK_VAL,
  PACX_WTIME_VAL, /* 110 */
  PACX_ADDRESS_VAL,
  PACX_BSEND_INIT_VAL,
  PACX_BUFFER_ATTACH_VAL,
  PACX_BUFFER_DETACH_VAL,
  PACX_REQUEST_FREE_VAL,
  PACX_RECV_INIT_VAL,
  PACX_SEND_INIT_VAL,
  PACX_GET_COUNT_VAL,
  PACX_GET_ELEMENTS_VAL,
  PACX_PACK_SIZE_VAL, /* 120 */
  PACX_RSEND_INIT_VAL,
  PACX_SSEND_INIT_VAL,
  PACX_START_VAL,
  PACX_STARTALL_VAL,
  PACX_TESTALL_VAL,
  PACX_TESTANY_VAL,
  PACX_TEST_CANCELLED_VAL,
  PACX_TESTSOME_VAL,
  PACX_TYPE_COMMIT_VAL,
  PACX_TYPE_CONTIGUOUS_VAL, /* 130 */
  PACX_TYPE_EXTENT_VAL,
  PACX_TYPE_FREE_VAL,
  PACX_TYPE_HINDEXED_VAL,
  PACX_TYPE_HVECTOR_VAL,
  PACX_TYPE_INDEXED_VAL,
  PACX_TYPE_LB_VAL,
  PACX_TYPE_SIZE_VAL,
  PACX_TYPE_STRUCT_VAL,
  PACX_TYPE_UB_VAL,
  PACX_TYPE_VECTOR_VAL, /* 140 */
  PACX_FILE_OPEN_VAL,
  PACX_FILE_CLOSE_VAL,
  PACX_FILE_READ_VAL,
  PACX_FILE_READ_ALL_VAL,
  PACX_FILE_WRITE_VAL,
  PACX_FILE_WRITE_ALL_VAL,
  PACX_FILE_READ_AT_VAL,
  PACX_FILE_READ_AT_ALL_VAL,
  PACX_FILE_WRITE_AT_VAL,
  PACX_FILE_WRITE_AT_ALL_VAL /* 150 */
}
PACXVal;

/* ==========================================================================
   ==== MPI Event Labels
   ========================================================================== */

#define  MPIEND_LABEL                      "End"
#define  PACX_SEND_LABEL                    "PACX_Send"
#define  PACX_RECV_LABEL                    "PACX_Recv"
#define  PACX_ISEND_LABEL                   "PACX_Isend"
#define  PACX_IRECV_LABEL                   "PACX_Irecv"
#define  PACX_WAIT_LABEL                    "PACX_Wait"
#define  PACX_WAITALL_LABEL                 "PACX_Waitall"

#define  PACX_BCAST_LABEL                   "PACX_Bcast"
#define  PACX_BARRIER_LABEL                 "PACX_Barrier"
#define  PACX_REDUCE_LABEL                  "PACX_Reduce"
#define  PACX_ALLREDUCE_LABEL               "PACX_Allreduce"

#define  PACX_ALLTOALL_LABEL                "PACX_Alltoall"
#define  PACX_ALLTOALLV_LABEL               "PACX_Alltoallv"
#define  PACX_GATHER_LABEL                  "PACX_Gather"
#define  PACX_GATHERV_LABEL                 "PACX_Gatherv"
#define  PACX_SCATTER_LABEL                 "PACX_Scatter"
#define  PACX_SCATTERV_LABEL                "PACX_Scatterv"
#define  PACX_ALLGATHER_LABEL               "PACX_Allgather"
#define  PACX_ALLGATHERV_LABEL              "PACX_Allgatherv"

#define  PACX_SCAN_LABEL                    "PACX_Scan"

#define  PACX_INIT_LABEL                    "PACX_Init"
#define  PACX_FINALIZE_LABEL                "PACX_Finalize"
#define  PACX_BSEND_LABEL                   "PACX_Bsend"
#define  PACX_SSEND_LABEL                   "PACX_Ssend"
#define  PACX_RSEND_LABEL                   "PACX_Rsend"
#define  PACX_IBSEND_LABEL                  "PACX_Ibsend"
#define  PACX_ISSEND_LABEL                  "PACX_Issend"
#define  PACX_IRSEND_LABEL                  "PACX_Irsend"
#define  PACX_TEST_LABEL                    "PACX_Test"
#define  PACX_CANCEL_LABEL                  "PACX_Cancel"
#define  PACX_SENDRECV_LABEL                "PACX_Sendrecv"
#define  PACX_SENDRECV_REPLACE_LABEL        "PACX_Sendrecv_replace"
#define  PACX_CART_CREATE_LABEL             "PACX_Cart_create"
#define  PACX_CART_SHIFT_LABEL              "PACX_Cart_shift"
#define  PACX_CART_COORDS_LABEL             "PACX_Cart_coords"
#define  PACX_CART_GET_LABEL                "PACX_Cart_get"
#define  PACX_CART_MAP_LABEL                "PACX_Cart_map"
#define  PACX_CART_RANK_LABEL               "PACX_Cart_rank"
#define  PACX_CART_SUB_LABEL                "PACX_Cart_sub"
#define  PACX_CARTDIM_GET_LABEL             "PACX_Cartdim_get"
#define  PACX_DIMS_CREATE_LABEL             "PACX_Dims_create"
#define  PACX_GRAPH_GET_LABEL               "PACX_Graph_get"
#define  PACX_GRAPH_MAP_LABEL               "PACX_Graph_map"
#define  PACX_GRAPH_CREATE_LABEL            "PACX_Graph_create"
#define  PACX_GRAPH_NEIGHBORS_LABEL         "PACX_Graph_neighbors"
#define  PACX_GRAPHDIMS_GET_LABEL           "PACX_Graphdims_get"
#define  PACX_GRAPH_NEIGHBORS_COUNT_LABEL   "PACX_Graph_neighbors_count"
#define  PACX_WAITANY_LABEL                 "PACX_Waitany"
#define  PACX_TOPO_TEST_LABEL               "PACX_Topo_test"
#define  PACX_WAITSOME_LABEL                "PACX_Waitsome"
#define  PACX_PROBE_LABEL                   "PACX_Probe"
#define  PACX_IPROBE_LABEL                  "PACX_Iprobe"

#define  PACX_WIN_CREATE_LABEL              "PACX_Win_create"
#define  PACX_WIN_FREE_LABEL                "PACX_Win_free"
#define  PACX_PUT_LABEL                     "PACX_Put"
#define  PACX_GET_LABEL                     "PACX_Get"
#define  PACX_ACCUMULATE_LABEL              "PACX_Accumulate"
#define  PACX_WIN_FENCE_LABEL               "PACX_Win_fence"
#define  PACX_WIN_START_LABEL               "PACX_Win_complete"
#define  PACX_WIN_COMPLETE_LABEL            "PACX_Win_start"
#define  PACX_WIN_POST_LABEL                "PACX_Win_post"
#define  PACX_WIN_WAIT_LABEL                "PACX_Win_wait"
#define  PACX_WIN_TEST_LABEL                "PACX_Win_test"
#define  PACX_WIN_LOCK_LABEL                "PACX_Win_lock"
#define  PACX_WIN_UNLOCK_LABEL              "PACX_Win_unlock"

#define  PACX_PACK_LABEL                    "PACX_Pack"
#define  PACX_UNPACK_LABEL                  "PACX_Unpack"

#define  PACX_OP_CREATE_LABEL               "PACX_Op_create"
#define  PACX_OP_FREE_LABEL                 "PACX_Op_free"
#define  PACX_REDUCE_SCATTER_LABEL          "PACX_Reduce_scatter"

#define  PACX_ATTR_DELETE_LABEL             "PACX_Attr_delete"
#define  PACX_ATTR_GET_LABEL                "PACX_Attr_get"
#define  PACX_ATTR_PUT_LABEL                "PACX_Attr_put"

#define  PACX_COMM_RANK_LABEL               "PACX_Comm_rank"
#define  PACX_COMM_SIZE_LABEL               "PACX_Comm_size"
#define  PACX_COMM_CREATE_LABEL             "PACX_Comm_create"
#define  PACX_COMM_DUP_LABEL                "PACX_Comm_dup"
#define  PACX_COMM_SPLIT_LABEL              "PACX_Comm_split"
#define  PACX_COMM_GROUP_LABEL              "PACX_Comm_group"
#define  PACX_COMM_FREE_LABEL               "PACX_Comm_free"
#define  PACX_COMM_REMOTE_GROUP_LABEL       "PACX_Comm_remote_group"
#define  PACX_COMM_REMOTE_SIZE_LABEL        "PACX_Comm_remote_size"
#define  PACX_COMM_TEST_INTER_LABEL         "PACX_Comm_test_inter"
#define  PACX_COMM_COMPARE_LABEL            "PACX_Comm_compare"

#define  PACX_GROUP_DIFFERENCE_LABEL        "PACX_Group_difference"
#define  PACX_GROUP_EXCL_LABEL              "PACX_Group_excl"
#define  PACX_GROUP_FREE_LABEL              "PACX_Group_free"
#define  PACX_GROUP_INCL_LABEL              "PACX_Group_incl"
#define  PACX_GROUP_INTERSECTION_LABEL      "PACX_Group_intersection"
#define  PACX_GROUP_RANK_LABEL              "PACX_Group_rank"
#define  PACX_GROUP_RANGE_EXCL_LABEL        "PACX_Group_range_excl"
#define  PACX_GROUP_RANGE_INCL_LABEL        "PACX_Group_range_incl"
#define  PACX_GROUP_SIZE_LABEL              "PACX_Group_size"
#define  PACX_GROUP_TRANSLATE_RANKS_LABEL   "PACX_Group_translate_ranks"
#define  PACX_GROUP_UNION_LABEL             "PACX_Group_union"
#define  PACX_GROUP_COMPARE_LABEL           "PACX_Group_compare"

#define  PACX_INTERCOMM_CREATE_LABEL        "PACX_Intercomm_create"
#define  PACX_INTERCOMM_MERGE_LABEL         "PACX_Intercomm_merge"
#define  PACX_KEYVAL_FREE_LABEL             "PACX_Keyval_free"
#define  PACX_KEYVAL_CREATE_LABEL           "PACX_Keyval_create"
#define  PACX_ABORT_LABEL                   "PACX_Abort"
#define  PACX_ERROR_CLASS_LABEL             "PACX_Error_class"
#define  PACX_ERRHANDLER_CREATE_LABEL       "PACX_Errhandler_create"
#define  PACX_ERRHANDLER_FREE_LABEL         "PACX_Errhandler_free"
#define  PACX_ERRHANDLER_GET_LABEL          "PACX_Errhandler_get"
#define  PACX_ERROR_STRING_LABEL            "PACX_Error_string"
#define  PACX_ERRHANDLER_SET_LABEL          "PACX_Errhandler_set"
#define  PACX_GET_PROCESSOR_NAME_LABEL      "PACX_Get_processor_name"
#define  PACX_INITIALIZED_LABEL             "PACX_Initialized"
#define  PACX_WTICK_LABEL                   "PACX_Wtick"
#define  PACX_WTIME_LABEL                   "PACX_Wtime"
#define  PACX_ADDRESS_LABEL                 "PACX_Address"
#define  PACX_BSEND_INIT_LABEL              "PACX_Bsend_init"
#define  PACX_BUFFER_ATTACH_LABEL           "PACX_Buffer_attach"
#define  PACX_BUFFER_DETACH_LABEL           "PACX_Buffer_detach"
#define  PACX_REQUEST_FREE_LABEL            "PACX_Request_free"
#define  PACX_RECV_INIT_LABEL               "PACX_Recv_init"
#define  PACX_SEND_INIT_LABEL               "PACX_Send_init"
#define  PACX_GET_COUNT_LABEL               "PACX_Get_count"
#define  PACX_GET_ELEMENTS_LABEL            "PACX_Get_elements"
#define  PACX_PACK_SIZE_LABEL               "PACX_Pack_size"
#define  PACX_RSEND_INIT_LABEL              "PACX_Rsend_init"
#define  PACX_SSEND_INIT_LABEL              "PACX_Ssend_init"
#define  PACX_START_LABEL                   "PACX_Start"
#define  PACX_STARTALL_LABEL                "PACX_Startall"
#define  PACX_TESTALL_LABEL                 "PACX_Testall"
#define  PACX_TESTANY_LABEL                 "PACX_Testany"
#define  PACX_TEST_CANCELLED_LABEL          "PACX_Test_cancelled"
#define  PACX_TESTSOME_LABEL                "PACX_Testsome"
#define  PACX_TYPE_COMMIT_LABEL             "PACX_Type_commit"
#define  PACX_TYPE_CONTIGUOUS_LABEL         "PACX_Type_contiguous"
#define  PACX_TYPE_EXTENT_LABEL             "PACX_Type_extent"
#define  PACX_TYPE_FREE_LABEL               "PACX_Type_free"
#define  PACX_TYPE_HINDEXED_LABEL           "PACX_Type_hindexed"
#define  PACX_TYPE_HVECTOR_LABEL            "PACX_Type_hvector"
#define  PACX_TYPE_INDEXED_LABEL            "PACX_Type_indexed"
#define  PACX_TYPE_LB_LABEL                 "PACX_Type_lb"
#define  PACX_TYPE_SIZE_LABEL               "PACX_Type_size"
#define  PACX_TYPE_STRUCT_LABEL             "PACX_Type_struct"
#define  PACX_TYPE_UB_LABEL                 "PACX_Type_ub"
#define  PACX_TYPE_VECTOR_LABEL             "PACX_Type_vector"
#define  PACX_FILE_OPEN_LABEL               "PACX_File_open"
#define  PACX_FILE_CLOSE_LABEL              "PACX_File_close"
#define  PACX_FILE_READ_LABEL               "PACX_File_read"
#define  PACX_FILE_READ_ALL_LABEL           "PACX_File_read_all"
#define  PACX_FILE_WRITE_LABEL              "PACX_File_write"
#define  PACX_FILE_WRITE_ALL_LABEL          "PACX_File_write_all"
#define  PACX_FILE_READ_AT_LABEL            "PACX_File_read_at"
#define  PACX_FILE_READ_AT_ALL_LABEL        "PACX_File_read_at_all"
#define  PACX_FILE_WRITE_AT_LABEL           "PACX_File_write_at"
#define  PACX_FILE_WRITE_AT_ALL_LABEL       "PACX_File_write_at_all"


#endif /* _PACX_EVENTENCODING_H */
