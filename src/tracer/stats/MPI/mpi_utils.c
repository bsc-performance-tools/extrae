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

#include "mpi_utils.h"

static unsigned MPI_P2P_EVENT_TYPES[NUM_MPI_P2P_EVENT_TYPES] = {
    MPI_BSEND_EV,
    MPI_SSEND_EV,
    MPI_SEND_EV,
    MPI_SENDRECV_EV,
    MPI_SENDRECV_REPLACE_EV,
    MPI_RECV_EV,
    MPI_IBSEND_EV,
    MPI_ISSEND_EV,
    MPI_ISEND_EV,
    MPI_IRECV_EV,
    MPI_IRECVED_EV,
    MPI_TEST_EV,
    MPI_TESTALL_EV,
    MPI_TESTANY_EV,
    MPI_TESTSOME_EV,
    MPI_TEST_COUNTER_EV,
    MPI_WAIT_EV,
    MPI_RSEND_EV,
    MPI_IRSEND_EV,
    MPI_WAITALL_EV,
    MPI_WAITANY_EV,
    MPI_WAITSOME_EV,
    MPI_ALIAS_COMM_CREATE_EV,
    MPI_PERSIST_REQ_EV,
    MPI_RECV_INIT_EV,
    MPI_SEND_INIT_EV,
    MPI_BSEND_INIT_EV,
    MPI_RSEND_INIT_EV,
    MPI_SSEND_INIT_EV,
    MPI_START_EV,
    MPI_STARTALL_EV,
    MPI_PROBE_EV,  /* was computed as OTHER */
    MPI_IPROBE_EV, 
    MPI_IPROBE_COUNTER_EV,
    MPI_IBSEND_EV,
    MPI_ISSEND_EV,
    MPI_ISEND_EV,
    MPI_IRECV_EV,
    MPI_IRSEND_EV,
};

static unsigned MPI_COLLECTIVE_EVENT_TYPES[NUM_MPI_COLLECTIVE_EVENT_TYPES] = {
    MPI_BARRIER_EV,
    MPI_BCAST_EV,
    MPI_IRECV_EV,
    MPI_ALLTOALL_EV,
    MPI_ALLTOALLV_EV,
    MPI_ALLREDUCE_EV,
    MPI_REDUCE_EV,
    MPI_GATHER_EV,
    MPI_GATHERV_EV,
    MPI_SCATTER_EV,
    MPI_SCATTERV_EV,
    MPI_COMM_SPAWN_EV, /* was OTHERs */
    MPI_COMM_SPAWN_MULTIPLE_EV,
    MPI_ALLGATHER_EV,
    MPI_ALLGATHERV_EV,
    MPI_REDUCESCAT_EV, /* especially strange */
    MPI_SCAN_EV,
    MPI_TIME_OUTSIDE_IPROBES_EV,
};


static unsigned MPI_OTHER_EVENT_TYPES[NUM_MPI_OTHER_EVENT_TYPES] = {
    MPI_INIT_EV,
    MPI_FINALIZE_EV,
    MPI_CANCEL_EV,
    MPI_COMM_RANK_EV,
    MPI_COMM_SIZE_EV,
    MPI_COMM_CREATE_EV,
    MPI_COMM_DUP_EV,
    MPI_COMM_SPLIT_EV,
    MPI_CART_CREATE_EV,
    MPI_CART_SUB_EV,
    MPI_COMM_FREE_EV,
    MPI_REQUEST_FREE_EV,
    MPI_FILE_OPEN_EV,
    MPI_FILE_CLOSE_EV,
    MPI_FILE_READ_EV,
    MPI_FILE_READ_ALL_EV,
    MPI_FILE_WRITE_EV,
    MPI_FILE_WRITE_ALL_EV,
    MPI_FILE_READ_AT_EV,
    MPI_FILE_READ_AT_ALL_EV,
    MPI_FILE_WRITE_AT_EV,
    MPI_FILE_WRITE_AT_ALL_EV,
    MPI_GET_EV,
    MPI_PUT_EV,
    MPI_TIME_OUTSIDE_IPROBES_EV,
};

unsigned isMPI_Global(unsigned EvtType)
{
    int i;
    for (i = 0; i < NUM_MPI_COLLECTIVE_EVENT_TYPES; i++)
    {
        if (EvtType == MPI_COLLECTIVE_EVENT_TYPES[i])
            return TRUE;
    }
    return FALSE;
}

unsigned isMPI_P2P(unsigned EvtType)
{
    int i;
    for (i = 0; i < NUM_MPI_P2P_EVENT_TYPES; i++)
    {
        if (EvtType == MPI_P2P_EVENT_TYPES[i])
            return TRUE;
    }
    return FALSE;
}

unsigned isMPI_Others(unsigned EvtType)
{
    int i;
    for (i = 0; i < NUM_MPI_OTHER_EVENT_TYPES; i++)
    {
        if (EvtType == MPI_OTHER_EVENT_TYPES[i])
            return TRUE;
    }
    return FALSE;
}
