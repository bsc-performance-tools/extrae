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

#ifndef MPI_STATS_DEFINED
#define MPI_STATS_DEFINED

#include "stats_types.h"
#include "common.h"
#include <mpi.h>

//MPI statistics ids
enum {
   MPI_BURST_STATS_P2P_COUNT = 0,
   MPI_BURST_STATS_P2P_BYTES_SENT,
   MPI_BURST_STATS_P2P_BYTES_RECV,
   MPI_BURST_STATS_GLOBAL_COUNT,
   MPI_BURST_STATS_GLOBAL_BYTES_SENT,
   MPI_BURST_STATS_GLOBAL_BYTES_RECV,
   MPI_BURST_STATS_GLOBAL_COMM_WORLD_COUNT,
   MPI_BURST_STATS_TIME_IN_MPI, 
   MPI_BURST_STATS_P2P_INCOMING_COUNT,
   MPI_BURST_STATS_P2P_OUTGOING_COUNT,
   MPI_BURST_STATS_P2P_INCOMING_PARTNERS_COUNT,
   MPI_BURST_STATS_P2P_OUTGOING_PARTNERS_COUNT,
   MPI_BURST_STATS_TIME_IN_OTHER,
   MPI_BURST_STATS_TIME_IN_P2P,
   MPI_BURST_STATS_TIME_IN_GLOBAL,
   MPI_BURST_STATS_OTHER_COUNT, 
   MPI_BURST_STATS_COUNT
};

//categories to store some statistics in arrays
enum
{
  OTHER,
  P2P,
  COLLECTIVE,
  NUM_MPI_CATEGORIES
};

typedef struct stats_mpi_thread_data
{
  UINT64 begin_time[NUM_MPI_CATEGORIES];
  UINT64 elapsed_time[NUM_MPI_CATEGORIES];
  int ntasks;                     /* Number of tasks */
  int P2P_Bytes_Sent;             /* Sent bytes by point to point MPI operations */
  int P2P_Bytes_Recv;             /* Recv bytes by point to point MPI operations */
  int P2P_Communications;         /* Number of point to point communications */
  int P2P_Communications_In;      /* Number of input communication by point to point MPI operations */
  int P2P_Communications_Out;     /* Number of output communication by point to point MPI operations */
  int *P2P_Partner_In;            /* Number of partners in */
  int *P2P_Partner_Out;           /* Nuber of partners out */
  int COLLECTIVE_Bytes_Sent;      /* Sent "bytes" by MPI global operations */
  int COLLECTIVE_Bytes_Recv;      /* Recv "bytes" by MPI global operations */
  int COLLECTIVE_Communications;  /* Number of global operations */
  int COLL_CommWorld_Communications;  /* Number of global operations in CommWorld*/
  int MPI_Others_count;           /* Number of global operations */

}stats_mpi_thread_data_t;

/**
 * Statistic object, first field is  'xtr_stats_t' as required by the stats manager.
 */
typedef struct xtr_MPI_stats
{
  xtr_stats_t common_stats_field;
  int num_threads; //necessary to free the allocated memory 
}xtr_MPI_stats_t;


void *xtr_stats_MPI_init(void);
void xtr_stats_MPI_realloc(xtr_MPI_stats_t * mpi_stats, int new_num_threads);
void xtr_stats_MPI_reset(int threadid, xtr_MPI_stats_t * mpi_stats) ;
xtr_MPI_stats_t *xtr_stats_MPI_dup (xtr_MPI_stats_t * mpi_stats);
void xtr_stats_MPI_copy(int threadid, xtr_MPI_stats_t * stats_origin, xtr_MPI_stats_t * stats_destination);
void xtr_stats_MPI_subtract(int threadid, xtr_MPI_stats_t * mpi_stats, xtr_MPI_stats_t * subtrahend, xtr_MPI_stats_t * destination);
int xtr_stats_MPI_get_values(int threadid, xtr_MPI_stats_t * mpi_stats, INT32 * out_statistic_type, UINT64 * out_values);
int xtr_stats_MPI_get_positive_values(int threadid, xtr_MPI_stats_t * mpi_stats, INT32 * out_statistic_type, UINT64 * out_values);
stats_info_t *xtr_stats_MPI_get_types_and_descriptions(void);

void xtr_stats_MPI_update_P2P(UINT64 begin_time, UINT64 end_time, int partner, int inputSize, int outputSize);
void xtr_stats_MPI_update_collective(UINT64 begin_time, UINT64 end_time, int inputSize, int outputSize, MPI_Comm communicator);
void xtr_stats_MPI_update_other(UINT64 begin_time, UINT64 end_time );

void xtr_stats_MPI_free (xtr_MPI_stats_t * mpi_stats );
void xtr_print_debug_mpi_stats ( int threadid );

#endif /* End of MPI_STATS_DEFINED */