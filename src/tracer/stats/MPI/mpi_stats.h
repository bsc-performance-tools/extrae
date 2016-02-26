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

#include <stdlib.h>
#include "common.h"
#include "mpi_utils.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct mpi_stats_t
{
/* MPI Stats */
    int ntasks;             /* Just to create vectors with the proper size */
    int P2P_Bytes_Sent;      /* Sent bytes by point to point MPI operations */
    int P2P_Bytes_Recv;      /* Recv bytes by point to point MPI operations */
    int COLLECTIVE_Bytes_Sent;      /* Sent "bytes" by MPI global operations */
    int COLLECTIVE_Bytes_Recv;      /* Recv "bytes" by MPI global operations */
    int P2P_Communications;      /* Number of point to point communications */
    int COLLECTIVE_Communications;      /* Number of global operations */
    int MPI_Others_count;      /* Number of global operations */
    unsigned long long Elapsed_Time_In_MPI;     /* Elapsed time in MPI */

    int P2P_Communications_In;      /* Number of input communication by point to point MPI operations */
    int P2P_Communications_Out; /* Number of output communication by point to point MPI operations */
    int * P2P_Partner_In;              /* Number of partners in */
    int * P2P_Partner_Out;             /* Nuber of partners out */
    unsigned long long Elapsed_Time_In_P2P_MPI; /* Time inside P2P MPI calls */
    unsigned long long Elapsed_Time_In_COLLECTIVE_MPI; /* Time inside global MPI calls */
} mpi_stats_t;

extern mpi_stats_t *global_mpi_stats;

mpi_stats_t * mpi_stats_init(int num_tasks);
void mpi_stats_reset(mpi_stats_t * mpi_stats);
void mpi_stats_free(mpi_stats_t * mpi_stats);
void mpi_stats_sum(mpi_stats_t * base, mpi_stats_t * extra);
void updateStats_P2P(mpi_stats_t * mpi_stats, int partner, int inputSize, int outputSize);
void updateStats_COLLECTIVE(mpi_stats_t * mpi_stats, int inputSize, int outputSize);
void updateStats_OTHER(mpi_stats_t * mpi_stats);
int mpi_stats_get_num_partners(mpi_stats_t * mpi_stats, int * partners_vector);
void mpi_stats_update_elapsed_time(mpi_stats_t * mpi_stats, unsigned EvtType, unsigned long long elapsedTime);


#if defined(__cplusplus)
}
#endif

#endif /* End of MPI_STATS_DEFINED */
