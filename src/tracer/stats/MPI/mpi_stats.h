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
  MPI_BURST_STATS_COLLECTIVE_COUNT,
  MPI_BURST_STATS_COLLECTIVE_BYTES_SENT,
  MPI_BURST_STATS_COLLECTIVE_BYTES_RECV,
  MPI_BURST_STATS_COLLECTIVE_WORLD_COUNT,
  MPI_BURST_STATS_COLLECTIVE_WORLD_BYTES_SENT,
  MPI_BURST_STATS_COLLECTIVE_WORLD_BYTES_RECV,
  MPI_BURST_STATS_TIME_IN_MPI, 
  MPI_BURST_STATS_P2P_INCOMING_COUNT,
  MPI_BURST_STATS_P2P_OUTGOING_COUNT,
  MPI_BURST_STATS_P2P_INCOMING_PARTNERS_COUNT,
  MPI_BURST_STATS_P2P_OUTGOING_PARTNERS_COUNT,
  MPI_BURST_STATS_TIME_IN_OTHER,
  MPI_BURST_STATS_TIME_IN_P2P,
  MPI_BURST_STATS_TIME_IN_COLLECTIVE,
  MPI_BURST_STATS_TIME_IN_COLLECTIVE_WORLD,
  MPI_BURST_STATS_OTHER_COUNT, 
  MPI_BURST_STATS_COUNT
};

//categories to store some statistics in arrays
enum
{
  OTHER,
  P2P,
  COLLECTIVE,
  COLLECTIVE_COMM_WORLD,
  NUM_MPI_CATEGORIES
};

typedef struct stats_mpi_thread_data
{
  UINT64 begin_time[NUM_MPI_CATEGORIES];
  UINT64 elapsed_time[NUM_MPI_CATEGORIES];
  int p2p_bytes_sent;                  /* Sent bytes by point to point MPI operations */
  int p2p_bytes_recv;                  /* Recv bytes by point to point MPI operations */
  int p2p_communications;              /* Number of point to point communications */
  int p2p_communications_in;           /* Number of input communication by point to point MPI operations */
  int p2p_communications_out;          /* Number of output communication by point to point MPI operations */
  int *p2p_partner_in;                 /* Number of partners in */
  int *p2p_partner_out;                /* Nuber of partners out */
  int collective_bytes_sent;           /* Sent "bytes" by MPI global operations */
  int collective_bytes_recv;           /* Recv "bytes" by MPI global operations */
  int collective_communications;       /* Number of global operations */
  int collective_world_bytes_sent;     /* Sent bytes in COMM_WORLD collectives*/
  int collective_world_bytes_recv;     /* Recieved bytes in COMM_WORLD collectives*/
  int collective_world_communications; /* Number of global operations in CommWorld*/
  int others_count;                    /* Number of global operations */
} stats_mpi_thread_data_t;

/**
 * Statistic object, first field is  'xtr_stats_t' as required by the stats manager.
 */
typedef struct xtr_MPI_stats
{
  xtr_stats_t common_stats_field;
  unsigned int num_threads; // necessary to free the allocated memory 
  int world_size;  // number of ranks in MPI_COMM_WORLD
} xtr_MPI_stats_t;


void *xtr_stats_MPI_init( void );
void xtr_stats_MPI_realloc( xtr_stats_t *stats, unsigned int new_num_threads );
void xtr_stats_MPI_reset( unsigned int threadid, xtr_stats_t *stats ) ;
xtr_stats_t *xtr_stats_MPI_dup ( xtr_stats_t *stats );
void xtr_stats_MPI_copy( unsigned int threadid, xtr_stats_t *stats_origin, xtr_stats_t *stats_destination );
void xtr_stats_MPI_subtract( unsigned int threadid, xtr_stats_t *stats, xtr_stats_t *subtrahend, xtr_stats_t *destination );
unsigned int xtr_stats_MPI_get_values( unsigned int threadid, xtr_stats_t * stats, INT32 *out_statistic_type, UINT64 *out_values );
unsigned int xtr_stats_MPI_get_positive_values( unsigned int threadid, xtr_stats_t *stats, INT32 *out_statistic_type, UINT64 *out_values );
stats_info_t *xtr_stats_MPI_get_types_and_descriptions( void );

void _xtr_stats_MPI_update_P2P( UINT64 begin_time, UINT64 end_time, int partner, int input_size, int output_size );
void _xtr_stats_MPI_update_collective( UINT64 begin_time, UINT64 end_time, int input_size, int output_size, int comm_size );
void _xtr_stats_MPI_update_other( UINT64 begin_time, UINT64 end_time );

void xtr_stats_MPI_free ( xtr_stats_t * mpi_stats );
void xtr_print_debug_mpi_stats ( unsigned int threadid );

#if defined(HAVE_BURST)
# define xtr_stats_MPI_update_P2P(...) _xtr_stats_MPI_update_P2P(__VA_ARGS__)
# define xtr_stats_MPI_update_collective(...) _xtr_stats_MPI_update_collective(__VA_ARGS__)
# define xtr_stats_MPI_update_other(...) _xtr_stats_MPI_update_other(__VA_ARGS__)
#else
# define xtr_stats_MPI_update_P2P(...)
# define xtr_stats_MPI_update_collective(...)
# define xtr_stats_MPI_update_other(...)
#endif

#endif /* End of MPI_STATS_DEFINED */
