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

#include "PhaseStats.h"
#include "taskid.h"
#include "utils.h"
#if defined(BACKEND)
# include "timesync.h"
# include "online_buffers.h"
# if USE_HARDWARE_COUNTERS
#  include "hwc.h"
# endif
#endif

using std::make_pair;

PhaseStats::PhaseStats(int num_tasks)
{
  MPI_Stats = mpi_stats_init( num_tasks );

#if USE_HARDWARE_COUNTERS && defined(BACKEND)
  HWC_Stats.clear();

  /* Query the counters read in each counter set */
  int num_sets = HWC_Get_Num_Sets();

  for (int set_id=0; set_id < num_sets; set_id++)
  {
    int *array_hwc_ids = NULL;
    int  num_hwcs  = HWC_Get_Set_Counters_ParaverIds(set_id, &array_hwc_ids);
    vector<int> vector_hwc_ids;
    for (int hwc=0; hwc < num_hwcs; hwc++)
    {
      vector_hwc_ids.push_back( array_hwc_ids[hwc] );
    }
    xfree(array_hwc_ids);
    HWCSetToIds[set_id] = vector_hwc_ids;
  }
#endif
}

PhaseStats::~PhaseStats()
{
  mpi_stats_free( MPI_Stats );
}

void PhaseStats::UpdateMPI(event_t *MPIBeginEv, event_t *MPIEndEv)
{
  unsigned int EvType = 0;
  int me    = 0;
  int root  = 0;
  int csize = 0;

  if ((MPIBeginEv != NULL) && (MPIEndEv != NULL))
  {
    EvType = Get_EvEvent(MPIBeginEv);
    switch(EvType)
    {
      /* Peer-to-peer */
      case MPI_BSEND_EV:
      case MPI_SSEND_EV:
      case MPI_RSEND_EV:
      case MPI_SEND_EV:
      case MPI_IBSEND_EV:
      case MPI_ISEND_EV:
      case MPI_ISSEND_EV:
      case MPI_IRSEND_EV:
        updateStats_P2P( MPI_Stats, Get_EvTarget(MPIEndEv), 0, Get_EvSize(MPIEndEv) );
        break;
      case MPI_RECV_EV:
        updateStats_P2P( MPI_Stats, Get_EvTarget(MPIEndEv), Get_EvSize(MPIEndEv), 0 );
        break;
      case MPI_SENDRECV_EV:
      case MPI_SENDRECV_REPLACE_EV:
        updateStats_P2P( MPI_Stats, Get_EvTarget(MPIBeginEv), 0, Get_EvSize(MPIBeginEv) );
        updateStats_P2P( MPI_Stats, Get_EvTarget(MPIEndEv), Get_EvSize(MPIEndEv), 0 );
        break;
      
      /* Collectives */
      case MPI_REDUCE_EV:
        me   = Get_EvTag(MPIBeginEv);
        root = Get_EvAux(MPIBeginEv);
        if (me == root)
          updateStats_COLLECTIVE( MPI_Stats, Get_EvSize(MPIBeginEv), 0);
        else
          updateStats_COLLECTIVE( MPI_Stats, 0, Get_EvSize(MPIBeginEv));
        break;
  
      case MPI_ALLREDUCE_EV:
        updateStats_COLLECTIVE( MPI_Stats, Get_EvSize(MPIBeginEv), Get_EvSize(MPIBeginEv) );
        break;
  
      case MPI_BARRIER_EV:
        updateStats_COLLECTIVE( MPI_Stats, 0, 0 );
        break;
      
      case MPI_BCAST_EV:
        me   = Get_EvTag(MPIBeginEv);
        root = Get_EvTarget(MPIBeginEv);
        if (me == root)
          updateStats_COLLECTIVE( MPI_Stats, 0, Get_EvSize(MPIBeginEv));
        else
          updateStats_COLLECTIVE( MPI_Stats, Get_EvSize(MPIBeginEv), 0);
        break;
    
      case MPI_ALLTOALL_EV:
      case MPI_ALLTOALLV_EV:
        updateStats_COLLECTIVE( MPI_Stats, Get_EvTarget(MPIBeginEv), Get_EvSize(MPIBeginEv) );
        break;
  
      case MPI_ALLGATHER_EV:
      case MPI_ALLGATHERV_EV:
        updateStats_COLLECTIVE( MPI_Stats, Get_EvAux(MPIBeginEv), Get_EvSize(MPIBeginEv) );
        break;

      case MPI_GATHER_EV:
      case MPI_GATHERV_EV:
        me   = Get_EvTag(MPIBeginEv);
        root = Get_EvTarget(MPIBeginEv);
        if (me == root)
          updateStats_COLLECTIVE( MPI_Stats, Get_EvAux(MPIBeginEv), 0);
        else
          updateStats_COLLECTIVE( MPI_Stats, 0, Get_EvSize(MPIBeginEv) );
        break;
  
      case MPI_SCATTER_EV:
      case MPI_SCATTERV_EV:
        me   = Get_EvTag(MPIBeginEv);
        root = Get_EvTarget(MPIBeginEv);
        if (me == root)
          updateStats_COLLECTIVE( MPI_Stats, 0, Get_EvSize(MPIBeginEv) );
        else
          updateStats_COLLECTIVE( MPI_Stats, Get_EvAux(MPIBeginEv), 0 );
        break;
  
      case MPI_REDUCESCAT_EV: 
        me   = Get_EvTag(MPIBeginEv);
        if (me == 0)
          updateStats_COLLECTIVE( MPI_Stats, Get_EvSize(MPIBeginEv), Get_EvSize(MPIBeginEv) );
        else
          updateStats_COLLECTIVE( MPI_Stats, Get_EvAux(MPIBeginEv), Get_EvSize(MPIBeginEv) );
        break;
  
      case MPI_SCAN_EV:
        me    = Get_EvTag(MPIBeginEv);
        csize = Get_EvSize(MPIEndEv);
        if (me == csize - 1)
          updateStats_COLLECTIVE( MPI_Stats, 0, Get_EvSize(MPIBeginEv) );
        else
          updateStats_COLLECTIVE( MPI_Stats, Get_EvSize(MPIBeginEv), 0 );
        break;
  
      /* Others */
      case MPI_INIT_EV:
      case MPI_PROBE_EV:
      case MPI_CANCEL_EV:
      case MPI_COMM_RANK_EV:
      case MPI_COMM_SIZE_EV:
      case MPI_COMM_CREATE_EV:
      case MPI_COMM_FREE_EV:
      case MPI_COMM_DUP_EV:
      case MPI_COMM_SPLIT_EV:
      case MPI_COMM_SPAWN_EV:
      case MPI_REQUEST_FREE_EV:
      case MPI_RECV_INIT_EV:
      case MPI_SEND_INIT_EV:
      case MPI_BSEND_INIT_EV:
      case MPI_RSEND_INIT_EV:
      case MPI_SSEND_INIT_EV:
      case MPI_CART_SUB_EV:
      case MPI_CART_CREATE_EV:
      case MPI_FILE_OPEN_EV:
      case MPI_FILE_CLOSE_EV:
      case MPI_FILE_READ_EV:
      case MPI_FILE_READ_ALL_EV:
      case MPI_FILE_WRITE_EV:
      case MPI_FILE_WRITE_ALL_EV:
      case MPI_FILE_READ_AT_EV:
      case MPI_FILE_READ_AT_ALL_EV:
      case MPI_FILE_WRITE_AT_EV:
      case MPI_FILE_WRITE_AT_ALL_EV:
      case MPI_GET_EV:
      case MPI_PUT_EV:
      case MPI_FINALIZE_EV:
        updateStats_OTHER( MPI_Stats );
      break;
    }
    /* Update the time in MPI */
    mpi_stats_update_elapsed_time( MPI_Stats, EvType, Get_EvTime(MPIEndEv) - Get_EvTime(MPIBeginEv) );
  }
}

void PhaseStats::UpdateMPI(event_t *SingleMPIEv)
{
  unsigned int EvType = Get_EvEvent(SingleMPIEv);
  switch(EvType)
  {
    case MPI_IRECVED_EV: /* TEST*, WAIT* */
      updateStats_P2P( MPI_Stats, Get_EvTarget(SingleMPIEv), Get_EvSize(SingleMPIEv), 0 );
      break;

    case MPI_PERSIST_REQ_EV:
      unsigned int type = Get_EvValue(SingleMPIEv);
      if (type == MPI_ISEND_EV)
        updateStats_P2P( MPI_Stats, Get_EvTarget(SingleMPIEv), 0, Get_EvSize(SingleMPIEv) );
      break;
  }
}


#if USE_HARDWARE_COUNTERS && defined(BACKEND)
void PhaseStats::UpdateHWC(event_t *Ev)
{
  if (Get_EvHWCRead(Ev))
  {
    unsigned long long ts          = Get_EvTime(Ev);
    int                current_set = Get_EvHWCSet(Ev);  

    if (HWC_Stats.rbegin() != HWC_Stats.rend())
    {
      int previous_set = HWC_Stats.rbegin()->second.first;

      if (current_set == previous_set)
      {
        /* Erase the accumulated counters in the last timestamp */
        HWC_Stats.erase( HWC_Stats.rbegin()->first );
      }
    }

    if (HWC_Stats.find( ts ) == HWC_Stats.end())
    {
      /* Store the accumulated counters in the current timestamp */
      HWC_Stats[ts]  = make_pair( Get_EvHWCSet(Ev), Get_EvHWCVal(Ev) );
    }
  }
}
#endif

void PhaseStats::Reset()
{
  mpi_stats_reset( MPI_Stats );

#if USE_HARDWARE_COUNTERS && defined(BACKEND)
  HWC_Stats.clear();
#endif
}


void PhaseStats::Concatenate(PhaseStats *NextPhase)
{

  mpi_stats_sum( this->GetMPIStats(), NextPhase->GetMPIStats() );

#if USE_HARDWARE_COUNTERS && defined(BACKEND)
  map< timestamp_t, hwc_set_val_pair_t >::iterator it;

  /* If the set of counters read last in the current phase is the same than the one read first in the next phase, discard the first and keep the latter */
  if ((this->HWC_Stats.size() > 0) && (NextPhase->HWC_Stats.size() > 0))
  {
    int this_set, next_set;

    this_set = this->HWC_Stats.rbegin()->second.first;
    next_set = NextPhase->HWC_Stats.begin()->second.first;

    if (this_set == next_set)
    {
      this->HWC_Stats.erase( this->HWC_Stats.rbegin()->first );
    }
  }

  for (it = NextPhase->HWC_Stats.begin(); it != NextPhase->HWC_Stats.end(); ++it)
  {
    this->HWC_Stats[ it->first ] = it->second;
  }
#endif
}

mpi_stats_t * PhaseStats::GetMPIStats()
{
  return MPI_Stats;
}

#if defined(BACKEND)
#if USE_HARDWARE_COUNTERS

void PhaseStats::GetAllCounters(map<unsigned int, long unsigned int> &Counters)
{
  map< timestamp_t, hwc_set_val_pair_t >::iterator it;

  for (it = HWC_Stats.begin(); it != HWC_Stats.end(); ++it)
  {
    int        set_id = it->second.first;
    long long *values = it->second.second;

    for (int current_hwc=0; current_hwc < MAX_HWC; current_hwc++)
    {
      int       type  = HWCSetToIds[set_id][current_hwc];
      long long value = values[current_hwc];

      if (type != -1)
      {
        if (Counters.find( type ) == Counters.end())
        {
          Counters[type] = value;
        }
        else
        {
          Counters[type] += value;
        }
      }
    }
  }
}

void PhaseStats::GetCommonCounters(map<unsigned int, long unsigned int> &Counters)
{
  map< timestamp_t, hwc_set_val_pair_t >::iterator it;

  for (it = HWC_Stats.begin(); it != HWC_Stats.end(); ++it)
  {
    int        set_id = it->second.first;
    long long *values = it->second.second;

    for (int current_hwc=0; current_hwc < MAX_HWC; current_hwc++)
    {
      int       type  = HWCSetToIds[set_id][current_hwc];
      long long value = values[current_hwc];


      if ((type != -1) && (HWC_IsCommonToAllSets(set_id, current_hwc)))
      {
        if (Counters.find( type ) == Counters.end())
        {
          Counters[type] = value;
        }
        else
        {
          Counters[type] += value;
        }
      }
    }
  }
}

void PhaseStats::GetLastAllCounters(map<unsigned int, long unsigned int> &Counters)
{
  map< timestamp_t, hwc_set_val_pair_t >::reverse_iterator it;

  it = HWC_Stats.rbegin();
  if (it != HWC_Stats.rend())
  {
    int        set_id = it->second.first;
    long long *values = it->second.second;

    for (int current_hwc=0; current_hwc < MAX_HWC; current_hwc++)
    {
      int       type  = HWCSetToIds[set_id][current_hwc];
      long long value = values[current_hwc];

      if (type != -1)
      {
        if (Counters.find( type ) == Counters.end())
        {
          Counters[type] = value;
        }
        else
        {
          Counters[type] += value;
        }
      }
    }
  }
}

void PhaseStats::GetLastCommonCounters(map<unsigned int, long unsigned int> &Counters)
{
  map< timestamp_t, hwc_set_val_pair_t >::reverse_iterator it;

  it = HWC_Stats.rbegin();
  if (it != HWC_Stats.rend())
  {
    int        set_id = it->second.first;
    long long *values = it->second.second;
    
    for (int current_hwc=0; current_hwc < MAX_HWC; current_hwc++)
    {
      int       type  = HWCSetToIds[set_id][current_hwc];
      long long value = values[current_hwc];

      if ((type != -1) && (HWC_IsCommonToAllSets(set_id, current_hwc)))
      {
        if (Counters.find( type ) == Counters.end())
        {
          Counters[type] = value;
        }
        else
        {
          Counters[type] += value;
        }
      }
    }
  }
}

int PhaseStats::GetLastSet()
{
  int set_id = -1;
  map< timestamp_t, hwc_set_val_pair_t >::reverse_iterator it;

  it = HWC_Stats.rbegin();
  if (it != HWC_Stats.rend())
  {
    set_id = it->second.first;
  }
  return set_id;
}

int PhaseStats::GetFirstSet()
{
  int set_id = -1;
  map< timestamp_t, hwc_set_val_pair_t >::iterator it;

  it = HWC_Stats.begin();
  if (it != HWC_Stats.end())
  {
    set_id = it->second.first;
  }
  return set_id;
}

#endif /* USE_HARDWARE_COUNTERS */

void PhaseStats::DumpZeros( unsigned long long timestamp )
{
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_COUNT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_BYTES_SENT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_BYTES_RECV_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_GLOBAL_COUNT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_GLOBAL_BYTES_SENT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_GLOBAL_BYTES_RECV_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_MPI_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_INCOMING_COUNT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_OUTGOING_COUNT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_INCOMING_PARTNERS_COUNT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_OUTGOING_PARTNERS_COUNT_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_OTHER_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_P2P_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_GLOBAL_EV, 0 );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_OTHER_COUNT_EV, 0 );
}

void PhaseStats::DumpToTrace( unsigned long long timestamp, bool dump_hwcs )
{
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_COUNT_EV, MPI_Stats->P2P_Communications );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_BYTES_SENT_EV, MPI_Stats->P2P_Bytes_Sent );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_BYTES_RECV_EV, MPI_Stats->P2P_Bytes_Recv );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_GLOBAL_COUNT_EV, MPI_Stats->COLLECTIVE_Communications );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_GLOBAL_BYTES_SENT_EV, MPI_Stats->COLLECTIVE_Bytes_Sent );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_GLOBAL_BYTES_RECV_EV, MPI_Stats->COLLECTIVE_Bytes_Recv );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_MPI_EV, MPI_Stats->Elapsed_Time_In_MPI );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_INCOMING_COUNT_EV, MPI_Stats->P2P_Communications_In );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_OUTGOING_COUNT_EV, MPI_Stats->P2P_Communications_Out );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_INCOMING_PARTNERS_COUNT_EV, mpi_stats_get_num_partners(MPI_Stats, MPI_Stats->P2P_Partner_In) );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_P2P_OUTGOING_PARTNERS_COUNT_EV, mpi_stats_get_num_partners(MPI_Stats, MPI_Stats->P2P_Partner_Out) );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_OTHER_EV, MPI_Stats->Elapsed_Time_In_MPI - MPI_Stats->Elapsed_Time_In_P2P_MPI - MPI_Stats->Elapsed_Time_In_COLLECTIVE_MPI );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_P2P_EV, MPI_Stats->Elapsed_Time_In_P2P_MPI );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_TIME_IN_GLOBAL_EV, MPI_Stats->Elapsed_Time_In_COLLECTIVE_MPI );
  TRACE_ONLINE_EVENT( timestamp, MPI_STATS_OTHER_COUNT_EV, MPI_Stats->MPI_Others_count );

#if USE_HARDWARE_COUNTERS
  if (dump_hwcs)
  {
    map< timestamp_t, hwc_set_val_pair_t >::iterator it;

    int count = 0;
    for (it=HWC_Stats.begin(); it!=HWC_Stats.end(); ++it)
    {
      unsigned long long hwc_ts   = it->first;
      int                hwc_set  = it->second.first;
      long long *        hwc_val  = it->second.second; 

/*
if (TASKID == 0)
  fprintf(stderr, "[DEBUG] PhaseStats::DumpToTrace: emitting HWCs at %llu set=%d\n", hwc_ts, hwc_set);
*/

      TRACE_ONLINE_COUNTERS( hwc_ts, hwc_set, hwc_val );

      count ++;
    }
  }
#endif /* USE_HARDWARE_COUNTERS */
}

void PhaseStats::Dump()
{
    map< timestamp_t, hwc_set_val_pair_t >::iterator it;
    int count = 1;
    for (it=HWC_Stats.begin(); it!=HWC_Stats.end(); ++it)
    {
      unsigned long long hwc_ts  = it->first;
      fprintf(stderr, "[DEBUG] HWC %d ts=%llu global_ts=%lu\n", count, hwc_ts, TIMESYNC(0, TASKID, hwc_ts));
      count ++;
    }
}

#endif /* BACKEND */
