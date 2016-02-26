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

#include "common.h"

#include <iostream>
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "Bursts.h"
#include "tags.h"
#if defined(BACKEND)
# include "taskid.h"
# include "timesync.h"
# include "online_buffers.h"
#if USE_HARDWARE_COUNTERS
# include "num_hwc.h"
# include "hwc.h"
#endif /* USE_HARDWARE_COUNTERS */
#endif /* BACKEND */
#include "utils.h"

Bursts::Bursts()
{
  NumberOfBursts = 0;
  MaxBursts      = 0;

  Timestamps = NULL;
  Durations  = NULL;

  BurstStats.clear();
  AccumulatedStats.clear();
}

Bursts::~Bursts()
{
  if (NumberOfBursts > 0)
  {
    free(Timestamps);
    free(Durations);
    for (int i=0; i<NumberOfBursts; i++)
    {
      delete BurstStats[i];
      delete AccumulatedStats[i];
    }
  }
}

void Bursts::Insert(unsigned long long timestamp, unsigned long long duration, PhaseStats *PreviousPhase, PhaseStats *CurrentPhase)
{
  if (NumberOfBursts == MaxBursts)
  {
    MaxBursts += BURSTS_CHUNK;
    Timestamps = (unsigned long long *)realloc(Timestamps, MaxBursts * sizeof(unsigned long long));
    Durations  = (unsigned long long *)realloc(Durations, MaxBursts * sizeof(unsigned long long));
  }
  Timestamps[ NumberOfBursts ] = timestamp;
  Durations [ NumberOfBursts ] = duration;
  NumberOfBursts ++;

  AccumulatedStats.push_back( PreviousPhase );
  BurstStats.push_back( CurrentPhase );
}


/**
 * Returns the number of bursts in the container.
 * \return The total count of bursts.
 */
int Bursts::GetNumberOfBursts()
{
  return NumberOfBursts;
}

/**
 * Returns the time of the specified burst.
 * \return The timestamp of the burst.
 */
unsigned long long Bursts::GetBurstTime(int burst_id)
{
  if (burst_id >= NumberOfBursts)
    return 0;
  else
    return Timestamps[burst_id];
}

/**
 * Returns the duration of the specified burst.
 * \return The duration of the burst.
 */
unsigned long long Bursts::GetBurstDuration(int burst_id)
{
  if (burst_id >= NumberOfBursts)
    return 0;
  else
    return Durations[burst_id];
}

#if defined(BACKEND)
#if USE_HARDWARE_COUNTERS 
using std::cerr;
using std::cout;
using std::endl;

void Bursts::GetCounters(int burst_id, map<unsigned int, long unsigned int> &BurstHWCs)
{
  map<unsigned int, long unsigned int> HWCsAtBegin, HWCsAtEnd;
  map<unsigned int, long unsigned int>::iterator it;

  int last_set = AccumulatedStats[burst_id]->GetLastSet();
  int first_set = BurstStats[burst_id]->GetFirstSet();

  if (last_set != first_set)
  {
    /* The counter set was changed just at the beginning of the burst, and so they were reset to 0.
     * Then we don't need to compute any differences, just read the counters at the end of the burst. 
     */
    //BurstStats[burst_id]->GetCommonCounters( BurstHWCs );
    BurstStats[burst_id]->GetAllCounters( BurstHWCs );
  }
  else
  {
    /* Compute the difference between the end and the beginning of the burst */
    
    //AccumulatedStats[burst_id]->GetLastCommonCounters(HWCsAtBegin); 
    //BurstStats[burst_id]->GetCommonCounters(HWCsAtEnd);
    AccumulatedStats[burst_id]->GetLastAllCounters(HWCsAtBegin); 
    BurstStats[burst_id]->GetAllCounters(HWCsAtEnd);

    BurstHWCs.clear();
    for (it=HWCsAtEnd.begin(); it!=HWCsAtEnd.end(); ++it)
    {
      BurstHWCs[ it->first ] = it->second - HWCsAtBegin[ it->first ];
    }
  }
}

#endif

void Bursts::EmitPhase(unsigned long long from, unsigned long long to, bool initialize, bool dump_hwcs)
{
  unsigned long long global_from = TIMESYNC( 0, TASKID, from ); 
  unsigned long long global_to   = TIMESYNC( 0, TASKID, to ); 
  PhaseStats *Phase = new PhaseStats( Extrae_get_num_tasks() );

//if (TASKID == 0) fprintf(stderr, "[DEBUG] EmitPhase BEGINS ============================== from=%llu (global_from=%llu) to=%llu (global_to=%llu)\n", from, global_from, to, global_to);

  for (int i=0; i<NumberOfBursts; i++)
  {
    if ((Timestamps[i] >= global_from) && (Timestamps[i] + Durations[i] <= global_to))
    {
//if (TASKID == 0) fprintf(stderr, "[DEBUG] Burst from %llu to %llu\n", TIMEDESYNC(0, TASKID, Timestamps[i]), TIMEDESYNC(0, TASKID, Timestamps[i] + Durations[i]));
      if (i > 0)
        Phase->Concatenate( AccumulatedStats[i] );
      Phase->Concatenate( BurstStats[i] );
    }
    if (Timestamps[i] > global_to) break;
  }

//if (TASKID == 0) Phase->Dump();

  if (initialize)
    Phase->DumpZeros( from );
  Phase->DumpToTrace( to, dump_hwcs );
}

void Bursts::EmitBursts(unsigned long long local_from, unsigned long long local_to, unsigned long long threshold)
{


  unsigned long long global_from = TIMESYNC( 0, TASKID, local_from );
  unsigned long long global_to   = TIMESYNC( 0, TASKID, local_to );
  PhaseStats *Accum = new PhaseStats( Extrae_get_num_tasks() );

  for (int i=0; i<NumberOfBursts; i++)
  {
    if ((Timestamps[i] >= global_from) && (Timestamps[i] + Durations[i] <= global_to))
    {
      if (Durations[i] > threshold)
      {
        unsigned long long burst_ini_ts = TIMEDESYNC( 0, TASKID, Timestamps[i] );
        unsigned long long burst_end_ts = burst_ini_ts + Durations[i];

        Accum->Concatenate( AccumulatedStats[i] );
        TRACE_ONLINE_EVENT( burst_ini_ts, CPU_BURST_EV, 1);
	Accum->DumpToTrace( burst_ini_ts, true );
        TRACE_ONLINE_EVENT( burst_end_ts, CPU_BURST_EV, 0);
        BurstStats[i]->DumpToTrace( burst_end_ts, true );
        Accum->Reset();
      }
      else
      {
        Accum->Concatenate( AccumulatedStats[i] );
        Accum->Concatenate( BurstStats[i] );
      }
    }
  }
  delete Accum;
}

#endif
