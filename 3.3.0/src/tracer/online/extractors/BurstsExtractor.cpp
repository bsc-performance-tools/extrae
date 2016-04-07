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

#include "BurstsExtractor.h"
#include "timesync.h"
#include "taskid.h"
#include "hwc.h"

/**
 * Constructor. Allocates a new container to store the bursts extracted from the buffer.
 *
 * @param min_duration The minimum duration of a CPU burst to consider.
 */
BurstsExtractor::BurstsExtractor(unsigned long long min_duration, bool sync_times)
{
  BurstBegin = BurstEnd = LastMPIBegin = LastMPIEnd = NULL;
  DurationFilter    = min_duration;
  SynchronizeTimes  = sync_times;
  ExtractedBursts   = new Bursts();

  CurrentPhase      = new PhaseStats( Extrae_get_num_tasks() );
  PreviousPhase     = NULL;
}

BurstsExtractor::~BurstsExtractor()
{
  delete ExtractedBursts;
}

/**
 * Callback for every event parsed from the buffer when the super-class method Extract() is called.
 * Here, we look for entries and exits of MPI's to delimit the CPU bursts and store them in 
 * the ExtractedBursts container.
 *
 * @param evt The event being parsed.
 */
/* XXX Information in burst mode not supported -- Take into account MPI_STATS_EV because there will not be MPI_EVs */
#if 1
int BurstsExtractor::ParseEvent(UNUSED int thread_id, event_t *evt)
{
  if (isBurstBegin(evt))
  {
    /* Here begins a burst */
    BurstBegin = LastMPIEnd = evt;

    CurrentPhase->UpdateMPI(LastMPIBegin, LastMPIEnd);
    CurrentPhase->UpdateHWC(BurstBegin);

    PreviousPhase = CurrentPhase;
    CurrentPhase  = new PhaseStats( Extrae_get_num_tasks() );
  }
  else if ((isBurstEnd(evt)) && (BurstBegin != NULL))
  {
    /* Here ends a burst */
    BurstEnd = LastMPIBegin = evt;

    unsigned long long ts_ini      = Get_EvTime(BurstBegin);
    unsigned long long ts_ini_sync = TIMESYNC(0, TASKID, ts_ini);
    unsigned long long ts_end      = Get_EvTime(BurstEnd);
    unsigned long long duration    = ts_end - ts_ini;
    unsigned long long ts          = (SynchronizeTimes ? ts_ini_sync : ts_ini);

    /* Only store those bursts that are longer than the given threshold */
    if (duration >= DurationFilter)
    {
      CurrentPhase->UpdateHWC(BurstEnd);

      ExtractedBursts->Insert(ts, duration, PreviousPhase, CurrentPhase);

      CurrentPhase  = new PhaseStats( Extrae_get_num_tasks() );
      PreviousPhase = NULL;
    }
    else
    {
      PreviousPhase->Concatenate(CurrentPhase);
      delete CurrentPhase;
      CurrentPhase = PreviousPhase;
    }
  }
  else
  {
    CurrentPhase->UpdateMPI( evt );
    CurrentPhase->UpdateHWC( evt );
  }
  return 0;
}
#endif


/**
 * @return the extracted bursts.
 */
Bursts * BurstsExtractor::GetBursts()
{
  return ExtractedBursts;
}

void BurstsExtractor::DetailToCPUBursts(unsigned long long t1, unsigned long long t2)
{
  int num_bursts = ExtractedBursts->GetNumberOfBursts();

  for (int i=0; i<num_bursts; i++)
  {
    unsigned long long ts_ini = ExtractedBursts->GetBurstTime(i);
    unsigned long long ts_end = ts_ini + ExtractedBursts->GetBurstDuration(i);

    if ((ts_ini >= t1) && (ts_end <= t2))
    {

    }
  }
}

#include <algorithm>
#include <vector>

using std::sort;
using std::vector;

unsigned long long BurstsExtractor::AdjustThreshold(double KeepThisPercentageOfComputingTime)
{
  unsigned int i         = 0;
  unsigned int NumBursts = 0;
  unsigned long long AggregatedTime = 0;
  unsigned long long BurstThreshold = 0;
  vector< unsigned long long > DurationsArray;
  vector< unsigned long long > AggregatedTimeArray;
  float MinAggregatedTime = 0;

  NumBursts = ExtractedBursts->GetNumberOfBursts();
  if (NumBursts > 0)
  {
    for (i=0; i<NumBursts; i++)
    {
      unsigned long long CurrentBurstDuration = ExtractedBursts->GetBurstDuration(i);

      DurationsArray.push_back( CurrentBurstDuration );
    }

    sort( DurationsArray.begin(), DurationsArray.end() );

    for (i=0; i<DurationsArray.size(); i++)
    {
      AggregatedTime += DurationsArray[i];
      AggregatedTimeArray.push_back( AggregatedTime );
    }

    MinAggregatedTime = AggregatedTime * (100 - KeepThisPercentageOfComputingTime) * 0.01;

    for (i=0; i<AggregatedTimeArray.size(); i++)
    {
      if (AggregatedTimeArray[i] > MinAggregatedTime) break;
    }

    BurstThreshold = DurationsArray[i];
  }
  return BurstThreshold;
}
