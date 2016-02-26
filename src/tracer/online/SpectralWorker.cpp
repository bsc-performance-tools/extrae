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

#include <sstream>

#include "trace_buffers.h"
#include "taskid.h"
#include "timesync.h"
#include "SpectralWorker.h"
#include "Chopper.h"
#include "Signal.h"
#include "OnlineControl.h"
#include "online_buffers.h"
#include "tags.h"

/**
 * Receives the streams used in this protocol.
 */
void SpectralWorker::Setup()
{
  Register_Stream(stSpectral);
}

/**
 * Back-end side of the Spectral Analysis algorithm. 
 *
 * Extracts data from the tracing buffer to build a signal, 
 * and sends it to the root. Waits for the root to execute 
 * the analysis and waits for the results. The selected
 * periods will be traced, and the rest of the trace buffer 
 * discarded.
 *
 * @return 0 on success; -1 if errors;
 */
int SpectralWorker::Run()
{
  int        tag;
  PACKET_PTR p;
  int        NumDetectedPeriods = 0;

  /* Extract all bursts since the last analysis */
  BurstsExtractor *ExtractedBursts = new BurstsExtractor(0); 

  ExtractedBursts->ParseBuffer(
    0,
    Online_GetAppResumeTime(),
    Online_GetAppPauseTime()
  ); 

  /* Generate the DurBurst signal from the bursts information */
  Signal *DurBurstSignal = new Signal( ExtractedBursts->GetBursts() );
 
  /* DEBUG -- Dump the signal generated at the back-end 
  stringstream ss;
  ss << "signal_backend_" << WhoAmI() << ".txt";
  Spectral_DumpSignal( DurBurstSignal->GetSignal(), (char *)ss.str().c_str() ); */

  /* Serialize the data and send to the front-end */
  DurBurstSignal->Serialize(stSpectral);

  /* Receive how many periods were detected */
  MRN_STREAM_RECV(stSpectral, &tag, p, SPECTRAL_DETECTED_PERIODS);
  PACKET_unpack(p, "%d", &NumDetectedPeriods);

  /* Receive each period */
  if (NumDetectedPeriods > 0)
  {
    vector<Period> DetectedPeriods( NumDetectedPeriods );

    for (int i=0; i<NumDetectedPeriods; i++)
    {
      MRN_STREAM_RECV(stSpectral, &tag, p, SPECTRAL_PERIOD);
      PACKET_unpack(p, "%f %ld %lf %lf %lf %ld %ld %ld %ld %d %d",
        &(DetectedPeriods[i].info.iters),
        &(DetectedPeriods[i].info.length),
        &(DetectedPeriods[i].info.goodness),
        &(DetectedPeriods[i].info.goodness2),
        &(DetectedPeriods[i].info.goodness3),
        &(DetectedPeriods[i].info.ini),
        &(DetectedPeriods[i].info.end),
        &(DetectedPeriods[i].info.best_ini),
        &(DetectedPeriods[i].info.best_end),
        &(DetectedPeriods[i].traced),
        &(DetectedPeriods[i].id)
      );      
    }

    ProcessPeriods(DetectedPeriods, ExtractedBursts);
  }

  delete DurBurstSignal;
  delete ExtractedBursts; 

  return 0;
}

void SpectralWorker::ProcessPeriods(vector<Period> &AllPeriods, BurstsExtractor *ExtractedBursts)
{
  Buffer_t          *buffer                   = ExtractedBursts->GetBuffer();
  unsigned long long FirstEvtTime             = Get_EvTime(Buffer_GetFirstEvent(buffer));
  unsigned long long LastEvtTime              = Get_EvTime(Buffer_GetLastEvent(buffer));
  int                LevelAtPeriodZone        = Online_GetSpectralPeriodZoneLevel();
  int                LevelAtNonPeriodZone     = Online_GetSpectralNonPeriodZoneLevel();
  unsigned long long NonPeriodZoneMinDuration = Online_GetSpectralNonPeriodZoneMinDuration();
  unsigned long long PreviousPeriodZoneEnd    = 0;
  unsigned long long AutomaticBurstThreshold  = 100000;
  int                NumDetectedPeriods       = AllPeriods.size();
  Bursts            *BurstsData               = ExtractedBursts->GetBursts();
  Chopper           *Chop                     = new Chopper();

  if (FirstEvtTime < Online_GetAppResumeTime()) FirstEvtTime = Online_GetAppResumeTime();
  if (LastEvtTime  > Online_GetAppPauseTime())  LastEvtTime  = Online_GetAppPauseTime();

  /* Mask out all traced data */
  Chop->MaskAll();

  /* Compute the burst threshold that keeps the configured percentage of computations */
  if (LevelAtNonPeriodZone == BURST_MODE)
  {
    AutomaticBurstThreshold = ExtractedBursts->AdjustThreshold( Online_GetSpectralBurstThreshold() );
  }
 
  /* Process each period */
  for (int i=0; i<NumDetectedPeriods; i++)
  {
    Period *CurrentPeriod  = &(AllPeriods[i]);
    Period *PreviousPeriod = (i > 0 ? &(AllPeriods[i-1]) : NULL);
    Period *NextPeriod     = (i < NumDetectedPeriods - 1 ? &(AllPeriods[i+1]) : NULL);
    unsigned long long         PeriodZoneIni      = TIMEDESYNC(0, TASKID, CurrentPeriod->info.ini);
    unsigned long long         PeriodZoneEnd      = TIMEDESYNC(0, TASKID, CurrentPeriod->info.end);
    unsigned long long         BestItersIni       = TIMEDESYNC(0, TASKID, CurrentPeriod->info.best_ini);
    unsigned long long         BestItersEnd       = TIMEDESYNC(0, TASKID, CurrentPeriod->info.best_end);
    long long int              PeriodLength       = CurrentPeriod->info.length * 1000000; /* Nanoseconds resolution */
    int                        AllIters           = (int)(CurrentPeriod->info.iters);
    int                        DetailedIters      = (int)( (BestItersEnd - BestItersIni) / PeriodLength );
    int                        DetailedItersLeft  = DetailedIters;
    unsigned long long         CurrentTime        = 0;
    unsigned long long         CurrentTimeFitted  = 0;
    vector<unsigned long long> ItersStartingTimes;
    vector<bool>               ItersInDetail;
    unsigned long long         NonPeriodZoneStart = 0;
    unsigned long long         NonPeriodZoneEnd   = 0;
    bool                       InsideDetailedZone = false;

/* XXX */    if ((AllIters < 1) || (DetailedIters < 1)) continue;

    /* Project the starting time of each iteration from the first detailed one */
    CurrentTime = BestItersIni;
    while (CurrentTime >= PeriodZoneIni)
    {
      CurrentTime -= PeriodLength;
    }
    CurrentTime += PeriodLength; /* Now points to the start time of the first iteration in the periodic zone */

    /* Adjust the starting time of the iteration to the closest running burst starting time */
    for (int count = 0; count <= AllIters; count ++)
    {
      CurrentTimeFitted = Chop->FindCloserRunningTime(0, CurrentTime );
      ItersStartingTimes.push_back( CurrentTimeFitted );

      if ( (CurrentTime >= BestItersIni) && (DetailedItersLeft > 0) )
      {
        ItersInDetail.push_back( true );
        DetailedItersLeft --;
      }
      else
      {
        ItersInDetail.push_back( false );
      }
      CurrentTime += PeriodLength;
    }
    /*
     * vector ItersStartingTimes contains the starting time adjusted to the closest running for each iteration in the periodic zone.
     * vector ItersInDetail marks for each iteration if it has to be traced in detail or not. 
     */
 
    /* Discard the first and/or last iterations if the time adjustment exceeds the timestamps when the application was resumed/paused */
    if (ItersStartingTimes[0] < Online_GetAppResumeTime())
    {
      ItersStartingTimes.erase( ItersStartingTimes.begin() );
    }
    if (ItersStartingTimes[ ItersStartingTimes.size() - 1 ] > Online_GetAppPauseTime())
    {
      ItersStartingTimes.pop_back();
    }

    /* Now emit events marking the phases, the structure is as follows :
     *  ----------------------------------------------------------------------------------------------------------------
     *  ... ANALYSIS | NON-PERIOD | PERIOD #1 | BEST ITERS | PERIOD #1 | NON-PERIOD | PERIOD #2 | NON-PERIOD | ANALYSIS |
     *  ----------------------------------------------------------------------------------------------------------------
     *               |                                                                                       |
     *            FirstEvt                                                                                LastEvt
     */
    if (PreviousPeriod == NULL) /* First period in the current block of trace data */
    {
      if (Online_GetAppResumeTime() > 0) /* Not the first round of analysis */
      {
        /* Mark NOT TRACING from the last analysis */ 
        TRACE_ONLINE_EVENT(Online_GetAppResumeTime(), DETAIL_LEVEL_EV, NOT_TRACING);
      }
      /* NON-PERIODIC zone from the first event in the buffer until the start of the first periodic zone */
      NonPeriodZoneStart    = FirstEvtTime;
      NonPeriodZoneEnd      = ItersStartingTimes[0];
    }
    else /* Further periods in the current block of trace data */
    {
      /* NON-PERIODIC zone from the end of the previous period to the start of the current one */
      NonPeriodZoneStart    = PreviousPeriodZoneEnd;
      NonPeriodZoneEnd      = ItersStartingTimes[0];
    }
    Summarize( NonPeriodZoneStart, NonPeriodZoneEnd, NonPeriodZoneMinDuration, LevelAtNonPeriodZone, BurstsData, AutomaticBurstThreshold );

    /* PERIODIC zone */
    TRACE_ONLINE_EVENT(ItersStartingTimes[0], DETAIL_LEVEL_EV, LevelAtPeriodZone);
    for (unsigned int current_iter = 0; current_iter < ItersStartingTimes.size() - 1; current_iter ++)
    {
      if (ItersInDetail[current_iter] && !InsideDetailedZone)
      {
        InsideDetailedZone = true;
        TRACE_ONLINE_EVENT(ItersStartingTimes[current_iter], DETAIL_LEVEL_EV, DETAIL_MODE); /* Entry of detailed zone */

      }
      else if (!ItersInDetail[current_iter] && InsideDetailedZone)
      {
        InsideDetailedZone = false;
        TRACE_ONLINE_EVENT(ItersStartingTimes[current_iter], DETAIL_LEVEL_EV, LevelAtPeriodZone); /* Exit of detailed zone */
      }

      /* Emit the period ID at the start of each iteration */
      TRACE_ONLINE_EVENT(ItersStartingTimes[current_iter], PERIODICITY_EV, REPRESENTATIVE_PERIOD + CurrentPeriod->id);

      if (LevelAtPeriodZone == PHASE_PROFILE)
      {
        BurstsData->EmitPhase( ItersStartingTimes[current_iter], ItersStartingTimes[current_iter + 1], (current_iter == 0), !InsideDetailedZone );
      }
    }
  
    /* NON-PERIODIC zone after the PERIODIC zone */
    TRACE_ONLINE_EVENT(ItersStartingTimes[ ItersStartingTimes.size() - 1 ], PERIODICITY_EV, NON_PERIODIC_ZONE);
    PreviousPeriodZoneEnd = ItersStartingTimes[ ItersStartingTimes.size() - 1 ];

    if (NextPeriod == NULL)
    {
      NonPeriodZoneStart    = PreviousPeriodZoneEnd;
      NonPeriodZoneEnd      = LastEvtTime;

      Summarize( NonPeriodZoneStart, NonPeriodZoneEnd, NonPeriodZoneMinDuration, LevelAtNonPeriodZone, BurstsData, AutomaticBurstThreshold );

      TRACE_ONLINE_EVENT( NonPeriodZoneEnd, DETAIL_LEVEL_EV, NOT_TRACING );
    }

    if (CurrentPeriod->traced)
    {
      Chop->UnmaskAll( BestItersIni, BestItersEnd );
    }

    /* DEBUG -- Raw values returned by the spectral analysis 
    TRACE_ONLINE_EVENT(PeriodZoneIni, RAW_PERIODICITY_EV, REPRESENTATIVE_PERIOD + CurrentPeriod->id);
    TRACE_ONLINE_EVENT(PeriodZoneEnd, RAW_PERIODICITY_EV, 0);
    TRACE_ONLINE_EVENT(BestItersIni, RAW_BEST_ITERS_EV, REPRESENTATIVE_PERIOD + CurrentPeriod->id);
    TRACE_ONLINE_EVENT(BestItersEnd, RAW_BEST_ITERS_EV, 0); */

  }
  delete Chop;
}

void SpectralWorker::Summarize(unsigned long long NonPeriodZoneStart, unsigned long long NonPeriodZoneEnd, unsigned long long DurationThreshold, int LevelAtNonPeriodZone, Bursts *BurstsData, unsigned long long BurstsThreshold)
{
  unsigned long long NonPeriodZoneDuration = NonPeriodZoneEnd - NonPeriodZoneStart;

  if (NonPeriodZoneDuration > DurationThreshold)
  {
    /* The NON-PERIODIC zone is long, transform detail into bursts/profile */
    TRACE_ONLINE_EVENT( NonPeriodZoneStart, DETAIL_LEVEL_EV, LevelAtNonPeriodZone );
    if (LevelAtNonPeriodZone == BURST_MODE)
    {
      BurstsData->EmitBursts( NonPeriodZoneStart, NonPeriodZoneEnd, BurstsThreshold );
    }
    else if (LevelAtNonPeriodZone == PHASE_PROFILE)
    {
      BurstsData->EmitPhase( NonPeriodZoneStart, NonPeriodZoneEnd, true, true );
    }
  }    
  else
  {
    /* The NON-PERIODIC zone is short, discard it */
    TRACE_ONLINE_EVENT( NonPeriodZoneStart, DETAIL_LEVEL_EV, NOT_TRACING );
  }
}

