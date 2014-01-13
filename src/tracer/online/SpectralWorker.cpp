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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include <sstream>

#include "trace_buffers.h"
#include "taskid.h"
#include "timesync.h"
#include "BurstsExtractor.h"
#include "SpectralWorker.h"
#include "Chopper.h"
#include "Signal.h"
#include "OnlineControl.h"
#include "online_buffers.h"
#include "tags.h"

int LevelAtPeriodicZone    = NOT_TRACING;
int LevelAtNonPeriodicZone = NOT_TRACING;
unsigned long long AutomaticBurstThreshold = 100000;

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
  int        NumDetectedPeriods = 0;
  int        tag;
  PACKET_PTR p;
  Period_t  *DetectedPeriods = NULL;
  Buffer_t  *buffer = TRACING_BUFFER(0);

  LevelAtPeriodicZone    = Online_GetSpectralPZoneDetail();
  LevelAtNonPeriodicZone = Online_GetSpectralNPZoneDetail();

  /* Mask out all the trace data */
  Mask_SetRegion (buffer, Buffer_GetHead(buffer), Buffer_GetTail(buffer), MASK_NOFLUSH);

  /* Extract all bursts since the last analysis */
  BurstsExtractor *Bursts = new BurstsExtractor(0); 

  Bursts->Extract(
    Online_GetAppResumeTime(),
    Online_GetAppPauseTime()
  ); 
  AutomaticBurstThreshold = Bursts->AdjustThreshold( Online_GetSpectralBurstThreshold() );
  fprintf(stderr, "[DEBUG] AutomaticBurstThreshold=%llu\n", AutomaticBurstThreshold);

  /* Generate the DurBurst signal from the bursts information */
  Signal *DurBurstSignal = new Signal(Bursts->GetBursts());
 
  /* DEBUG -- Dump the signal generated at the back-end 
  stringstream ss;
  ss << "signal_backend_" << WhoAmI() << ".txt";
  Spectral_DumpSignal( DurBurstSignal->GetSignal(), (char *)ss.str().c_str() ); */

  /* Serialize the data and send to the front-end */
  DurBurstSignal->Serialize(stSpectral);

  /* Receive how many periods were detected */
  MRN_STREAM_RECV(stSpectral, &tag, p, SPECTRAL_DETECTED_PERIODS);
  PACKET_unpack(p, "%d", &NumDetectedPeriods);

  /* Mark the detail level as bursts from the first first timestamp that has tracing data until the first periodic zone is found */
  unsigned long long PrevNonPeriodicZoneStartTime = Get_EvTime(Buffer_GetFirstEvent(buffer));
  unsigned long long NextNonPeriodicZoneStartTime = 0;
  unsigned long long LastNonPeriodicZoneEndTime = 0;
  TRACE_ONLINE_EVENT(PrevNonPeriodicZoneStartTime, DETAIL_LEVEL_EV, LevelAtNonPeriodicZone);

  /* Receive each period */
  DetectedPeriods = (Period_t *)malloc(NumDetectedPeriods * sizeof(Period_t));
  for (int i=0; i<NumDetectedPeriods; i++)
  {
    int trace_this_period = 0;
    int rep_period_id     = 0;
  
    Period_t *CurrentPeriod = &(DetectedPeriods[i]);
    MRN_STREAM_RECV(stSpectral, &tag, p, SPECTRAL_PERIOD);
    PACKET_unpack(p, "%f %ld %lf %lf %lf %ld %ld %ld %ld %d %d",
      &CurrentPeriod->iters,
      &CurrentPeriod->length,
      &CurrentPeriod->goodness,
      &CurrentPeriod->goodness2,
      &CurrentPeriod->goodness3,
      &CurrentPeriod->ini,
      &CurrentPeriod->end,
      &CurrentPeriod->best_ini,
      &CurrentPeriod->best_end,
      &trace_this_period,
      &rep_period_id
    );      

    ProcessPeriod( Bursts->GetBursts(), CurrentPeriod, trace_this_period, rep_period_id, PrevNonPeriodicZoneStartTime, NextNonPeriodicZoneStartTime );
    PrevNonPeriodicZoneStartTime = NextNonPeriodicZoneStartTime;
  }

  LastNonPeriodicZoneEndTime = Get_EvTime(Buffer_GetLastEvent(buffer));

  /* Mark that there's no data since the last event in the buffer until the first event in the next step */
  TRACE_ONLINE_EVENT(LastNonPeriodicZoneEndTime, DETAIL_LEVEL_EV, NOT_TRACING); 

  /* Emit bursts in the last non-periodic zone */
  if (LastNonPeriodicZoneEndTime - PrevNonPeriodicZoneStartTime > Online_GetSpectralNPZoneMinDuration())
  {
    if (LevelAtNonPeriodicZone == BURST_MODE)
    {
      Bursts->GetBursts()->EmitBursts( PrevNonPeriodicZoneStartTime, LastNonPeriodicZoneEndTime, AutomaticBurstThreshold );
    }
    else if (LevelAtNonPeriodicZone == PHASE_PROFILE)
    {
      Bursts->GetBursts()->EmitPhase( PrevNonPeriodicZoneStartTime, LastNonPeriodicZoneEndTime, true, true );
    }
  }

  delete Bursts; 

  return 0;
}


void SpectralWorker::ProcessPeriod(Bursts *AllBursts, Period_t *CurrentPeriod, int trace_this_period, int rep_period_id, unsigned long long PrevNonPeriodicZoneStartTime, unsigned long long &NextNonPeriodicZoneStartTime)
{
  unsigned long long periodic_zone_ini     = TIMEDESYNC(0, TASKID, CurrentPeriod->ini);
  unsigned long long periodic_zone_end     = TIMEDESYNC(0, TASKID, CurrentPeriod->end);
  unsigned long long best_iters_ini        = TIMEDESYNC(0, TASKID, CurrentPeriod->best_ini);
  unsigned long long best_iters_end        = TIMEDESYNC(0, TASKID, CurrentPeriod->best_end);
  unsigned long long best_iters_ini_fitted = best_iters_ini;
  unsigned long long best_iters_end_fitted = best_iters_end;
  unsigned long long current_time          = 0;
  unsigned long long current_time_fitted   = 0;
  long long int      period_length_ns      = CurrentPeriod->length * 1000000;
  vector<unsigned long long> iters_start_time;
  vector<bool>               iters_in_detail;
  Chopper *EventFinder = new Chopper();
  int remaining_detailed_iters = 0;
    event_t *best_iters_ini_ev = NULL;
    event_t *best_iters_end_ev   = NULL;
   
  int all_iters      = (int)(CurrentPeriod->iters);
  int detailed_iters = (int)((best_iters_end - best_iters_ini) / period_length_ns);
  if ((all_iters < 1) || (detailed_iters < 1)) return;

  remaining_detailed_iters = detailed_iters;

  /* Project the starting time of each iteration from the first detailed one */
  current_time = best_iters_ini;
  while (current_time >= periodic_zone_ini)
  {
    current_time -= period_length_ns;
  }
  current_time += period_length_ns; /* Now points to the start time of the first iteration in the periodic zone */

  for (int i=0; i<all_iters + 1; i++)
  {
    event_t *evt = EventFinder->EventCloserRunning( current_time );
    current_time_fitted = (evt != NULL ? Get_EvTime(evt) : current_time);

    iters_start_time.push_back( current_time_fitted );

    if ((current_time >= best_iters_ini) && (remaining_detailed_iters > 0))
    {
      if (best_iters_ini_ev == NULL) best_iters_ini_ev = evt;
      iters_in_detail.push_back( true );
      remaining_detailed_iters --;
    }
    else
    {
      if ((best_iters_end_ev == NULL) && (remaining_detailed_iters == 0)) best_iters_end_ev = evt;
      iters_in_detail.push_back( false );
    }
    current_time += period_length_ns;
  }

  /* Previous non-periodic zone from PrevNonPeriodicZoneStartTime to iters_start_time[0] */
  if (iters_start_time[0] - PrevNonPeriodicZoneStartTime > Online_GetSpectralNPZoneMinDuration())
  {
    if (LevelAtNonPeriodicZone == BURST_MODE)
    {
      AllBursts->EmitBursts( PrevNonPeriodicZoneStartTime, iters_start_time[0], AutomaticBurstThreshold );
    }
    else if (LevelAtNonPeriodicZone == PHASE_PROFILE)
    {
      AllBursts->EmitPhase( PrevNonPeriodicZoneStartTime, iters_start_time[0], true, true );
    }
  }

  NextNonPeriodicZoneStartTime = iters_start_time[ iters_start_time.size() - 1 ];

  bool detailed_zone_reached = false;
  bool dump_iteration = true;

  TRACE_ONLINE_EVENT(iters_start_time[0], DETAIL_LEVEL_EV, LevelAtPeriodicZone);
  for (int i=0; i<iters_start_time.size() - 1; i++)
  {
/*
    if (dump_iteration)
    {
      TRACE_ONLINE_EVENT(iters_start_time[i], PERIODICITY_EV, REPRESENTATIVE_PERIOD + rep_period_id);
      if (TASKID == 0) fprintf(stderr, "[DEBUG] SpectralWorker::ProcessPeriod iter=%d calling EmitPhase\n", i);
      AllBursts->EmitPhase( iters_start_time[i], iters_start_time[i+1], ((i == 0) ? true : false) );
    }
*/

    if (iters_in_detail[i] && !detailed_zone_reached)
    {
      detailed_zone_reached = true;
      dump_iteration = false;
      TRACE_ONLINE_EVENT(iters_start_time[i], DETAIL_LEVEL_EV, DETAIL_MODE);
    }
    else if (!iters_in_detail[i] && detailed_zone_reached)
    {
      detailed_zone_reached = false;
      dump_iteration = true;
      TRACE_ONLINE_EVENT(iters_start_time[i], DETAIL_LEVEL_EV, LevelAtPeriodicZone);
    }

    TRACE_ONLINE_EVENT(iters_start_time[i], PERIODICITY_EV, REPRESENTATIVE_PERIOD + rep_period_id);

    if (LevelAtPeriodicZone == PHASE_PROFILE)
    {
      AllBursts->EmitPhase( iters_start_time[i], iters_start_time[i+1], ((i == 0) ? true : false), !detailed_zone_reached );
    }
  }
  TRACE_ONLINE_EVENT(iters_start_time[ iters_start_time.size() - 1 ], PERIODICITY_EV, NON_PERIODIC_ZONE);
  TRACE_ONLINE_EVENT(iters_start_time[ iters_start_time.size() - 1 ], DETAIL_LEVEL_EV, LevelAtNonPeriodicZone);

  if (trace_this_period)
  {
//    fprintf(stderr, "[DEBUG] New_Trace_Period %llu %llu\n", Get_EvTime(best_iters_ini_ev), Get_EvTime(best_iters_end_ev));
    New_Trace_Period(best_iters_ini_ev, best_iters_end_ev);
  }

  /* DEBUG -- Raw values returned by the spectral analysis */
  TRACE_ONLINE_EVENT(periodic_zone_ini, ORIGINAL_PERIODICITY_EV, REPRESENTATIVE_PERIOD + rep_period_id);
  TRACE_ONLINE_EVENT(periodic_zone_end, ORIGINAL_PERIODICITY_EV, 0);
  TRACE_ONLINE_EVENT(best_iters_ini, ORIGINAL_BEST_ITERS_EV, REPRESENTATIVE_PERIOD + rep_period_id);
  TRACE_ONLINE_EVENT(best_iters_end, ORIGINAL_BEST_ITERS_EV, 0);
}

/**
 * Unsets the NO_FLUSH mask of all the events between the given times, that represent
 * the best iterations of a period. 
 *
 * @param CurrentPeriod The period to trace.
 *
 * @return best_ini_out The local timestamp for the first event of the traced region.
 * @return best_end_out The local timestamp for the last event of the traced region.
 */
void SpectralWorker::New_Trace_Period(event_t *start, event_t *end)
{
  Buffer_t *buffer = TRACING_BUFFER(0);

  if (start != NULL && end != NULL && start != end)
  {
    Mask_UnsetRegion(buffer, start, end, MASK_NOFLUSH);
  }
}

