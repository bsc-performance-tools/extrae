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
#include "SpectralWorker.h"
#include "BurstsExtractor.h"
#include "Chopper.h"
#include "Signal.h"
#include "OnlineControl.h"
#include "trace_buffers.h"
#include "online_buffers.h"
#include "taskid.h"
#include "trace_buffers.h"
#include "timesync.h"
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
  int        NumDetectedPeriods = 0;
  int        tag;
  PACKET_PTR p;
  Period_t  *DetectedPeriods = NULL;
  Buffer_t  *buffer = TRACING_BUFFER(0);

  /* Mask out all the trace data */
  Mask_SetRegion (buffer, Buffer_GetHead(buffer), Buffer_GetTail(buffer), MASK_NOFLUSH);

  /* Extract all bursts since the last analysis */
  BurstsExtractor *Bursts = new BurstsExtractor(0); 

  Bursts->Extract(
    Online_GetAppResumeTime(),
    Online_GetAppPauseTime()
  ); 

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

  /* Receive each period */
  DetectedPeriods = (Period_t *)malloc(NumDetectedPeriods * sizeof(Period_t));
  for (int i=0; i<NumDetectedPeriods; i++)
  {
    int trace_this_period = 0;
    int rep_period_id = 0;

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

    if (trace_this_period)
    {
      /* Mask in all trace data that belongs to this period */
      long long int fit_best_ini = 0;
      long long int fit_best_end = 0;
      Trace_Period(CurrentPeriod, &fit_best_ini, &fit_best_end);

      /* Mark the traced period in the online buffer */
      TRACE_ONLINE_EVENT(fit_best_ini, REP_PERIOD_EV, rep_period_id+1);
      TRACE_ONLINE_EVENT(fit_best_end, REP_PERIOD_EV, 0);
    }

  }

  delete Bursts; 

  return 0;
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
void SpectralWorker::Trace_Period(Period_t *CurrentPeriod, long long int *best_ini_out, long long int *best_end_out)
{
  Buffer_t *buffer = TRACING_BUFFER(0);

  /* Transform the global times of the boundaries of the period into local times */
  long long int real_best_ini = TIMEDESYNC(0, TASKID, CurrentPeriod->best_ini);
  long long int real_best_end = TIMEDESYNC(0, TASKID, CurrentPeriod->best_end);

  /* Get the first and last events for the given time interval */
  event_t *period_ini_evt = NULL;
  event_t *period_end_evt = NULL;

  Chopper *ChopPeriod = new Chopper();

  ChopPeriod->Chop(real_best_ini, real_best_end, period_ini_evt, period_end_evt);

  /* Mask in all the buffer data from the first to the last event */
  if ((period_ini_evt != NULL) && (period_end_evt != NULL) && (period_ini_evt != period_end_evt))
  {
    *best_ini_out = Get_EvTime(period_ini_evt);
    *best_end_out = Get_EvTime(period_end_evt);
    Mask_UnsetRegion(buffer, period_ini_evt, period_end_evt, MASK_NOFLUSH);
  }
  else
  {
    /* Should not happen */
    *best_ini_out = real_best_ini;
    *best_end_out = real_best_end;
  }
}

