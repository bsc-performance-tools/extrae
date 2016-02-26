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
#include <sstream>

using std::cerr;
using std::endl;
using std::stringstream;

#include "utils.h"

#include "SpectralRoot.h"
#include "OnlineConfig.h"
#include "OnlineControl.h"
#include "Signal.h"
#include "Messaging.h"


SpectralRoot::SpectralRoot()
{
  Step = 0;
  TotalPeriodsTraced = 0;
}

/**
 * Register the stream that will perform the signal reduction 
 */
void SpectralRoot::Setup()
{
  /* stSpectral uses the filter OnlineSpectral */
  stSpectral = Register_Stream("OnlineSpectral", SFILTER_WAITFORALL);
}

/**
 * Front-end side of the Spectral Analysis algorithm. Gets the 
 * summed signal from the back-ends, performs the spectral analysis,
 * and sends the periods detected back to the back-ends to 
 * trace those that are new.
 *
 * @return 1 if the analysis target is achieved; 0 otherwise.
 */
int SpectralRoot::Run()
{
  int          tag;
  PACKET_PTR   p; 
  stringstream ss;
  Messaging   *Msgs = new Messaging();

  Step ++;

  /* Receive the added signal from the back-ends */
  MRN_STREAM_RECV( stSpectral, &tag, p, REDUCE_SIGNAL );
  Signal   *DurBurstSignal = new Signal( p );
  signal_t *sig_dur_burst  = DurBurstSignal->GetSignal();

  /* DEBUG -- Dump the signal to disk */
  ss << "signal_step_" << Step << ".txt";
  Spectral_DumpSignal( sig_dur_burst, (char *)ss.str().c_str() );

  /* Run the spectral analysis */
  int NumDetectedPeriods     = 0;
  int NumValidPeriods        = 0;
  Period_t **DetectedPeriods = NULL;
  vector<signal_t *> ListOfChops;

  NumDetectedPeriods = Spectral_ExecuteAnalysis( 
    sig_dur_burst, 
    Online_GetSpectralNumIters(),
    (Step == 1 ? WINDOWING_10PCT : WINDOWING_NONE),
    &DetectedPeriods);

  for (int i=0; i<NumDetectedPeriods; i++)
  {	
    Period_t *CurrentPeriod = DetectedPeriods[i];
    signal_t *CurrentChop   = Spectral_ChopSignal( sig_dur_burst, CurrentPeriod->best_ini, CurrentPeriod->best_end );

    if (CurrentChop != NULL)
    {
      NumValidPeriods ++;
    }
    ListOfChops.push_back( CurrentChop );
  }

  Msgs->debug(cerr, "Detected %d period(s) - %d valid", NumDetectedPeriods, NumValidPeriods);

  /* Send the number of valid periods (if a chop is null, that period is not counted) */
  MRN_STREAM_SEND(stSpectral, SPECTRAL_DETECTED_PERIODS, "%d", NumValidPeriods);

  /* Process every period */
  for (int i=0; i<NumDetectedPeriods; i++)
  {
    Period_t *CurrentPeriod          =  DetectedPeriods[i];
    int       TraceThisPeriod        =  0;
    int       RepresentativePeriodID = -1;

    /* Skip the period if the chop is empty */
    if (ListOfChops[i] != NULL)
    {
      /* Check if this period has been seen before */
      RepresentativePeriodID = FindRepresentative( ListOfChops[i], CurrentPeriod );

      /* Decide if current period will be traced */
      TraceThisPeriod = (
        ((TotalPeriodsTraced < Online_GetSpectralMaxPeriods()) ||
         (Online_GetSpectralMaxPeriods() == 0))                             && /* Not traced enough */
        (!Get_RepIsTraced(RepresentativePeriodID))                          && /* Not traced before */
        (Get_RepIsSeen(RepresentativePeriodID) > Online_GetSpectralMinSeen())  /* Seen enough times */
      ); 

      if (TraceThisPeriod)
      {
        /* Mark that this period is going to be traced */
        Set_RepIsTraced(RepresentativePeriodID, 1);
      }
    
      /* Transfer the period information to the back-ends */
      MRN_STREAM_SEND(stSpectral, SPECTRAL_PERIOD, "%f %ld %lf %lf %lf %ld %ld %ld %ld %d %d",
              CurrentPeriod->iters,
              CurrentPeriod->length,
              CurrentPeriod->goodness,
              CurrentPeriod->goodness2,
              CurrentPeriod->goodness3,
              CurrentPeriod->ini,
              CurrentPeriod->end,
              CurrentPeriod->best_ini,
              CurrentPeriod->best_end,
              TraceThisPeriod,
              RepresentativePeriodID
      );
    }
    xfree (DetectedPeriods[i]);
  }
  xfree (DetectedPeriods);

  for (int i=0; i<ListOfChops.size(); i++)
  {
    Spectral_FreeSignal( ListOfChops[i] );
  }

  delete DurBurstSignal;

  if (NumValidPeriods <= 0)
  {
    Online_UpdateFrequency(50);
  }

  return ( Done() ? 1 : 0 );
}

/**
 * Checks if the specified period for the signal is a new type of period
 * or it's been seen before. If this type of period has been seen before,
 * returns the representative identifier. If this is a new type of period,
 * it saves this period as a new representative and returns the new 
 * identifier.
 *
 * @param signal The spectral signal.
 * @param period A period detected in the signal.
 *
 * @returns the representative period identifier.
 */
int SpectralRoot::FindRepresentative( signal_t *chop, Period_t *period )
{
  int found = -1;

  if (chop != NULL)
  {
    for (unsigned int i=0; i<RepresentativePeriods.size(); i++)
    {
      double likeness = Spectral_CompareSignals( RepresentativePeriods[i].chop, chop, WINDOWING_NONE );
      if (likeness >= Online_GetSpectralMinLikeness())
      {
        found = i;
        RepresentativePeriods[i].seen ++;
        break;
      }
    }

    if (found == -1)
    {
      /* Save a new representative period */
      RepresentativePeriod_t rep_period;    

      rep_period.period = (Period_t *)malloc(sizeof(Period_t));
      memcpy(rep_period.period, period, sizeof(Period_t));
      rep_period.seen   = 1;
      rep_period.traced = 0;
      rep_period.chop   = Spectral_CloneSignal(chop);
 
      RepresentativePeriods.push_back( rep_period );
      found = RepresentativePeriods.size() - 1;
    
      /* Write this representative to disk */
      stringstream ss;
      ss << "rep_period_" << found+1 << ".txt";
      Spectral_DumpSignal( chop, (char *)ss.str().c_str() );
    }
  }
  return found;
}

/**
 * Check if the given representative period has already been traced.
 *
 * @param rep_period_id The period identifier.
 * @return 1 if traced; 0 otherwise.
 */
int SpectralRoot::Get_RepIsTraced(int rep_period_id)
{
  if (rep_period_id > (int)RepresentativePeriods.size())
  {
    return 0;
  }
  else
  {
    return RepresentativePeriods[rep_period_id].traced;
  }
}

/**
 * Marks the given representative period as already traced.
 *
 * @param rep_period_id The period identifier.
 * @param traced        1 traced; 0 not traced.
 */
void SpectralRoot::Set_RepIsTraced(int rep_period_id, int traced)
{
  if ((rep_period_id >= 0) && (rep_period_id <= (int)RepresentativePeriods.size()))
  {
    TotalPeriodsTraced ++;
    RepresentativePeriods[rep_period_id].traced = traced;
  }
}

/**
 * Get how many times the given representative period has been seen.
 *
 * @param rep_period_id The period identifier.
 * @return the number of times the given period has been seen.
 */
int SpectralRoot::Get_RepIsSeen(int rep_period_id)
{
  if (rep_period_id > (int)RepresentativePeriods.size())
  {
    return 0;
  }
  else
  {
    return RepresentativePeriods[rep_period_id].seen;
  }
}

/**
 * Checks whether the analysis objectives have been met.
 *
 * @return true if the target number of periods have been traced; false otherwise.
 */
bool SpectralRoot::Done()
{
  int RequestedPeriods = Online_GetSpectralMaxPeriods();

  if ((TotalPeriodsTraced < RequestedPeriods) || (RequestedPeriods == 0))
  {
    return false;
  }
  return true;
}


