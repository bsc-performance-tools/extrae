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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/tracer/online/SpectralWorker.cpp $
 | @last_commit: $Date: 2014-01-31 14:13:36 +0100 (vie, 31 ene 2014) $
 | @version:     $Revision: 2459 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>


static char UNUSED rcsid[] = "$Id: GremlinsWorker.cpp 2459 2014-01-31 13:13:36Z gllort $";

#include "GremlinsWorker.h"
#include "clock.h"
#include "online_buffers.h"
#include "OnlineConfig.h"

/**
 * Receives the streams used in this protocol.
 */
void GremlinsWorker::Setup()
{
  Register_Stream(stGremlins);

  char *env_max_gremlins = getenv("N_CONTS");
  
  MinGremlins      = 0;
  MaxGremlins      = atoi((const char *)env_max_gremlins);
  NumberOfGremlins = Online_GetGremlinsStartCount();
  TargetGremlins   = (Online_GetGremlinsIncrement() > 0 ? MaxGremlins : MinGremlins );
  Roundtrip        = (TargetGremlins > MinGremlins ? 1 : -1);

  if (NumberOfGremlins > MaxGremlins)
  {
    NumberOfGremlins = MaxGremlins;
  }

  TRACE_ONLINE_EVENT(TIME, GREMLIN_EV, NumberOfGremlins);
  
  fprintf(stderr, "GremlinsWorker:: StartingGremlins=%d\n", NumberOfGremlins);
  SwitchSome( NumberOfGremlins );

  Loops = 0;
}

/**
 * Back-end side of the Gremlins analysis.
 */
int GremlinsWorker::Run()
{
  int CurrentGremlins = NumberOfGremlins;

  if ((CurrentGremlins == TargetGremlins) && (Online_GetGremlinsRoundtrip()))
  {
    Roundtrip = Roundtrip * -1;
    TargetGremlins = CurrentGremlins + (MaxGremlins * Roundtrip);
    Loops ++;
    if (Loops == 2)
    {
      return 1;
    }
  }

  int GremlinsToChange = Online_GetGremlinsIncrement() * Roundtrip;
  if (CurrentGremlins + GremlinsToChange < MinGremlins)
  {
    GremlinsToChange = CurrentGremlins * -1;
  }
  else if (CurrentGremlins + GremlinsToChange > MaxGremlins)
  {
    GremlinsToChange = MaxGremlins - CurrentGremlins; 
  }

  NumberOfGremlins = CurrentGremlins + GremlinsToChange;
  fprintf(stderr, "GremlinsWorker:: Run: CurrentGremlins=%d NextGremlins=%d Increment=%d\n", CurrentGremlins, NumberOfGremlins, GremlinsToChange);

  TRACE_ONLINE_EVENT(TIME, GREMLIN_EV, NumberOfGremlins);

  SwitchSome( GremlinsToChange );

  return 0;
}

void GremlinsWorker::SwitchSome(int GremlinsToChange)
{
  char env_extrae_online_gremlins[1024];
  snprintf(env_extrae_online_gremlins, 1024, "%s=%d", "EXTRAE_ONLINE_GREMLINS", GremlinsToChange);
  putenv(env_extrae_online_gremlins);

  if (GremlinsToChange != 0)
  {
    kill(getpid(), SIGUSR1);
  }
}

#if 0
int GremlinsWorker::Run()
{
  if ((StopPhase) && (NumberOfGremlins == 0))
  {
    return 1;
  }

  fprintf(stderr, "[DEBUG-GREMLINS %d] Online CurrentGremlins=%d MaxGremlins=%d\n", getpid(), NumberOfGremlins, MaxGremlins);

  if (StartPhase)
  {
    if (NumberOfGremlins >= MaxGremlins)
    {
      StopPhase = true;
      StartPhase = false;
    }
    else
    {
      NumberOfGremlins ++;
      kill(getpid(), SIGUSR1);
    }

  }
  if (StopPhase)
  {
    if (NumberOfGremlins > 0)
    {
      NumberOfGremlins --;
      kill(getpid(), SIGUSR2);
    }
  }
  TRACE_ONLINE_EVENT(TIME, GREMLIN_EV, NumberOfGremlins);

  return 0;
}
#endif

