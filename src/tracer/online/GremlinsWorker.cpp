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
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>

#include "GremlinsWorker.h"
#include "clock.h"
#include "online_buffers.h"
#include "OnlineConfig.h"

/**
 * Receives the streams used in this protocol.
 */

void GremlinsWorker::SetInitialConditions()
{
  char *env_max_gremlins = getenv("N_CONTS");

  if (env_max_gremlins != NULL)
  {
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
  }
}

void GremlinsWorker::Setup()
{
  Register_Stream(stGremlins);

  SetInitialConditions();

  Sweeps = 0;

  if (!Online_GetGremlinsLoop()) 
  {
    LastSweep = 1;

    if (Online_GetGremlinsRoundtrip())
    {
      LastSweep ++;
    }
  }
  else
  {
    LastSweep = -1;
  }
}

/**
 * Back-end side of the Gremlins analysis.
 */
int GremlinsWorker::Run()
{
  int CurrentGremlins  = NumberOfGremlins;
  int GremlinsToChange = 0;

  if (CurrentGremlins == TargetGremlins) 
  {
    Sweeps ++;

    if (Online_GetGremlinsRoundtrip())
    {
      Roundtrip *= -1;
      GremlinsToChange = Online_GetGremlinsIncrement() * Roundtrip;
      TargetGremlins = CurrentGremlins + (MaxGremlins * Roundtrip);
    }
    else
    {
      GremlinsToChange = ( CurrentGremlins - Online_GetGremlinsStartCount() ) * -1;
      TargetGremlins = (Online_GetGremlinsIncrement() > 0 ? MaxGremlins : MinGremlins );
    }

    if ((LastSweep != -1 ) && (Sweeps >= LastSweep))
    {
      return 1;
    }
  }
  else
  {
    GremlinsToChange = Online_GetGremlinsIncrement() * Roundtrip;
  }

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
