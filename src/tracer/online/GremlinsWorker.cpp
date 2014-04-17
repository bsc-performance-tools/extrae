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

/**
 * Receives the streams used in this protocol.
 */
void GremlinsWorker::Setup()
{
  Register_Stream(stGremlins);
}

int NumberOfGremlins = 0;

/**
 * Back-end side of the Gremlins analysis.
 */
int GremlinsWorker::Run()
{
  NumberOfGremlins ++;

  TRACE_ONLINE_EVENT(TIME, GREMLIN_EV, NumberOfGremlins);
  fprintf(stderr, "[DEBUG-GREMLINS %d] Online sending signal\n", getpid());

  kill(getpid(), SIGUSR1);

  return 0;
}

