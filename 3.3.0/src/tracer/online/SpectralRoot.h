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

#ifndef __SPECTRAL_ROOT_H__
#define __SPECTRAL_ROOT_H__

#include <vector>
#include <Signal.h>
#include "FrontProtocol.h"
#include "tags.h"

using std::vector;
using namespace Synapse;

typedef struct
{
  Period_t *period;
  int       seen;
  int       traced;
  char     *file;
  signal_t *chop;
} RepresentativePeriod_t;

class SpectralRoot : public FrontProtocol
{
  public:
    SpectralRoot();

    string ID (void) { return "SPECTRAL"; } /* ID matches the back-end protocol */
    void Setup(void);
    int  Run  (void);

  private:
    STREAM *stSpectral;

    int     Step;
    int     TotalPeriodsTraced;

    vector<RepresentativePeriod_t> RepresentativePeriods;

    int  FindRepresentative( signal_t *chop, Period_t *period );
    int  Get_RepIsTraced(int rep_period_id);
    void Set_RepIsTraced(int rep_period_id, int traced);
    int  Get_RepIsSeen  (int rep_period_id);
    bool Done           (void);
};

#endif /* __SPECTRAL_ROOT_H__ */

