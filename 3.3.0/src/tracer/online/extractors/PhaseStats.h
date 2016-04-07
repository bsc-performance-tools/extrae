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

#ifndef __PHASE_STATS_H__
#define __PHASE_STATS_H__

#include <map>
#include <vector>
#include "mpi_stats.h"
#include "record.h"
#include "common.h"

using std::map;
using std::pair;
using std::vector;

class PhaseStats
{
  public:
    PhaseStats(int num_tasks);
    ~PhaseStats();

    void UpdateMPI(event_t *MPIBeginEv, event_t *MPIEndEv);
    void UpdateMPI(event_t *SingleMPIEv);
    void Reset();
    void Concatenate(PhaseStats *OtherStats);
    mpi_stats_t * GetMPIStats();

#if defined(BACKEND)
    void DumpToTrace( unsigned long long timestamp, bool dump_hwcs);
    void DumpZeros( unsigned long long timestamp);
void Dump();

#if USE_HARDWARE_COUNTERS 
    void UpdateHWC(event_t *Ev);
    void GetCommonCounters(map<unsigned int, long unsigned int> &Counters);
    void GetAllCounters(map<unsigned int, long unsigned int> &Counters);
    void GetLastAllCounters(map<unsigned int, long unsigned int> &Counters);
    void GetLastCommonCounters(map<unsigned int, long unsigned int> &Counters);
    int  GetLastSet();
    int  GetFirstSet();
#endif
#endif


  private:
    mpi_stats_t *MPI_Stats; 

#if USE_HARDWARE_COUNTERS && defined(BACKEND)
    typedef unsigned long long           timestamp_t;
    typedef pair<int, long long *>       hwc_set_val_pair_t;
    map<timestamp_t, hwc_set_val_pair_t> HWC_Stats;
    map<timestamp_t, int>                HWC_Events;

    map< int, vector<int> > HWCSetToIds;
#endif

};

#endif /* __PHASE_STATS_H__ */
