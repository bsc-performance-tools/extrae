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

#ifndef __SIGNAL_H__
#define __SIGNAL_H__

#include <vector>
#include <string>
#include <MRNet_wrappers.h>
#include <spectral-api.h>
#include "Bursts.h"

using std::vector;
using std::string;

#define SIGNAL_XMIT_FORMAT "%ald %ald %alf" 

class Signal 
{
  public:
    Signal();
    Signal(string file);
    Signal(Bursts *bursts);
    Signal(PACKET_PTR InputPacket);
    ~Signal();

    void       Serialize(STREAM *OutputStream);
    PACKET_PTR Serialize(int OutputStreamID);
    void Unpack(PACKET_PTR InputPacket);
    signal_t * GetSignal();
    int        GetSize();
    void       Sum(vector<Signal *> SignalsList);

  private:
    signal_t *SpectralSignal;

};

#endif /* __SIGNAL_H__ */
