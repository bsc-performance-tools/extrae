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

#ifndef __ONLINE_DEBUG_H__
#define __ONLINE_DEBUG_H__

#include <ostream>
#include <string>

using std::ostream;
using std::string;

class Messaging
{
  public:
    Messaging();
    Messaging(int be_rank, bool is_master);

    void say    (ostream &out, const char *fmt, ...);
    void say_one(ostream &out, const char *fmt, ...);

    void debug    (ostream &out, const char *fmt, ...);
    void debug_one(ostream &out, const char *fmt, ...);

    void error(const char *fmt, ...);

    bool debugging();

  private:
    bool   I_am_FE;
    bool   I_am_BE;
    bool   I_am_master_BE;
    string ProcessLabel;
    bool   DebugEnabled;
};

#endif /* __ONLINE_DEBUG_H__ */
