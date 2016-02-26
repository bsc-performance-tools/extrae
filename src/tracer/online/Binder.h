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

#ifndef __BINDER_H__
#define __BINDER_H__

#include <string>
#include <vector>
#include "Messaging.h"

using std::string;
using std::vector;

#define MAX_WAIT_RETRIES 60

#define OUT_PREFIX ".extrae-online-"
#define TMP_PREFIX ".tmp-extrae-online-"

#define TMP_ONLINE_RESOURCES_FILE   TMP_PREFIX"rlist.txt"
#define ONLINE_RESOURCES_FILE       OUT_PREFIX"rlist.txt"

#define ONLINE_TOPOLOGY_FILE        OUT_PREFIX"topology.txt"

#define TMP_ONLINE_ATTACHMENTS_FILE TMP_PREFIX"attach.txt"
#define ONLINE_ATTACHMENTS_FILE     OUT_PREFIX"attach.txt"

#define ONLINE_TERMINATION_FILE     OUT_PREFIX"kill.txt"

class Binder
{
  public:
    void WipeExchangeData();

    /* Root side */
    Binder();
    bool WaitForResources(vector<string> &Backends);
    void SendAttachments();
    bool WaitForTermination();

    /* Master back-end side */
    Binder(int rank);
    void SendResources(int NumberOfNodes, char **ListOfNodes);
    bool WaitForAttachments(int ExpectedAttachments);
    void SendTermination();

    string GetResourcesFile();
    string GetResourcesTmpFile();
    string GetTopologyFile();
    string GetAttachmentsFile();
    string GetAttachmentsTmpFile();
    string GetTerminationFile();

  private:
    Messaging *Msgs;
    string     GPFSPath;

    string PathTo(string FileName);
    bool WaitForFile(string FileName, int MaxRetries, int StallTime);
    bool WaitForFilePolling(string FileName, int MaxRetries, int StallTime);
};

#endif /* __BINDER_H__ */
