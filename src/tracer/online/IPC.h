#ifndef __IPC_H__
#define __IPC_H__

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

class IPC
{
  public:
    void WipeExchangeData();

    /* Root side */
    IPC();
    bool WaitForResources(vector<string> &Backends);
    void SendAttachments();
    bool WaitForTermination();

    /* Master back-end side */
    IPC(int rank);
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
};

#endif /* __IPC_H__ */
