#ifndef __SYNCING_H__
#define __SYNCING_H__

#include "OnlineControl.h"

#if defined(__cplusplus)
extern "C" {
#endif

int  SyncPendingConnections(
  int   rank, 
  int   root, 
  int   rdy_to_connect, 
  char *sendbuf, 
  int  *sendcnts, 
  int  *displs, 
  BE_thread_data_t *BE_args);

int  SyncOk(int this_be_ok);

void SyncWaitAll();

#if defined(__cplusplus)
}
#endif

#endif /* __SYNCING_H__ */
