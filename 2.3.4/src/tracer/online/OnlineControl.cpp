#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <FrontEnd.h>
#include <BackEnd.h>
#include <PendingConnections.h>
#include "OnlineControl.h"
#include "Syncing.h"

using std::string;
using std::stringstream;

/**
 * Global variables
 */
int this_BE_rank = 0;
int I_am_root    = 0;

/**
 * These are only defined in the MPI process that runs the front-end (FRONTEND_RANK define)
 */
FrontEnd        *FE = NULL;
pthread_t        FE_thread;             // Thread running the front-end 
int              FE_thread_started = 0; // 1 if the FE thread is spawned successfully
FE_thread_data_t FE_data;               // Arguments for the front-end
pthread_mutex_t  FE_lock;

/**
 * Each task runs its own back-end
 */
BackEnd         *BE = NULL; 
pthread_t        BE_thread;             // Thread running the back-end analysis loop
int              BE_thread_started = 0; // 1 if the BE thread is spawned successfully
BE_thread_data_t BE_data;               // Connection information for the back-end


/**
 * Online_Start
 */
int Online_Start(int rank, int world_size, char **node_list)
{
  int me_init_ok = 0;
  int we_init_ok = 0;

  int ok_dist    = 0;
  int ok_be_init = 0;

  this_BE_rank = rank;
  I_am_root    = (rank == FRONTEND_RANK(world_size));

  ONLINE_DBG("Starting the on-line analysis ...\n");

  /* These get values in the root process, who distributes 
   * the pending connections to the rest 
   */
  int   rdy_to_connect = 0;
  char *sendbuf  = NULL;
  int  *sendcnts = NULL;
  int  *displs   = NULL;

  if (I_am_root)
  {
    pid_t pid = getpid();

#if defined(ONLINE_DEBUG)
    int i = 0;
    ONLINE_DBG_1("I will start the front-end (world_size=%d, node_list=", world_size); 
    for (i=0; i<world_size; i++)
    {
      fprintf(stderr, "%s", node_list[i]);
      if (i<world_size-1) fprintf(stderr, ",");
    }
    fprintf(stderr, ")\n");
#endif /* ONLINE_DEBUG */

    snprintf(FE_data.resources_file, 128, "%d.rlist",  (int)pid);
    snprintf(FE_data.topology_file,  128, "%d.top",    (int)pid);
    snprintf(FE_data.attach_file,    128, "%d.attach", (int)pid);
    FE_data.num_backends = world_size;

    ONLINE_DBG_1("Generating the network topology... %s\n", FE_data.topology_file);
    Generate_Topology(world_size, node_list, FE_data.resources_file, FE_data.topology_file);

    ONLINE_DBG_1("Launching the front-end...\n");
    FE = new FrontEnd();
    FE->Init(
      FE_data.topology_file,
      FE_data.num_backends,
      FE_data.attach_file,
      false /* Don't wait for BEs to connect */ );

    /* When Init() returns the pending connections file has been written.
     * Only the root parses this file to avoid stressing the filesystem.
     */ 
    PendingConnections BE_connex(string(FE_data.attach_file));
    if (BE_connex.ParseForMPIDistribution(world_size, sendbuf, sendcnts, displs) == 0)
    {
      /* The pending connections file was parsed successfully. The root will 
       * distribute the information to the rest of ranks */
      rdy_to_connect = 1;
    }
  }

  ONLINE_DBG_1("Distributing pending connections info to all back-ends...\n");
  /* All back-ends wait for their pending connection information */
  ok_dist = SyncPendingConnections(
              rank, 
              FRONTEND_RANK(world_size), 
              rdy_to_connect, 
              sendbuf, 
              sendcnts, 
              displs, 
              &BE_data);

  if (ok_dist)
  {
    if (I_am_root)
    {
#if !defined(ONLINE_DEBUG)
      /* Temporary files are deleted unless there's an error or debug is enabled*/
      unlink(FE_data.resources_file);
      unlink(FE_data.topology_file);
      unlink(FE_data.attach_file);
#endif

      /* The front-end thread waits for the back-ends to connect and enters the main loop */
      FE_thread_started = (pthread_create(&FE_thread, NULL, FE_main_loop, NULL) == 0);
      if (!FE_thread_started)
      {
        perror("pthread_create: Front-end analysis thread: ");
      }
    }

    /* Start the back-ends */
    ONLINE_DBG("Launching the back-end...\n");
    BE = new BackEnd();
    ok_be_init = (BE->Init( BE_data.my_rank,
                            BE_data.parent_hostname,
                            BE_data.parent_port,
                            BE_data.parent_rank) == 0);
    if (ok_be_init)
    {
      ONLINE_DBG("Back-end is ready!\n");

      /* The back-end enters the main loop */
      BE_thread_started = (pthread_create(&BE_thread, NULL, BE_main_loop, NULL) == 0);
      if (!BE_thread_started)
      {
        perror("pthread_create: Back-end analysis thread: ");
      }
    }
  }

  /* Final allreduce to check whether all back-ends are ok */
  me_init_ok = (ok_dist && ok_be_init && BE_thread_started && ((I_am_root && FE_thread_started) || (!I_am_root)));

  we_init_ok = SyncOk(me_init_ok);

  if (!we_init_ok) 
  {
    /* Any task had errors... shutdown everything! */ 
    if (rank == 0) fprintf(stderr, "Online_Start:: FATAL ERROR: Initializing the on-line analysis (see errors above).\n");
    Online_Disable();
    Online_Stop();
  }

  return (we_init_ok);
}

void * FE_main_loop(void *context)
{
  /* Stall the front-end until all back-ends have connected */
  FE->Connect();

  ONLINE_DBG_1("Front-end entering the main analysis loop...\n");
  while (1) 
  {
    pthread_mutex_lock(&FE_lock);
    pthread_mutex_unlock(&FE_lock);
    sleep(Online_GetFrequency());
  }
  
  return NULL;
}

void * BE_main_loop(void *context)
{
  ONLINE_DBG("Back-end entering the main analysis loop...\n");
  BE->Loop();
  ONLINE_DBG("Back-end exiting the main analysis loop...\n");
  return NULL;
}


/**
 * Online_Stop
 */
int Online_Stop()
{
  ONLINE_DBG("Stopping the online-analysis...\n");

  if (I_am_root)
  {
    ONLINE_DBG_1("Shutting down the front-end...\n");
    /* Take the mutex to prevent a shutdown while a protocol is being computed */
    pthread_mutex_lock(&FE_lock);
    FE->Shutdown();
    pthread_mutex_unlock(&FE_lock);
  }
  ONLINE_DBG("Waiting for back-end to close...\n");

  if (BE_thread_started) 
    pthread_join(BE_thread, NULL);

  ONLINE_DBG("Back-end closed!\n");

  SyncWaitAll();

  return 0;
}


/**
 * Generate_Topology
 */
int Generate_Topology(int world_size, char **node_list, char *ResourcesFile, char *TopologyFile)
{
  int i = 0;
  FILE *fd = NULL;

  fd = fopen(ResourcesFile, "w+");
  for (i=0; i<world_size; i++)
  {
    fprintf(fd, "%s\n", node_list[i]);
  }
  fclose(fd);

  fd = fopen(TopologyFile, "w+");
  fprintf(fd, "%s:0 ;\n", node_list[FRONTEND_RANK(world_size)]);
  fclose(fd);

  return 0;
}
