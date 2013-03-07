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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/tracer/xml-parse.c $
 | @last_commit: $Date: 2013-01-25 15:56:47 +0100 (Fri, 25 Jan 2013) $
 | @version:     $Revision: 1464 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id: threadid.c 1311 2012-10-25 11:05:07Z harald $";

#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_DLFCN_H
# include <dlfcn.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#include <string>
#include <sstream>
#include <FrontEnd.h>
#include <BackEnd.h>
#include <PendingConnections.h>
#include "threadid.h"
#include "clock.h"
#include "wrapper.h"
#include "utils.h"
#include "trace_buffers.h"
#include "online_buffers.h"
#include "OnlineControl.h"
#include "Syncing.h"
#include "SpectralRoot.h"
#include "SpectralWorker.h"

using std::string;
using std::stringstream;

/**
 * Global variables
 */
int  this_BE_rank   = 0;
int  I_am_root      = 0;

/**
 * These are only significant in the MPI process that runs the front-end (FRONTEND_RANK define)
 */
FrontEnd        *FE = NULL;
pthread_t        FE_thread;             // Thread running the front-end 
int              FE_thread_started = 0; // 1 if the FE thread is spawned successfully
FE_thread_data_t FE_data;               // Arguments for the front-end
pthread_mutex_t  FE_running_prot_lock;

static void Stop_FE();

/**
 * Each task runs a back-end
 */
BackEnd         *BE = NULL; 
pthread_t        BE_thread;             // Thread running the back-end analysis loop
int              BE_thread_started = 0; // 1 if the BE thread is spawned successfully
BE_thread_data_t BE_data;               // Connection information for the back-end
unsigned long long AppPausedAt  = 0;
unsigned long long AppResumedAt = 0;

static void Stop_BE();
static void BE_pre_protocol (string prot_id, Protocol *p);
static void BE_post_protocol(string prot_id, Protocol *p);

/**
 * Control that the main app thread and the back-end thread don't stop the network twice.
 */
int STOP_Ongoing = 0;
pthread_mutex_t STOP_lock;

/**
 * Buffer for the online events
 */
Buffer_t *OnlineBuffer = NULL;
char      tmpBufferFile[TMP_DIR];
char      finalBufferFile[TMP_DIR];

/**
 * Generates the paths for the temporary and final trace buffers.
 */
static void Online_GenerateOutputFiles()
{
  int initial_TASKID = Extrae_get_initial_TASKID();
  FileName_PTT(tmpBufferFile, Get_TemporalDir(initial_TASKID), Get_ApplName(), getpid(), initial_TASKID, 0, EXT_TMP_ONLINE);
  FileName_PTT(finalBufferFile, Get_FinalDir(TASKID), Get_ApplName(),  getpid(), TASKID, 0, EXT_ONLINE);
}


/*****************************************************************\
|***                      BACK-END SIDE                        ***|
\*****************************************************************/

/**
 * Online_Start
 *
 * The root task specified by FRONTEND_RANK initializes the front-end,
 * and all tasks initialize a back-end. The routine returns when
 * the network is connected and the FE and BE analysis threads have 
 * been created.
 *
 * @param rank       This MPI task rank.
 * @param world_size Total number of MPI tasks.
 * @param node_list  List of hostnames where each task is running.
 *
 * @return 1 if all tasks joined the network without errors; 0 otherwise.
 */
int Online_Start(int rank, int world_size, char **node_list)
{
  int me_init_ok     = 0;
  int we_init_ok     = 0;
  int ok_dist        = 0;
  int ok_init        = 0;
  int rdy_to_connect = 0;

  char *sendbuf  = NULL;
  int  *sendcnts = NULL;
  int  *displs   = NULL;

  ONLINE_DBG("Starting the on-line analysis ...\n");

  this_BE_rank = rank;
  I_am_root    = (rank == FRONTEND_RANK(world_size));

  if (I_am_root)
  {
    /* Start the front-end */
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

    ONLINE_DBG_1("Generating the network topology...\n");
    FE_data.num_backends = world_size;
    Generate_Topology(world_size, node_list, FE_data.resources_file, FE_data.topology_file);

    ONLINE_DBG_1("Launching the front-end...\n");
    FE = new FrontEnd();
    FE->Init(
      FE_data.topology_file,
      FE_data.num_backends,
      FE_data.attach_file,
      false /* Don't wait for BEs to attach */ );

    /* When Init() returns the attachments file has been written.
     * Only the root parses this file to avoid stressing the filesystem.
     */ 
    PendingConnections BE_connex(string(FE_data.attach_file));
    if (BE_connex.ParseForMPIDistribution(world_size, sendbuf, sendcnts, displs) == 0)
    {
      /* The attachments file was parsed successfully. The root will 
       * distribute the information to the rest of ranks */
      rdy_to_connect = 1;
    }
  }

  ONLINE_DBG_1("Distributing attachments information to all back-ends...\n");
  /* All back-ends wait for their attachment information */
  ok_dist = SyncAttachments(
              rank, 
              FRONTEND_RANK(world_size), 
              rdy_to_connect, 
              sendbuf,
              sendcnts,
              displs,
              &BE_data);

  if (ok_dist)
  {
    /* The attachments were successfully distributed */
    if (I_am_root)
    {
#if !defined(ONLINE_DEBUG)
      /* Temporary files are deleted unless there's an error or debug is enabled*/
      unlink(FE_data.resources_file);
      unlink(FE_data.topology_file);
      unlink(FE_data.attach_file);
#endif

      /* Start the front-end analysis thread */
      FE_thread_started = (pthread_create(&FE_thread, NULL, FE_main_loop, NULL) == 0);
      if (!FE_thread_started)
      {
        perror("pthread_create: Front-end analysis thread: ");
      }
    }

    /* Start the back-ends */
    ONLINE_DBG("Launching the back-end...\n");
    BE = new BackEnd();
    ok_init = (BE->Init( BE_data.my_rank,
                         BE_data.parent_hostname,
                         BE_data.parent_port,
                         BE_data.parent_rank) == 0);
    if (ok_init)
    {
      ONLINE_DBG("Back-end is ready!\n");

      /* Start the back-end analysis thread */
      BE_thread_started = (pthread_create(&BE_thread, NULL, BE_main_loop, NULL) == 0);
      if (!BE_thread_started)
      {
        perror("pthread_create: Back-end analysis thread: ");
      }
    }
  }

  /* Final synchronization to check whether all back-ends are ok */
  me_init_ok = (ok_dist && ok_init && BE_thread_started && ((I_am_root && FE_thread_started) || (!I_am_root)));

  we_init_ok = SyncOk(me_init_ok);

  if (!we_init_ok) 
  {
    /* Any task had errors... shutdown everything! */ 
    if (rank == 0) fprintf(stderr, "Online_Start:: FATAL ERROR: Initializing the on-line analysis (see errors above).\n");
    Online_Disable();
    Online_Stop();
  }
  else
  {
    /* Create the online buffer */
    Online_GenerateOutputFiles();
    OnlineBuffer = new_Buffer(1000, tmpBufferFile, 0);
  }
  return (we_init_ok);
}


/**
 * Online_Stop
 *
 * The root task starts the network shutdown, and all tasks
 * stall waiting for their analysis threads to die.
 */
void Online_Stop()
{
  ONLINE_DBG("Stopping the online-analysis...\n");

  if (I_am_root)
  {
    /* Start the network shutdown */
    Stop_FE();
  }

  ONLINE_DBG("Waiting for back-end to finish...\n");
  
  /* Wait for the analysis threads to die */
  if (FE_thread_started)
    pthread_join(FE_thread, NULL);
  if (BE_thread_started)
    pthread_join(BE_thread, NULL);

  ONLINE_DBG("Back-end closed!\n");

  /* Barrier synchronization */
  SyncWaitAll(); 
}


/**
 * Stop_FE
 *
 * Initiates the network shutdown, which will cause the analysis threads to quit.
 * This routine can be called either by the main thread (app has finished) or 
 * the front-end analysis thread (analyis is over).
 */
static void Stop_FE()
{
  /* Take the mutex to prevent a shutdown while a protocol is being computed */
  pthread_mutex_lock(&FE_running_prot_lock);
  if (FE->isUp())
  {
    ONLINE_DBG_1("Shutting down the front-end...\n");
    FE->Shutdown(); /* All backends will leave the main analysis loop */
  }
  pthread_mutex_unlock(&FE_running_prot_lock);
}


/**
 * Stop_BE
 *
 * This routine is called when the back-end analysis thread quits, 
 * either because the application or the analysis has finished.
 */
static void Stop_BE()
{
  Online_Flush();
}


/**
 * Front-end analysis thread
 *
 * Waits for all back-ends to attach, loads the analysis protocols and 
 * enters the main analysis loop, dispatching commands periodically.
 */
void * FE_main_loop(UNUSED void *context)
{
  int done = 0;

  /* Stall the front-end until all back-ends have connected */
  FE->Connect();

  /* Load the FE-side analysis protocols */
  FrontProtocol *fp = new SpectralRoot();
  FE->LoadProtocol( fp );

  ONLINE_DBG_1("Front-end entering the main analysis loop...\n");
  do 
  {
    ONLINE_DBG_1("Front-end going to sleep for %d seconds...\n", Online_GetFrequency());
    sleep(Online_GetFrequency());

    /* Take the mutex to prevent starting a new protocol if the application has just finished */
    pthread_mutex_lock(&FE_running_prot_lock);
    if (FE->isUp())
    {
      FE->Dispatch("SPECTRAL", done);
    }
    pthread_mutex_unlock(&FE_running_prot_lock);

  } while ((FE->isUp()) && (!done));

  ONLINE_DBG_1("Front-end exiting the main analysis loop...\n");

  /* The analysis is over, start the network shutdown */
  Stop_FE();

  return NULL;
}


/**
 * Back-end analysis thread
 *
 * Loads the counterpart analysis protocols and enters an infinite loop waiting for commands
 * from the front-end. This thread exists when the front-end initiates the network shutdown
 * calling to Stop_FE().
 */
void * BE_main_loop(UNUSED void *context)
{
  /* Load the BE-side analysis protocols */
  BackProtocol *bp = new SpectralWorker();
  BE->LoadProtocol( bp );

  /* Enter the main analysis loop */
  ONLINE_DBG("Back-end entering the main analysis loop...\n");
  BE->Loop(&BE_pre_protocol, &BE_post_protocol);
  ONLINE_DBG("Back-end exiting the main analysis loop...\n");

  /* At this point, the front-end did a shutdown and quit, and the back-ends exit the listening loop and quit */
  Stop_BE();

  ONLINE_DBG("Bye!\n");

  return NULL;
}


/**
 * Callback called before the back-end runs a protocol. 
 * We pause the application, then emit events into the online buffer.
 *
 * @param prot_id The protocol identifier.
 * @param p       The protocol object.
 */
static void BE_pre_protocol(UNUSED string prot_id, UNUSED Protocol *p)
{
  Online_PauseApp();
  TRACE_ONLINE_EVENT(TIME, ONLINE_STATE_EV, ONLINE_PAUSE_APP);
}


/**
 * Callback called after the back-end runs a protocol.
 * We emit events into the online buffer, then resume the application.
 *
 * @param prot_id The protocol identifier.
 * @param p       The protocol object.
 */
static void BE_post_protocol(UNUSED string prot_id, UNUSED Protocol *p)
{
  TRACE_ONLINE_EVENT(TIME, ONLINE_STATE_EV, ONLINE_RESUME_APP);
  Online_ResumeApp();
}


/**
 * Pause the application by locking the mutex on all tracing buffers.
 */
void Online_PauseApp()
{
  unsigned int thread = 0;
  for (thread=0; thread<Backend_getMaximumOfThreads(); thread++)
  {
    Buffer_Lock(TRACING_BUFFER(thread));
  }
  AppPausedAt = TIME;
}


/**
 * Flushes the tracing buffers to disk after the analysis, and resumes
 * the application by releasing the mutex on all tracing buffers.
 */
void Online_ResumeApp()
{
  unsigned int thread = 0;
  AppResumedAt = TIME;

  for (thread=0; thread<Backend_getMaximumOfThreads(); thread++)
  {
    Buffer_Flush(TRACING_BUFFER(thread));
  }
  for (thread=0; thread<Backend_getMaximumOfThreads(); thread++)
  {
    Buffer_Unlock(TRACING_BUFFER(thread));
  }
}


/**
 * Returns the timestamp when the application was last paused.
 *
 * @return the last pause time.
 */
unsigned long long Online_GetAppPauseTime()
{
  return AppPausedAt;
}


/**
 * Returns the timestamp when the application was last resumed.
 *
 * @return the last resume time.
 */
unsigned long long Online_GetAppResumeTime()
{
  return AppResumedAt;;
}


/**
 * Generates the network topology specification and writes it into a file.
 *
 * @return 0 on success; -1 otherwise.
 */
int Generate_Topology(int world_size, char **node_list, char *ResourcesFile, char *TopologyFile)
{
  int  i = 0, k = 0;
  FILE *fd = NULL;
  char *selected_topology = NULL;
  char *env_topgen = NULL;
  string Topology = "";
  string TopologyType = "";

  /* Write the available resources in a file */
  fd = fopen(ResourcesFile, "w+");
  for (i=0; i<world_size; i++)
  {
    fprintf(fd, "%s\n", node_list[i]);
  }
  fclose(fd);

#if 1
  /* Check the topology specified in the XML */
  selected_topology = Online_GetTopology();
  if (strcmp(selected_topology, "auto") == 0)
  {
    /* Build an automatic topology with the default fanout */
    int cps_x_level = world_size;
    int tree_depth  = 0;
    stringstream ssTopology; 

    while (cps_x_level > DEFAULT_FANOUT)
    {
      cps_x_level = cps_x_level / DEFAULT_FANOUT;
      tree_depth ++;
    }

    for (k=0; k<tree_depth; k++)
    {
      ssTopology << DEFAULT_FANOUT;
      if (k < tree_depth - 1) ssTopology << "x"; 
    }
    Topology = ssTopology.str();
    TopologyType = "b";
 
    ONLINE_DBG_1("Using an automatic topology: %s\n", (Topology.length() == 0 ? "root-only" : Topology.c_str()));
  }
  else if (strcmp(selected_topology, "root") == 0)
  {
    /* All backends will connect directly to the front-end */
    Topology = "";
    ONLINE_DBG_1("Using a root-only topology\n");
  }
  else
  {
    /* Use the topology specified by the user */
    Topology = string(selected_topology);
    TopologyType = "g";
    ONLINE_DBG_1("Using the user topology: %s\n", Topology.c_str());
  }

  /* Write the topology file */
  env_topgen = getenv("MRNET_TOPGEN");
  if ((Topology.length() == 0) || (env_topgen == NULL))
  {
    /* Writing a root-only topology */
    fd = fopen(TopologyFile, "w+");
    fprintf(fd, "%s:0 ;\n", node_list[FRONTEND_RANK(world_size)]);
    fclose(fd);
  }
  else
  {
    /* Invoking mrnet_topgen to build the topology file */
    string cmd;
    cmd = string(env_topgen) + " -f " + node_list[FRONTEND_RANK(world_size)] + " --hosts=" + string(ResourcesFile) + " --topology=" + TopologyType + ":" + Topology + " -o " + string(TopologyFile);
    ONLINE_DBG_1("Invoking the topology generator: %s\n", cmd.c_str());
    system(cmd.c_str());

  }
  ONLINE_DBG_1("Topology written at '%s'\n", TopologyFile);
#endif

#if 0
#if 0
    fd = fopen(TopologyFile, "w+");
    fprintf(fd, "%s:0; \n", node_list[FRONTEND_RANK(world_size)]);
    fclose(fd);
#else
    fd = fopen(TopologyFile, "w+");
    fprintf(fd, "%s:0 => \n", node_list[FRONTEND_RANK(world_size)]);
    fprintf(fd, "  %s:1  \n", node_list[FRONTEND_RANK(world_size)]);
    fprintf(fd, "  %s:2  \n", node_list[FRONTEND_RANK(world_size)]);
    fprintf(fd, "  %s:3  \n", node_list[FRONTEND_RANK(world_size)]);
    fprintf(fd, "  %s:4  \n", node_list[FRONTEND_RANK(world_size)]);
    fprintf(fd, "  %s:5; \n", node_list[FRONTEND_RANK(world_size)]);
    fclose(fd);
#endif
#endif

  return 0;
}


/**
 * Closes all tracing buffers and dumps the online buffer to disk. 
 */
void Online_Flush()
{
  /* Close the application buffers */
  Online_PauseApp();
  Backend_Finalize_close_files();
  Online_ResumeApp();

  /* Flush and close the online buffer */
  Buffer_Flush(OnlineBuffer);
  Buffer_Close(OnlineBuffer);

  /* Rename the temporary file into the final online buffer */
  if (file_exists (tmpBufferFile))
  {
    rename_or_copy (tmpBufferFile, finalBufferFile);
    fprintf (stdout, PACKAGE_NAME": Online trace file created : %s\n", finalBufferFile);
    fflush(stdout);
  }
}


/**
 * Removes the online buffer temporary files.
 */
void Online_CleanTemporaries()
{
  if (file_exists (tmpBufferFile))
    if (unlink(tmpBufferFile) == -1)
      fprintf (stderr, PACKAGE_NAME": Error removing online file (%s)\n", tmpBufferFile);
}


/**
 * @return the path to the temporary online buffer file.
 */
char * Online_GetTmpBufferName()
{
  return tmpBufferFile;
}


/**
 * @return the path to the final online buffer file.
 */
char * Online_GetFinalBufferName()
{
  return finalBufferFile;
}
