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

#include "common.h"

#include <string>
#include <iostream>

using std::string;
using std::cerr;
using std::cout;
using std::endl;

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif

#include <BackEnd.h>
#include <PendingConnections.h>

#include "OnlineControl.h"
#include "Syncing.h"
#include "Binder.h"
#include "Messaging.h"
#include "KnownProtocols.h"
#include "online_buffers.h"
#include "clock.h"
#include "wrapper.h"
#include "utils.h"


/************************\
 *** Global variables ***
\************************/

/* Ranks */
int  this_BE_rank       = 0;
bool I_am_master_BE     = false;
int  NumberOfBackends   = 0;

/* Messaging */
Messaging *Msgs = NULL;

/* Synchronization with the root process */
Binder *BindToRoot = NULL;

/* Backend running in this process */
BackEnd  *BE = NULL;                  // Handler for the Synapse back-end library
pthread_t BE_thread;                  // Thread running the back-end analysis loop
bool      BE_thread_started  = false; // 1 if the BE thread is spawned successfully
bool      FE_process_started = false; // true if the FE process is spawned successfully
BE_data_t BE_data;                    // Connection information for the back-end

/* Timers */
unsigned long long AppPausedAt  = 0;  // Timestamp when the current analysis started
unsigned long long AppResumedAt = 0;  // Timestamp when the previous analysis finished

/* Tracing buffer for the online events */
Buffer_t *OnlineBuffer = NULL;
char      tmpBufferFile[TMP_DIR];
char      finalBufferFile[TMP_DIR];

/* Prototypes for private functions */
static void Stop_FE();
static void Stop_BE();

static void BE_load_known_protocols();
static void BE_pre_protocol (string prot_id, Protocol *p);
static void BE_post_protocol(string prot_id, Protocol *p);

static void Online_GenerateOutputFiles();
static void Initialize_Online_Buffers();


/*************************\
 *** Back-end controls ***
\*************************/

/**
 * Initializes the rank variables and the inter-process
 * communicator to the root process in the master back-end.
 *
 * @param rank ID of this MPI process (back-end).
 * @param world_size Number of MPI processes (back-ends).

 * @return 0 on success; -1 otherwise.
 */
int Online_Init(int rank, int world_size) 
{
  this_BE_rank     = rank;
  I_am_master_BE   = (rank == MASTER_BACKEND_RANK(world_size));
  NumberOfBackends = world_size;

  Msgs = new Messaging(this_BE_rank, I_am_master_BE);

  if (I_am_master_BE)
  {
    BindToRoot = new Binder(this_BE_rank);
  }
  return 0;
}

/**
 * Passes the resources to the root, waits for the network to be created
 * and attaches the back-ends.
 *
 * @param node_list List of nodes where each MPI task is executing.
 * 
 * @return 1 on success; 0 otherwise;
 */
int Online_Start(char **node_list)
{
  /* Variables to control the status of the different initialization stages */
  int ok_sync     = 0;
  int ok_init     = 0;
  int me_init_ok  = 0;
  int all_init_ok = 0;
  int BE_ready_to_connect = 0;

  /* Buffers to distribute the attachments information through MPI */
  char *send_buffer   = NULL;
  int  *send_counts   = NULL;
  int  *displacements = NULL;

  Msgs->debug(cerr, "Starting the on-line analysis...");

  if (I_am_master_BE)
  {
    bool AttachmentsReady = false;

    /* 
     * Write the available resources in a file. The MRNet root process
     * is waiting for this file to start the network. 
     */
    BindToRoot->SendResources( NumberOfBackends, node_list );

    /* 
     * At this point, the root will wake up, generate the topology, start
     * the network and write the attachments file that is required by the back-ends.
     */
    AttachmentsReady = BindToRoot->WaitForAttachments( NumberOfBackends );

    if (AttachmentsReady)
    {
      FE_process_started = 1;

      /* The master back-end parses the attachments file and distributes the information through MPI */
      Msgs->debug_one(cerr, "Distributing attachments information to all back-ends...");
      PendingConnections BE_connex(string( BindToRoot->GetAttachmentsFile()));
      if (BE_connex.ParseForMPIDistribution(NumberOfBackends, send_buffer, send_counts, displacements) == 0)
      {
        BE_ready_to_connect = 1;
      }
    }
  }

  /* All back-ends wait for their attachment information */
  ok_sync = SyncAttachments(
              this_BE_rank,
              MASTER_BACKEND_RANK(NumberOfBackends),
              BE_ready_to_connect,
              send_buffer,
              send_counts,
              displacements,
              &BE_data);

  if (ok_sync)
  {
    /* Start the back-end */
    Msgs->debug(cerr, "Launching the back-end...");
    BE = new BackEnd();
    ok_init = (BE->Init( BE_data.my_rank,
                         BE_data.parent_hostname,
                         BE_data.parent_port,
                         BE_data.parent_rank) == 0);

    if (ok_init)
    {
      Msgs->debug(cerr, "Back-end is ready!");

      /* Start the back-end analysis thread */
      BE_thread_started = (pthread_create(&BE_thread, NULL, BE_main_loop, NULL) == 0);
      if (!BE_thread_started)
      {
        perror("pthread_create: Back-end analysis thread: ");
      }
    }
  }

  /* Final synchronization to check whether all back-ends are ok */
  me_init_ok = (ok_sync && ok_init && BE_thread_started && ((I_am_master_BE && FE_process_started) || (!I_am_master_BE)));

  all_init_ok = SyncOk(me_init_ok);
  if (!all_init_ok)
  {
    /* One or more tasks had errors... shutdown everything! */
    Msgs->debug_one(cerr, "Online_Start:: FATAL ERROR: Initializing the on-line analysis (see errors above).");
    Online_Stop();
  }
  else
  {
    Initialize_Online_Buffers();
  }

  return (all_init_ok);
}


/**
 * Communicates with the root process to start the network shutdown, 
 * and all backends stall waiting for their analysis threads to die.
 */
void Online_Stop()
{
  Msgs->debug(cerr, "Stopping the online-analysis");

  Online_Disable();

  if (I_am_master_BE)
  {
    /* Tell the root to quit */
    Stop_FE();
  }

  Msgs->debug(cerr, "Waiting for back-end to finish...");
  
  if (BE_thread_started)
    pthread_join(BE_thread, NULL);

  Msgs->debug(cerr, "Back-end closed!");

  /* Barrier synchronization */
  SyncWaitAll(); 
}


/**
 * Sends a termination notice to the MRNet root process which will cause this process to quit.
 */
static void Stop_FE()
{
  Msgs->debug_one(cerr, "Sending termination notice to root process");
  BindToRoot->SendTermination();
}


/**
 * This routine is called when the back-end analysis thread quits, 
 * either because the application or the analysis has finished.
 */
static void Stop_BE()
{
  Online_Flush();
}


/**
 * Main analysis loop executed by the back-end thread. Loads the counterpart analysis protocols and 
 * enters an infinite loop waiting for commands from the front-end. This thread exits when the master 
 * back-end initiates the network shutdown calling to Stop_FE() at the end of the execution, or when 
 * the root process decides to end the analysis. 
 */
void * BE_main_loop(UNUSED void *context)
{
  /* Load the BE-side analysis protocols */
  BE_load_known_protocols(); 

  /* Enter the main analysis loop */
  Msgs->debug(cerr, "Back-end entering the main analysis loop...");
  BE->Loop(&BE_pre_protocol, &BE_post_protocol);
  Msgs->debug(cerr, "Back-end exiting the main analysis loop...");

  /* At this point, the front-end did a shutdown and quit, and the back-ends exit the listening loop and quit */
  Stop_BE();

  Msgs->debug(cerr, "Bye!");

  return NULL;
}


/**
 * Loads all back-end side protocols
 */
static void BE_load_known_protocols()
{
#if defined(HAVE_SPECTRAL)
  BE->LoadProtocol( (Protocol *)(new SpectralWorker()) );
#endif

#if defined(HAVE_CLUSTERING)
  BE->LoadProtocol( (Protocol *)(new ClusteringWorker()) );
#endif

  BE->LoadProtocol( (Protocol *)(new GremlinsWorker()) );
}


/**
 * Callback called before the back-end runs a protocol. 
 * We pause the application.
 *
 * @param prot_id The protocol identifier.
 * @param p       The protocol object.
 */
static void BE_pre_protocol(UNUSED string prot_id, UNUSED Protocol *p)
{
  Online_PauseApp();
}


/**
 * Callback called after the back-end runs a protocol.
 * We resume the application.
 *
 * @param prot_id The protocol identifier.
 * @param p       The protocol object.
 */
static void BE_post_protocol(UNUSED string prot_id, UNUSED Protocol *p)
{
  Online_ResumeApp();
}


/************************************************\
 *** Main application pause / resume controls ***
\************************************************/

/**
 * Pause the application by locking the mutex on all tracing buffers.
 */
void Online_PauseApp(bool emit_events)
{
  unsigned int thread = 0;

  for (thread=0; thread<Backend_getMaximumOfThreads(); thread++)
  {
    Buffer_Lock(TRACING_BUFFER(thread));
  }
  AppPausedAt = TIME;
  if (emit_events) 
  {
    TRACE_ONLINE_EVENT(LAST_READ_TIME, ONLINE_STATE_EV, ONLINE_PAUSE_APP);
  }
}


/**
 * Flushes the tracing buffers to disk after the analysis, and resumes
 * the application by releasing the mutex on all tracing buffers.
 */
void Online_ResumeApp(bool emit_events)
{
  unsigned int thread = 0;

  AppResumedAt = TIME;
  if (emit_events)
  {
    TRACE_ONLINE_EVENT(LAST_READ_TIME, ONLINE_STATE_EV, ONLINE_RESUME_APP);
  }
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


/*******************************\
 *** Online buffers controls ***
\*******************************/

/**
 * Generates the paths for the temporary and final trace buffers.
 */
static void Online_GenerateOutputFiles()
{
	char hostname[1024];
	int initial_TASKID = Extrae_get_initial_TASKID();
	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");
	FileName_PTT(tmpBufferFile, Get_TemporalDir(initial_TASKID), Get_ApplName(),
	  hostname, getpid(), initial_TASKID, 0, EXT_TMP_ONLINE);
	FileName_PTT(finalBufferFile, Get_FinalDir(TASKID), Get_ApplName(), hostname,
	  getpid(), TASKID, 0, EXT_ONLINE);
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


/**
 * Creates the tracing buffers to hold the on-line events.
 */
static void Initialize_Online_Buffers()
{
  /* Create the online tracing buffer */
  Online_GenerateOutputFiles();

  OnlineBuffer = new_Buffer(1000, Online_GetTmpBufferName(), 0);
}


/**
 * Closes all tracing buffers and dumps the online buffer to disk. 
 */
void Online_Flush()
{
  /* Close the application buffers */
  Online_PauseApp(false);
  Backend_Finalize_close_files();
  Online_ResumeApp(false);

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

