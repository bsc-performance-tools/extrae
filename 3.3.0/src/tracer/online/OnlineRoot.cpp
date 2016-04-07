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

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::ifstream;
using std::getline;
using std::vector;
using std::stringstream;
using std::cerr;
using std::cout;
using std::endl;

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif

#include <FrontEnd.h>

#include "OnlineRoot.h"
#include "OnlineConfig.h"
#include "OnlineUtils.h"
#include "Messaging.h"
#include "Binder.h"
#include "KnownProtocols.h"
#include "xml-parse-online.h"


/************************\
 *** Global variables ***
\************************/

FrontEnd *FE = NULL;                   /* Front-end handler for the Synapse front-end library */

pthread_t       TerminationThread;     /* Thread listening for termination signals from the main back-end */

/* Messaging */
Messaging *Msgs = NULL;

/* Synchronization with the master back-end process */
Binder *BindToBackend = NULL;

bool TargetMet = false;
bool QuitNow   = false;

/**
 * Writes the topology file required to create the MRNet network.
 *
 * @param NumberOfBackends The total number of back-ends (MPI tasks).
 * @param TopologySpec The topology specification can be auto, root or a user-defined one.
 * @param TopologyFile The name of the file where the topology will be written.
 *
 * @return 0 on success; -1 otherwise.
 */
int Generate_Topology(int NumberOfBackends, string TopologySpec, string TopologyFile)
{
  string Topology     = "";
  string TopologyType = "";

  if (TopologySpec == "auto")
  {
    if (NumberOfBackends > 512)
    {
      /* Build an automatic topology with the default fanout */
      int cps_x_level = NumberOfBackends;
      int tree_depth  = 0;
      stringstream ssTopology;

      while (cps_x_level > DEFAULT_FANOUT)
      {
        cps_x_level = cps_x_level / DEFAULT_FANOUT;
        tree_depth ++;
      }

      for (int i=0; i < tree_depth; i++)
      {
        ssTopology << DEFAULT_FANOUT;
        if (i < tree_depth - 1) ssTopology << "x";
      }
      Topology     = ssTopology.str();
      TopologyType = "b";
    }
    else
    {
      /* Set a fixed number of CPs for small executions */
      TopologyType = "g";
      Topology     = "";
      if (NumberOfBackends >= 32)  Topology = "2";
      if (NumberOfBackends >= 64)  Topology = "4";
      if (NumberOfBackends >= 128) Topology = "8";
      if (NumberOfBackends >= 256) Topology = "16";
    }
    Msgs->say(cerr, "Using an automatic topology: %s: %s", TopologyType.c_str(), (Topology.length() == 0 ? "root-only" : Topology.c_str()));
  }
  else if (TopologySpec == "root")
  {
    /* All backends will connect directly to the front-end */
    Topology = "";
    Msgs->say(cerr, "Using a root-only topology");
  }
  else
  {
    /* Use the topology specified by the user */
    Topology = TopologySpec;
    TopologyType = "g";
    Msgs->say(cerr, "Using the user topology: %s", Topology.c_str());
  }

  /* Write the topology file */
  char *env_topgen = getenv("MRNET_TOPGEN");
  if ((Topology.length() == 0) || (env_topgen == NULL))
  {
    FILE *fd = NULL;

    /* Writing a root-only topology */
    char myHostname[64];
    while( gethostname(myHostname, 64) == -1 ) {}
    myHostname[63] = '\0';

    fd = fopen(TopologyFile.c_str(), "w+");
    fprintf(fd, "%s:0 ;\n", Select_NIC(myHostname).c_str());
    fclose(fd);
  }
  else
  {
    /* Get the localhost for the front-end */
    char myHostname[64];
    while( gethostname(myHostname, 64) == -1 ) {}
    myHostname[63] = '\0';

    /* Invoking mrnet_topgen to build the topology file */
    string cmd;
    cmd = string(env_topgen) + " --fehost=" + Select_NIC(myHostname) + " --hosts=" + BindToBackend->GetResourcesFile() + " --topology=" + TopologyType + ":" + Topology + " -o " + TopologyFile;
    Msgs->say(cerr, "Invoking the topology generator: %s", cmd.c_str());
    system(cmd.c_str());

    /* Check that mrnet_topgen didn't fail and wrote the topology file */
    ifstream fd(TopologyFile.c_str());
    if (!fd.good())
    {
      Msgs->error("mrnet_topgen failed! Check there's enough hosts for the requested topology");
      return -1;
    }
  }

  Msgs->say(cerr, "Topology written at '%s'", TopologyFile.c_str());
  return 0;
}


/**
 * Loads the front-end side of the analysis protocols
 */
void FE_load_known_protocols()
{
#if defined(HAVE_SPECTRAL)
  FE->LoadProtocol( (Protocol *)(new SpectralRoot()) );
#endif

#if defined(HAVE_CLUSTERING)
  FE->LoadProtocol( (Protocol *)(new ClusteringRoot()) );
#endif

  FE->LoadProtocol( (Protocol *)(new GremlinsRoot()) );
}


/**
 * A secondary thread stalls waiting for a termination notice from the master back-end
 */
void * ListenForTermination(void UNUSED *context)
{
  /* Stalls until the termination notice is received */
  BindToBackend->WaitForTermination();

  Msgs->debug(cerr, "Termination notice received");

  QuitNow = true;

  return NULL;
}


/** 
 * The main thread loops dispatching new analyses until the targets are met, 
 * or the application finishes.
 */
void AnalysisLoop()
{
  int RoundCounter = 1;

  /* Load the FE-side analysis protocols */
  FE_load_known_protocols(); 

  Msgs->debug(cerr, "Entering the main analysis loop");
  do
  {
    int NextAnalysis = Online_GetFrequency();

    if (NextAnalysis > 0)
      Msgs->debug(cerr, "Sleeping for %d seconds...", NextAnalysis);
    else
      Msgs->debug(cerr, "Sleeping until the end of the execution...");

    while ((NextAnalysis != 0) && (!QuitNow))
    {
      sleep(1);
      //Msgs->debug(cerr, "%d seconds left...", NextAnalysis);
      NextAnalysis --;
    }
    Msgs->debug(cerr, "Awake!");

    if (!TargetMet) 
    {
      Msgs->debug(cerr, "Starting analysis round #%d...", RoundCounter);
      TargetMet = AnalysisRound();
      Msgs->debug(cerr, "Analysis round #%d is over! Targets were met? %s!", RoundCounter, (TargetMet ? "yes" : "no"));
      RoundCounter++;
    }
  } while ((!TargetMet) && (!QuitNow));

  Msgs->debug(cerr, "Starting network shutdown...");
  Stop_FE();
}


/**
 * Dispatches a new round of analysis
 */
bool AnalysisRound()
{
  int done = 0;

  if (FE->isUp())
  {
    if (Online_GetAnalysis() == ONLINE_DO_SPECTRAL)
    {
      FE->Dispatch("SPECTRAL", done);
    }
    else if (Online_GetAnalysis() == ONLINE_DO_CLUSTERING)
    {
      FE->Dispatch("TDBSCAN", done);
    }
    else if (Online_GetAnalysis() == ONLINE_DO_GREMLINS)
    {
      FE->Dispatch("GREMLINS", done);
    }
    return done;
  }
  else
  {
    return 1;
  }
}


int main(int UNUSED argc, char UNUSED **argv)
{
  vector<string>      BackendNodes;
  int                 NumberOfBackends = 0;
  bool                ResourcesReady   = false;
  string              TopologyFile;

  /* Initialize the messaging system */
  Msgs = new Messaging();

  /* Initialize the inter-process communicator with the master back-end */
  BindToBackend = new Binder();
  
  /* Read the environment variable EXTRAE_CONFIG_FILE */
  char *env_extrae_config_file = getenv("EXTRAE_CONFIG_FILE");

  if (env_extrae_config_file == NULL)
  {
    Msgs->say(cerr, "ERROR: Environment variable EXTRAE_CONFIG_FILE is not set!");
    exit(-1);
  }

  /* Parse the Extrae on-line configuration */
  Parse_XML_Online_From_File( env_extrae_config_file );  

  /* Quit if the on-line analysis is disabled in the configuration */
  if (!Online_isEnabled())
  {
    Msgs->say(cerr, "On-line analysis is disabled in the configuration. Bye!");
    exit(0);
  }

  /* Start a secondary thread listening for a termination notice from the master back-end */
  pthread_create(&TerminationThread, NULL, ListenForTermination, NULL);

  /* Wait for the list of available resources */
  ResourcesReady = BindToBackend->WaitForResources(BackendNodes);
  if (!ResourcesReady)
  {
    Msgs->error("Resources are not ready!");
    exit (-1);
  }
  NumberOfBackends = BackendNodes.size();
  Msgs->say(cerr, "Resources are ready!");

  /* Generate the network topology */
  TopologyFile = BindToBackend->GetTopologyFile();

  if (Generate_Topology(NumberOfBackends, string(Online_GetTopology()), TopologyFile) == -1)
  {
    /* Send empty attachments to wake up the back-ends before their time-out */
    BindToBackend->SendAttachments();
    Msgs->error("Generate_Topology failed: Quitting the front-end now!");
    exit(-1);
  }

  /* Start the MRNet */
  Msgs->say(cerr, "Starting the network...");

  FE = new FrontEnd();

  FE->Init(
      TopologyFile.c_str(),
      NumberOfBackends,
      BindToBackend->GetAttachmentsTmpFile().c_str(),
      false /* false = Don't wait for BEs to attach */ );

  /* Send the attachments to the master back-end */
  BindToBackend->SendAttachments();

  /* Wait for all back-ends to attach */
  FE->Connect(); 

  /* Enter the analyis loop */
  AnalysisLoop();
}


/**
 * Initiates the network shutdown and quits the program. We enter here either 
 * because the analysis is over, or because the application has ended.
 */
void Stop_FE()
{
  if (FE->isUp())
  {
    Msgs->debug(cerr, "Shutting down the front-end...");
    FE->Shutdown(); /* All backends will leave the main analysis loop */
  }

  /* Clean all exchanged files */
  if (!Msgs->debugging())
  {
    BindToBackend->WipeExchangeData();
  }
  Msgs->debug(cerr, "Bye!");
}

