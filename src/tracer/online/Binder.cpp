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
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "OnlineUtils.h"
#include "Binder.h"
#if defined(HAVE_INOTIFY)
#include <sys/select.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <limits.h>
#include <errno.h>
#include <stdio.h>
#include <libgen.h>
#include <string.h>
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;


/***************************************************************************\
 ***        This class contains several methods for inter-process        ***
 ***        communication between the root and the master-backend.       ***
\***************************************************************************/


/**
 * Root constructor
 */
Binder::Binder()
{
  GPFSPath = "";
  WipeExchangeData();
  Msgs = new Messaging();
  
  char *env_gpfs_path = getenv("EXTRAE_ONLINE_GPFS_PATH");
  if ((env_gpfs_path != NULL) && (dir_exists(env_gpfs_path)))
  {
    GPFSPath = string(env_gpfs_path) + "/";
  }
}


/**
 * Master back-end constructor 
 */
Binder::Binder(int rank)
{
  GPFSPath = "";
  WipeExchangeData();
  Msgs = new Messaging(rank, true);

  char *env_gpfs_path = getenv("EXTRAE_ONLINE_GPFS_PATH");
  if ((env_gpfs_path != NULL) && (dir_exists(env_gpfs_path)))
  {
    GPFSPath = string(env_gpfs_path) + "/";
  }
}


/** 
 * Returns the full path to the given file name taking into account the definition of the environment variable EXTRAE_ONLINE_GPFS_PATH
 */
string Binder::PathTo(string FileName)
{
  string FullPath = GPFSPath + FileName;
  return FullPath;
}


/**
 * Returns the full path to the ONLINE_RESOURCES_FILE 
 */
string Binder::GetResourcesFile()
{
  return PathTo(string(ONLINE_RESOURCES_FILE));
}


/**
 * Returns the full path to the TMP_ONLINE_RESOURCES_FILE 
 */
string Binder::GetResourcesTmpFile()
{
  return PathTo(string(TMP_ONLINE_RESOURCES_FILE));
}


/**
 * Returns the full path to the ONLINE_TOPOLOGY_FILE
 */
string Binder::GetTopologyFile()
{
  return PathTo(string(ONLINE_TOPOLOGY_FILE));
}


/**
 * Returns the full path to the ONLINE_ATTACHMENTS_FILE
 */
string Binder::GetAttachmentsFile()
{
  return PathTo(string(ONLINE_ATTACHMENTS_FILE));
}


/**
 * Returns the full path to the TMP_ONLINE_ATTACHMENTS_FILE
 */
string Binder::GetAttachmentsTmpFile()
{
  return PathTo(string(TMP_ONLINE_ATTACHMENTS_FILE));
}


/**
 * Returns the full path to the ONLINE_TERMINATION_FILE
 */
string Binder::GetTerminationFile()
{
  return PathTo(string(ONLINE_TERMINATION_FILE));
}


/**
 * Sets inotify to watch for the given control file until it is created or a timeout expires.
 *
 * @param FileName Name of the file to check for.
 * @param MaxRetries Maximum number of retries.
 * @param StallTime Number of seconds for each stall.
 * @returns true if file is found; false otherwise.
 */
#if defined(HAVE_INOTIFY)

# define MAX_EVENTS 1024                                      /* Max number of events to process at one go */
# define LEN_NAME   16                                        /* Assuming that the length of the filename won't exceed 16 bytes */
# define EVENT_SIZE ( sizeof (struct inotify_event) )         /* Size of one event */
# define BUF_LEN    ( MAX_EVENTS * ( EVENT_SIZE + LEN_NAME )) /* Buffer to store the data of events */

bool Binder::WaitForFile(string FileName, int MaxRetries, int StallTime)
{
  int            fd, wd, ready;
  fd_set         descriptors;
  struct timeval timeout;
  bool           done  = false;
  bool           found = false;
  char           buffer[BUF_LEN];

  /* Initialize inotify*/
  fd = inotify_init();
  if ( fd < 0 ) 
  {
    Msgs->error("inotify: error initializing");
    return WaitForFilePolling(FileName, MaxRetries, StallTime);
  }
  
  /* Add watch to starting directory */
  wd = inotify_add_watch(fd, dirname((char *)FileName.c_str()), IN_CREATE | IN_MOVED_TO );

  if (wd == -1)
  {
    Msgs->error("inotify: couldn't add watch to directory '%s'", dirname((char *)FileName.c_str()));
    return WaitForFilePolling(FileName, MaxRetries, StallTime);
  }
  else
  {
    Msgs->debug(cerr, "inotify: watching directory '%s' for file '%s'", dirname((char *)FileName.c_str()), basename((char *)FileName.c_str()));
  }

  FD_ZERO ( &descriptors );
  FD_SET ( fd, &descriptors );

  timeout.tv_sec = MaxRetries * StallTime;
  timeout.tv_usec = 0;

  while (!done)
  {
    if (MaxRetries < 0)
      ready = select ( fd + 1, &descriptors, NULL, NULL, NULL);
    else
      ready = select ( fd + 1, &descriptors, NULL, NULL, &timeout);

    if ( ready < 0 )
    {
      perror("select");
      return WaitForFilePolling(FileName, MaxRetries, StallTime);
    }
    else if ( !ready )
    {
      /* Timeout expired */
      found = false;
      done  = true;
    }
    else if ( FD_ISSET( fd, &descriptors ))
    {
      int i, length;
      
      /* Process the inotify events */
      length = read( fd, buffer, BUF_LEN );

      if ( length < 0 ) 
      {
        perror( "read" );
        return false;
      }

      i = 0;
      while ( i < length ) 
      {
        struct inotify_event *event = ( struct inotify_event * ) &buffer[ i ];
        if ( event->len ) 
        {
          if (( event->mask & IN_CREATE ) || ( event->mask & IN_MOVED_TO ))
          {
            if (strcmp(event->name, basename((char *)FileName.c_str())) == 0)
            {
              Msgs->debug(cerr, "inotify: detected file '%s'", basename((char *)FileName.c_str()));
              found = true; 
              done  = true;
            }
          }
          i += EVENT_SIZE + event->len;
        }
      }
    }
  }

  inotify_rm_watch ( fd, wd );
  close( fd );

  return found;
}
#else
bool Binder::WaitForFile(string FileName, int MaxRetries, int StallTime)
{
  return WaitForFilePolling(FileName, MaxRetries, StallTime);
}
#endif


/**
 * Stalls 'StallTime' seconds a maximum of 'MaxRetries' times or until file 'FileName' is found.
 *
 * @param FileName Name of the file to check for.
 * @param MaxRetries Maximum number of retries.
 * @param StallTime Number of seconds for each stall.
 * @returns true if file is found; false otherwise.
 */
bool Binder::WaitForFilePolling(string FileName, int MaxRetries, int StallTime)
{
  bool found = false;
  int  retry = 0;

  Msgs->debug(cerr, "Waiting for file '%s'", FileName.c_str());

  while ( (!found) && ((MaxRetries == -1) || (retry < MaxRetries)) ) 
  {
    ifstream fd(FileName.c_str());
    if (fd.good())
    {
      fd.close(); 
      found = true;
    }
    else
    {
      sleep(StallTime);
      retry ++;
    }
  }
  if (found)
    Msgs->debug(cerr, "File '%s' found after %d seconds!", FileName.c_str(), retry * StallTime);
  else
    Msgs->debug(cerr, "File '%s' NOT found after %d seconds!", FileName.c_str(), retry * StallTime);

  return found;
}


/**
 * The master-backend writes the list of available resources in a file for the root.
 *
 * @param NumberOfNodes Total count of nodes (number of MPI tasks).
 * @param ListOfNodes Array containing the name of all nodes.
 */
void Binder::SendResources(int NumberOfNodes, char **ListOfNodes)
{
  FILE *fd = NULL;

  /* Write the available resources in a temporary file. */
  fd = fopen(PathTo(TMP_ONLINE_RESOURCES_FILE).c_str(), "w+");
  
  for (int i=0; i<NumberOfNodes; i++)
  {
    fprintf(fd, "%s\n", Select_NIC(ListOfNodes[i]).c_str());
  }
  fclose(fd);

  /* Rename the temporary when all the data has been written. 
   * This wakes up the MRNet root who is waiting on this file to start the network. 
   */
  rename(PathTo(TMP_ONLINE_RESOURCES_FILE).c_str(), PathTo(ONLINE_RESOURCES_FILE).c_str());
}


/**
 * The root stalls waiting for the resources file.
 *
 * @param Backends (out) Will contain the list of node names after the call.
 */
bool Binder::WaitForResources(vector<string> &Backends)
{
  bool found = WaitForFile( PathTo(ONLINE_RESOURCES_FILE), MAX_WAIT_RETRIES, 1 );

  if (found)
  {
    ifstream fd( PathTo(ONLINE_RESOURCES_FILE).c_str() );
    if (fd.good())
    {
      string Node;

      while ( getline(fd, Node) )
      {
        Backends.push_back( Node );
      }
      fd.close();
    }
    else
    {
      found = false;
    }
  }
  return found;
}


/**
 * The root writes the attachments information in a file for the master back-end. FE->Init() must have been 
 * called first, which produces this information in a temporary file. Here we rename the file, and we need 
 * this step so that the back-end does not see the file while it is being written! 
 */
void Binder::SendAttachments()
{
  /* A previous call to FE->Init() in the root must have produced the temporary file TMP_ONLINE_ATTACHMENTS_FILE */
  ifstream fd( PathTo(TMP_ONLINE_ATTACHMENTS_FILE).c_str() );
  if (fd.good())
  { 
    fd.close();

    Msgs->debug(cerr, "Passing attachments file to the master back-end...");

    /* Rename the temporary when all the data has been written. 
     * This wakes up the master back-end who is waiting on this file 
     * to distribute the attachments information and start connecting. 
     */
    rename(PathTo(TMP_ONLINE_ATTACHMENTS_FILE).c_str(), PathTo(ONLINE_ATTACHMENTS_FILE).c_str());
  }
  else
  {
    /* Generate an empty attachments file to wake up the back-ends. When they see it empty, they will generate errors */
    Msgs->error("Attachments file could not be generated!");
    FILE *fd = fopen(PathTo(ONLINE_ATTACHMENTS_FILE).c_str(), "w+");
    fclose(fd);
  }
}


/**
 * The master-backend stalls waiting for the attachments information.
 * 
 * @param ExpectedAttachments Number of lines that should contain the attachments file (equal to the number of back-ends).
 */
bool Binder::WaitForAttachments(int ExpectedAttachments)
{
  int  NumberOfAttachments = 0;
  bool found = WaitForFile( PathTo(ONLINE_ATTACHMENTS_FILE), MAX_WAIT_RETRIES, 1 );

  if (found)
  {
    ifstream fd( PathTo(ONLINE_ATTACHMENTS_FILE).c_str() );
    if (fd.good())
    {
      string Attachment;
      
      while ( getline(fd, Attachment) )
      {
        NumberOfAttachments ++;
      }
      fd.close();

      if (NumberOfAttachments != ExpectedAttachments)
      {
        found = false;
        Msgs->error("Attachments file is incomplete! Expected %d attachments, but %d found!", ExpectedAttachments, NumberOfAttachments);
      }
    }
  }
  return found;
}
 

/**
 * The master-backend sends a termination notice to the root when the application has reached the MPI_Finalize.
 */
void Binder::SendTermination()
{
  FILE *fd = NULL;

  fd = fopen(PathTo(ONLINE_TERMINATION_FILE).c_str(), "w+");
  fclose(fd);
}


/**
 * A thread in the root stalls waiting for the termination notice, when it is received, the root will quit.
 */
bool Binder::WaitForTermination()
{
  bool found = WaitForFile( PathTo(ONLINE_TERMINATION_FILE), -1, 1 );
  return found;
}


/* 
 * Can be called either at root or back-end to delete the exchanged files.  
 */
void Binder::WipeExchangeData()
{
  unlink( PathTo(TMP_ONLINE_RESOURCES_FILE).c_str() );
  unlink( PathTo(ONLINE_RESOURCES_FILE).c_str() );
  unlink( PathTo(ONLINE_TOPOLOGY_FILE).c_str() );
  unlink( PathTo(TMP_ONLINE_ATTACHMENTS_FILE).c_str() );
  unlink( PathTo(ONLINE_ATTACHMENTS_FILE).c_str() );
  unlink( PathTo(ONLINE_TERMINATION_FILE).c_str() );
}

