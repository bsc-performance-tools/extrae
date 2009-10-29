/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/mrnet_be.C,v $
 | 
 | @last_commit: $Date: 2009/10/29 10:10:19 $
 | @version:     $Revision: 1.14 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id: mrnet_be.C,v 1.14 2009/10/29 10:10:19 gllort Exp $";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_MMAN_H
# include <sys/mman.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif

#include <mrnet/MRNet.h>
#include "mrnet_commands.h"
#include "mrnet_be.h"
#include "utils.h"
#include "common.h"
#include "be_protocol.h"
#include "clock.h"
#include "hwc.h"
#if defined(HAVE_MPI)
# include <mpi.h>
#endif
#include "mrn_config.h"

using namespace MRN;

/* Global variables */
Network *net = NULL;
Stream *downStream = NULL, *upStream = NULL, *p2pStream = NULL;
pthread_t CmdHandler_thread;
Buffer_t *MRNetBuffer = NULL;
Condition_t SyncFlush_Completion;

int BE_ID; 

/* Private variables */
static int MRN_isEnabled = FALSE;
#if defined(HAVE_MPI)
static int MRNet_AllConnected = FALSE;
#endif

/**
 * Marks this module as enabled.
 */
void Enable_MRNet()
{
	MRN_isEnabled = TRUE;
}

/**
 * Returns whether this module is enabled.
 * @return Returns 1 if this module is enabled, 0 otherwise.
 */
int MRNet_isEnabled()
{
   return MRN_isEnabled;
}

/* XXX: Move this to mrn_config.c */
void Config_Startup_BE()
{
	if (BE_ID == 0)
	{
		int set, max_sets = HWC_Get_Num_Sets();
		int *hwc_ids = NULL;
		int num_hwc = 0;

		/* Send HWC sets configuration */
		MRN_STREAM_SEND(p2pStream, MRN_CONFIG, "%d", max_sets);
		for (set=0; set<max_sets; set++)
		{
			num_hwc = HWC_Get_Set_Counters_ParaverIds(set, &hwc_ids);
			MRN_STREAM_SEND(p2pStream, MRN_CONFIG, "%ad", hwc_ids, num_hwc);
		}

		/* Send target trace size */
		MRN_STREAM_SEND(p2pStream, MRN_CONFIG, "%d %d %d", MRNCfg_GetTargetTraceSize(), MRNCfg_GetAnalysisType(), MRNCfg_GetStartAfter());
	}
}

/**
 * Connects to the MRNet and creates a thread listening for commands received through it.
 * @param[in] rank The MPI task identifier.
 * @return Returns 1 upon successful connection to the MRNet, or 0 if an error occurred.
 */
int Join_MRNet(int rank)
{
	int rc = 0;
	BE_ID = rank;

	/* Stagger the connection of sockets */
	usleep(1000000 * (rank % 10));

	/* XXX: Only the task 0 should read the file and broadcast the information to the other tasks */
	if (Connect_Backend(rank, "./BE.map") == -1) 
	{
		fprintf(stderr, "[BE %d] Join_MRNet: Error while joining the network.\n", rank);
		rc = 0;
	}
	else 
	{
		char tmp[128];

		/* Initialize the streams */
		Register_Streams(rank);

		Config_Startup_BE ();

		/* Allocate the events buffer for the MRNet */
		FileName_PTT(tmp, ".", "TRACE", getpid(), rank, 0, EXT_MRN);
		MRNetBuffer = new_Buffer(1000, tmp);

		/* Start the commands handler thread */
		if (pthread_create(&CmdHandler_thread, NULL, Commands_Handler, &rank) != 0)
		{
			fprintf(stderr, "[BE %d] Join_MRNet: Error starting the commands handler thread.\n", rank);
			rc = 0;
		}
		else rc = 1;

#if defined(HAVE_MPI)
		/* Check all processes connected successfully to the MRNet */
		PMPI_Allreduce(&rc, &MRNet_AllConnected, 1, MPI_INTEGER, MPI_LAND, MPI_COMM_WORLD);
		if (!MRNet_AllConnected)
		{
			if (rank == 0)
			{
				fprintf(stderr, "[BE %d] Join_MRNet: Some back-ends did not connect to the network due to previous errors.\n", rank);
			}
			rc = 0;
		}
#endif
	}

	return rc;
}

void MRN_CloseFiles ()
{
	Buffer_Flush (MRNetBuffer);
	Buffer_Close (MRNetBuffer);
}

/**
 * Connects the backend to its parent communication node.
 * @param[in] rank The MPI task identifier.
 * @param[in] be_topology_file The file specifying where do the back-ends have to connect. 
 * @return Returns 0 upon successful connection to the MRNet, or -1 if an error occurred.
 */
int Connect_Backend(int rank, char *be_topology_file)
{
	int fd;
	struct stat sbuf;
	char *be_topology;
	int num_backends;
	char **backends;
	int fields;
	char **parent_info;
	char *parent_host;
	char be_host[HOST_NAME_MAX];
	Port parent_port;
	Rank parent_rank;

	if (be_topology_file == (char *)NULL)
	{
		fprintf(stderr, "[BE %d] Connect_Backend: Back-ends topology file was not specified.\n", rank);
		return -1;    
	}

	/* Map the file into memory */
	if ((fd = open(be_topology_file, O_RDONLY)) == -1)
	{
		fprintf(stderr, "[BE %d] Connect_Backend: Error opening back-ends topology file '%s'\n", rank, be_topology_file);
		perror("Connect_Backend: open");
		return -1;
	}
	if (fstat(fd, &sbuf) == -1)
	{
		perror("Connect_Backend: stat");
		return -1;
	}
	if ((be_topology = (char *)mmap((caddr_t)0, sbuf.st_size, PROT_READ, MAP_SHARED, fd, 0)) == (caddr_t)-1)
	{
		perror("Connect_Backend: mmap");
		return -1;
	}
	if (close(fd) == -1)
	{
		perror("Connect_Backend: close");
		return -1;
	}

#if 0
#if defined(HAVE_MPI)
	/* Delete the topology file when all tasks have read it */
	PMPI_Barrier (MPI_COMM_WORLD);
	if (rank == 0)
	{
		unlink(be_topology_file);
	}
#endif
#endif

	/* Parse the backends connection information */
	num_backends = explode(be_topology, "\n", &backends);
	if (rank >= num_backends)
	{
		fprintf(stderr, "[BE %d] Connect_Backend: Back-ends topology file '%s' is incomplete (rank=%d, num_backends=%d)\n",
			rank, rank, num_backends);
		return -1;
	}
	fields = explode(backends[rank], " ", &parent_info);

	/* Get the parent host, port and rank */
	parent_host = parent_info[0];
	parent_port = (Port)strtoul(parent_info[1], NULL, 10);
	parent_rank = (Rank)strtoul(parent_info[2], NULL, 10);

	/* Connect to the network */
	gethostname(be_host, sizeof(be_host));
	fprintf(stderr, "[BE %d] Connect_Backend: Back-end %s:%d connecting to %s:%d:%d\n", 
		rank, be_host, MRN_RANK(rank), parent_host, parent_port, parent_rank);
	net = new Network(parent_host, parent_port, parent_rank, be_host, MRN_RANK(rank));
	if (net == NULL)
	{
		fprintf(stderr, "[BE %d] Connect_Backend: Back-end initialization failed.\n", rank);
		return -1;
	}

	return 0;
}

/**
 * Initializes the streams that have been created at the front-end side 
 * @param[in] rank The MPI task identifier. 
 */
void Register_Streams (int rank)
{
	int tag;
	PacketPtr data;
   
	MRN_NETWORK_RECV(net, &tag, data, MRN_REGISTER_STREAM, &downStream, true);
	MRN_NETWORK_RECV(net, &tag, data, MRN_REGISTER_STREAM, &upStream, true);
	MRN_NETWORK_RECV(net, &tag, data, MRN_REGISTER_STREAM, &p2pStream, true);

#if defined(MRN_DEBUG)
	fprintf(stderr, "[BE %d] Registered downStream (ID: %d)\n", rank, downStream->get_Id());
	fprintf(stderr, "[BE %d] Registered upStream (ID: %d)\n", rank, upStream->get_Id());
	fprintf(stderr, "[BE %d] Registered p2pStream (ID: %d)\n", rank, p2pStream->get_Id());
	fflush(stderr);
#endif
}

#if ! defined(NEW_DYNAMIC_STREAMS)
/**
 * Enters an infinite loop waiting for and processing commands sent from the MRNet root.
 * @param[in] rank_ptr A pointer to the task identifier.
 */
void * Commands_Handler (void * rank_ptr)
{
	int tag;
	PacketPtr data;
//	int rank = *((int *)rank_ptr);
	int rank = BE_ID;

#if defined(TRACE_MRN_ACTIVITY)
	TRACE_MRN_EVENT(TIME, MRNET_EV, 1);
#endif
	do
	{
		/* Block waiting for the next command */
		MRN_STREAM_RECV(downStream, &tag, data, MRN_ANY);

#if defined(TRACE_MRN_ACTIVITY)
		TRACE_MRN_EVENT(TIME, MRNET_EV, 2);
#endif
		/* Execute the back-end side protocol associated to the received command (tag) */
		Execute_Backend_Protocol (tag, downStream);
#if defined(TRACE_MRN_ACTIVITY)
		TRACE_MRN_EVENT(TIME, MRNET_EV, 0);
#endif

	} while (tag != MRN_TERMINATE);
#if defined(TRACE_MRN_ACTIVITY)
	TRACE_MRN_EVENT(TIME, MRNET_EV, 0);
#endif

	return NULL;
}
#else
#include "StreamPublisher.h"
void * Commands_Handler (void * rank_ptr)
{
    int tag;
    PacketPtr data;
//    int rank = *((int *)rank_ptr);
//	int *be_mask, num_be;
	int rank = BE_ID;

#if defined(TRACE_MRN_ACTIVITY)
	TRACE_MRN_EVENT(TIME, MRNET_EV, 1);
#endif

	do
	{
//		int new_stream_id;
		Stream *new_stream;
		StreamPublisher sp(rank, net, downStream);

#if 0
		/* Block waiting for the next request */
		MRN_STREAM_RECV(downStream, &tag, data, MRN_REGISTER_STREAM);
		data->unpack("%d %ad", &new_stream_id, &be_mask, &num_be);

		fprintf(stderr, "[BE %d] new_stream_id=%d be_mask[%d]=%d\n", rank, new_stream_id, rank, be_mask[rank]);
		/* Check whether this back-end has to process this request */
		if (be_mask[rank])
		{
			/* Retrieve the specific command */
			new_stream = net->get_Stream(new_stream_id);
			MRN_STREAM_RECV(new_stream, &tag, data, MRN_ANY);

			/* Run the back-end side protocol for this command */
			sleep(3);
#if defined(TRACE_MRN_ACTIVITY)
			TRACE_MRN_EVENT(TIME, MRNET_EV, 2);
#endif
			Execute_Backend_Protocol (tag, new_stream);
#if defined(TRACE_MRN_ACTIVITY)
			TRACE_MRN_EVENT(TIME, MRNET_EV, 0);
#endif

			delete new_stream;
		} 
#endif
		new_stream = sp.Recv();
		if (new_stream != NULL)
		{
            /* Retrieve the specific command */
			MRN_STREAM_RECV(new_stream, &tag, data, MRN_ANY);

			fprintf(stderr, "BE %d processes command %d\n", rank, tag);
			

            /* Run the back-end side protocol for this command */
#if defined(TRACE_MRN_ACTIVITY)
            TRACE_MRN_EVENT(TIME, MRNET_EV, 2);
#endif
            Execute_Backend_Protocol (tag, new_stream);
#if defined(TRACE_MRN_ACTIVITY)
            TRACE_MRN_EVENT(TIME, MRNET_EV, 0);
#endif
			delete new_stream;
		}
		else fprintf(stderr, "BE %d ignores command\n", rank);

	} while (tag != MRN_TERMINATE);

#if defined(TRACE_MRN_ACTIVITY)
	TRACE_MRN_EVENT(TIME, MRNET_EV, 0);
#endif

	return NULL;
}
#endif

int MRNet_Notify_Root(int rank, int thread, int be_request)
{
	if (MRNet_isEnabled())
	{
		/* Notify the root */
		MRN_STREAM_SEND(upStream, be_request, "%d %d", rank, thread);

		/* Request is processed through the P2P stream of this task */
		return Execute_Backend_Protocol(be_request, p2pStream);	
	}
	return 0;
}

/****************************************************************************************************/
/*************************** FUNCTIONS BELOW SHOULD USE MRNet_Notify_Root ***************************/
/****************************************************************************************************/

/**
 * Disconnects from the MRNet
 * @param[in] rank The MPI task identifier.
 */
void Quit_MRNet(int rank)
{
	if (MRNet_isEnabled())	
	{
		/* Notify process finalization */
		MRN_STREAM_SEND(upStream, MRN_QUIT, "");

		/* Wait for MRNet to be shutdown */
		pthread_join (CmdHandler_thread, NULL);
		fprintf(stderr, "[BE %d] Back-end terminated!\n", rank);
	}
}

int MRNet_Sync_Flush (Buffer_t *buffer)
{
	if (Buffer_IsFull (buffer))
	{
		/* Initialize the condition variable */
		Signals_CondInit (&SyncFlush_Completion);

		/* Send the flush request */
		MRN_STREAM_SEND(upStream, MRN_SYNC_FLUSH_BE_REQUEST, "");

		Buffer_Unlock (buffer);

		/* Waits for the synchronized flush to be executed */
		Signals_CondWait (&SyncFlush_Completion);
	}
	return 1;
}

int MRNet_Notify_Flush (Buffer_t *buffer)
{
	int tag;
	PacketPtr data;

	if (Buffer_IsFull (buffer))
	{
		MRN_STREAM_SEND_ID(upStream, MRN_NOTIFY_FLUSH, "");

		Buffer_Unlock (buffer);

		MRN_STREAM_RECV(p2pStream, &tag, data, MRN_NOTIFY_FLUSH);
	}
	return 1;
}

