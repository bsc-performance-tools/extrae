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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/mrnet_root.C,v $
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_SYS_MMAN_H
# include <sys/mman.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_WAIT_H
# include <sys/wait.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_LIMITS_H
# include <limits.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
# include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
# include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
# include <netdb.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif

#include <mrnet/MRNet.h>
#include "mrnet_commands.h"
#include "utils.h"
#include "common.h"
#include "mrnet_root.h"
#include "root_protocol.h"
#include "commands_queue.h"
#include "MRNetClustering.h"
#include "BEMask.h"
#include "mrn_config.h"

#define CONCURRENT_DISPATCHERS 1

using namespace MRN;

/*** STREAMS ***
 * downStream: Requests starting on the front-end side, sent to the back-ends
 * upStream: Requests starting on the back-ends, sent to the front-end
 * p2pStream: Used to answer/process requests started at the upStream
 */

/* Global variables */
Stream *downStream=NULL, *upStream=NULL, **p2pStreams=NULL;
pthread_t ListenMonitor_thread, ListenBackends_thread, ListenTimer_thread;

/**
 * Builds a topology file with the specified balanced structure.
 * @param[in] topology A balanced topology spefication in the form F^D.
 * @param[in] hostfile A list of available hosts (must not contain repetitions).
 * @return Path to the file defining the topology.
 */
char * build_Topology(char *topology, char *hostfile)
{
    int fd, fanout, depth;
	FILE *fs;
    MRN::Tree *tree = NULL;
    char *topfile = NULL;
   
    /* Create the topology file */
    topfile = (char *)malloc((strlen(TOPOLOGY_FILE_TEMPLATE)+1) * sizeof(char));
    strcpy(topfile, TOPOLOGY_FILE_TEMPLATE);
    if ((fd = mkstemp(topfile)) == -1)
    {
        perror("mkstemp");
        exit(1);
    }
	if ((fs = fdopen(fd, "w+")) == NULL)
	{
		perror("fdopen");
		exit(1);
	}

	if ((topology != NULL) && (strcmp(topology, "") != 0))
	{
		std::string Topology = topology;
		std::string Hosts = hostfile;

		if (sscanf(topology, "%d^%d", &fanout, &depth) == 2)
		{
			fprintf(stderr, "[FE] Creating balanced topology \"%s\"\n", topology);
			tree = new MRN::BalancedTree (Topology, Hosts, 1, 4, 4);
		}
		else 
		{
			fprintf(stderr, "[FE] Creating other topology \"%s\"\n", topology);
			tree = new MRN::GenericTree (Topology, Hosts);
		}

		/* Write the topology into a file */
		tree->create_TopologyFile(fs);
		delete tree;
	}
	else 
	{
		char this_host[256]; 

		fprintf(stderr, "[FE] Creating root-only topology.\n");
		gethostname(this_host, sizeof(this_host));
		fprintf(fs, "%s:0 ;\n", this_host);
	}

	if (fclose(fs) != 0)
	{
        perror("fclose");
        exit(1);
	}
	fprintf(stderr, "[FE] Tree topology written into '%s'.\n", topfile);

	/* Return the file name where the topology has been written */
	return topfile;
}

#if 0
unsigned int build_Topology(char *topology, char *hostfile, char **io_topfile)
{   
	int fd, fanout, depth;
	unsigned int num_be=0;
	std::string Topology;
	std::string Hosts;
	MRN::Tree *tree = NULL;
	char *topfile = NULL;
    
	if (sscanf(topology, "%d^%d", &fanout, &depth) != 2)
	{
		fprintf(stderr, "Invalid topology specified. Use F^D.\n");
		exit(1);
	}
	num_be = (unsigned int)pow(fanout, depth+1);

	Topology = topology;
	Hosts = hostfile;
    
	/* Build a balanced topology */
	fprintf(stderr, "[FE] Creating balanced '%s' topology\n", topology);
	tree = new MRN::BalancedTree (Topology, Hosts, 1, 4, 4); 

	/* Write the topology into a file */
	topfile = (char *)malloc((strlen(TOPOLOGY_FILE_TEMPLATE)+1) * sizeof(char));
	strcpy(topfile, TOPOLOGY_FILE_TEMPLATE);
	if ((fd = mkstemp(topfile)) == -1)
	{
		perror("mkstemp");
		exit(1);
	}
	tree->create_TopologyFile(topfile);
	if (close(fd) == -1)
	{
		perror("close");
		exit(1);
	}
	fprintf(stderr, "[FE] Balanced '%s' topology written into '%s'.\n", topology, topfile);

	/* Return the file name where the topology has been written */
	*io_topfile = topfile;
	return num_be;
}
#endif

/**
 * Creates the front-end side of the network with the specified topology.
 * @param[in] num_be The number of back-ends of the network.
 * @param[in] topology_file Path to the file defining the topology.
 * @param[in] be_connect_file Name of the file where the back-ends connection information will be written.
 * @return The network. 
 */
Network * Create_MRNet(unsigned int num_be, char *topology_file, char *be_connect_file)
{
	Network *net = NULL;

	fprintf(stderr, "[FE] Creating MRNet with the topology defined in '%s' (%u back-ends).\n", topology_file, num_be);
	net = new Network(topology_file, NULL, NULL);

#if 1 /* Wait for the back-ends to connect */
	FILE *befile;
	if ((befile = fopen(be_connect_file, (const char *)"w+")) == NULL)
	{
		perror("fopen");
		exit(1);
	}

	std::vector< MRN::NetworkTopology::Node * > leaves;
	MRN::NetworkTopology* nettop = net->get_NetworkTopology();
	nettop->get_Leaves( leaves );
	fprintf(stderr, "MRNet network topology has %u leaves\n", leaves.size());

	unsigned int orig_net_size = nettop->get_NumNodes();
	fprintf(stderr, "MRNet network topology has %u nodes\n", orig_net_size);

	unsigned int num_leaves = leaves.size();
	unsigned int be_per_leaf = num_be / num_leaves;
	unsigned int curr_leaf = 0;
	for (unsigned int i=0; (i < num_be) && (curr_leaf < num_leaves); i++)
	{
		if ( i && ( i % be_per_leaf ==0 ) && (curr_leaf != (num_leaves-1)))
			curr_leaf++;

		fprintf(stderr, "Task %d will connect to %s:%d:%d\n",
			i,
			leaves[curr_leaf]->get_HostName().c_str(),
			leaves[curr_leaf]->get_Port(),
			leaves[curr_leaf]->get_Rank() );

		/* Write into a file where do the backends have to connect */
		fprintf(befile, "%s %d %d\n",
			leaves[curr_leaf]->get_HostName().c_str(),
			leaves[curr_leaf]->get_Port(),
			leaves[curr_leaf]->get_Rank());
	}
	fclose(befile);
	fprintf(stderr, "[FE] Back-ends topology file written into '%s'\n", be_connect_file);

	unsigned int net_size = 0, retry = 0;
	do {
		sleep(1);
		net_size = nettop->get_NumNodes();
		fprintf(stderr, "[FE] Waiting for back-ends to connect... (%d left)\n", (orig_net_size + num_be) - net_size);
		fflush(stderr);
		retry ++;
	} while ((net_size < (orig_net_size + num_be)));
#endif

	fprintf(stderr, "[FE] All back-ends connected!\n");
	fflush(stderr);

	return net;
}

/**
 * Opens a socket connection to the given host and port
 * @param[in] host The host to connect to.
 * @param[in] port The port to connect to.
 * @return The socket fd of the established connection.
 */
int Connect_To_Host (char * host, int port)
{
	int sock_fd;
	struct hostent * server;
	struct sockaddr_in serv_addr;

	sock_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (sock_fd < 0)
	{
		perror("[FE] Connect_To_Host: Error opening socket");
		return -1;
	}

	server = gethostbyname(host);
	if (server == NULL)
	{
		fprintf(stderr, "[FE] Connect_To_Host: No such host '%s'\n", host);
		return -1;
	}

	bzero((char *) &serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
	serv_addr.sin_port = htons(port);

	if (connect(sock_fd, (const  struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
		fprintf(stderr, "[FE] Connect_To_Host: Error connecting to %s:%d\n", host, port);
		return -1;
	}

	return sock_fd;
}


/**
 * Announces the valid streams to the back-ends
 * param[in] The network.
 */
void Create_Streams (Network * net)
{
	Communicator *comm_BC = net->get_BroadcastCommunicator();
	unsigned int comm_BC_size = comm_BC->get_EndPoints().size();

	downStream = net->new_Stream(comm_BC, TFILTER_NULL, SFILTER_DONTWAIT);
	MRN_STREAM_SEND(downStream, MRN_REGISTER_STREAM, "");
	fprintf(stderr, "[FE] Created downStream (ID: %d, SIZE: %d)\n", downStream->get_Id(), downStream->size());

	upStream = net->new_Stream(comm_BC, TFILTER_NULL, SFILTER_DONTWAIT);
	MRN_STREAM_SEND(upStream, MRN_REGISTER_STREAM, "");
	fprintf(stderr, "[FE] Created upStream (ID: %d, SIZE: %d)\n", upStream->get_Id(), upStream->size());

	/* Create peer to peer streams */
	p2pStreams = (Stream **)malloc(comm_BC_size * sizeof(Stream *));
	for (unsigned int i=0; i<comm_BC_size; i++)
	{
		Communicator *comm_i = net->new_Communicator();
		comm_i->add_EndPoint(MRN_RANK(i));
		p2pStreams[i] = net->new_Stream(comm_i, TFILTER_NULL, SFILTER_DONTWAIT);
		MRN_STREAM_SEND(p2pStreams[i], MRN_REGISTER_STREAM, "");
		fprintf(stderr, "[FE] Created p2pStreams[%u] (ID: %d, SIZE: %d)\n", i, p2pStreams[i]->get_Id(), p2pStreams[i]->size());
	}
}	

/**
 * Auxiliary thread listening for commands sent interactively with the monitor.
 */
void * Listen_Monitor (void *)
{
	int monitor_fd;
	char *monitor_host = NULL;
	char *monitor_port_str = NULL;
	int monitor_port;

	/* Read environment variables */
	monitor_host = getenv("MONITOR_HOST");
	monitor_port_str = getenv("MONITOR_PORT");
	if ((monitor_host == NULL) || (monitor_port_str == NULL))
	{
		fprintf(stderr, "[FE-MONITOR] Where's the monitor running? Please define MONITOR_HOST and MONITOR_PORT variables.\n");
		return NULL;
	}
	monitor_port = atoi(monitor_port_str);

	/* Establish connection with the monitor */
	monitor_fd = Connect_To_Host (monitor_host, monitor_port);
	if (monitor_fd < 0)
	{
		fprintf(stderr, "[FE-MONITOR] Can't connect to monitor application. Please make sure it is running at %s:%d.\n",
			monitor_host,
			monitor_port);
		fflush(stderr);
	}
	else
	{
		int rbytes = 0;
		do
		{
#if 0
			int cmd_id;
			/* Read the command sent */
			rbytes = read(monitor_fd, &cmd_id, sizeof(cmd_id));
			/* Enqueue the command */
#if defined(DEAD_CODE)
			CmdQueue_Insert (cmd_id, NULL);
#else
			CmdQueue_Insert (cmd_id);
#endif
#endif
			int MinutesToTraceFromNowOn = 0;
            /* Read the command sent */
            rbytes = read(monitor_fd, &MinutesToTraceFromNowOn, sizeof(MinutesToTraceFromNowOn));
            /* Enqueue the command */
			CmdQueue_Insert (MRN_WHEN_FULL_FLUSH);
			sleep(MinutesToTraceFromNowOn * 60);
			CmdQueue_Insert (MRN_WHEN_FULL_OVERWRITE);
		} while (rbytes != 0);
	}
	return NULL;
}

#if ! defined(NEW_DYNAMIC_STREAMS)
/** 
 * Auxiliary thread listening for any back-end request.
 * @param[in] stream_ptr The stream listening for requests.
 */ 
void * Listen_Backends (void *stream_ptr)
{
	int tag;
	PacketPtr data;
	Stream *stream = (Stream *)stream_ptr;
	int rc;

	fprintf(stderr, "[FE] Listening for back-ends at stream %d\n", stream->get_Id());
	do 
	{
		int task, thread;

		MRN_STREAM_RECV(stream, &tag, data, MRN_ANY);

		MRN_ID(data, &task, &thread);

		/* Execute the front-end side protocol of this request */
		rc = Execute_Root_Protocol (tag, p2pStreams[task]);

	} while (rc != -1);

	return NULL;
}
#else
void * Listen_Backends (void *stream_ptr)
{
    int tag;
    PacketPtr data;
    Stream *stream = (Stream *)stream_ptr;
    int rc;

    fprintf(stderr, "[FE] Listening for back-ends at stream %d\n", stream->get_Id());
    do
    {
        int task, thread;

        MRN_STREAM_RECV(stream, &tag, data, MRN_ANY);
		data->unpack("%d %d", &task, &thread);

        /* Execute the front-end side protocol of this request */
        rc = Execute_Root_Protocol (tag, p2pStreams[task]);

    } while (rc != -1);

    return NULL;
}
#endif

/**
 * Auxiliary thread listening for timed commands.
 */
void * Listen_Timer (void *)
{
	char *timer_reps_str = NULL;
	char *timer_freq_str = NULL;
	int timer_reps = 0;
	int timer_freq = 0;
	int i = 0;

	timer_reps_str = getenv("MPITRACE_MRNET_TIMER_REPS");
	timer_freq_str = getenv("MPITRACE_MRNET_TIMER_FREQ");
	if ((timer_reps_str != NULL) && (timer_freq_str != NULL))
	{
		timer_reps = atoi(timer_reps_str);
		timer_freq = atoi(timer_freq_str);

		fprintf(stderr, "[FE-TIMER] Timer thread started (reps=%d, freq=%d)\n", timer_reps, timer_freq);

		for (i=0; i<timer_reps; i++)
		{
			sleep(timer_freq);
			/* Enqueue the command */
#if defined(DEAD_CODE)
			CmdQueue_Insert (MRN_PING, NULL);
#else
			CmdQueue_Insert (MRN_CLUSTERS);
#if 0
			CmdQueue_Insert (MRN_PING);
			sleep(timer_freq);
			CmdQueue_Insert (MRN_CLUSTERS);
			sleep(timer_freq);
			CmdQueue_Insert (MRN_SPECTRAL);
			sleep(timer_freq);
			CmdQueue_Insert (MRN_LONG_GLOPS);
			sleep(timer_freq);
			CmdQueue_Insert (MRN_FEEDBACK_SPECTRAL);
#endif
#endif
		}
	}
	else
	{
		fprintf(stderr, "[FE-TIMER] Timer thread cannot be started. Please define MPITRACE_MRNET_TIMER_REPS and MPITRACE_MRNET_TIMER_FREQ.\n");
	}
	return NULL;
}

/**
 * Main commands dispatcher. Fetches commands from the queue, that have been inserted by the different threads.
 * @param[in] stream The stream where the commands will be broadcasted through.
 */

/* XXX */
extern unsigned long long last_sync_flush_time;

#if 0
Stream * Announce_Stream (Network *n, Stream *bcast_stream, BEMask *be_mask, int up_transfilter_id, int up_syncfilter_id)
{
	Communicator *new_comm = n->new_Communicator();
	Stream *new_stream = NULL;

	for (int i=0; i<be_mask->size(); i++)
	{
		if (be_mask->Check(i))
		{
			new_comm->add_EndPoint(MRN_RANK(i));
		}	
	}

	/* Create a new stream comprising the selected tasks to process this command */
	new_stream = n->new_Stream(new_comm, up_transfilter_id, up_syncfilter_id);

fprintf(stderr, "[FE] Announcing stream '%d' through '%d'\n", new_stream->get_Id(), bcast_stream->get_Id());

	/* Announce the newly created stream */
	MRN_STREAM_SEND(bcast_stream, MRN_REGISTER_STREAM, "%d %ad", new_stream->get_Id(), be_mask->get_Selection(), be_mask->size());

	return new_stream;
}
#endif

#if ! defined(NEW_DYNAMIC_STREAMS)
void Start_Commands_Dispatcher (Stream *stream)
{
	int cmd_id;
	cmd_t command;

	do 
	{
		/* Fetch the next command from the queue */
		CmdQueue_FetchCmd (&command);
		cmd_id = command.id;

		fprintf(stderr, "[FE] Dispatching command=%d through stream=%d\n", cmd_id, stream->get_Id());

		if ((cmd_id == MRN_SYNC_FLUSH) && (command.time < last_sync_flush_time))
		{
			fprintf(stderr, "[FE] DISCARDING overlapped flush sync request\n");
			continue;
		}

		/* Run the front-end side protocol for this command */
		Execute_Root_Protocol(cmd_id, stream);

#if defined(DEAD_CODE)
		CmdQueue_Free (command);
#endif
	} while (cmd_id != MRN_TERMINATE);

#if defined(DEAD_CODE)
	/* Wait for all workers */
	WaitForWorkers ();
#endif
}
#else
#include "StreamPublisher.h"
void Start_Commands_Dispatcher (Stream *s, MRN::Network *net)
{
	int i, num_be = downStream->size();
	int *be_mask = NULL;
	cmd_t cmd;
//	BEMask *mask = new BEMask(net);

	be_mask = (int *)malloc(num_be * sizeof(int));
//	fprintf(stderr, "[FE] num_be=%d, be_mask_size=%d\n", num_be, num_be * sizeof(int));

#if 1 /* GENERIC FILTER */
    int filter_id;
    char so_file[512];

    snprintf(so_file, sizeof(so_file), "%s/bin/mrn_filters.so", getenv("MPITRACE_HOME"));

    filter_id = net->load_FilterFunc( so_file, "BigFilter" );
    if (filter_id == -1)
    {
        fprintf(stderr, "[FE] Stream::load_FilterFunc(\"%s\", \"BigFilter\") failed.\n", so_file);
        exit(1);
    }
	fprintf(stderr, "[FE] Filters loaded successfully (filter_id=%d).\n", filter_id);
#endif

	sleep (MRNCfg_GetStartAfter());

	do
	{
#if 0
		Communicator *new_comm = net->new_Communicator();
		Stream *new_stream = NULL;

		bzero(be_mask, num_be * sizeof(int));

		/* Fetch the next command from the queue */
		CmdQueue_FetchCmd (&cmd);

		/* Filter which tasks receive the command */
		for (i=0; i<num_be; i++)
		{
			if (1) /* MASK[i] */ 
			{
				be_mask[i] = TRUE;
//				mask->Set(i);
				new_comm->add_EndPoint(MRN_RANK(i));
			}
			else be_mask[i] = FALSE;
		}

		/* Create a new stream comprising the selected tasks to process this command */
		new_stream = net->new_Stream(new_comm, filter_id, SFILTER_WAITFORALL);
		//new_stream = net->new_Stream(new_comm, TFILTER_NULL, SFILTER_DONTWAIT);

		/* Announce the newly created stream */
		MRN_STREAM_SEND(downStream, MRN_REGISTER_STREAM, "%d %ad", new_stream->get_Id(), be_mask, num_be);

//		new_stream = Announce_Stream (net, downStream, mask, filter_id, SFILTER_WAITFORALL);
//		delete mask;
#endif
		/* Fetch the next command from the queue */
		CmdQueue_FetchCmd (&cmd);

		StreamPublisher sp(net, s);
		std::set<int> be_list;
		Stream *new_stream;
		for (i=0; i<num_be; i++)
		{
			if (1) /* MASK */
			{
				be_list.insert(i);
			}
		}
		new_stream = sp.Announce (be_list, filter_id, SFILTER_WAITFORALL);
		

		/* Broadcast the command */
        fprintf(stderr, "[FE] Dispatching command=%d through stream=%d\n", cmd.id, new_stream->get_Id());
		MRN_STREAM_SEND(new_stream, cmd.id, "");

		/* Run the front-end side protocol for this command */
		Execute_Root_Protocol(cmd.id, new_stream);

		delete new_stream;
//		delete new_comm;
	} while (cmd.id != MRN_TERMINATE);
}
#endif

#include "mrn_config.h"
std::map< int, std::vector<int> > ConfigHWCSet;

/* XXX: Move this to mrn_config.c */
void Config_Startup_FE ()
{
	int tag, max_sets, num_hwc, *hwc_ids, set, i;
	PacketPtr data;
	int TargetTraceSize, Analysis, StartAfter;

	/* Receive the HWC sets configuration from task 0 */
	MRN_STREAM_RECV(p2pStreams[0], &tag, data, MRN_CONFIG);
	data->unpack("%d", &max_sets);
	fprintf(stderr, "[FE] Receiving %d HWC sets\n", max_sets);
	for (set=0; set<max_sets; set++)
	{
		MRN_STREAM_RECV(p2pStreams[0], &tag, data, MRN_CONFIG);
		data->unpack("%ad", &hwc_ids, &num_hwc);
		fprintf(stderr, "[FE] HWC Ids of set %d: ", set);
		for (i=0; i<num_hwc; i++)
		{
			fprintf(stderr, "%d ", hwc_ids[i]);
			ConfigHWCSet[set].push_back(hwc_ids[i]);
		}
		fprintf(stderr, "\n");
	}

	/* Receive the target trace size and analysis type */
	MRN_STREAM_RECV(p2pStreams[0], &tag, data, MRN_CONFIG);
	data->unpack("%d %d %d", &TargetTraceSize, &Analysis, &StartAfter);
	MRNCfg_SetTargetTraceSize (TargetTraceSize);
	MRNCfg_SetAnalysisType (Analysis, StartAfter);
	fprintf(stderr, "[FE] Starting %s analysis in %d seconds. Requested trace size: %d Mb.\n", 
		(Analysis == MRN_ANALYSIS_CLUSTER) ? "CLUSTERING" : "SPECTRAL",
		StartAfter,
		TargetTraceSize);

	if (Analysis == MRN_ANALYSIS_CLUSTER) CmdQueue_Insert (MRN_CLUSTERS);
	else if (Analysis == MRN_ANALYSIS_SPECTRAL) CmdQueue_Insert (MRN_SPECTRAL);
}

MRN::Network *globnet = NULL;

int Start_MRNet(int num_be, char *topology, char *hostfile, char *be_connect_file)
{
	MRN::Network *net = NULL;
	char *topology_file = NULL;

	/* Build the topology file */
//	num_be = build_Topology(topology, hostfile, &topology_file);
	topology_file = build_Topology(topology, hostfile);

	/* Start the network */
	net = Create_MRNet (num_be, topology_file, be_connect_file);
globnet = net;

	/* Initialize the streams */
	Create_Streams(net);

	/* Initialize the commands queue */
	CmdQueue_Initialize ();

	Config_Startup_FE();

	/* Create threads listening for the different sources of commands */
//	pthread_create(&ListenMonitor_thread, NULL, Listen_Monitor, NULL);
	pthread_create(&ListenBackends_thread, NULL, Listen_Backends, (void *)upStream);
//	pthread_create(&ListenTimer_thread, NULL, Listen_Timer, NULL);

	/* Run the commands dispatcher */

#if ! defined(NEW_DYNAMIC_STREAMS)
	Start_Commands_Dispatcher(downStream);
#else
	Start_Commands_Dispatcher(downStream, net);
#endif

	net->shutdown_Network();

	fprintf(stderr, "[FE] Dispatcher terminated!\n");
	return 0;
}

int main(int argc, char ** argv)
{
	char *env_mrnet_topology = NULL;
    char *env_mrnet_num_be = NULL;
	int num_be = 0;

	if (argc != 3)
	{
		fprintf(stderr, "Invalid syntax.\nUsage: %s <hostfile> <backends_output_file>", argv[0]);
		exit(1);
	}

	/* Read the MRNET_TOPOLOGY environment variable */
	if ((env_mrnet_topology = getenv("MRNET_TOPOLOGY")) == NULL)
	{
		fprintf(stderr, "[FE] Warning: MRNET_TOPOLOGY topology is not set. Using a root-only topology.\n");
	}

	/* Read the MRNET_NUM_BE environment variable 
	 * XXX: I'd like to query the number of backends once the topology has been built 
	 * and not to define this variable.
	 */
	if ((env_mrnet_num_be = getenv("MRNET_NUM_BE")) == NULL)
	{
		fprintf(stderr, "[FE] Error: MRNET_NUM_BE is not set! Set it to the number of procesess of the application.\n");
		exit(1);
	}
	num_be = atoi(env_mrnet_num_be);
	if (num_be <= 0)
	{
		fprintf(stderr, "[FE] Error: Invalid value '%d' for MRNET_NUM_BE. Set it to the number of procesess of the application.\n", num_be);
		exit(1);
	}

	Start_MRNet(num_be, env_mrnet_topology, argv[1], argv[2]);

	return 0;
}

