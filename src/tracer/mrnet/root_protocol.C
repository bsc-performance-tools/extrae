/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/root_protocol.C,v $
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_SEMAPHORE_H
# include <semaphore.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif

#include <mrnet/MRNet.h>
#include <semaphore.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include "utils.h"
#include "mrnet_commands.h"
#include "protocol.h"
#include "mrnet_root.h"
#include "commands_queue.h"
#include "root_protocol.h"
/* #include "clustering.h" DEAD_CODE */
#include "BurstInfo_FE.h"
#include "ClusterTool.h"
#include "signal_interface.h"
#include "mrn_config.h"


using namespace MRN;

int do_Ping (Stream * stream)
{
	int i;
	int num_be = stream->size();

#if ! defined(NEW_DYNAMIC_STREAMS)
	fprintf(stderr, "[FE] do_Ping: Broadcasting PING\n");
	MRN_STREAM_SEND(stream, MRN_PING, "");
#endif

	for (i=0; i<num_be; i++)
	{
		int tag;
		PacketPtr data;

		fprintf(stderr, "[FE] Receiving PONG from BE %d\n", i);
		MRN_STREAM_RECV(stream, &tag, data, MRN_PONG);
	}
	fprintf(stderr, "[FE] Received PONG from %d back-ends\n", num_be);

	return 0;
}

typedef struct 
{
	int firstID;
	int lastID;
	int NumGlops;
	unsigned long long *Durations;
} GlopsInfo_t;

#define LONG_GLOPS_THRESHOLD 800000

int do_Long_Glops (Stream * stream)
{
	int i, k, num_be = stream->size();
	GlopsInfo_t * GlopsInfo = NULL;
	int commonFirstGlop = 0, commonLastGlop = 0, NumCommonGlops = 0;
	unsigned int * Selected_Glops = NULL;
	int NumSelectedGlops = 0;

#if ! defined(NEW_DYNAMIC_STREAMS)
	fprintf(stderr, "[FE] do_Long_Glops: Broadcasting MRN_LONG_GLOPS\n");
	MRN_STREAM_SEND(stream, MRN_LONG_GLOPS, "");
#endif
	
	GlopsInfo = (GlopsInfo_t *)malloc(num_be * sizeof(GlopsInfo_t));

	for (i=0; i<num_be; i++)
	{
		int tag;
		PacketPtr data;
		int firstGlopID = 0, lastGlopID = 0, NumGlops = 0;
		unsigned long long *Glops_Durations = NULL;

		MRN_STREAM_RECV(stream, &tag, data, MRN_GLOPS_INFO);

        data->unpack("%d %d %ald",
            &firstGlopID, &lastGlopID, &Glops_Durations, &NumGlops);

		fprintf(stderr, "[FE] do_Long_Glops: Received GlOps info (firstID=%d, lastID=%d, NumGlops=%d).\n", 
			firstGlopID, lastGlopID, NumGlops);

		GlopsInfo[i].firstID = firstGlopID;
		GlopsInfo[i].lastID = lastGlopID;
		GlopsInfo[i].NumGlops = NumGlops;
		GlopsInfo[i].Durations = Glops_Durations;

		commonFirstGlop = MAX(commonFirstGlop, firstGlopID);
		commonLastGlop = ((i == 0) ? lastGlopID : MIN(commonLastGlop, lastGlopID));
	}

	/* Build the selection mask */
	NumCommonGlops = (commonLastGlop - commonFirstGlop + 1);
	fprintf(stderr, "[FE] do_Long_Glops: commonFirstGlop=%d, commonLastGlop=%d, NumCommonGlops=%d\n", 
		commonFirstGlop, commonLastGlop, NumCommonGlops);

	if (NumCommonGlops <= 0)
	{
		NumCommonGlops = commonFirstGlop = commonLastGlop = 0;
		Selected_Glops = NULL;
	}
	else
	{
		Selected_Glops = (unsigned int *)malloc(NumCommonGlops * sizeof(unsigned int));

		/* Select the collectives where any of the tasks took longer than a given threshold */
		for (i=0; i<NumCommonGlops; i++)
		{
			unsigned int MaxTaskDuration = 0;
			for (k=0; k<num_be; k++)
			{
				unsigned int taskFirstGlop = GlopsInfo[k].firstID;

				MaxTaskDuration = MAX(MaxTaskDuration,
				                      GlopsInfo[k].Durations[commonFirstGlop - taskFirstGlop + i]);
			}

			if (MaxTaskDuration > LONG_GLOPS_THRESHOLD) /* ns */
			{
				Selected_Glops[i] = TRUE;
				NumSelectedGlops ++;	
			}
			else
			{
				Selected_Glops[i] = FALSE;
			}			
		}
		fprintf(stderr, "[FE] do_Long_Glops: NumSelectedGlops=%d\n", NumSelectedGlops);
	}

	MRN_STREAM_SEND(stream, MRN_GLOPS_SELECTED, "%d %d %aud", commonFirstGlop, commonLastGlop, Selected_Glops, NumCommonGlops);

	for (i=0; i<num_be; i++)
	{
		xfree(GlopsInfo[i].Durations);
	}
	xfree(GlopsInfo);
	xfree(Selected_Glops);

	return 0;
}

int Receive_Bursts_Info (int num_be, Stream *stream, BurstInfo_t ***bi_list_io, int *io_bytes, unsigned long long *io_ns, double *io_mb_min)
{
	int i;
	int tag;
	PacketPtr data;
	long long min_time, max_time;
	int bytes;
	double mb, ns, secs, mins, mb_per_min;
	BurstInfo_t ** bi_list = (BurstInfo_t **)malloc(sizeof(BurstInfo_t *) * num_be);
	int got_data = FALSE;

	fprintf(stderr, "[FE] Receive_Burst_Info: Retrieving BI's from BackEnds\n");
	for (i=0; i<num_be; i++)
	{
		int TaskID, ThreadID, num_Bursts, num_HWCperBurst, tmp;
		long long *Timestamp, *Durations, *HWCValues;
		int *HWCSet;

		MRN_STREAM_RECV(stream, &tag, data, MRN_BURSTS_INFO);

		data->unpack("%d %d %d %d %ald %ald %ald %ad",
			&TaskID, &ThreadID, &num_Bursts, &num_HWCperBurst,
			&Timestamp, &tmp,
			&Durations, &tmp,
			&HWCValues, &tmp,
			&HWCSet, &tmp);	

		bi_list[i] = BurstInfo_Assemble (TaskID, ThreadID, num_Bursts, num_HWCperBurst, Timestamp, Durations, HWCValues, HWCSet);

		fprintf(stderr, "[FE] Receive_Bursts_Info: Data received from Task=%d (countBursts=%d, HWCperBurst=%d)\n",
			TaskID, num_Bursts, num_HWCperBurst);
	}
	fprintf(stderr, "[FE] Receive_Bursts_Info: All data retrieved from %d back-ends\n", num_be);

	MRN_STREAM_SEND(stream, MRN_ACK, "");

	/* fprintf(stderr, "[FE] Receive_Burst_Info: Reducing MB's analyzed\n"); */
	MRN_STREAM_RECV(stream, &tag, data, REDUCE_INT_ADD);
	data->unpack("%d", &bytes);
	mb = bytes / (1024*1024);

/*
	MRN_STREAM_RECV(stream, &tag, data, REDUCE_ULL_MAX);
	data->unpack("%uld", &min_time);
	MRN_STREAM_RECV(stream, &tag, data, REDUCE_ULL_MAX);
	data->unpack("%uld", &max_time);

	fprintf(stderr, "[FE] min_time=%llu max_time=%llu\n", min_time, max_time);
*/
	
#if 0
	min_time = -1;
	max_time = -1;
	for (i=0; i<num_be; i++)
	{
		long long local_min_time, local_max_time;

		MRN_STREAM_RECV(stream, &tag, data, MRN_BURSTS_INFO);
		data->unpack("%ld %ld", &local_min_time, &local_max_time);
		
		if ((local_min_time > 0) && (local_max_time > 0))
		{
			if (min_time == -1) min_time = local_min_time;
			if (max_time == -1) max_time = local_max_time;
			/* MIN(mins) ... MAX(maxs) 
			if (min_time != -1) min_time = MIN(min_time, local_min_time);
			if (max_time != -1) max_time = MAX(max_time, local_max_time);
			*/

			/* MAX(mins) ... MIN(maxs) */
            if (min_time != -1) min_time = MAX(min_time, local_min_time);
			if (max_time != -1) max_time = MIN(max_time, local_max_time);
		}
	}
#endif

	MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MIN_POSITIVE);
	data->unpack("%ld", &min_time);
	MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
	data->unpack("%ld", &max_time);

	if ((min_time < 0) || (max_time < 0) || (min_time >= max_time))
	{
		fprintf(stderr, "[FE] Error: There is no data to analyze\n");
		mb_per_min = 0;
		bytes = 0;
		ns = 0;
		got_data = FALSE;
	}
	else 
	{
		ns = max_time - min_time;
		secs = ns / 1000000000;
		mins = secs / 60;
		mb_per_min = mb / mins;
		got_data = TRUE;
		fprintf(stderr, "[FE] Analyzing %.2f Mb of data (%.2f seconds of trace, Avg %.2f Mb/min) TimeInterval=(%lld, %lld).\n", mb, secs, mb_per_min, min_time, max_time);
	}

    *io_bytes = bytes;
    *io_ns = ns;
	*bi_list_io = bi_list;
	*io_mb_min = mb_per_min;
	return got_data;
}

#if 0
#include <sys/time.h>
#define timerclear(tvp) ((tvp)->tv_sec = (tvp)->tv_usec = 0)
#define diff_time(entry_time, exit_time) (long)(exit_time.tv_sec - entry_time.tv_sec)
#endif
extern std::map< int, std::vector<int> > ConfigHWCSet;
#include <iostream>
#include <sstream>
using namespace std;
extern Network *globnet;
#include "StreamPublisher.h"

#if 0
#define CLUSTERING_BASENAME "CLUSTERING_STEP_"
#define MAX_BASENAME_LENGTH 256
#endif

int CurrentClusteringStep = 0;

#if 0
char * Get_Clustering_BaseName(char *prefix, int step, char *suffix)
{
	char *basename = NULL;

	basename = (char *)malloc(MAX_BASENAME_LENGTH);
	snprintf(basename, MAX_BASENAME_LENGTH, "%s%d%s", prefix, step, suffix);

	return basename;
}
#endif

void Clusters_SendConfig (ClusterTool *CT, Stream *stream)
{
    /* Broadcast the duration filter */
    MRN_STREAM_SEND(stream, MRN_CLUSTERS, "%uld", CT->getClustering()->GetDurationFilter());
}

int Clusters_FetchInput (Stream *stream, const char *dumpFile, BurstInfo_t ***bi_list_io, double *mb_per_min_io, int *bytes_io, unsigned long long *ns_io)
{
	int num_be = stream->size();
    BurstInfo_t **bi_list = NULL;
	double mb_per_min;
	int bytes;
	unsigned long long ns;
	int got_data;

    /* Read data from back-ends */
    got_data = Receive_Bursts_Info (num_be, stream, &bi_list, &bytes, &ns, &mb_per_min);

   	/* Store input in disk */
    if ((DUMP_CLUSTERS_DATA) && (got_data))
	{
        BurstInfo_DumpArray (bi_list, num_be, dumpFile, ConfigHWCSet);
		//free (dumpFile);
	}
	*bi_list_io = bi_list;
	*bytes_io = bytes;
	*ns_io = ns;
	*mb_per_min_io = mb_per_min;
	return got_data;
}

#if 0
void Convert (BurstInfo_t *bi, int nb, INT32 *task_id, INT32 *thread_id, UINT64 *timestamp, UINT64 *duration, vector<INT64> *hwc_values)
{
	*task_id = bi->TaskID;
	*thread_id = bi->ThreadID;
	*timestamp = bi->Timestamp[nb];
	*duration = bi->Durations[nb];
	for (int i=0; i<bi->num_HWCperBurst; i++)
	{
		(*hwc_values).push_back(bi->HWCValues[(nb*bi->num_HWCperBurst)+i]);
	}
}

//#define BOTH_REPRESENTATIVES_AND_RANDOM
#define CLUSTERING_FED_RANDOMLY
#define MAX_REPRESENTATIVES 8
#define PCT_BURSTS 15
#define MAX_EASILY_HANDLES 8000


void Clusters_FeedWithData (MRNetClustering *C, int num_be, BurstInfo_t **bi_list)
{
    int i, j, k;
	int representatives = 0;
	int abs_total_bursts = 0;

	for (i=0; i<num_be; i++)
	{
		abs_total_bursts += bi_list[i]->num_Bursts;
	}
	int EASILY_HANDLES = MIN(MAX_EASILY_HANDLES, abs_total_bursts);

#if defined(BOTH_REPRESENTATIVES_AND_RANDOM) || !defined(CLUSTERING_FED_RANDOMLY)
	/* Select a few tasks to clusterize */

	representatives = MIN(num_be, MAX_REPRESENTATIVES);
	fprintf(stderr, "[FE] Selecting as representatives: ");
    for (i=0; i<representatives; i++)
    {
        BurstInfo_t *bi = bi_list[i];
        for (j=0; j<bi->num_Bursts; j++)
        {
			INT32  TaskId, ThreadId;
			UINT64 BeginTime, Duration;
			vector<INT64> HWCValues;

			Convert (bi, j, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);
            C->SetHWCountersGroup(ConfigHWCSet[bi->HWCSet[j]]);
            C->NewPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);
        }
		fprintf(stderr, "%d ", bi->TaskID);
    }
	fprintf(stderr, "\n");
#endif

#if defined(BOTH_REPRESENTATIVES_AND_RANDOM) || defined(CLUSTERING_FED_RANDOMLY)
	/* Select random bursts from all tasks to clusterize */

	for (i=representatives; i<num_be; i++)
	{
		BurstInfo_t *bi = bi_list[i];
	    int total_bursts = bi->num_Bursts;
	    int num_samples = (total_bursts * PCT_BURSTS) / 100;

		double task_contribution_pct = (double)(total_bursts * 100) / (double)abs_total_bursts;
		num_samples = (EASILY_HANDLES * task_contribution_pct) / 100;

	    double range = total_bursts / num_samples;

		fprintf(stderr, "[FE] Task %d: Selecting %d samples out of %d bursts (range=%d, abs_total_bursts=%d, task_contribution_pct=%f)\n",
			bi->TaskID,
			num_samples,
			total_bursts,
			range, abs_total_bursts, task_contribution_pct);

	    srand(getpid());

	    for (j=0; j<num_samples; j++)
	    {
	        int sample;

	        sample = (int) (range*rand()/(RAND_MAX+1.0));
	        sample += j * range;

	        if ((sample >= 0) || (sample < total_bursts))
	        {
	            INT32  TaskId, ThreadId;
    	        UINT64 BeginTime, Duration;
        	    vector<INT64> HWCValues;

				Convert (bi, sample, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);
				C->SetHWCountersGroup(ConfigHWCSet[bi->HWCSet[sample]]);
				C->NewPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);
	        }
	        else
	        {
	            fprintf(stderr, "BAD_SAMPLE!!\n");
	        }
	    }
	}
#endif
}

void Clusters_ExecuteAnalysis (MRNetClustering *C)
{
    struct timeval entry_time; 
    struct timeval exit_time;
	bool ok = false;
    char cmd[2048];

    /* Invoke the clustering tool */
    fprintf(stderr, "[FE] do_Clusters: Invoking the clustering tool...\n");
    fflush(stderr);
    timerclear(&entry_time);
    timerclear(&exit_time);
    gettimeofday(&entry_time, 0);

	char *basename = Get_Clustering_BaseName(CLUSTERING_BASENAME, CurrentClusteringStep, "");
    ok = C->ExecuteClustering (true, basename);
    C->PrintGNUPlot (basename);

    gettimeofday(&exit_time, 0);
    fprintf(stderr, "[FE] do_Clusters: Clustering tool returns %s after %ld seconds.\n", (ok ? "successfully" : "with errors"), diff_time(entry_time, exit_time));

    /* Compute CPI Stack statistics */
    snprintf(cmd, sizeof(cmd), "%s/bin/cpistack_stats.pl %s.clusters_info.csv", getenv("MPITRACE_HOME"), basename);
    fprintf(stderr, "[FE] Computing CPI Stack statistics: %s\n", cmd);
    system (cmd);
	free(basename);
}

#include "extern.h" /* PARAVER_OFFSET ... GetClusterId & ClassifyPoint -> wrappers to encapsulate this offset */

typedef std::map< std::pair< int, int >, std::vector< int > > ClusterIDs_t;

void Classify_All_Bursts (MRNetClustering *C, int num_be, BurstInfo_t **bi_list, ClusterIDs_t & cids)
{
    struct timeval entry_time;
    struct timeval exit_time;

    timerclear(&entry_time);
    timerclear(&exit_time);

    gettimeofday(&entry_time, 0);

    /* Classify all points */
    for (int i=0; i<num_be; i++)
    {
        BurstInfo_t *bi = bi_list[i];

        std::pair< int, int > key = std::make_pair(bi->TaskID, bi->ThreadID);
        for (int j=0; j<bi->num_Bursts; j++)
        {
            INT32  TaskId, ThreadId;
            UINT64 BeginTime, Duration;
            vector<INT64> HWCValues;

            Convert (bi, j, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);

            C->SetHWCountersGroup(ConfigHWCSet[bi->HWCSet[j]]);
            cids[key].push_back( C->ClassifyPoint( TaskId, ThreadId, BeginTime, Duration, HWCValues ) + PARAVER_OFFSET );
        }
    }
    gettimeofday(&exit_time, 0);
    fprintf(stderr, "[FE] do_Clusters: Classification lasted %ld seconds.\n", diff_time(entry_time, exit_time));

    char *basename = Get_Clustering_BaseName(CLUSTERING_BASENAME, CurrentClusteringStep, ".classify");
    C->PrintGNUPlot (basename);
    free(basename);
}
#endif

void Clusters_TransferCIDS (Stream *stream, ClusterIDs_m & CIDs)
{
    StreamPublisher sp(globnet, stream);
    std::vector<Stream *> *stream_list = NULL;
    std::vector<Stream *>::iterator it;

	fprintf(stderr, "[FE] Transferring Cluster ID's\n");

    stream_list = sp.AnnounceP2P(stream);
    for ( it = stream_list->begin(); it != stream_list->end(); it ++ )
    {
        Stream *p2p = *it;
        std::set<Rank> ep = p2p->get_EndPoints();
        int r = BE_RANK(*(ep.begin()));

        std::pair< int, int > key = std::make_pair(r, 0);
        fprintf(stderr, "[FE] Sending %d CIDS to %d\n", CIDs[key].size(), r);
        MRN_STREAM_SEND(p2p, MRN_CLUSTERS, "%ad", &CIDs[key][0], CIDs[key].size());
    }
    delete stream_list;
}

#if 0
void OLD_Clusters_TransferCIDS (MRNetClustering *C, Stream *stream, int num_be, BurstInfo_t **bi_list)
{
    MRNetClusteringResults *CR = NULL;
    std::map< std::pair< int, int >, std::vector< int > > cids;

	fprintf(stderr, "[FE] Clusters_TransferCIDS\n");

	/* XXX Both Clustering & Classification results should be gathered right after the analysis has finished and just transfer here */
#if !defined(BOTH_REPRESENTATIVES_AND_RANDOM) && !defined(CLUSTERING_FED_RANDOMLY)
	/* Get the results of the clustering */
    C->InitResultsWalk();
    while ((CR = C->GetNextResults()) != NULL)
    {
        std::pair< int, int > key = std::make_pair(CR->GetTaskId(), CR->GetThreadId());

        cids[key].push_back(CR->GetClusterId()+PARAVER_OFFSET);

        //fprintf(stderr, "[CIDS] %d %d %llu %llu %d\n", CR->GetTaskId(), CR->GetThreadId(), CR->GetBeginTime(), CR->GetDuration(), CR->GetClusterId());
    }
#endif

#if 1
    struct timeval entry_time;
    struct timeval exit_time;

    timerclear(&entry_time);
    timerclear(&exit_time);

    gettimeofday(&entry_time, 0);
#if !defined(BOTH_REPRESENTATIVES_AND_RANDOM) && !defined(CLUSTERING_FED_RANDOMLY)
	/* Classify only those that were not clusterized */
	for (int i=MIN(num_be, MAX_REPRESENTATIVES); i<num_be; i++)
#else
	/* Classify all points */
	for (int i=0; i<num_be; i++)
#endif
	{
		BurstInfo_t *bi = bi_list[i];

		std::pair< int, int > key = std::make_pair(bi->TaskID, bi->ThreadID);
		for (int j=0; j<bi->num_Bursts; j++)
		{
            INT32  TaskId, ThreadId;
            UINT64 BeginTime, Duration;
            vector<INT64> HWCValues;

            Convert (bi, j, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);

			C->SetHWCountersGroup(ConfigHWCSet[bi->HWCSet[j]]);
			cids[key].push_back( C->ClassifyPoint( TaskId, ThreadId, BeginTime, Duration, HWCValues ) + PARAVER_OFFSET );
		}
	}
    gettimeofday(&exit_time, 0);
    fprintf(stderr, "[FE] do_Clusters: Classification lasted %ld seconds.\n", diff_time(entry_time, exit_time));

	char *basename = Get_Clustering_BaseName(CLUSTERING_BASENAME, CurrentClusteringStep, ".classify");
	C->PrintGNUPlot (basename);
	free(basename);
#endif

    StreamPublisher sp(globnet, stream);
    std::vector<Stream *> *stream_list = NULL;
    std::vector<Stream *>::iterator it;

    stream_list = sp.AnnounceP2P(stream);
    for ( it = stream_list->begin(); it != stream_list->end(); it ++ )
    {
        Stream *p2p = *it;
        std::set<Rank> ep = p2p->get_EndPoints();
        int r = BE_RANK(*(ep.begin()));

        std::pair< int, int > key = std::make_pair(r, 0);
        fprintf(stderr, "[FE] Sending %d CIDS to %d\n", cids[key].size(), r);
        MRN_STREAM_SEND(p2p, MRN_CLUSTERS, "%ad", &cids[key][0], cids[key].size());
    }
    delete stream_list;
	delete CR;
}
#endif

ClusterTool * SingleClusterAnalysis (Stream *stream, double *mb_per_min_io, int *bytes_io, unsigned long long *ns_io, int *num_be_io, BurstInfo_t ***bi_list_io, ClusterIDs_m & CIDs_io)
{
	ClusterTool * CT;
	BurstInfo_t **bi_list = NULL;
    int num_be = stream->size();
	double mb_per_min;
	int bytes;
	unsigned long long ns;
	int got_data;
   
	fprintf(stderr, "[FE] Sending ACK for SingleClusterAnalysis\n");
	MRN_STREAM_SEND(stream, MRN_ACK, "");

    CurrentClusteringStep ++;

	std::ostringstream stm;
	stm << CurrentClusteringStep;

	string basename(CLUSTERING_BASENAME);
	string step(stm.str());
	string outPrefix = basename + step;

	CT = new ClusterTool ( outPrefix, &ConfigHWCSet);

    Clusters_SendConfig(CT, stream);

    got_data = Clusters_FetchInput(stream, outPrefix.c_str(), &bi_list, &mb_per_min, &bytes, &ns);
	if (!got_data)
	{
		/* The app did not produce data (probably ended) */
		delete CT;
		CT = NULL;
	}
	else
	{
		CT->feed(num_be, bi_list);
		CT->cluster();
		CT->print_plots("");
		CT->classify(num_be, bi_list, CIDs_io);
		CT->print_plots(".classify");
	}
	*mb_per_min_io = mb_per_min;
    *bytes_io = bytes;
    *ns_io = ns;
	*bi_list_io = bi_list;
	*num_be_io = num_be;
	return CT;
}

#define MIN_HITS_IN_A_ROW       3
#define MIN_PCT_EQUAL           85

bool ClusterStable(ClusterTool *c1, ClusterTool *c2)
{
	char cmd[1024];
	int rc;
	static int hits_in_row = 0, min_hits_in_row = MIN_HITS_IN_A_ROW;
	static int accepted_pct_equal = MIN_PCT_EQUAL;
	static int best_so_far = 0;
	static int tries_so_far = 0;
	int min_tries_before_lower;
	
	string curBaseName, prevBaseName;

    if ((c1 == NULL) || (c2 == NULL)) return false;

    /* Compute CPI Stack statistics */
	curBaseName = c2->getBasename();
	prevBaseName = c1->getBasename();
    snprintf(cmd, sizeof(cmd), "%s/bin/compare_clusters.pl %s %s", getenv("MPITRACE_HOME"), curBaseName.c_str(), prevBaseName.c_str());

    rc = system (cmd);
	rc = rc >> 8; /* Perl exit code */
    fprintf(stderr, "[FE] Comparing clusters ... PctEqual=%d%\n", rc);

/*
	hits = ((rc == 1) ? hits-1 : MIN_HITS_IN_A_ROW);
	return (hits == 0);
*/
	min_tries_before_lower = min_hits_in_row * 2;

	hits_in_row = ((rc > accepted_pct_equal) ? hits_in_row+1 : 0);
	tries_so_far ++;
	best_so_far = MAX(best_so_far, rc);

	fprintf(stderr, "[FE] STABILITY %=%d hits_in_row=%d min_hits_in_row=%d best_so_far=%d tries_so_far=%d tries_before_lower=%d accepted_pct=%d\n", 
		rc, hits_in_row, min_hits_in_row, best_so_far, tries_so_far, min_tries_before_lower, accepted_pct_equal);

	if ((tries_so_far + (min_hits_in_row - hits_in_row)) > min_tries_before_lower)
	{
		accepted_pct_equal = MIN(accepted_pct_equal, best_so_far) - 5;
		min_hits_in_row = MAX(min_hits_in_row-1, 1);
		tries_so_far = 0;
		best_so_far = 0;
	}
	return (hits_in_row == min_hits_in_row);
}

//#define SIZE_RATIO_MPIT_PRV 0.6
#define SIZE_RATIO_MPIT_PRV 1

bool Appl_Ended = false;

ClusterTool * MultiClusterAnalysis(Stream *stream, int *num_be_io, BurstInfo_t ***bi_list_io, ClusterIDs_m & CIDs_io)
{
	int freq = -1;
	ClusterTool *prevCT = NULL, *currCT = NULL; 
	int stability = FALSE;
	char *mrn_trace_max_size;
	double mb_per_min, b_per_ns, max_size;
	int num_be;
	BurstInfo_t **bi_list;
	int bytes;
 	unsigned long long ns, change_hwc_freq;
	long long freq_ns;

/*
	if ((mrn_trace_max_size = getenv ("MPITRACE_MRNET_TRACE_MAX_SIZE")) == NULL)
	{
		fprintf(stderr, "MPITRACE_MRNET_TRACE_MAX_SIZE not defined! Exiting...\n");
		exit(1);
	}
	max_size = (double)atol(mrn_trace_max_size);	
*/
	max_size = MRNCfg_GetTargetTraceSize();

	while (! stability)
	{
		if (prevCT != NULL) delete prevCT;
		prevCT = currCT;
		currCT = SingleClusterAnalysis(stream, &mb_per_min, &bytes, &ns, &num_be, &bi_list, CIDs_io);
		if (currCT == NULL)
		{
			/* The app did not produce data (probably ended) */
			//return NULL;
			freq_ns = (long long)MRNCfg_GetStartAfter() * (long long)1000000000;
			fprintf(stderr, "[FE] Waiting another %lld ns for data\n", freq_ns);
			change_hwc_freq = freq_ns;
			stability = FALSE;
		}
		else
		{
			stability = ClusterStable(currCT, prevCT);

			/* delete if to recalc freq after every clustering -- beware! at the very end of the app, very few data in very short time might result in a huge freq! 
			if (freq == -1) { */ 
				freq = ((max_size * SIZE_RATIO_MPIT_PRV) / mb_per_min) * 60;

				b_per_ns = (double)bytes / (double)ns;

				freq_ns = ((max_size * 1024 * 1024 * SIZE_RATIO_MPIT_PRV) / b_per_ns);
				freq_ns = MIN(freq_ns, (long long)150 * (long long)1000000000);
				change_hwc_freq = (unsigned long long)((unsigned long long)(ns) / (unsigned long long)(10 * 2)); /* freq / num_hwc_sets * num_rotations_of_hwc */

				/* fprintf(stderr, "[FE] max_size=%f mb_per_min=%f b_per_ns=%f freq=%d ns=%llu bytes=%d freq_ns=%llu change_hwc_freq=%llu\n", 
					max_size, mb_per_min, b_per_ns, freq, ns, bytes, freq_ns, change_hwc_freq); */
				fprintf(stderr, "[FE] Next analysis step will be computed in %llu ns (%d seconds).\n", freq_ns, freq);
				fprintf(stderr, "[FE] HWC change frequency set at %llu ns.\n", change_hwc_freq);

	//		}
		}
		fprintf(stderr, "[FE] Broadcasting STABILITY status: %d\n", stability);
		MRN_STREAM_SEND(stream, MRN_CLUSTERS, "%d %uld", stability, change_hwc_freq);

		if (!stability) 
		{
			BurstInfo_FreeArray(bi_list, num_be);

			CIDs_io.clear();

			while (freq_ns > 0)
			{
				usleep(100000);
				/* If app has ended */
				if (Appl_Ended) {
					MRN_STREAM_SEND(stream, MRN_NACK, "");
					*bi_list_io = NULL;
					*num_be_io = num_be;
					return NULL;
				}
				freq_ns -=  (long long) (100000 * 1000);
			}
		}
	}
	if (prevCT != NULL) delete prevCT;
	*num_be_io = num_be;
	*bi_list_io = bi_list;
	return currCT;
}

int do_Clusters (Stream *stream)
{
	int num_be;
	BurstInfo_t **bi_list;
	int bytes;
	unsigned long long ns;
	ClusterIDs_m CIDs;

//    ClusterTool *CT = SingleClusterAnalysis(stream, &mb_per_min, &bytes, &ns, &num_be, &bi_list, CIDs);
    ClusterTool *CT = MultiClusterAnalysis(stream, &num_be, &bi_list, CIDs);

	if (CT != NULL)
	{
		char cmd[2048];

		Clusters_TransferCIDS(stream, CIDs);
		BurstInfo_FreeArray(bi_list, num_be);
		delete CT;

		/* Generate sequence of plots */
		snprintf(cmd, sizeof(cmd), "%s/bin/animate.pl CLUSTERING_STEP", getenv("MPITRACE_HOME"));
		fprintf(stderr, "[FE] Generating sequence of plots: %s\n", cmd);
		system (cmd);
	}

	return 0;
}

#if 0
void Feed_ClusterInput (MRNetClustering *C, BurstInfo_t **bi_list, int count)
{
    int i, j, k;

    for (i=0; i<count; i++)
    {
        BurstInfo_t *bi = bi_list[i];
        for (j=0; j<bi->num_Bursts; j++)
        {
            vector<INT64> hwc_values;
            for (k=0; k<bi->num_HWCperBurst; k++)
            {
                hwc_values.push_back(bi->HWCValues[(j*bi->num_HWCperBurst)+k]);
            }
            C->SetHWCountersGroup(ConfigHWCSet[bi->HWCSet[j]]);
            C->NewPoint(bi->TaskID, bi->ThreadID, bi->Timestamp[j], bi->Durations[j], hwc_values);
        }
    }
}

int do_Clusters (Stream *stream)
{
	bool ok;
	int num_be = stream->size();
	BurstInfo_t **bi_list = NULL;
	static int CurrentClusteringStep = 0;
	char OutputFileName[256];
	ClusteringBurstsInfo_t *ClusterInput = NULL;
	int numClusterSets = 0;
	char cmd[2048];
	int i;
	MRNetClustering *C;

	struct timeval entry_time; 
	struct timeval exit_time;

	unsigned long long min_burst_length = 0;

	C = new MRNetClustering();
	C->InitClustering("./cl.I.IPC.xml", true);

	/* Broadcast the duration filter */
	min_burst_length = C->GetDurationFilter();
	MRN_STREAM_SEND(stream, MRN_CLUSTERS, "%uld", min_burst_length);

#if ! defined(NEW_DYNAMIC_STREAMS)
    /* Broadcast request to back-ends */
    fprintf(stderr, "[FE] do_Clusters: Broadcasting request to %d back-ends...\n", num_be);
    MRN_STREAM_SEND(stream, MRN_CLUSTERS, "");
#endif

    /* Read data from back-ends */
    bi_list = Receive_Bursts_Info (num_be, stream);

	/* Store input in disk */
    CurrentClusteringStep ++;
    snprintf(OutputFileName, sizeof(OutputFileName), "%s%d", "CLUSTER", CurrentClusteringStep);
    if (DUMP_CLUSTERS_DATA)
    {
		BurstInfo_DumpArray (bi_list, num_be, OutputFileName);
	}

	/* Feed cluster library */
	Feed_ClusterInput (	C, bi_list, num_be );

    /* Invoke the clustering tool */
    fprintf(stderr, "[FE] do_Clusters: Invoking the clustering tool...\n");
    fflush(stderr);
    timerclear(&entry_time);
    timerclear(&exit_time);
    gettimeofday(&entry_time, 0);
    ok = C->ExecuteClustering (OutputFileName, true);
    gettimeofday(&exit_time, 0);
    fprintf(stderr, "[FE] do_Clusters: Clustering tool returns %s after %ld seconds.\n", (ok ? "successfully" : "with errors"), diff_time(entry_time, exit_time));

    /* Compute CPI Stack statistics */
    snprintf(cmd, sizeof(cmd), "%s/bin/cpistack_stats.pl %s.clusters_info.csv", getenv("MPITRACE_HOME"), OutputFileName);
    fprintf(stderr, "[FE] Computing CPI Stack statistics: %s\n", cmd);
    system (cmd);

	/* Free memory */
	for (i=0; i<num_be; i++) free_BurstInfo (bi_list[i]);

	C->InitResultsWalk();
	MRNetClusteringResults *CR = NULL;

	std::map< std::pair< int, int >, std::vector< int > > cids;


	while ((CR = C->GetNextResults()) != NULL)
	{
		std::pair< int, int > key = std::make_pair(CR->GetTaskId(), CR->GetThreadId());

		cids[key].push_back(CR->GetClusterId());

		//fprintf(stderr, "[CIDS] %d %d %llu %llu %d\n", CR->GetTaskId(), CR->GetThreadId(), CR->GetBeginTime(), CR->GetDuration(), CR->GetClusterId());
	}

	StreamPublisher sp(globnet, stream);
	std::vector<Stream *> *stream_list = NULL;
	std::vector<Stream *>::iterator it;

	stream_list = sp.AnnounceP2P(stream);
	for ( it = stream_list->begin(); it != stream_list->end(); it ++ )
	{
		Stream *p2p = *it;
		std::set<Rank> ep = p2p->get_EndPoints();
		int r = BE_RANK(*(ep.begin()));

        std::pair< int, int > key = std::make_pair(r, 0);
		fprintf(stderr, "[FE] Sending %d CIDS to %d\n", cids[key].size(), r);
        MRN_STREAM_SEND(p2p, MRN_CLUSTERS, "%ad", &cids[key][0], cids[key].size());
	}
	delete stream_list;

	delete C;
	delete CR;

	return 0;
}
#endif

/*
int do_Cluster_META_Analysis (Stream *stream)
{
	ClusterAnalysis *ca = new ClusterAnalysis(), *last_ca;

	ca->Fetch(stream);
	ca->Volume();
	compute freq
	ca->Execute();
	
	last_ca = ca;
	while (!stable)
	{
		sleep(freq);

		ca = new ClusterAnalysis();
		ca->Fetch(stream);
		ca->Volume();
		adjust freq
		ca->Execute();
		
		stable = ca.Compare(last_ca);

		delete last_ca;
		last_ca = ca;
	}
	
	last_ca.Send_Feedback (stream);

	delete last_ca;
	return 0;
}
*/

#include "SpectralTool.h"

int do_Spectral (Stream *stream)
{
	int num_be = stream->size();
	BurstInfo_t **bi_list = NULL;
	static int NumSpectralExecuted = 0;
	char OutputFileName[256];
	int num_periods;
	double mb_per_min;
	int bytes;
	unsigned long long ns;
	int got_data;

#if ! defined(NEW_DYNAMIC_STREAMS)
	/* Broadcast request to back-ends */
	fprintf(stderr, "[FE] do_Spectral: Broadcasting request to %d back-ends...\n", num_be);
	MRN_STREAM_SEND(stream, MRN_SPECTRAL, "");
#endif

	/* Read data from back-ends */
	got_data = Receive_Bursts_Info (num_be, stream, &bi_list, &bytes, &ns, &mb_per_min);

	/* Invoke the spectral analysis tool */
	NumSpectralExecuted ++;
	snprintf(OutputFileName, sizeof(OutputFileName), "%s%d", "SPECTRAL", NumSpectralExecuted);

	if (DUMP_SPECTRAL_DATA)
	{
		BurstInfo_DumpArray (bi_list, num_be, OutputFileName, ConfigHWCSet);
	}

	fprintf(stderr, "[FE] do_Spectral: Invoking the spectral analysis tool...\n");
	SpectralTool * st = new SpectralTool (bi_list, num_be, OutputFileName);

	st->execute();

	BurstInfo_FreeArray (bi_list, num_be);

	int numPeriods = st->get_NumPeriods();
	MRN_STREAM_SEND(stream, MRN_SPECTRAL, "%d", numPeriods);
	for (int i=0; i<numPeriods; i++)
	{
		MRN_STREAM_SEND(stream, MRN_SPECTRAL, "%f %ld %lf %lf %lf %ld %ld %ld %ld", 
			st->get_Period(i)->iters,
			st->get_Period(i)->length,
			st->get_Period(i)->goodness,
			st->get_Period(i)->goodness2,
			st->get_Period(i)->goodness3,
			st->get_Period(i)->ini,
			st->get_Period(i)->end,
			st->get_Period(i)->best_ini,
			st->get_Period(i)->best_end);
	}

    for (int i=0; i<num_be; i++)
    {
        int tag;
        PacketPtr data;

        MRN_STREAM_RECV(stream, &tag, data, MRN_ACK);
    }
    fprintf(stderr, "[FE] do_Spectral: ACK's received! -- numPeriods=%d\n", numPeriods);

	return 0;
}

#if 0
int do_Spectral_Feedback (Stream *stream)
{
    int i, num_be = stream->size();
    BurstInfo_t **bi_list = NULL;
    static int NumSpectralExecuted = 0;
    char OutputFileName[256];
    int num_periods;
    Period_t *Periods;
	SpectralInput_t *SpectralInput;
	double mb_per_min;
	int bytes;
	unsigned long long ns;

#if ! defined(NEW_DYNAMIC_STREAMS)
    /* Broadcast request to back-ends */
    fprintf(stderr, "[FE] do_Spectral_Feedback: Broadcasting request to %d back-ends...\n", num_be);
    MRN_STREAM_SEND(stream, MRN_FEEDBACK_SPECTRAL, "");
#endif

    /* Read data from back-ends */
    mb_per_min = Receive_Bursts_Info (num_be, stream, &bi_list, &bytes, &ns);

#if 0
    /* Invoke the spectral analysis tool */
    NumSpectralExecuted ++;
    snprintf(OutputFileName, sizeof(OutputFileName), "%s%d", "SPECTRAL", NumSpectralExecuted);

    if (DUMP_SPECTRAL_DATA)
    {
        BI_Dump (bi, OutputFileName);
    }

    fprintf(stderr, "[FE] do_Spectral_Feedback: Invoking the spectral analysis tool...\n");
	BI_2_SpectralInput (bi, &SpectralInput);
    num_periods = ExecuteSpectralAnalysis (SpectralInput, BI_NumSets(bi), OutputFileName, &Periods);
    fprintf(stderr, "[FE] do_Spectral_Feedback: Spectral analysis tool returns (%d periods found).\n", num_periods);

    fprintf(stderr, "[FE] do_Spectral_Feedback: Sending periods information to all back-ends...\n");
    MRN_STREAM_SEND(stream, MRN_NUM_PERIODS, "%d", num_periods);
    for (i=0; i<num_periods; i++)
    {
        MRN_STREAM_SEND(stream, MRN_PERIOD_INFO, "%f %ld %lf %lf %lf %ld %ld", 
			Periods[i].iters,
			Periods[i].main_period_duration,
			Periods[i].goodness,
			Periods[i].goodness2,
			Periods[i].goodness3,
			Periods[i].ini,
			Periods[i].end);
    }

	/* Wait for ACK's */
	for (i=0; i<num_be; i++)
	{
		int tag;
		PacketPtr data;

        MRN_STREAM_RECV(stream, &tag, data, MRN_ACK);
	}
	fprintf(stderr, "[FE] do_Spectral_Feedback: ACK's received!\n");

    /* Free memory */
	SI_Free(SpectralInput);
    BI_Free(bi);
#endif
    return 0;
}
#endif

int do_When_Full_Flush (Stream * stream)
{
#if ! defined(NEW_DYNAMIC_STREAMS)
	fprintf(stderr, "[FE] do_When_Full_Flush\n");
	MRN_STREAM_SEND(stream, MRN_WHEN_FULL_FLUSH, "");
#endif
	return 0;
}

int do_When_Full_Overwrite (Stream * stream)
{
#if ! defined(NEW_DYNAMIC_STREAMS)
	fprintf(stderr, "[FE] do_When_Full_Overwrite\n");
	MRN_STREAM_SEND(stream, MRN_WHEN_FULL_OVERWRITE, "");
#endif
	return 0;
}

int do_Sync_Flush_BE_Request (Stream * stream)
{
	CmdQueue_Insert(MRN_SYNC_FLUSH);
	return 0;
}

unsigned long long TIME()
{
	struct timeval t;
	unsigned long long usec = 0;
	gettimeofday(&t, 0);
	usec = (t.tv_sec * 1000000) + (t.tv_usec);
	return usec;
}

unsigned long long last_sync_flush_time = 0;

int do_Sync_Flush (Stream * stream)
{
    int i, num_be = stream->size();

#if ! defined(NEW_DYNAMIC_STREAMS)
	fprintf(stderr, "[FE] do_Sync_Flush\n");
	MRN_STREAM_SEND(stream, MRN_SYNC_FLUSH, "");
#endif

	/* Wait for ACK's */
	for (i=0; i<num_be; i++)
	{
		int tag;
		PacketPtr data;

		MRN_STREAM_RECV(stream, &tag, data, MRN_ACK);
	}

	last_sync_flush_time = TIME();

	return 0;
}

int do_Quit (Stream * stream)
{
	static unsigned int NumOfFinishedProcs = 0;

	NumOfFinishedProcs ++;
	fprintf(stderr, "[FE] Received process QUIT notification (NumOfFinishedProcs=%d, NumBackEnds=%d)\n", NumOfFinishedProcs, stream->size());
	if (NumOfFinishedProcs == stream->size())
	{
#if defined(DEAD_CODE)
		CmdQueue_Insert(MRN_TERMINATE, NULL);
#else
		CmdQueue_Insert(MRN_TERMINATE);
		Appl_Ended = true;
#endif
		return -1;
	}
	else return 0;
}

int do_Terminate (Stream * stream)
{
	int i;
    int num_be = stream->size();
	
#if ! defined(NEW_DYNAMIC_STREAMS)
	fprintf(stderr, "[FE] Broadcasting TERMINATE command\n");
	MRN_STREAM_SEND(stream, MRN_TERMINATE, "");
#endif

	for (i=0; i<num_be; i++)
	{
		int tag;
		PacketPtr data;

		/* Wait for ACK's */
		MRN_STREAM_RECV(stream, &tag, data, MRN_ACK);
	}

	return 0;
}

int do_BE_Notifies_Flush (Stream *stream) /* Stream is P2P */
{
#if 0
	if (sync_flush)
		CmdQueue_Insert ...
	else
		if (stack_data)
			gather data
		else
			MRN_STREAM_SEND(P2P_stream[task], MRN_NOTIFY_FLUSH, "");

	/* Need to know which task sent the msg!!! */
	MRN_STREAM_SEND(stream, MRN_NOTIFY_FLUSH, "");
#endif

	return 0;
}

int do_Test(Stream *stream)
{
	int add;
	int tag;
	PacketPtr data;
	long long y, z, amin, amax, bmin, bmax, cmin, cmax;

    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MIN_POSITIVE);
    data->unpack("%ld", &y);
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
    data->unpack("%ld", &z);
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MIN_POSITIVE);
    data->unpack("%ld", &amin);
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
    data->unpack("%ld", &amax);

    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MIN_POSITIVE);
    data->unpack("%ld", &bmin);
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
    data->unpack("%ld", &bmax);

    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MIN_POSITIVE);
    data->unpack("%ld", &cmin);
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
    data->unpack("%ld", &cmax);
    fprintf(stderr, "[FE] do_Test: Received y=%lld z=%lld amin=%lld amax=%lld bmin=%lld bmax=%lld cmin=%lld cmax=%lld\n", y, z, amin, amax, bmin, bmax, cmin, cmax);

	return 0;
}

#include "FE_ProtScope.h"
int do_Calc_Mb_Min (Stream *stream)
{
#if 0
	int tag;
	PacketPtr data;
	long long min_common, max_common;
	double total_mb_min, total_bytes_ns;
	int total_num_common_events = 0;

	/* Reduce the common region */
	MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
	data->unpack("%ld", &min_common);
	MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
	data->unpack("%ld", &max_common);

	/* Broadcast the common region */
	MRN_STREAM_SEND(stream, MRN_CALC_MB_MIN, "%ld %ld", min_common, max_common);

	MRN_STREAM_RECV(stream, &tag, data, REDUCE_INT_ADD);
	data->unpack("%d", &total_num_common_events);

	MRN_STREAM_RECV(stream, &tag, data, REDUCE_DOUBLE_ADD);	
	data->unpack("%lf", &total_mb_min);
	MRN_STREAM_RECV(stream, &tag, data, REDUCE_DOUBLE_ADD);	
	data->unpack("%lf", &total_bytes_ns);
	fprintf(stderr, "[FE] total_num_common_events=%d total_mb_min=%lf\n", total_num_common_events, total_mb_min);

	#define RATIO 1
	int desired_mb = 100;
	double freq_secs = ((desired_mb * RATIO) / total_mb_min) * 60;
	double freq_ns = ((desired_mb * 1024 * 1024 * RATIO) / total_bytes_ns);
	fprintf(stderr, "[FE] freq_secs=%lf freq_ns=%lf\n", freq_secs, freq_ns);

	return 0;
#else
	FE_ProtScope *p = new FE_ProtScope();
	p->run(stream);
#endif
}


/* Table associating commands with functions */
cmd_handler_t Root_Handlers_List[] = {
	{ MRN_TEST, do_Test },
	{ MRN_PING, do_Ping },
	{ MRN_LONG_GLOPS, do_Long_Glops },
	{ MRN_CLUSTERS, do_Clusters },
	{ MRN_SPECTRAL, do_Spectral },
	{ MRN_CALC_MB_MIN, do_Calc_Mb_Min },
#if 0
	{ MRN_FEEDBACK_SPECTRAL, do_Spectral_Feedback },
#endif
	{ MRN_WHEN_FULL_FLUSH, do_When_Full_Flush },
	{ MRN_WHEN_FULL_OVERWRITE, do_When_Full_Overwrite },
	{ MRN_SYNC_FLUSH_BE_REQUEST, do_Sync_Flush_BE_Request },
	{ MRN_SYNC_FLUSH, do_Sync_Flush },
	{ MRN_NOTIFY_FLUSH, do_BE_Notifies_Flush },
	{ MRN_QUIT, do_Quit },
	{ MRN_TERMINATE, do_Terminate },
	{ MRN_ALL_COMMANDS, NULL }
};

int Execute_Root_Protocol(int cmd, Stream * stream)
{
	return Execute_Command_Handler(Root_Handlers_List, cmd, stream);
}


#if 0
/* Below is the first integration with the clustering, there was a semaphore mechanism for concurrency we may rescue back */
#include <pthread.h>
void DetachWorker(pthread_t);
void WaitForWorkers();

typedef struct
{
    int NumBackends;
    BurstsInfo_t *BurstsInfo;
    char OutputFileName[256];
    int WaitForSlot;
} ClusterInput_t;

#define MAX_CONCURRENT_CLUSTERINGS  1
#define DISCARD_CONCURRENT_REQUESTS 0
#define WRITE_CLUSTERING_DATA_TO_DISK 1

pthread_t *JoinableThreads = NULL;
int CountJoinableThreads = 0;

void DetachWorker(pthread_t thd)
{
	CountJoinableThreads ++;
	JoinableThreads = (pthread_t *)realloc(JoinableThreads, CountJoinableThreads * sizeof(pthread_t));
	JoinableThreads[CountJoinableThreads - 1] = thd;
}

void WaitForWorkers()
{
	int i;

	for (i=0; i<CountJoinableThreads; i++)
	{
		pthread_join(JoinableThreads[i], NULL);
	}
	xfree (JoinableThreads);
	JoinableThreads = NULL;
	CountJoinableThreads = 0;
}

sem_t ClusterOngoing;

void * ClusteringWorker (void *InputData_ptr)
{
	int i, rc;
	ClusterInput_t *InputData = (ClusterInput_t *)InputData_ptr;

	if (InputData->WaitForSlot)
	{
		/* Wait in semaphore for an open spot */
		sem_wait(&ClusterOngoing);
	}

	/* Invoke the clustering tool */
	fprintf(stderr, "[FE] Calling ExecuteClustering\n");
	fflush(stderr);

	rc = ExecuteClustering("./cl.I.IPC.xml", InputData->BurstsInfo, InputData->NumBackends, InputData->OutputFileName, true);

	fprintf(stderr, "[FE] ExecuteClustering returns=%d\n", rc);
	fflush(stderr);

	/* Free data */
	for (i=0; i<InputData->NumBackends; i++)
	{
		xfree(InputData->BurstsInfo[i].HWCTypes);
		xfree(InputData->BurstsInfo[i].Timestamp);
		xfree(InputData->BurstsInfo[i].Durations);
		xfree(InputData->BurstsInfo[i].HWCValues);
	}
	xfree(InputData->BurstsInfo);
	xfree(InputData);

	/* Open slot */
	sem_post (&ClusterOngoing);

	return NULL;
}

int do_Clustering (Stream *stream)
{
	pthread_t worker;
	ClusterInput_t *InputData;
	int i, slot_open, rc, num_be = stream->size();
	static int CurrentClusteringStep = 0;
	static int ClusterInitialized = FALSE;
	BurstsInfo_t *BurstsInfo = NULL;

	if (!ClusterInitialized)
	{
		rc = sem_init (&ClusterOngoing, 0, MAX_CONCURRENT_CLUSTERINGS);
		if (rc == -1)
		{
			fprintf(stderr, "[FE] do_Clustering: Error initializing semaphore, errno=%d\n", errno);
			exit (1);
		}
		ClusterInitialized = TRUE;
	}

	/* Check if semaphore is open */
	slot_open = (sem_trywait (&ClusterOngoing) == 0);

	if ((slot_open) || (!DISCARD_CONCURRENT_REQUESTS))
	{
		/* Broadcast request to back-ends */
        fprintf(stderr, "[FE] do_Clustering: Broadcasting request to %d back-ends...\n", num_be);
        fflush(stderr);
        if ((stream->send(MRN_CLUSTERIZE, "") == -1) || (stream->flush() == -1))
        {
            fprintf(stderr, "[FE] stream::send(CLUSTERING) failed\n");
            fflush(stderr);
        }

        BurstsInfo = (BurstsInfo_t *)malloc(num_be * sizeof(BurstsInfo_t));

        /* Read clustering data from back-ends */
        for (i=0; i<num_be; i++)
        {
            int tag;
            PacketPtr data;
            int *HWCIds;
            long long *Timestamp, *Durations, *HWCValues;
            int TaskID, count_bursts, count_hwc, len_Timestamp, len_Durations, len_HWCValues;

            MRN_STREAM_RECV(stream, &tag, data, 0);

            data->unpack("%d %d %ad %ald %ald %ald", 
                &TaskID, &count_bursts, &HWCIds, &count_hwc, 
                &Timestamp, &len_Timestamp, &Durations, &len_Durations, &HWCValues, &len_HWCValues);

/*
            fprintf(stderr, "[FE] do_Clustering: Data received from TASK=%d (count_bursts=%d, count_hwc=%d, len_Tstamp=%d, len_Durations=%d, len_HWCVals=%d)\n", 
                TaskID, count_bursts, count_hwc, len_Timestamp, len_Durations, len_HWCValues);
            fflush(stderr);
*/
            BurstsInfo[TaskID].num_Bursts = count_bursts;
            BurstsInfo[TaskID].num_CountersPerBurst = count_hwc;
            BurstsInfo[TaskID].HWCTypes = HWCIds;
            BurstsInfo[TaskID].Timestamp = Timestamp;
            BurstsInfo[TaskID].len_Timestamp = len_Timestamp;
            BurstsInfo[TaskID].Durations = Durations;
            BurstsInfo[TaskID].len_Durations = len_Durations;
            BurstsInfo[TaskID].HWCValues = HWCValues;
            BurstsInfo[TaskID].len_HWCValues = len_HWCValues;
        }
        fprintf(stderr, "[FE] do_Clustering: All data retrieved from %d back-ends\n", num_be);

		InputData = (ClusterInput_t *)malloc(sizeof(ClusterInput_t));
		InputData->NumBackends = num_be;
		InputData->BurstsInfo = BurstsInfo;
		snprintf(InputData->OutputFileName, sizeof(InputData->OutputFileName), "%s%d", "CLUSTER", CurrentClusteringStep); 
        CurrentClusteringStep ++;

		fprintf(stderr, "WRITE_CLUSTERING_DATA_TO_DISK = %d\n", WRITE_CLUSTERING_DATA_TO_DISK);
		if (WRITE_CLUSTERING_DATA_TO_DISK)
		{
			/* Write data to disk */
			Dump_Clustering_Data(InputData);	
		}
//		else
		{
			/* Create a slave thread which will handle this request */
			InputData->WaitForSlot = !slot_open;

			pthread_create (&worker, NULL, ClusteringWorker, (void *)InputData);
			DetachWorker (worker);
		}
	}
	else
	{
		fprintf(stderr, "[FE] do_Clustering: Request discarded (Semaphore closed).\n");
	}
	return 0;
}
#endif

