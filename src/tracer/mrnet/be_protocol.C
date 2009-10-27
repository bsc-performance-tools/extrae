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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/be_protocol.C,v $
 | 
 | @last_commit: $Date: 2009/06/10 17:41:56 $
 | @version:     $Revision: 1.15 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: be_protocol.C,v 1.15 2009/06/10 17:41:56 gllort Exp $";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#include <mrnet/MRNet.h>
#include "mrnet_commands.h"
#include "mrnet_be.h"
#include "signals.h"
#include "protocol.h"
#include "utils.h"
#include "be_protocol.h"
#include "clock.h"
#include "hwc.h"
#include "trace_buffers.h"
#include "mrnet_buffers.h"
#include "libextract/BurstInfo_BE.h"
#include "libextract/LongGlops.h"

using namespace MRN;

int Send_Bursts_Info (Stream *stream, BurstInfo_t *bi, int bytes)
{
	int i;
	int TaskID=0, ThreadID=0, num_Bursts=0, num_HWCperBurst=0;
	long long *Timestamp=NULL, *Durations=NULL, *HWCValues=NULL;
	int *HWCSet=NULL;
	int tag; 
	PacketPtr data;

	BurstInfo_Serialize (bi, &TaskID, &ThreadID, &num_Bursts, &num_HWCperBurst, 
		&Timestamp, &Durations, &HWCValues, &HWCSet);

	fprintf(stderr, "[BE %d] Sending BurstsInfo to FE (num_Bursts=%d)\n", TaskID, num_Bursts);
	/* Send data to front-end */
	MRN_STREAM_SEND(stream, MRN_BURSTS_INFO, "%d %d %d %d %ald %ald %ald %ad",
		TaskID, ThreadID, num_Bursts, num_HWCperBurst, 
		Timestamp, num_Bursts, 
		Durations, num_Bursts,
		HWCValues, num_Bursts * num_HWCperBurst,
		HWCSet, num_Bursts);

	MRN_STREAM_RECV(stream, &tag, data, MRN_ACK);
	MRN_STREAM_SEND(stream, REDUCE_INT_ADD, "%d", bytes);

/*
	unsigned long long min_time = TIME, max_time = 0;
	if (num_Bursts > 0)
	{
		min_time = Timestamp[0];
		max_time = Timestamp[num_Bursts-1];
	}

fprintf(stderr, "[BE %d] xxxx MINTIME %lld MAXTIME %lld\n", TaskID, min_time, max_time);

	MRN_STREAM_SEND(stream, REDUCE_ULL_MAX, "%uld", min_time);
	MRN_STREAM_SEND(stream, REDUCE_ULL_MAX, "%uld", max_time);
*/
/*
	long long min_time = -1, max_time = -1;
	if (num_Bursts > 0)
	{
		min_time = Timestamp[0];
		max_time = Timestamp[num_Bursts-1] + Durations[num_Bursts-1];
	}
	MRN_STREAM_SEND(stream, MRN_BURSTS_INFO, "%ld %ld", min_time, max_time);
*/
	long long min_time = -1, max_time = -1;
	if (num_Bursts > 0)
	{
		min_time = Timestamp[0];
		max_time = Timestamp[num_Bursts-1] + Durations[num_Bursts-1];
	}
	fprintf(stderr, "[BE %d] LOCAL_TIME_MIN %lld LOCAL_TIME_MAX %lld\n", TaskID, min_time, max_time);
	MRN_STREAM_SEND(stream, REDUCE_LL_MIN_POSITIVE, "%ld", min_time);
	MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", max_time);

	return 0;
}

int do_Ping (Stream *stream)
{
	MRN_STREAM_SEND(stream, MRN_PONG, "%d %d %d", 1, 2, 3);
	return 0;
}

int do_Long_Glops (Stream *stream)
{
	unsigned long long * Glops_Durations = NULL;
	int NumGlops, firstGlopID, lastGlopID;
	int NumCommonGlops, commonFirstGlop, commonLastGlop;
	unsigned int * Selected_Glops = NULL;
	int tag;
	PacketPtr data;

	Buffer_Lock(TRACING_BUFFER(0));
	Signals_PauseApplication();

	/* Get data from tracing buffer */
	NumGlops = Extract_LongGlops (TASKID, 0, &Glops_Durations, &firstGlopID, &lastGlopID);

	MRN_STREAM_SEND(stream, MRN_GLOPS_INFO, "%d %d %ald",
		firstGlopID, lastGlopID, Glops_Durations, NumGlops);

	xfree(Glops_Durations);

	MRN_STREAM_RECV(stream, &tag, data, MRN_GLOPS_SELECTED);
	data->unpack("%d %d %aud", &commonFirstGlop, &commonLastGlop, &Selected_Glops, &NumCommonGlops);

	fprintf(stderr, "[BE %d] commonFirstGlop=%d, commonLastGlop=%d, NumCommonGlops=%d\n", TASKID, commonFirstGlop, commonLastGlop, NumCommonGlops);

	Filter_Long_Glops (TASKID, 0, commonFirstGlop, commonLastGlop, Selected_Glops);

	Buffer_Flush (TRACING_BUFFER(0));

	Signals_ResumeApplication();
	Buffer_Unlock(TRACING_BUFFER(0));

	xfree(Selected_Glops);

	return 0;
}

extern Network *net;
#include "StreamPublisher.h"
#include "timesync.h"

void Clusters_TraceCIDS(Stream *stream, BurstInfo_t *bi)
{
	int *cids = NULL, len;
	int tag;
	PacketPtr data;

    /* Read CIDs */
    StreamPublisher sp(TASKID, net, stream);
    Stream *p2p = NULL;
    p2p = sp.Recv();
    if (p2p != NULL)
    {
        MRN_STREAM_RECV(p2p, &tag, data, MRN_CLUSTERS);

        data->unpack("%ad", &cids, &len);
        fprintf(stderr, "[BE %d] Received %d CIDS\n", TASKID, len);

        for (int i=0; i<len; i++)
        {
			unsigned long long t = TIMEDESYNC(TASKID, bi->Timestamp[i]);

            TRACE_MRN_EVENT(t, CLUSTER_ID_EV, cids[i]);
            TRACE_MRN_EVENT(t+bi->Durations[i], CLUSTER_ID_EV, 0);

        }
    }

    /* Free data */
    xfree(cids);
}

int Clusters_ReadConfig (Stream *stream)
{
	int tag;
	PacketPtr data;
	unsigned long long min_duration;

    /* Read the duration filter */
    MRN_STREAM_RECV(stream, &tag, data, MRN_CLUSTERS);
    data->unpack("%uld", &min_duration);

	return min_duration;
}

#if 0
int new_Clusters_ExtractData_FromCommonRegion (Stream *stream, int thread, unsigned long long min_duration, BurstInfo_t **bi)
{
    long long min_time = -1, max_time = -1;
	Buffer_t *buffer = TRACING_BUFFER(thread);
	BufferIterator_t *it;

	/* Find the local minimum event time */
	it = BIT_NewForward(buffer);
	if (!BIT_OutOfBounds(it)) min_time = TIMESYNC(TASKID, Get_EvTime(BIT_GetEvent(it)));
	/* Find the minimum common event time */
    MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", min_time);
	MRN_STREAM_RECV(stream, &tag, data, MRN_CLUSTERS);
	data->unpack("%ld", &min_time);
	min_time = TIME_DESYNC(TASKID, min_time);
	min_time = MAX(min_time, lastClusteringStartedAt);

	/* Extract data from min_time ... last event */
	MbDataAnalyzed = Extract_BurstInfo (TASKID, 0, min_time, min_duration, bi);

	return MbDataAnalyzed;
}
#endif 

unsigned long long lastClusteringStartedAt = 0;

int Clusters_ExtractData (int thread, unsigned long long min_duration, BurstInfo_t **bi)
{
	int MbDataAnalyzed;

    MbDataAnalyzed = Extract_BurstInfo (TASKID, 0, lastClusteringStartedAt, TIME, min_duration, bi);
	fprintf(stderr, "[BE %d] Clusters_ExtractData NULL? %d\n", TASKID, (bi == NULL));

	return MbDataAnalyzed;
}

int SingleClusterAnalysis(Stream *stream, BurstInfo_t **io_bi)
{
    unsigned long long min_duration;
    BurstInfo_t *bi = NULL;
    int bytes;
	int tag;
	PacketPtr data;
	int interrupted = FALSE;

	fprintf(stderr, "[BE %d]BEFORE ACK RECEIVE SingleClusterAnalysis\n", TASKID);
	MRN_STREAM_RECV(stream, &tag, data, MRN_ANY);
	fprintf(stderr, "[BE %d]SingleClusterAnalysis ACK received\n", TASKID);
	if (tag == MRN_ACK)
	{
	    /* Read the duration filter */
	    min_duration = Clusters_ReadConfig(stream);

		/* Lock the application (until the results are sent back) */
		Buffer_Lock(TRACING_BUFFER(0));

	    /* Extract data from tracing buffer */
	    bytes = Clusters_ExtractData(0, min_duration, &bi);

	    /* Send data to root */
	    Send_Bursts_Info (stream, bi, bytes);

		lastClusteringStartedAt = TIME;
	}
	else
	{
		interrupted = TRUE;
	}
	*io_bi = bi;
    return interrupted;
}

int MultiClusterAnalysis(Stream *stream, BurstInfo_t **io_bi)
{
	BurstInfo_t *bi = NULL;
	int i, stability = FALSE;
	int tag;
	PacketPtr data;
	unsigned long long change_hwc_freq;
	int interrupted = FALSE;

	while (! stability)
	{
		/* Let the application run until the next cluster is requested */
		Buffer_Unlock(TRACING_BUFFER(0));

		BurstInfo_Free(bi);

		interrupted = SingleClusterAnalysis(stream, &bi);
		if (interrupted)
		{
			/* The application reached the end and no data was analyzed */
			*io_bi = NULL;
			return TRUE;
		}
		/* if ( (bi == NULL) || (bi->num_Bursts == 0) ) A task may have 0 bursts while the others don't!!!
		{
			return bi;
		}*/

		MRN_STREAM_RECV(stream, &tag, data, MRN_CLUSTERS);
		data->unpack("%d %uld", &stability, &change_hwc_freq);
		fprintf(stderr, "[BE %d] Stability? %d\n", TASKID, stability);
		if (!stability) 
		{
			Buffer_DiscardAll(TRACING_BUFFER(0));
			
			for (i=0; i<HWC_Get_Num_Sets(); i++)
			{
				HWC_Set_ChangeAtTime_Frequency(i, change_hwc_freq);
			}
			fprintf(stderr, "[BE %d] CHANGE_HWC (new_freq=%llu, num_sets=%d)\n", TASKID, change_hwc_freq, HWC_Get_Num_Sets());
		}
	}
	*io_bi = bi;
	return FALSE;
}

int do_Clusters (Stream *stream)
{
	BurstInfo_t *bi = NULL;
	int interrupted;

	//interrupted = SingleClusterAnalysis(stream, &bi);
	interrupted = MultiClusterAnalysis(stream, &bi);

	if (!interrupted)
//	if ((bi != NULL) && (bi->num_Bursts > 0)) A task may have 0 bursts while the others don't!!!
	{
	    /* Recv & trace cluster ids */
		Clusters_TraceCIDS (stream, bi);
	}

	Buffer_Flush(TRACING_BUFFER(0));
	close_mpits(0);
    Buffer_Unlock(TRACING_BUFFER(0));

    /* Free data */
    BurstInfo_Free (bi);

	return 0;
}

int do_Spectral (Stream *stream)
{
	int MbDataAnalyzed;
	BurstInfo_t *bi = NULL;
	int tag;
	PacketPtr data;
	int numPeriods = 0;
	Period_t * listPeriods;

    /* Extract data from tracing buffer */
    Buffer_Lock(TRACING_BUFFER(0));
    MbDataAnalyzed = Extract_BurstInfo (TASKID, 0, 0, TIME, 0, &bi);

    Send_Bursts_Info (stream, bi, MbDataAnalyzed);
	BurstInfo_Free (bi);

    MRN_STREAM_RECV(stream, &tag, data, MRN_SPECTRAL);
	data->unpack("%d", &numPeriods);


    listPeriods = (Period_t *)malloc(numPeriods * sizeof(Period_t));
	for (int i=0; i<numPeriods; i++)
	{
    	MRN_STREAM_RECV(stream, &tag, data, MRN_SPECTRAL);
		data->unpack("%f %ld %lf %lf %lf %ld %ld %ld %ld",
			&listPeriods[i].iters,
			&listPeriods[i].length,
			&listPeriods[i].goodness,
			&listPeriods[i].goodness2,
			&listPeriods[i].goodness3,
			&listPeriods[i].ini,
			&listPeriods[i].end,
			&listPeriods[i].best_ini,
			&listPeriods[i].best_end);
fprintf(stderr, "[BE %d] Period=%d Iters=%f best_ini=%lld best_end=%lld\n", TASKID, i, listPeriods[i].iters, listPeriods[i].best_ini, listPeriods[i].best_end);
	}

	if (numPeriods > 0)
	{
		/* If no periods are found, the whole buffer is traced ... for debugging purposes right now ! */
		Filter_Periods (TASKID, 0, numPeriods, listPeriods);
	}

    xfree (listPeriods);

	Buffer_Flush(TRACING_BUFFER(0));
	close_mpits(0);

    MRN_STREAM_SEND(stream, MRN_ACK, "");

    Buffer_Unlock(TRACING_BUFFER(0));

	return 0;
}

#if 0
#include "spectral.h"
int do_Spectral_Feedback (Stream *stream)
{
	int i, num_periods, tag;
	PacketPtr data;
	Period_t *Periods = NULL;
	BurstInfo_t *bi = NULL;
	int MbDataAnalyzed;
	
    /* Extract data from tracing buffer */
	Buffer_Lock (TRACING_BUFFER(0));
	Signals_PauseApplication();

    MbDataAnalyzed = Extract_BurstInfo (TASKID, 0, 0, &bi);
    Buffer_Unlock(TRACING_BUFFER(0));

	Send_Bursts_Info (stream, bi, MbDataAnalyzed);
	BurstInfo_Free (bi);

	MRN_STREAM_RECV(stream, &tag, data, MRN_NUM_PERIODS);
    data->unpack("%d", &num_periods);

	Periods = (Period_t *)malloc(num_periods * sizeof(Period_t));
	for (i=0; i<num_periods; i++)
	{
		MRN_STREAM_RECV(stream, &tag, data, MRN_PERIOD_INFO);
		data->unpack("%f %ld %lf %lf %lf %ld %ld", 
			&Periods[i].iters,
            &Periods[i].main_period_duration,
            &Periods[i].goodness,
            &Periods[i].goodness2,
            &Periods[i].goodness3,
            &Periods[i].ini,
            &Periods[i].end);
	}

//	Filter_Periods (TASKID, 0, num_periods, Periods);

	MRN_STREAM_SEND(stream, MRN_ACK, "");

	xfree (Periods);

	Buffer_Flush (TRACING_BUFFER(0));

	Signals_ResumeApplication();

	return 0;
}
#endif

int do_When_Full_Flush (Stream * stream)
{
	Buffer_t *buffer = TRACING_BUFFER(0);
    Buffer_Lock (buffer);
    Buffer_DiscardAll (buffer);
    Buffer_SetFlushCallback (buffer, CALLBACK_FLUSH);
    Buffer_Unlock (buffer);
    return 0;
}

int do_When_Full_Overwrite (Stream * stream)
{
	Buffer_t *buffer = TRACING_BUFFER(0);
    Buffer_Lock (buffer);
    Buffer_Flush (buffer);
//  Buffer_SetFlushCallback (buffer, CALLBACK_OVERWRITE);
    close_mpits (0);
    Buffer_Unlock (buffer);
    return 0;
}

int do_Sync_Flush (Stream *stream)
{
	Buffer_Lock (TRACING_BUFFER(0));

    TRACE_MRN_EVENT(TIME, MRNET_EV, 3);

	Buffer_Flush (TRACING_BUFFER(0));

	/* Wake up the application's main thread that may be waiting on this */
	Signals_CondWakeUp (&SyncFlush_Completion);

	MRN_STREAM_SEND(stream, MRN_ACK, "");

    TRACE_MRN_EVENT(TIME, MRNET_EV, 0);

	Buffer_Unlock (TRACING_BUFFER(0));

	return 0;
}

int do_Terminate (Stream *stream)
{
	MRN_STREAM_SEND(stream, MRN_ACK, "");
	return 0;
}

#include "StreamPublisher.h"
int do_Test (Stream *stream)
{
	int tag;
	PacketPtr data;
	long long y = 1000 * TASKID;
 	long long z = 1000 * TASKID;
	long long a = 12345, b = -1, c = (TASKID % 2 ? -1 : 5678);

	fprintf(stderr, "[BE %d] do_Test: Sending y=%lld z=%lld a=%lld b=%lld c=%lld\n", TASKID, y, z, a, b, c);
	MRN_STREAM_SEND(stream, REDUCE_LL_MIN_POSITIVE, "%ld", y);
	MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", z);

	MRN_STREAM_SEND(stream, REDUCE_LL_MIN_POSITIVE, "%ld", a);
	MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", a);

	MRN_STREAM_SEND(stream, REDUCE_LL_MIN_POSITIVE, "%ld", b);
	MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", b);

	MRN_STREAM_SEND(stream, REDUCE_LL_MIN_POSITIVE, "%ld", c);
	MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", c);

	return 0;
}

#include "BE_ProtScope.h"
#include "timesync.h"
int do_Calc_Mb_Min (Stream *stream)
{
#if 0
	/* Common region defined from MAX(min_time) to MAX(max_time) */
	Buffer_t *buffer = TRACING_BUFFER(0);
	BufferIterator_t *itf, *itb, *itr;
	long long min_local_time = -1, max_local_time = -1; 
	unsigned long long min_common_time, max_common_time;
	int num_common_events = 0;
	int tag;
	PacketPtr data;

	Buffer_Lock(buffer);

	itf = BIT_NewForward(buffer);
	itb = BIT_NewBackward(buffer);

	if (!BIT_OutOfBounds(itf)) min_local_time = TIMESYNC(TASKID, Get_EvTime(BIT_GetEvent(itf)));

	if (!BIT_OutOfBounds(itb)) max_local_time = TIMESYNC(TASKID, Get_EvTime(BIT_GetEvent(itb)));

	MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", min_local_time);
	MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", max_local_time);

	MRN_STREAM_RECV(stream, &tag, data, MRN_CALC_MB_MIN)
	data->unpack("%ld %ld", &min_common_time, &max_common_time);
	min_common_time = TIMEDESYNC(TASKID, min_common_time);
	max_common_time = TIMEDESYNC(TASKID, max_common_time);
	
	/* Now calc Mb/min ratio */
	itr = BIT_NewRange(buffer, min_common_time, max_common_time);
	num_common_events = 0;
	while (!BIT_OutOfBounds(itr))
	{
		num_common_events ++;
		BIT_Next(itr);
	}	
	MRN_STREAM_SEND(stream, REDUCE_INT_ADD, "%d", num_common_events);

	double ns = max_common_time - min_common_time;
	double bytes = (num_common_events * sizeof(event_t));
	double mb, secs, mins, mb_min, bytes_ns;

    mb = bytes / (1024*1024);
    secs = ns / 1000000000;
    mins = secs / 60;
    mb_min = mb / mins;
	bytes_ns = bytes / ns;

	fprintf(stderr, "[BE %d] MRN_CALC_MB_MIN min_local=%lld max_local=%lld n_events=%d -- min_common_time=%llu max_common_time=%llu n_events=%d -- bytes=%llu ns=%llu mb_min=%.3lf\n", 
		TASKID, min_local_time, max_local_time, Buffer_GetFillCount(buffer), min_common_time, max_common_time, num_common_events, bytes, ns, mb_min);

	MRN_STREAM_SEND(stream, REDUCE_DOUBLE_ADD, "%lf", mb_min);
	MRN_STREAM_SEND(stream, REDUCE_DOUBLE_ADD, "%lf", bytes_ns);

	Buffer_Unlock(buffer);
	return 0;
#else
    BE_ProtScope *p = new BE_ProtScope();
    p->run(stream);
#endif
}


/* Table associating commands with functions */
cmd_handler_t Backend_Handlers_List[] = {
	{ MRN_TEST, do_Test },
	{ MRN_PING, do_Ping },
	{ MRN_LONG_GLOPS, do_Long_Glops },
	{ MRN_CLUSTERS, do_Clusters },
	{ MRN_SPECTRAL, do_Spectral },
#if 0
	{ MRN_FEEDBACK_SPECTRAL, do_Spectral_Feedback },
#endif
	{ MRN_CALC_MB_MIN, do_Calc_Mb_Min },
    { MRN_WHEN_FULL_FLUSH, do_When_Full_Flush },
    { MRN_WHEN_FULL_OVERWRITE, do_When_Full_Overwrite },
	{ MRN_SYNC_FLUSH, do_Sync_Flush },
	{ MRN_TERMINATE, do_Terminate },
	{ MRN_ALL_COMMANDS, NULL }
};

int Execute_Backend_Protocol(int command, Stream * stream)
{
	return Execute_Command_Handler(Backend_Handlers_List, command, stream);
}

