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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __MRNET_COMMANDS_H__
#define __MRNET_COMMANDS_H__

#include <mrnet/MRNet.h>

#define NEW_DYNAMIC_STREAMS

enum 
{ 
	MRN_TERMINATE=FirstApplicationTag, 
	MRN_ANY,
	MRN_REGISTER_STREAM,
	MRN_CONFIG,
	MRN_ACK,
	MRN_NACK,
	MRN_PING,
	MRN_PONG,
	MRN_LONG_GLOPS,
	MRN_GLOPS_INFO,
	MRN_GLOPS_SELECTED,
	MRN_CLUSTERS,
	MRN_SPECTRAL,
	MRN_FEEDBACK_SPECTRAL,
	MRN_BURSTS_INFO,
	MRN_NUM_PERIODS,
	MRN_PERIOD_INFO,
	MRN_WHEN_FULL_FLUSH,
	MRN_WHEN_FULL_OVERWRITE,
	MRN_SYNC_FLUSH_BE_REQUEST,
	MRN_SYNC_FLUSH,
	MRN_NOTIFY_FLUSH,
	MRN_QUIT,
	MRN_CALC_MB_MIN,
	MRN_TEST,
	REDUCE_INT_ADD, 
/*
	REDUCE_ULL_MIN, 
	REDUCE_ULL_MAX,
*/
	REDUCE_LL_MIN_POSITIVE,
	REDUCE_LL_MAX_POSITIVE,
	REDUCE_DOUBLE_ADD,
	MRN_SCATTER,
	MRN_ALL_COMMANDS
};

/* #define MRN_DEBUG */
#define MRN_RANK(rank) (Rank)(rank+1000000)
#define BE_RANK(rank) (rank-1000000)

#if defined(FE)
# define PRINT_WHERE fprintf(stderr, "[FE] %s:%d: ", __FUNCTION__, __LINE__);
#elif defined(BE)
# include "taskid.h"
# define PRINT_WHERE fprintf(stderr, "[BE %d] %s:%d: ", TASKID, __FUNCTION__, __LINE__);
#endif

/* XXX: Should "rc" be returned in the macros below? */

#define BLOCKING_STREAM_RECV

/* Receive from a specific stream */
#if defined(BLOCKING_STREAM_RECV)
# define MRN_STREAM_RECV(stream, tag, data, expected)                             \
{                                                                                 \
	int rc;                                                                       \
	rc = stream->recv(tag, data, true);                                           \
	if (rc == -1)                                                                 \
	{                                                                             \
		PRINT_WHERE;                                                              \
		fprintf(stderr, "stream::recv() failed (stream_id=%d).",                  \
			stream->get_Id());                                                    \
		exit(1);                                                                  \
	}                                                                             \
    if ((expected != MRN_ANY) && (*tag != expected))                              \
    {                                                                             \
        PRINT_WHERE;                                                              \
        fprintf(stderr, "stream::recv() tag received %d, but expected %d (%s)\n", \
            *tag, expected, #expected);                                           \
    }                                                                             \
}
#else
# define MRN_STREAM_RECV(stream, tag, data, expected)                             \
{                                                                                 \
	int rc;                                                                       \
	while ((rc = stream->recv(tag, data, false)) == 0)                            \
		usleep(500000)                                                            \
	if (rc == -1)                                                                 \
	{                                                                             \
        PRINT_WHERE;                                                              \
        fprintf(stderr, "stream::recv() failed (stream_id=%d).",                  \
            stream->get_Id());                                                    \
        exit(1);                                                                  \
	}                                                                             \
    if ((expected != MRN_ANY) && (*tag != expected))                              \
	{                                                                             \
		PRINT_WHERE;                                                              \
		fprintf(stderr, "stream::recv() tag received %d, but expected %d (%s)\n", \
			*tag, expected, #expected);                                           \
	}                                                                             \
}
#endif

/* Receives from any stream of the network */
#define MRN_NETWORK_RECV(net, tag, data, expected, stream, blocking)              \
{                                                                                 \
	int rc;                                                                       \
	rc = net->recv(tag, data, stream, blocking);                                  \
	if (rc == -1) {                                                               \
		PRINT_WHERE;                                                              \
		fprintf(stderr, "network::recv() failed.\n");                             \
		exit(1);                                                                  \
	}                                                                             \
    if ((expected != MRN_ANY) && (*tag != expected))                              \
    {                                                                             \
        PRINT_WHERE;                                                              \
        fprintf(stderr, "stream::recv() tag received %d, but expected %d (%s)\n", \
            *tag, expected, #expected);                                           \
    }                                                                             \
}

/* Sends message and flushes stream */
#define MRN_STREAM_SEND(stream, tag, format, args...)                                \
{                                                                                    \
	int rc;                                                                          \
	rc = stream->send(tag, format, ## args);                                         \
	if (rc == -1) {                                                                  \
		PRINT_WHERE;                                                                 \
		fprintf(stderr, "stream::send(%s, \"%s\") failed (stream_id=%d, tag=%d).\n", \
			 #tag, format, stream->get_Id(), tag);                                   \
		exit(1);                                                                     \
	}                                                                                \
	else {                                                                           \
		rc = stream->flush();                                                        \
		if (rc == -1) {                                                              \
		    PRINT_WHERE;                                                             \
			fprintf(stderr, "stream::flush() failed (stream_id=%d).\n",              \
				stream->get_Id());                                                   \
			exit(1);                                                                 \
		}                                                                            \
    }                                                                                \
} 

#define MRN_STREAM_SEND_ID(stream, tag, format, args...)                             \
{                                                                                    \
	MRN_STREAM_SEND(stream, tag, "%d %d "format, TASKID, /* THREADID */ 0, ## args);       \
} 

#define MRN_UNPACK(data, format, args...) \
{ \
	data->unpack(format, ## args); \
}

#define MRN_ID(data, task, thread) \
{ \
	MRN_UNPACK(data, "%d %d", task, thread); \
}



#endif /* __MRNET_COMMANDS_H__ */
