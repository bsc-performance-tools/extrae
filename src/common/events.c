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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/common/events.c,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:38:53 $
 | @version:     $Revision: 1.12 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: events.c,v 1.12 2009/05/28 13:38:53 harald Exp $";

#include "events.h"

#define MPI_EVENTS 61
static unsigned mpi_events[] = {
  BSEND_EV, SSEND_EV, BARRIER_EV, BCAST_EV, SEND_EV, RECV_EV, SENDRECV_EV,
  SENDRECV_REPLACE_EV, IBSEND_EV, ISSEND_EV, ISEND_EV, IRECV_EV,
  TEST_EV, WAIT_EV, CANCEL_EV, RSEND_EV, IRSEND_EV, ALLTOALL_EV, ALLTOALLV_EV,
  ALLREDUCE_EV, REDUCE_EV, WAITALL_EV, PROBE_EV, IPROBE_EV, GATHER_EV,
  GATHERV_EV, SCATTER_EV, SCATTERV_EV, REDUCESCAT_EV, SCAN_EV, WAITANY_EV,
  WAITSOME_EV, FINALIZE_EV, MPIINIT_EV, ALLGATHER_EV, ALLGATHERV_EV,
  PERSIST_REQ_EV, START_EV, STARTALL_EV, REQUEST_FREE_EV, RECV_INIT_EV,
  SEND_INIT_EV, BSEND_INIT_EV, RSEND_INIT_EV, SSEND_INIT_EV, COMM_RANK_EV,
  COMM_SIZE_EV, IPROBE_COUNTER_EV, TIME_OUTSIDE_IPROBES_EV, TEST_COUNTER_EV,
  FILE_OPEN_EV, FILE_CLOSE_EV, FILE_READ_EV, FILE_READ_ALL_EV, FILE_WRITE_EV,
  FILE_WRITE_ALL_EV, FILE_READ_AT_EV, FILE_READ_AT_ALL_EV, FILE_WRITE_AT_EV,
  FILE_WRITE_AT_ALL_EV, IRECVED_EV };

/******************************************************************************
 ***  IsMPI
 ******************************************************************************/

unsigned IsMPI (unsigned EvType)
{
  unsigned evt;

  for (evt = 0; evt < MPI_EVENTS; evt++)
    if (mpi_events[evt] == EvType)
      return TRUE;
  return FALSE;
}

#define MISC_EVENTS 20
static unsigned misc_events[] = {FLUSH_EV, READ_EV, WRITE_EV, APPL_EV, USER_EV,
  HWC_DEF_EV, HWC_CHANGE_EV, HWC_EV, TRACING_EV, SET_TRACE_EV, MPI_CALLER_EV,
	CPU_BURST_EV, RUSAGE_EV, MPI_STATS_EV, USRFUNC_EV, SAMPLING_EV,
	HWC_SET_OVERFLOW_EV, TRACING_MODE_EV, MRNET_EV, CLUSTER_ID_EV };

/******************************************************************************
 ***  IsMISC
 ******************************************************************************/

unsigned IsMISC (unsigned EvType)
{
    unsigned evt;

    if (EvType>=MPI_CALLER_EV && EvType<=MPI_CALLER_EV+MAX_CALLERS)
        return TRUE;
    if (EvType>=SAMPLING_EV && EvType<=SAMPLING_EV+MAX_CALLERS)
        return TRUE;
    for (evt = 0; evt < MISC_EVENTS; evt++)
        if (misc_events[evt] == EvType)
            return TRUE;
  return FALSE;
}

#define OMP_EVENTS 8
static unsigned omp_events[] = { OMPFUNC_EV, PAR_EV, WSH_EV, BARRIEROMP_EV,
  UNNAMEDCRIT_EV, NAMEDCRIT_EV, WORK_EV, JOIN_EV };

/******************************************************************************
 ***  IsOpenMP
 ******************************************************************************/

unsigned IsOpenMP (unsigned EvType)
{
  unsigned evt;

  for (evt = 0; evt < OMP_EVENTS; evt++)
    if (omp_events[evt] == EvType)
      return TRUE;
  return FALSE;
}

#define PTHREAD_EVENTS 4
static unsigned pthread_events[] = { PTHREADCREATE_EV, PTHREADJOIN_EV,
  PTHREADDETACH_EV, PTHREADFUNC_EV };

/******************************************************************************
 ***  IsPthread
 ******************************************************************************/

unsigned IsPthread (unsigned EvType)
{
  unsigned evt;

  for (evt = 0; evt < PTHREAD_EVENTS; evt++)
    if (pthread_events[evt] == EvType)
      return TRUE;
  return FALSE;
}

#define TRT_EVENTS 3
static unsigned trt_events[] = { TRT_SPAWN_EV, TRT_READ_EV, TRT_USRFUNC_EV };

/******************************************************************************
 ***  IsTRT
 ******************************************************************************/

unsigned IsTRT (unsigned EvType)
{
  unsigned evt;

  for (evt = 0; evt < TRT_EVENTS; evt++)
    if (trt_events[evt] == EvType)
      return TRUE;
  return FALSE;
}

/******************************************************************************
 ***  IsBurst
 ******************************************************************************/
unsigned IsBurst (unsigned EvType)
{
	return (EvType == CPU_BURST_EV);
}

/******************************************************************************
 ***  IsHwcChange
 ******************************************************************************/
unsigned IsHwcChange(unsigned EvType)
{
	return (EvType == HWC_CHANGE_EV);
}

/******************************************************************************
 ***  IsCollective
 ******************************************************************************/
unsigned IsMPICollective(unsigned EvType)
{
   switch (EvType)
   {
      case BARRIER_EV:
      case BCAST_EV:
      case ALLTOALL_EV:
      case ALLTOALLV_EV:
      case REDUCE_EV:
      case ALLREDUCE_EV:
      case GATHER_EV:
      case GATHERV_EV:
      case ALLGATHER_EV:
      case ALLGATHERV_EV:
      case SCATTER_EV:
      case SCATTERV_EV:
      case REDUCESCAT_EV:
      case SCAN_EV:
         return TRUE;
      default:
         return FALSE;
   }
   return FALSE;
}


/******************************************************************************
 ***  getEventType
 ******************************************************************************/

EventType_t getEventType (unsigned EvType, unsigned *Type)
{
	if (IsMPI (EvType))
	{
		*Type = MPI_TYPE;
		return TRUE;
	}
	else if (IsMISC (EvType))
	{
		*Type = MISC_TYPE;
		return TRUE;
	}
	else if (IsOpenMP (EvType))
	{
		*Type = OPENMP_TYPE;
		return TRUE;
	}
	else if (IsPthread (EvType))
	{
		*Type = PTHREAD_TYPE;
		return TRUE;
	}
	else if (IsTRT (EvType))
	{
		*Type = TRT_TYPE;
		return TRUE;
	}
	else if ((EvType == COMM_CREATE_EV) || (EvType == COMM_DUP_EV) ||
	(EvType == COMM_SPLIT_EV) || (EvType == MPI_CART_CREATE_EV) ||
	(EvType == MPI_CART_SUB_EV))
	{
		*Type = COMM_ALIAS_TYPE;
		return TRUE;
	}
	return FALSE;
}

