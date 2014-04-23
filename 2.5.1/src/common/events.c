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
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include "events.h"

#define MPI_EVENTS 72
static unsigned mpi_events[] = {
	MPI_BSEND_EV, MPI_SSEND_EV, MPI_BARRIER_EV, MPI_BCAST_EV, MPI_SEND_EV,
	MPI_RECV_EV, MPI_SENDRECV_EV, MPI_SENDRECV_REPLACE_EV, MPI_IBSEND_EV,
	MPI_ISSEND_EV, MPI_ISEND_EV, MPI_IRECV_EV, MPI_TEST_EV, MPI_WAIT_EV,
	MPI_CANCEL_EV, MPI_RSEND_EV, MPI_IRSEND_EV, MPI_ALLTOALL_EV,
	MPI_ALLTOALLV_EV, MPI_ALLREDUCE_EV, MPI_REDUCE_EV, MPI_WAITALL_EV,
	MPI_TESTANY_EV, MPI_TESTALL_EV, MPI_TESTSOME_EV,
	MPI_PROBE_EV, MPI_IPROBE_EV, MPI_GATHER_EV, MPI_GATHERV_EV,
	MPI_SCATTER_EV, MPI_SCATTERV_EV, MPI_REDUCESCAT_EV, MPI_SCAN_EV,
	MPI_WAITANY_EV, MPI_WAITSOME_EV, MPI_FINALIZE_EV, MPI_INIT_EV,
	MPI_ALLGATHER_EV, MPI_ALLGATHERV_EV, MPI_PERSIST_REQ_EV, MPI_START_EV,
	MPI_STARTALL_EV, MPI_REQUEST_FREE_EV, MPI_RECV_INIT_EV, MPI_SEND_INIT_EV,
	MPI_BSEND_INIT_EV, MPI_RSEND_INIT_EV, MPI_SSEND_INIT_EV, MPI_COMM_RANK_EV,
	MPI_COMM_SIZE_EV, MPI_IPROBE_COUNTER_EV, MPI_TIME_OUTSIDE_IPROBES_EV,
	MPI_TEST_COUNTER_EV, MPI_FILE_OPEN_EV, MPI_FILE_CLOSE_EV, MPI_FILE_READ_EV,
	MPI_FILE_READ_ALL_EV, MPI_FILE_WRITE_EV, MPI_FILE_WRITE_ALL_EV, 
	MPI_FILE_READ_AT_EV, MPI_FILE_READ_AT_ALL_EV, MPI_FILE_WRITE_AT_EV,
	MPI_FILE_WRITE_AT_ALL_EV, MPI_IRECVED_EV, MPI_GET_EV, MPI_PUT_EV,
	MPI_COMM_CREATE_EV, MPI_COMM_DUP_EV, MPI_COMM_SPLIT_EV,
	MPI_CART_CREATE_EV, MPI_CART_SUB_EV, MPI_COMM_FREE_EV };

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

#define PACX_EVENTS 67
static unsigned pacx_events[] = {
	PACX_BSEND_EV, PACX_SSEND_EV, PACX_BARRIER_EV, PACX_BCAST_EV, PACX_SEND_EV,
	PACX_RECV_EV, PACX_SENDRECV_EV, PACX_SENDRECV_REPLACE_EV, PACX_IBSEND_EV,
	PACX_ISSEND_EV, PACX_ISEND_EV, PACX_IRECV_EV, PACX_TEST_EV, PACX_WAIT_EV,
	PACX_CANCEL_EV, PACX_RSEND_EV, PACX_IRSEND_EV, PACX_ALLTOALL_EV,
	PACX_ALLTOALLV_EV, PACX_ALLREDUCE_EV, PACX_REDUCE_EV, PACX_WAITALL_EV,
	PACX_PROBE_EV, PACX_IPROBE_EV, PACX_GATHER_EV, PACX_GATHERV_EV,
	PACX_SCATTER_EV, PACX_SCATTERV_EV, PACX_REDUCESCAT_EV, PACX_SCAN_EV,
	PACX_WAITANY_EV, PACX_WAITSOME_EV, PACX_FINALIZE_EV, PACX_INIT_EV,
	PACX_ALLGATHER_EV, PACX_ALLGATHERV_EV, PACX_PERSIST_REQ_EV, PACX_START_EV,
	PACX_STARTALL_EV, PACX_REQUEST_FREE_EV, PACX_RECV_INIT_EV, PACX_SEND_INIT_EV,
	PACX_BSEND_INIT_EV, PACX_RSEND_INIT_EV, PACX_SSEND_INIT_EV, PACX_COMM_RANK_EV,
	PACX_COMM_SIZE_EV, PACX_IPROBE_COUNTER_EV, PACX_TIME_OUTSIDE_IPROBES_EV,
	PACX_TEST_COUNTER_EV, PACX_FILE_OPEN_EV, PACX_FILE_CLOSE_EV, PACX_FILE_READ_EV,
	PACX_FILE_READ_ALL_EV, PACX_FILE_WRITE_EV, PACX_FILE_WRITE_ALL_EV, 
	PACX_FILE_READ_AT_EV, PACX_FILE_READ_AT_ALL_EV, PACX_FILE_WRITE_AT_EV,
	PACX_FILE_WRITE_AT_ALL_EV, PACX_IRECVED_EV,
	PACX_COMM_CREATE_EV, PACX_COMM_DUP_EV, PACX_COMM_SPLIT_EV,
	PACX_CART_CREATE_EV, PACX_CART_SUB_EV, PACX_COMM_FREE_EV };

/******************************************************************************
 ***  IsMPI
 ******************************************************************************/

unsigned IsPACX (unsigned EvType)
{
  unsigned evt;

  for (evt = 0; evt < PACX_EVENTS; evt++)
    if (pacx_events[evt] == EvType)
      return TRUE;
  return FALSE;
}


#define MISC_EVENTS 43
static unsigned misc_events[] = {FLUSH_EV, READ_EV, WRITE_EV, APPL_EV, USER_EV,
	HWC_DEF_EV, HWC_CHANGE_EV, HWC_EV, TRACING_EV, SET_TRACE_EV, CALLER_EV,
	CPU_BURST_EV, RUSAGE_EV, MEMUSAGE_EV, MPI_STATS_EV, USRFUNC_EV,
	SAMPLING_EV, SAMPLING_ADDRESS_EV, SAMPLING_ADDRESS_MEM_LEVEL_EV, SAMPLING_ADDRESS_TLB_LEVEL_EV,
	SAMPLING_ADDRESS_REFERENCE_COST_EV,
	HWC_SET_OVERFLOW_EV, TRACING_MODE_EV, MRNET_EV, CLUSTER_ID_EV, SPECTRAL_PERIOD_EV,
	USER_SEND_EV, USER_RECV_EV, RESUME_VIRTUAL_THREAD_EV, SUSPEND_VIRTUAL_THREAD_EV,
	TRACE_INIT_EV, REGISTER_STACKED_TYPE_EV, REGISTER_CODELOCATION_TYPE_EV,
	FORK_EV, WAIT_EV, WAITPID_EV, EXEC_EV, GETCPU_EV, SYSTEM_EV,
	MALLOC_EV, FREE_EV, CALLOC_EV, REALLOC_EV };

/******************************************************************************
 ***  IsMISC
 ******************************************************************************/

unsigned IsMISC (unsigned EvType)
{
    unsigned evt;

    if (EvType>=CALLER_EV && EvType<=CALLER_EV+MAX_CALLERS)
        return TRUE;
    if (EvType>=SAMPLING_EV && EvType<=SAMPLING_EV+MAX_CALLERS)
        return TRUE;
    for (evt = 0; evt < MISC_EVENTS; evt++)
        if (misc_events[evt] == EvType)
            return TRUE;
  return FALSE;
}

#define OMP_EVENTS 14
static unsigned omp_events[] = { OMPFUNC_EV, PAR_EV, WSH_EV, BARRIEROMP_EV,
	UNNAMEDCRIT_EV, NAMEDCRIT_EV, WORK_EV, JOIN_EV, OMPSETNUMTHREADS_EV,
	OMPGETNUMTHREADS_EV, TASK_EV, TASKWAIT_EV, TASKFUNC_EV, TASKFUNC_LINE_EV };

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

#define PTHREAD_EVENTS 14
static unsigned pthread_events[] = { PTHREAD_CREATE_EV, PTHREAD_JOIN_EV,
	PTHREAD_DETACH_EV, PTHREAD_FUNC_EV, PTHREAD_RWLOCK_WR_EV, PTHREAD_RWLOCK_RD_EV,
	PTHREAD_RWLOCK_UNLOCK_EV, PTHREAD_MUTEX_LOCK_EV, PTHREAD_MUTEX_UNLOCK_EV,
	PTHREAD_COND_SIGNAL_EV, PTHREAD_COND_BROADCAST_EV, PTHREAD_COND_WAIT_EV,
	PTHREAD_EXIT_EV, PTHREAD_BARRIER_WAIT_EV };

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

#define CUDA_EVENTS 14
static unsigned cuda_events[] = {
	/* Host events */
	CUDALAUNCH_EV, CUDACONFIGCALL_EV,
	CUDAMEMCPY_EV, CUDAMEMCPYASYNC_EV, CUDATHREADBARRIER_EV,
	CUDASTREAMBARRIER_EV, CUDASTREAMCREATE_EV, CUDADEVICERESET_EV,
	CUDATHREADEXIT_EV,
	/* Accelerator events */
    CUDAKERNEL_GPU_EV, CUDACONFIGKERNEL_GPU_EV, CUDAMEMCPY_GPU_EV,
	CUDAMEMCPYASYNC_GPU_EV, CUDATHREADBARRIER_GPU_EV };

/******************************************************************************
 ***  IsCUDA
 ******************************************************************************/

unsigned IsCUDA (unsigned EvType)
{
  unsigned evt;

  for (evt = 0; evt < CUDA_EVENTS; evt++)
    if (cuda_events[evt] == EvType)
      return TRUE;
  return FALSE;
}

#define OPENCL_EVENTS 65
static unsigned opencl_events[] = {
	OPENCL_CLCREATEBUFFER_EV, OPENCL_CLCREATECOMMANDQUEUE_EV, 
	OPENCL_CLCREATECONTEXT_EV, OPENCL_CLCREATECONTEXTFROMTYPE_EV,
	OPENCL_CLCREATESUBBUFFER_EV, OPENCL_CLCREATEKERNEL_EV,
	OPENCL_CLCREATEKERNELSINPROGRAM_EV, OPENCL_CLSETKERNELARG_EV,
	OPENCL_CLCREATEPROGRAMWITHSOURCE_EV, OPENCL_CLCREATEPROGRAMWITHBINARY_EV,
	OPENCL_CLCREATEPROGRAMWITHBUILTINKERNELS_EV, OPENCL_CLENQUEUEFILLBUFFER_EV,
	OPENCL_CLENQUEUECOPYBUFFER_EV, OPENCL_CLENQUEUECOPYBUFFERRECT_EV,
	OPENCL_CLENQUEUENDRANGEKERNEL_EV, OPENCL_CLENQUEUETASK_EV,
	OPENCL_CLENQUEUENATIVEKERNEL_EV, OPENCL_CLENQUEUEREADBUFFER_EV,
	OPENCL_CLENQUEUEREADBUFFERRECT_EV, OPENCL_CLENQUEUEWRITEBUFFER_EV,
	OPENCL_CLENQUEUEWRITEBUFFERRECT_EV, OPENCL_CLBUILDPROGRAM_EV,
	OPENCL_CLCOMPILEPROGRAM_EV, OPENCL_CLLINKPROGRAM_EV,
	OPENCL_CLFINISH_EV, OPENCL_CLFLUSH_EV, OPENCL_CLWAITFOREVENTS_EV,
	OPENCL_CLENQUEUEMARKERWITHWAITLIST_EV,
	OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV,	OPENCL_CLENQUEUEMAPBUFFER_EV,
	OPENCL_CLENQUEUEUNMAPMEMOBJECT_EV, OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_EV,
	OPENCL_CLENQUEUEMARKER_EV, OPENCL_CLENQUEUEBARRIER_EV,
	OPENCL_CLENQUEUEFILLBUFFER_ACC_EV, OPENCL_CLENQUEUECOPYBUFFER_ACC_EV,
	OPENCL_CLENQUEUECOPYBUFFERRECT_ACC_EV,
	OPENCL_CLENQUEUENDRANGEKERNEL_ACC_EV, OPENCL_CLENQUEUETASK_ACC_EV,
	OPENCL_CLENQUEUENATIVEKERNEL_ACC_EV, OPENCL_CLENQUEUEREADBUFFER_ACC_EV,
	OPENCL_CLENQUEUEREADBUFFERRECT_ACC_EV, 
	OPENCL_CLENQUEUEWRITEBUFFER_ACC_EV,
	OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV,
	OPENCL_CLENQUEUEMARKERWITHWAITLIST_ACC_EV, 
	OPENCL_CLENQUEUEBARRIERWITHWAITLIST_ACC_EV,
	OPENCL_CLENQUEUEMAPBUFFER_ACC_EV,
	OPENCL_CLENQUEUEUNMAPMEMOBJECT_ACC_EV,
	OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_ACC_EV,
	OPENCL_CLENQUEUEMARKER_ACC_EV, OPENCL_CLENQUEUEBARRIER_ACC_EV,
	OPENCL_CLRETAINCOMMANDQUEUE_EV, OPENCL_CLRELEASECOMMANDQUEUE_EV,
	OPENCL_CLRETAINCONTEXT_EV, OPENCL_CLRELEASECONTEXT_EV,
	OPENCL_CLRETAINDEVICE_EV, OPENCL_CLRELEASEDEVICE_EV,
	OPENCL_CLRETAINEVENT_EV, OPENCL_CLRELEASEEVENT_EV,
	OPENCL_CLRETAINKERNEL_EV, OPENCL_CLRELEASEKERNEL_EV,
	OPENCL_CLRETAINMEMOBJECT_EV, OPENCL_CLRELEASEMEMOBJECT_EV,
	OPENCL_CLRETAINPROGRAM_EV, OPENCL_CLRELEASEPROGRAM_EV
};


/******************************************************************************
 ***  IsOpenCL
 ******************************************************************************/

unsigned IsOpenCL (unsigned EvType)
{
  unsigned evt;

  for (evt = 0; evt < OPENCL_EVENTS; evt++)
    if (opencl_events[evt] == EvType)
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
 ***  IsMPICollective
 ******************************************************************************/
unsigned IsMPICollective(unsigned EvType)
{
   switch (EvType)
   {
      case MPI_BARRIER_EV:
      case MPI_BCAST_EV:
      case MPI_ALLTOALL_EV:
      case MPI_ALLTOALLV_EV:
      case MPI_REDUCE_EV:
      case MPI_ALLREDUCE_EV:
      case MPI_GATHER_EV:
      case MPI_GATHERV_EV:
      case MPI_ALLGATHER_EV:
      case MPI_ALLGATHERV_EV:
      case MPI_SCATTER_EV:
      case MPI_SCATTERV_EV:
      case MPI_REDUCESCAT_EV:
      case MPI_SCAN_EV:
         return TRUE;
      default:
         return FALSE;
   }
   return FALSE;
}

/******************************************************************************
 ***  IsPACXCollective
 ******************************************************************************/
unsigned IsPACXCollective(unsigned EvType)
{
   switch (EvType)
   {
      case PACX_BARRIER_EV:
      case PACX_BCAST_EV:
      case PACX_ALLTOALL_EV:
      case PACX_ALLTOALLV_EV:
      case PACX_REDUCE_EV:
      case PACX_ALLREDUCE_EV:
      case PACX_GATHER_EV:
      case PACX_GATHERV_EV:
      case PACX_ALLGATHER_EV:
      case PACX_ALLGATHERV_EV:
      case PACX_SCATTER_EV:
      case PACX_SCATTERV_EV:
      case PACX_REDUCESCAT_EV:
      case PACX_SCAN_EV:
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
	else if (IsPACX(EvType))
	{
		*Type = PACX_TYPE;
		return TRUE;
	}
	else if (IsCUDA(EvType))
	{
		*Type = CUDA_TYPE;
		return TRUE;
	}
	else if (IsOpenCL(EvType))
	{
		*Type = OPENCL_TYPE;
		return TRUE;
	}
	else if (EvType == MPI_ALIAS_COMM_CREATE_EV)
	{
		*Type = MPI_COMM_ALIAS_TYPE;
		return TRUE;
	}
	else if (EvType == PACX_ALIAS_COMM_CREATE_EV)
	{
		*Type = PACX_COMM_ALIAS_TYPE;
		return TRUE;
	}
	return FALSE;
}

