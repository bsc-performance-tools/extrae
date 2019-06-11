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

/* Additional credis to Vince Weaver, vincent.weaver _at_ maine.edu 
   For further reference see https://github.com/deater/perf_event_tests
   particularly sample_weight.c and sample_data_src.c */

#define _GNU_SOURCE 1

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <assert.h>

#include "sampling-common.h"
#include "sampling-intel-pebs.h"

#include "wrapper.h"
#include "trace_macros.h"

#ifndef __NR_perf_event_open
# if defined(__i386__)
#  define __NR_perf_event_open    336
# elif defined(__x86_64__)
#  define __NR_perf_event_open    298
# endif
#endif

static int PEBS_enabled = 0;
static int pebs_init_threads = 0;
static pthread_mutex_t pebs_init_lock = PTHREAD_MUTEX_INITIALIZER;
static int _PEBS_sampling_paused = FALSE;

#define LOAD_SAMPLE_TYPE         ( PERF_SAMPLE_IP | PERF_SAMPLE_WEIGHT | PERF_SAMPLE_DATA_SRC | PERF_SAMPLE_ADDR )
#define STORE_SAMPLE_TYPE        ( PERF_SAMPLE_IP | PERF_SAMPLE_DATA_SRC | PERF_SAMPLE_ADDR )
#define LOAD_L3M_SAMPLE_TYPE     ( PERF_SAMPLE_IP | PERF_SAMPLE_ADDR )

typedef enum SamplingType_e {
	LOAD_INDEX = 0,
	STORE_INDEX,
	LOAD_L3M_INDEX,
	NUM_SAMPLING_TYPES }
SamplingType_t;

#ifdef HAVE_LINUX_PERF_EVENT_H
# include <linux/perf_event.h>
#else
# error Missing linux/perf_event.h header file to compile PEBS support
#endif

int perf_event_open(struct perf_event_attr *hw_event_uptr, pid_t pid, int cpu,
	int group_fd, unsigned long flags)
{
	return syscall (__NR_perf_event_open,hw_event_uptr, pid, cpu, group_fd, flags);
}

#if defined(__x86_64)
# if defined(__KNC__)
#  define rmb() __sync_synchronize()
# else
#  define rmb() asm volatile("lfence":::"memory")
# endif
#else
# error Need to define rmb for this architecture!
# error See the kernel source directory: tools/perf/perf.h file
#endif

static int processor_type=-2;
static int processor_vendor=-2;

#define VENDOR_UNKNOWN -1
#define VENDOR_INTEL    1

#define PROCESSOR_UNKNOWN		-1
#define PROCESSOR_PENTIUM_PRO		1
#define PROCESSOR_PENTIUM_II		2
#define PROCESSOR_PENTIUM_III		3
#define PROCESSOR_PENTIUM_4		4
#define PROCESSOR_PENTIUM_M		5
#define PROCESSOR_COREDUO		6
#define PROCESSOR_CORE2			7
#define PROCESSOR_NEHALEM		8
#define PROCESSOR_NEHALEM_EX		9
#define PROCESSOR_WESTMERE		10
#define PROCESSOR_WESTMERE_EX		11
#define PROCESSOR_SANDYBRIDGE		12
#define PROCESSOR_ATOM			13
#define PROCESSOR_IVYBRIDGE		20
#define PROCESSOR_KNIGHTSCORNER		21
#define PROCESSOR_SANDYBRIDGE_EP	22
#define PROCESSOR_IVYBRIDGE_EP		24
#define PROCESSOR_HASWELL		25
#define PROCESSOR_ATOM_CEDARVIEW	26
#define PROCESSOR_ATOM_SILVERMONT	27
#define PROCESSOR_BROADWELL		28
#define PROCESSOR_HASWELL_EP		29
#define PROCESSOR_KNIGHTS_LANDING       30
#define PROCESSOR_SKYLAKE		31


static int detect_processor_cpuinfo(void)
{
	int cpu_family=0,model=0;
	char string[BUFSIZ];

	FILE * procinfo_file = fopen("/proc/cpuinfo","r");
	if (procinfo_file == NULL)
	{
		fprintf (stderr, PACKAGE_NAME ": Error! Can't open /proc/cpuinfo\n");
		return PROCESSOR_UNKNOWN;
	}

	while (TRUE)
	{
		if (fgets(string,BUFSIZ,procinfo_file)==NULL)
			break;

		/* vendor */
		if (strstr(string,"vendor_id"))
			if (strstr(string,"GenuineIntel"))
				processor_vendor=VENDOR_INTEL;

		/* family */
		if (strstr(string,"cpu family"))
			sscanf(string,"%*s %*s %*s %d",&cpu_family);

		/* model */
		if ((strstr(string,"model")) && (!strstr(string,"model name")))
			sscanf(string,"%*s %*s %d",&model);
	}

	fclose (procinfo_file);

	if (processor_vendor==VENDOR_INTEL)
	{
		if (cpu_family==6)
		{
			switch(model)
			{
				case 1:
					processor_type=PROCESSOR_PENTIUM_PRO;
					break;
				case 3:
				case 5:
				case 6:
					processor_type=PROCESSOR_PENTIUM_II;
					break;
				case 7:
				case 8:
				case 10:
				case 11:
					processor_type=PROCESSOR_PENTIUM_III;
					break;
				case 9:
				case 13:
					processor_type=PROCESSOR_PENTIUM_M;
					break;
				case 14:
					processor_type=PROCESSOR_COREDUO;
					break;
				case 15:
				case 22:
				case 23:
				case 29:
					processor_type=PROCESSOR_CORE2;
					break;
				case 28:
				case 38:
				case 39:
				case 53:
					processor_type=PROCESSOR_ATOM;
					break;
				case 54:
					processor_type=PROCESSOR_ATOM_CEDARVIEW;
					break;
				case 55:
				case 77:
					processor_type=PROCESSOR_ATOM_SILVERMONT;
					break;
				case 26:
				case 30:
				case 31:
					processor_type=PROCESSOR_NEHALEM;
					break;
				case 46:
					processor_type=PROCESSOR_NEHALEM_EX;
					break;
				case 37:
				case 44:
					processor_type=PROCESSOR_WESTMERE;
					break;
				case 47:
					processor_type=PROCESSOR_WESTMERE_EX;
					break;
				case 42:
					processor_type=PROCESSOR_SANDYBRIDGE;
					break;
				case 45:
					processor_type=PROCESSOR_SANDYBRIDGE_EP;
					break;
				case 58:
					processor_type=PROCESSOR_IVYBRIDGE;
					break;
				case 62:
					processor_type=PROCESSOR_IVYBRIDGE_EP;
					break;
				case 60:
				case 69:
				case 70:
					processor_type=PROCESSOR_HASWELL;
					break;
				case 63:
					processor_type=PROCESSOR_HASWELL_EP;
					break;
				case 61:
				case 71:
				case 79:
					processor_type=PROCESSOR_BROADWELL;
					break;
				case 85:
					processor_type=PROCESSOR_SKYLAKE;
					break;
				case 87:
					processor_type=PROCESSOR_KNIGHTS_LANDING;
					break;
				default:
					processor_type=PROCESSOR_UNKNOWN;
					break;
			}
			return 0;
		}

		if (cpu_family == 11)
		{
			processor_type=PROCESSOR_KNIGHTSCORNER;
			return 0;
		}

		if (cpu_family == 15)
		{
			processor_type=PROCESSOR_PENTIUM_4;
			return 0;
		}
	}

	processor_type=PROCESSOR_UNKNOWN;

	return 0;
}

static int detect_processor(void)
{
	if (processor_type==-2)
		detect_processor_cpuinfo();

	return processor_type;
}

static int get_latency_load_event(unsigned long long *config)
{
	int processor_notfound = 0;

	switch (detect_processor())
	{
		case PROCESSOR_SANDYBRIDGE:
		case PROCESSOR_SANDYBRIDGE_EP:
		case PROCESSOR_IVYBRIDGE:
		case PROCESSOR_IVYBRIDGE_EP:
		case PROCESSOR_HASWELL:
		case PROCESSOR_HASWELL_EP:
		case PROCESSOR_BROADWELL:
		case PROCESSOR_SKYLAKE:
			*config=0x1cd; // MEM_TRANS_RETIRED.LOAD_LATENCY
			break;
		case PROCESSOR_KNIGHTS_LANDING:
			*config=0x0404; /* Use 0x0204 for "MEM_UOPS_RETIRED:L2_HIT_LOADS; 0x0404 for "MEM_UOPS_RETIRED:L2_MISS_LOADS */
			break;
		default:
			*config=0x0;
			processor_notfound=-1;
			break;
	}
	return processor_notfound;
}

static int get_store_event (unsigned long long *config)
{
	int processor_notfound = 0;

	switch (detect_processor())
	{
		case PROCESSOR_SANDYBRIDGE:
		case PROCESSOR_SANDYBRIDGE_EP: 
		case PROCESSOR_IVYBRIDGE:
		case PROCESSOR_IVYBRIDGE_EP:
			*config=0x2cd; // MEM_TRANS_RETIRED.PRECISE_STORE
			break;
		case PROCESSOR_HASWELL:
		case PROCESSOR_HASWELL_EP:
		case PROCESSOR_BROADWELL:
			*config=0x82d0; // MEM_UOPS_RETIRED.ALL_STORES
			break;
		case PROCESSOR_SKYLAKE:
			*config=0x82d0; // MEM_INST_RETIRED.ALL_STORES
			break;
	        case PROCESSOR_KNIGHTS_LANDING:
		default:
			*config=0x0;
			processor_notfound=-1;
			break;
	}
	return processor_notfound;
}

static int get_load_l3m_event (unsigned long long *config)
{
	int processor_notfound = 0;

	switch (detect_processor())
	{
		case PROCESSOR_SKYLAKE:
			*config= 0x20d1; // MEM_LOAD_RETIRED.L3_MISS
			break;
		default:
			*config=0x0;
			processor_notfound=-1;
			break;
	}
	return processor_notfound;
}

static int PEBS_load_operates_in_frequency_mode     = 0;
static int PEBS_store_operates_in_frequency_mode    = 0;
static int PEBS_load_l3m_operates_in_frequency_mode = 0;
static int PEBS_load_frequency                      = 100;
static int PEBS_store_frequency                     = 100;
static int PEBS_load_l3m_frequency                  = 100;
static int PEBS_load_period                         = 1000000;
static int PEBS_store_period                        = 1000000;
static int PEBS_load_l3m_period                     = 1000000;

static int PEBS_load_enabled = FALSE;
static int PEBS_store_enabled = FALSE;
static int PEBS_load_l3m_enabled = FALSE;
static int PEBS_minimumLoadLatency = 3;

void Extrae_IntelPEBS_setLoadFrequency(int frequency)
{ 
  PEBS_load_operates_in_frequency_mode = 1;
  PEBS_load_frequency = frequency; 
}

void Extrae_IntelPEBS_setLoadPeriod(int period)
{ 
  PEBS_load_operates_in_frequency_mode = 0;
  PEBS_load_period = period;
}

void Extrae_IntelPEBS_setStoreFrequency (int frequency)
{ 
  PEBS_store_operates_in_frequency_mode = 1;
  PEBS_store_frequency = frequency; 
}

void Extrae_IntelPEBS_setStorePeriod(int period)
{
  PEBS_store_operates_in_frequency_mode = 0;
  PEBS_store_period = period;
}

void Extrae_IntelPEBS_setLoadL3MFrequency (int frequency)
{ 
  PEBS_load_l3m_operates_in_frequency_mode = 1;
  PEBS_load_l3m_frequency = frequency; 
}

void Extrae_IntelPEBS_setLoadL3MPeriod(int period)
{
  PEBS_load_l3m_operates_in_frequency_mode = 0;
  PEBS_load_l3m_period = period;
}

void Extrae_IntelPEBS_setLoadSampling (int enabled)
{ PEBS_load_enabled = enabled; }

void Extrae_IntelPEBS_setMinimumLoadLatency (int numcycles)
{ PEBS_minimumLoadLatency = numcycles; }

void Extrae_IntelPEBS_setStoreSampling (int enabled)
{ PEBS_store_enabled = enabled; }

void Extrae_IntelPEBS_setLoadL3MSampling (int enabled)
{ PEBS_load_l3m_enabled = enabled; }

#define MMAP_DATA_SIZE 8

static void ***extrae_intel_pebs_mmap = NULL;
static long long **prev_head = NULL;
static int **perf_pebs_fd = NULL;
static int mmap_pages=1+MMAP_DATA_SIZE;
static int *group_fd = NULL;

#define ALLOCATED_SIZE MMAP_DATA_SIZE*4096
static char **data_thread_buffer = NULL;

/* This function extracts PEBS entries from the previously allocated buffer 
   in data ptr. At this moment, we only should have 1 event in the buffer */
static long long extrae_perf_mmap_read_pebs (void *extrae_intel_pebs_mmap_thread,
	long long prev_head, int sample_type,
	long long *ip, long long *addr, long long *weight,
	union perf_mem_data_src *data_src)
{
	struct perf_event_mmap_page *control_page = extrae_intel_pebs_mmap_thread;
	long long head = control_page->data_head;
	rmb(); /* Must always follow read of data_head */
	void *data_mmap = extrae_intel_pebs_mmap_thread+sysconf(_SC_PAGESIZE);
	int size = head-prev_head;
#if defined(HAVE_PERF_EVENT_MMAP_PAGE_DATA_SIZE)
	long long bytesize = control_page->data_size;
#else
	long long bytesize = mmap_pages*sysconf(_SC_PAGESIZE);
#endif

	if (size > bytesize)
		fprintf (stderr, PACKAGE_NAME": Error! overflowed the mmap buffer %d>%lld bytes\n", size, bytesize);

	char * data = data_thread_buffer[THREADID];
	if (bytesize > ALLOCATED_SIZE)
	{
		fprintf (stderr, PACKAGE_NAME": Error! overflow in the allocated size for PEBS buffer\n");
		return -1;
	}

	long long prev_head_wrap = prev_head % bytesize;
	memcpy (data,(unsigned char*)data_mmap + prev_head_wrap, bytesize-prev_head_wrap);
	memcpy(data+(bytesize-prev_head_wrap),(unsigned char *)data_mmap, prev_head_wrap);

	long long offset = 0;

	while (offset < size)
	{
		struct perf_event_header * event = (struct perf_event_header *) & data[offset];

		offset += 8; /* skip header */

		/***********************/
		/* Print event Details */
		/***********************/

		if (event->type == PERF_RECORD_SAMPLE)
		{

			if (sample_type & PERF_SAMPLE_IP)
			{
				if (ip != NULL)
				{
					*ip = *((long long *) &data[offset]);
#if defined(DEBUG)
					printf("\tPERF_SAMPLE_IP, IP: %llx\n",*ip);
#endif
				}
				offset += 8;
			}
			if (sample_type & PERF_SAMPLE_ADDR)
			{
				if (addr != NULL)
				{
					*addr = *((long long *) &data[offset]);
#if defined(DEBUG)
					printf ("\tPERF_SAMPLE_ADDR, addr: %llx\n",*addr);
#endif
				}
				offset += 8;
			}
			if (sample_type & PERF_SAMPLE_WEIGHT)
			{
				if (weight != NULL)
				{
					*weight = *((long long *) &data[offset]);
#if defined(DEBUG)
					printf ("\tPERF_SAMPLE_WEIGHT, Weight: %lld\n",*weight);
#endif
				}
				offset += 8;
			}
			if (sample_type & PERF_SAMPLE_DATA_SRC)
			{
				if (data_src != NULL)
					memcpy (data_src, &data[offset],
					  sizeof(union perf_mem_data_src));

				offset += sizeof(union perf_mem_data_src);
			}
		}
		else
		{
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": Unhandled perf record type %d\n",
			  event->type);
#endif
		}
	}

	control_page->data_tail = head;

	return head;
}

/* This handler will deal with the PEBS buffer when it monitors LOADS
   and the buffer is full (1 entry only). It emits everythin Extrae needs
   timestamp, reference to memory, portion of the memory hierarchy that
   provides it, the access cost. */
static void extrae_intel_pebs_handler_load (int threadid)
{
	void * _mmap = extrae_intel_pebs_mmap[threadid][LOAD_INDEX];
	if (_mmap)
	{
		long long ip, addr = 0, weight;
		union perf_mem_data_src data_src;

		prev_head[threadid][LOAD_INDEX] = extrae_perf_mmap_read_pebs (_mmap, 
		  prev_head[threadid][LOAD_INDEX], LOAD_SAMPLE_TYPE, &ip, &addr, &weight, &data_src);

		if (tracejant && Extrae_isSamplingEnabled() && !Backend_inInstrumentation(threadid) && addr != 0)
		{
			unsigned memlevel, memhitormiss;
			unsigned tlblevel, tlbhitormiss;

			/* see linux/perf_event.h perf_mem_data_src */
			if (data_src.mem_lvl & PERF_MEM_LVL_HIT)
				memhitormiss = PEBS_MEMORYHIERARCHY_HIT;
			else if (data_src.mem_lvl & PERF_MEM_LVL_MISS)
				memhitormiss = PEBS_MEMORYHIERARCHY_MISS;
			else
				memhitormiss = PEBS_MEMORYHIERARCHY_UNKNOWN;
	
			if (data_src.mem_dtlb & PERF_MEM_TLB_HIT)
				tlbhitormiss = PEBS_MEMORYHIERARCHY_HIT;
			else if (data_src.mem_dtlb & PERF_MEM_TLB_MISS)
				tlbhitormiss = PEBS_MEMORYHIERARCHY_MISS;
			else
				tlbhitormiss = PEBS_MEMORYHIERARCHY_UNKNOWN;

			if (data_src.mem_lvl & PERF_MEM_LVL_L1)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_L1;
			else if (data_src.mem_lvl & PERF_MEM_LVL_LFB)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_LFB;
			else if (data_src.mem_lvl & PERF_MEM_LVL_L2)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_L2;
			else if (data_src.mem_lvl & PERF_MEM_LVL_L3)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_L3;
			else if (data_src.mem_lvl & PERF_MEM_LVL_REM_CCE1)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_RCACHE_1HOP;
			else if (data_src.mem_lvl & PERF_MEM_LVL_REM_CCE2)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_RCACHE_2HOP;
			else if (data_src.mem_lvl & PERF_MEM_LVL_LOC_RAM)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_LOCAL_RAM;
			else if (data_src.mem_lvl & PERF_MEM_LVL_REM_RAM1)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_REMOTE_RAM_1HOP;
			else if (data_src.mem_lvl & PERF_MEM_LVL_REM_RAM2)
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_REMOTE_RAM_2HOP;
			else
				memlevel = PEBS_MEMORYHIERARCHY_UNCACHEABLE_IO;
	
			/* PATCH #0 if data comes from dram, it can't be a hit! */
			/* Seems unclear, but from table 18-19:
			   http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3b-part-2-manual.pdf
			*/
			if (data_src.mem_lvl & PERF_MEM_LVL_LOC_RAM ||
			    data_src.mem_lvl & PERF_MEM_LVL_REM_RAM1 ||
			    data_src.mem_lvl & PERF_MEM_LVL_REM_RAM2)
			{
				memhitormiss = PEBS_MEMORYHIERARCHY_MISS;
			}
	
			/* PATCH #1 if data l3 & miss == data served by dram */
			/* Seems unclear, but from table 18-19:
			   http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3b-part-2-manual.pdf
			*/
			if (data_src.mem_lvl & PERF_MEM_LVL_MISS &&
			    data_src.mem_lvl & PERF_MEM_LVL_L3)
			{
				memhitormiss = PEBS_MEMORYHIERARCHY_MISS;
				memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_LOCAL_RAM;
			}
	
			if (data_src.mem_dtlb & PERF_MEM_TLB_L1)
				tlblevel = PEBS_MEMORYHIERARCHY_TLB_L1;
			else if (data_src.mem_dtlb & PERF_MEM_TLB_L2)
				tlblevel = PEBS_MEMORYHIERARCHY_TLB_L2;
			else
				tlblevel = PEBS_MEMORYHIERARCHY_TLB_OTHER;
	
			UINT64 t = Clock_getCurrentTime_nstore();
	
			SAMPLE_EVENT_HWC_PARAM(t, SAMPLING_ADDRESS_LD_EV, ip, addr);
			SAMPLE_EVENT_NOHWC_PARAM(t, SAMPLING_ADDRESS_MEM_LEVEL_EV, memhitormiss,
			  memlevel);
			SAMPLE_EVENT_NOHWC_PARAM(t, SAMPLING_ADDRESS_TLB_LEVEL_EV, tlbhitormiss,
			  tlblevel);
			SAMPLE_EVENT_NOHWC(t, SAMPLING_ADDRESS_REFERENCE_COST_EV, weight);
	
			Extrae_trace_callers (t, 5, CALLER_SAMPLING); 
		}
	}
}

/* This handler will deal with the PEBS buffer when it monitors STORES
   and the buffer is full (1 entry only). It emits everythin Extrae needs
   timestamp  and reference to memory. */
static void extrae_intel_pebs_handler_store (int threadid)
{
	void * _mmap = extrae_intel_pebs_mmap[threadid][STORE_INDEX];
	if (_mmap)
	{
		long long ip, addr = 0;
		union perf_mem_data_src data_src;

		prev_head[threadid][STORE_INDEX] = extrae_perf_mmap_read_pebs (_mmap,
		  prev_head[threadid][STORE_INDEX], STORE_SAMPLE_TYPE, &ip, &addr, NULL, &data_src);
	
		if (tracejant && Extrae_isSamplingEnabled() && !Backend_inInstrumentation(threadid) && addr != 0)
		{
			unsigned memlevel, memhitormiss;
		
			if (data_src.mem_lvl & PERF_MEM_LVL_HIT)
				memhitormiss = PEBS_MEMORYHIERARCHY_HIT;
			else if (data_src.mem_lvl & PERF_MEM_LVL_MISS)
				memhitormiss = PEBS_MEMORYHIERARCHY_MISS;
			else
				memhitormiss = PEBS_MEMORYHIERARCHY_UNKNOWN;
			
			memlevel = PEBS_MEMORYHIERARCHY_MEM_LVL_L1;

			UINT64 t = Clock_getCurrentTime_nstore();
		
			SAMPLE_EVENT_HWC_PARAM(t, SAMPLING_ADDRESS_ST_EV, ip, addr);
			SAMPLE_EVENT_NOHWC_PARAM(t, SAMPLING_ADDRESS_MEM_LEVEL_EV, memhitormiss, memlevel);
			Extrae_trace_callers (t, 5, CALLER_SAMPLING); 
		}
	}
}

/* This handler will deal with the PEBS buffer when it monitors LOAD L3 MISSES
   and the buffer is full (1 entry only). It emits everythin Extrae needs
   timestamp  and reference to memory. */
static void extrae_intel_pebs_handler_load_l3m (int threadid)
{
	void * _mmap = extrae_intel_pebs_mmap[threadid][LOAD_L3M_INDEX];
	if (_mmap)
	{
		long long ip, addr = 0;

		prev_head[threadid][LOAD_L3M_INDEX] = extrae_perf_mmap_read_pebs (_mmap,
		  prev_head[threadid][LOAD_L3M_INDEX], LOAD_L3M_SAMPLE_TYPE, &ip, &addr, NULL, NULL);
	
		if (tracejant && Extrae_isSamplingEnabled() && !Backend_inInstrumentation(threadid) && addr != 0)
		{
			UINT64 t = Clock_getCurrentTime_nstore();
			SAMPLE_EVENT_HWC_PARAM(t, SAMPLING_ADDRESS_LD_EV, ip, addr);
			SAMPLE_EVENT_NOHWC_PARAM(t, SAMPLING_ADDRESS_MEM_LEVEL_EV, PEBS_MEMORYHIERARCHY_MISS,
			  PEBS_MEMORYHIERARCHY_MEM_LVL_L3);
			Extrae_trace_callers (t, 5, CALLER_SAMPLING); 
		}
	}
}

static void extrae_intel_pebs_handler (int signum, siginfo_t *info, void *uc)
{
	int ret;
	int thid = THREADID;

	UNREFERENCED_PARAMETER(signum);
	UNREFERENCED_PARAMETER(uc);

	// If there's a thread initializing, do not emit the sample as the
	// reallocated data might have changed its addres.
	// Use pthread_mutex_trylock to avoid locking if mutex is already locked.
	if (pthread_mutex_trylock (&pebs_init_lock) == 0)
	{
		if (info->si_fd == perf_pebs_fd[thid][LOAD_INDEX])
			extrae_intel_pebs_handler_load (thid);
		else if (info->si_fd == perf_pebs_fd[thid][STORE_INDEX])
			extrae_intel_pebs_handler_store (thid);
		else if (info->si_fd == perf_pebs_fd[thid][LOAD_L3M_INDEX])
			extrae_intel_pebs_handler_load_l3m (thid);
		pthread_mutex_unlock (&pebs_init_lock);
	}

	// restart sampling on the given counter
	// If user did not request loads, try with stores
	// int group_fd = perf_pebs_fd[thid][LOAD_INDEX] >= 0 ? perf_pebs_fd[thid][LOAD_INDEX] :
	// 				(perf_pebs_fd[thid][STORE_INDEX] >= 0 ? perf_pebs_fd[thid][STORE_INDEX] : perf_pebs_fd[thid][LOAD_L3M_INDEX]);
	// ret = ioctl (group_fd, PERF_EVENT_IOC_REFRESH, 1);
	
	// Rather than restarting the group leader, we restart this specific counter
	ret = ioctl (info->si_fd, PERF_EVENT_IOC_REFRESH, 1);
	(void) ret;
}

/* Extrae_IntelPEBS_enable (void)
   initializes the sampling based on PEBS. If loads is TRUE, then the PEBS is
   setup to monitor LOAD instructions, otherwise it monitors STORE instructions.
*/
static int Extrae_IntelPEBS_enable (void)
{
	__u64 hwc;
	int ret;
	struct perf_event_attr pe;
	struct sigaction sa;
	int thread_id = THREADID;

	if (!PEBS_load_enabled && !PEBS_store_enabled && !PEBS_load_l3m_enabled)
		return 0;

	/* Need a lock as different threads may be initializing, thus, allocating structures simultaneously */
	pthread_mutex_lock (&pebs_init_lock);
	if (thread_id >= pebs_init_threads)
	{
		int i;

		/* Extend the data structures to the maximum number of threads seen so far */
		extrae_intel_pebs_mmap = (void ***) realloc(extrae_intel_pebs_mmap, (thread_id+1) * sizeof(void **));
		assert (extrae_intel_pebs_mmap);

		perf_pebs_fd = (int **)realloc(perf_pebs_fd, (thread_id+1) * sizeof(int*));
		assert (perf_pebs_fd);

		prev_head = (long long **) realloc (prev_head, (thread_id+1) * sizeof(long long*));
		assert (prev_head);

		group_fd = (int*) realloc (group_fd, (thread_id+1) * sizeof(int));
		assert (group_fd);

		data_thread_buffer = (char **) realloc (data_thread_buffer, (thread_id+1) * sizeof(char *));
		assert (data_thread_buffer);

		for (i=pebs_init_threads; i<(thread_id+1); i++)
		{
			extrae_intel_pebs_mmap[i] = malloc (sizeof(void*)*NUM_SAMPLING_TYPES);
			assert (extrae_intel_pebs_mmap[i]);
			extrae_intel_pebs_mmap[i][LOAD_INDEX] =
			  extrae_intel_pebs_mmap[i][STORE_INDEX] =
			  extrae_intel_pebs_mmap[i][LOAD_L3M_INDEX] = NULL;

			perf_pebs_fd[i] = malloc (sizeof(int)*NUM_SAMPLING_TYPES);
			assert (perf_pebs_fd[i]);
			perf_pebs_fd[i][LOAD_INDEX] =
			  perf_pebs_fd[i][STORE_INDEX] =
			  perf_pebs_fd[i][LOAD_L3M_INDEX] = -1;

			prev_head[i] = (long long*) malloc  (sizeof (long long)*NUM_SAMPLING_TYPES);
			assert (prev_head[i]);
			prev_head[i][LOAD_INDEX] =
			  prev_head[i][STORE_INDEX] =
			  prev_head[i][LOAD_L3M_INDEX] = 0;

			group_fd[i] = -1;

			data_thread_buffer[i] = malloc (ALLOCATED_SIZE);
			assert (data_thread_buffer[i]);
		}

		pebs_init_threads = thread_id+1;
	}
	pthread_mutex_unlock (&pebs_init_lock);

	memset(&sa, 0, sizeof(struct sigaction));
	sa.sa_sigaction = extrae_intel_pebs_handler;
	sa.sa_flags = SA_SIGINFO;
	if (sigaction( SIGIO, &sa, NULL) < 0)
	{
		fprintf (stderr, PACKAGE_NAME": Error setting up signal handler\n");
		return -1;
	}

	struct f_owner_ex owner;
	owner.type = F_OWNER_TID;
	owner.pid = syscall(SYS_gettid);

	if (PEBS_load_enabled && get_latency_load_event (&hwc) >= 0)
	{
		memset (&pe,0,sizeof(struct perf_event_attr));
		pe.config = hwc;
		pe.type = PERF_TYPE_RAW;
		pe.config1 = PEBS_minimumLoadLatency;
		pe.size = sizeof(struct perf_event_attr);
		pe.precise_ip = 2;
		pe.sample_type = LOAD_SAMPLE_TYPE;
		pe.disabled = 1;
		pe.pinned = 1;
		pe.exclude_kernel = 1;
		pe.exclude_hv = 1;
		pe.wakeup_events = 1;
		if (PEBS_load_operates_in_frequency_mode)
		{
			pe.freq = 1;
			pe.sample_freq = PEBS_load_frequency;
		}
		else // operates in period mode by default
		{
			pe.freq = 0;
			pe.sample_period = PEBS_load_period;
		}
	
		group_fd[thread_id] = perf_pebs_fd[thread_id][LOAD_INDEX] = perf_event_open (&pe, 0, -1, -1, 0);
		if (perf_pebs_fd[thread_id][LOAD_INDEX] < 0)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot open the perf_event file descriptor for loads\n");
			return -1;
		}

		extrae_intel_pebs_mmap[thread_id][LOAD_INDEX] = mmap (NULL, mmap_pages*sysconf(_SC_PAGESIZE),
		  PROT_READ|PROT_WRITE, MAP_SHARED, perf_pebs_fd[thread_id][LOAD_INDEX], 0);
		if (extrae_intel_pebs_mmap[thread_id][LOAD_INDEX] == MAP_FAILED)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot mmap for load events\n");
			close (perf_pebs_fd[thread_id][LOAD_INDEX]);
			return -1;
		}
	
		fcntl(perf_pebs_fd[thread_id][LOAD_INDEX], F_SETFL, fcntl(perf_pebs_fd[thread_id][LOAD_INDEX], F_GETFL, 0) | O_ASYNC);
		fcntl(perf_pebs_fd[thread_id][LOAD_INDEX], F_SETSIG, SIGIO);
		fcntl(perf_pebs_fd[thread_id][LOAD_INDEX], F_SETOWN, getpid());
		fcntl(perf_pebs_fd[thread_id][LOAD_INDEX], F_SETOWN_EX, &owner);
	}

	if (PEBS_store_enabled && get_store_event (&hwc) >= 0)
	{
		memset (&pe,0,sizeof(struct perf_event_attr));
		pe.config = hwc;
		pe.type = PERF_TYPE_RAW;
		pe.size = sizeof(struct perf_event_attr);
		pe.precise_ip = 2;
		pe.sample_type = STORE_SAMPLE_TYPE;
		pe.exclude_kernel = 1;
		pe.exclude_hv = 1;
		pe.wakeup_events = 1;
		if (PEBS_store_operates_in_frequency_mode)
		{
			pe.freq = 1;
			pe.sample_freq = PEBS_store_frequency;
		}
		else // operates in period mode by default
		{
			pe.sample_period = PEBS_store_period;
		}

		// If we're creating this group, make sure that we pin the group and that the
		// group starts disabled
		if (group_fd[thread_id] == -1)
		{
			pe.pinned = 1;
			pe.disabled = 1;
		}
	
		perf_pebs_fd[thread_id][STORE_INDEX] = perf_event_open (&pe, 0, -1, group_fd[thread_id], 0); // Chain on LOADS - if previously configured
		if (perf_pebs_fd[thread_id][STORE_INDEX] < 0)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot open the perf_event file descriptor for stores\n");
			return -1;
		}

		if (group_fd[thread_id] == -1)
			group_fd[thread_id] = perf_pebs_fd[thread_id][STORE_INDEX];

		extrae_intel_pebs_mmap[thread_id][STORE_INDEX] = mmap (NULL, mmap_pages*sysconf(_SC_PAGESIZE),
		  PROT_READ|PROT_WRITE, MAP_SHARED, perf_pebs_fd[thread_id][STORE_INDEX], 0);
		if (extrae_intel_pebs_mmap[thread_id][STORE_INDEX] == MAP_FAILED)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot mmap for store events\n");
			close (perf_pebs_fd[thread_id][STORE_INDEX]);
			return -1;
		}

	
		fcntl(perf_pebs_fd[thread_id][STORE_INDEX], F_SETFL, fcntl(perf_pebs_fd[thread_id][STORE_INDEX], F_GETFL, 0) | O_ASYNC);
		fcntl(perf_pebs_fd[thread_id][STORE_INDEX], F_SETSIG, SIGIO);
		fcntl(perf_pebs_fd[thread_id][STORE_INDEX], F_SETOWN, getpid());
		fcntl(perf_pebs_fd[thread_id][STORE_INDEX], F_SETOWN_EX, &owner);
	}

	if (PEBS_load_l3m_enabled && get_load_l3m_event (&hwc) >= 0)
	{
		memset (&pe,0,sizeof(struct perf_event_attr));
		pe.config = hwc;
		pe.type = PERF_TYPE_RAW;
		pe.size = sizeof(struct perf_event_attr);
		pe.precise_ip = 2;
		pe.sample_type = LOAD_L3M_SAMPLE_TYPE;
		pe.exclude_kernel = 1;
		pe.exclude_hv = 1;
		pe.wakeup_events = 1;
		if (PEBS_load_l3m_operates_in_frequency_mode)
		{
			pe.freq = 1;
			pe.sample_freq = PEBS_load_l3m_frequency;
		}
		else // operates in period mode by default
		{
			pe.freq = 0;
			pe.sample_period = PEBS_load_l3m_period;
		}

		// If we're creating this group, make sure that we pin the group and that the
		// group starts disabled
		if (group_fd[thread_id] == -1)
		{
			pe.pinned = 1;
			pe.disabled = 1;
		}
	
		perf_pebs_fd[thread_id][LOAD_L3M_INDEX] = perf_event_open (&pe, 0, -1, group_fd[thread_id], 0); // Chain on LOADS - if setup
		if (perf_pebs_fd[thread_id][LOAD_L3M_INDEX] < 0)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot open the perf_event file descriptor for loads L3M\n");
			return -1;
		}

		if (group_fd[thread_id] == -1)
			group_fd[thread_id] = perf_pebs_fd[thread_id][LOAD_L3M_INDEX];

		extrae_intel_pebs_mmap[thread_id][LOAD_L3M_INDEX] = mmap (NULL, mmap_pages*sysconf(_SC_PAGESIZE),
		  PROT_READ|PROT_WRITE, MAP_SHARED, perf_pebs_fd[thread_id][LOAD_L3M_INDEX], 0);
		if (extrae_intel_pebs_mmap[thread_id][LOAD_L3M_INDEX] == MAP_FAILED)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot mmap for load L3M events\n");
			close (perf_pebs_fd[thread_id][LOAD_L3M_INDEX]);
			return -1;
		}
	
		fcntl(perf_pebs_fd[thread_id][LOAD_L3M_INDEX], F_SETFL, fcntl(perf_pebs_fd[thread_id][LOAD_L3M_INDEX], F_GETFL, 0) | O_ASYNC);
		fcntl(perf_pebs_fd[thread_id][LOAD_L3M_INDEX], F_SETSIG, SIGIO);
		fcntl(perf_pebs_fd[thread_id][LOAD_L3M_INDEX], F_SETOWN, getpid());
		fcntl(perf_pebs_fd[thread_id][LOAD_L3M_INDEX], F_SETOWN_EX, &owner);
	}


	// Start sampling on the given counter -- which is a group leader.
	if (!_PEBS_sampling_paused)
	{
		ret = ioctl (group_fd[thread_id], PERF_EVENT_IOC_REFRESH, 1);
		if (ret < 0)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot enable the PEBS sampling file descriptor\n");
			return -1;
		}
	}

	return 1;
}

/*  Extrae_IntelPEBS_stopSampling
    Stops using PEBS. It stops the sampling mechanism */
void Extrae_IntelPEBS_stopSampling (void)
{
	int i = 0;

	if (PEBS_enabled != 1) return;

	pthread_mutex_lock (&pebs_init_lock);
	for (i=0; i<pebs_init_threads; i++)
	{
		// Stop Loads and unmap associated pages
		if (perf_pebs_fd[i][LOAD_INDEX] >= 0) {
			ioctl (perf_pebs_fd[i][LOAD_INDEX], PERF_EVENT_IOC_REFRESH, 0);
			close (perf_pebs_fd[i][LOAD_INDEX]);
		}
		if (extrae_intel_pebs_mmap[i][LOAD_INDEX] != NULL) {
			munmap (extrae_intel_pebs_mmap[i][LOAD_INDEX], mmap_pages*sysconf(_SC_PAGESIZE));
			extrae_intel_pebs_mmap[i][LOAD_INDEX] = NULL;
		}
		// Stop Stores and unmap associated pages
		if (perf_pebs_fd[i][STORE_INDEX] >= 0) {
			ioctl (perf_pebs_fd[i][STORE_INDEX], PERF_EVENT_IOC_REFRESH, 0);
			close (perf_pebs_fd[i][STORE_INDEX]);
		}
		if (extrae_intel_pebs_mmap[i][STORE_INDEX] != NULL) {
			munmap (extrae_intel_pebs_mmap[i][STORE_INDEX], mmap_pages*sysconf(_SC_PAGESIZE));
			extrae_intel_pebs_mmap[i][STORE_INDEX] = NULL;
		}
		// Stop LoadL3M and unmap associated pages
		if (perf_pebs_fd[i][LOAD_L3M_INDEX] >= 0) {
			ioctl (perf_pebs_fd[i][LOAD_L3M_INDEX], PERF_EVENT_IOC_REFRESH, 0);
			close (perf_pebs_fd[i][LOAD_L3M_INDEX]);
		}
		if (extrae_intel_pebs_mmap[i][LOAD_L3M_INDEX] != NULL) {
			munmap (extrae_intel_pebs_mmap[i][LOAD_L3M_INDEX], mmap_pages*sysconf(_SC_PAGESIZE));
			extrae_intel_pebs_mmap[i][LOAD_L3M_INDEX] = NULL;
		}
	}
	pthread_mutex_unlock (&pebs_init_lock);
}

/*  Extrae_IntelPEBS_startSampling
    Starts using PEBS. It starts the sampling mechanism */
void Extrae_IntelPEBS_startSampling (void)
{
	PEBS_enabled = Extrae_IntelPEBS_enable ();
}

void Extrae_IntelPEBS_pauseSampling (void)
{
	int i;

	if (PEBS_enabled != 1) return;

	pthread_mutex_lock (&pebs_init_lock);
	for (i=0; i<pebs_init_threads; i++)
		ioctl (group_fd[i], PERF_EVENT_IOC_REFRESH, 0);
	_PEBS_sampling_paused = TRUE;
	pthread_mutex_unlock (&pebs_init_lock);
}

void Extrae_IntelPEBS_resumeSampling (void)
{
	int i;

	if (PEBS_enabled != 1) return;

	pthread_mutex_lock (&pebs_init_lock);
	for (i=0; i<pebs_init_threads; i++)
		ioctl (group_fd[i], PERF_EVENT_IOC_REFRESH, 1);
	_PEBS_sampling_paused = FALSE;
	pthread_mutex_unlock (&pebs_init_lock);
}

void Extrae_IntelPEBS_stopSamplingThread (int thid)
{
	if (PEBS_enabled != 1) return;

	// Stop Loads and unmap associated pages
	if (perf_pebs_fd[thid][LOAD_INDEX] >= 0) {
		ioctl (perf_pebs_fd[thid][LOAD_INDEX], PERF_EVENT_IOC_REFRESH, 0);
		close (perf_pebs_fd[thid][LOAD_INDEX]);
	}
	if (extrae_intel_pebs_mmap[thid][LOAD_INDEX] != NULL) {
		munmap (extrae_intel_pebs_mmap[thid][LOAD_INDEX], mmap_pages*sysconf(_SC_PAGESIZE));
		extrae_intel_pebs_mmap[thid][LOAD_INDEX] = NULL;
	}
	// Stop Stores and unmap associated pages
	if (perf_pebs_fd[thid][STORE_INDEX] >= 0) {
		ioctl (perf_pebs_fd[thid][STORE_INDEX], PERF_EVENT_IOC_REFRESH, 0);
		close (perf_pebs_fd[thid][STORE_INDEX]);
	}
	if (extrae_intel_pebs_mmap[thid][STORE_INDEX] != NULL) {
		munmap (extrae_intel_pebs_mmap[thid][STORE_INDEX], mmap_pages*sysconf(_SC_PAGESIZE));
		extrae_intel_pebs_mmap[thid][STORE_INDEX] = NULL;
	}
	// Stop LoadL3M and unmap associated pages
	if (perf_pebs_fd[thid][LOAD_L3M_INDEX] >= 0) {
		ioctl (perf_pebs_fd[thid][LOAD_L3M_INDEX], PERF_EVENT_IOC_REFRESH, 0);
		close (perf_pebs_fd[thid][LOAD_L3M_INDEX]);
	}
	if (extrae_intel_pebs_mmap[thid][LOAD_L3M_INDEX] != NULL) {
		munmap (extrae_intel_pebs_mmap[thid][LOAD_L3M_INDEX], mmap_pages*sysconf(_SC_PAGESIZE));
		extrae_intel_pebs_mmap[thid][LOAD_L3M_INDEX] = NULL;
	}
}
