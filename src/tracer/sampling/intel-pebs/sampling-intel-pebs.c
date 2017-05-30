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

#include "perf_event.h"

int perf_event_open(struct perf_event_attr *hw_event_uptr, pid_t pid, int cpu,
	int group_fd, unsigned long flags)
{
	return syscall(__NR_perf_event_open,hw_event_uptr, pid, cpu, group_fd,
	  flags);
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


static int detect_processor_cpuinfo(void)
{
	FILE *procinfo_file;
	int cpu_family=0,model=0;
	char string[BUFSIZ];

	procinfo_file = fopen("/proc/cpuinfo","r");
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

static int get_latency_load_event(unsigned long long *config,
			int *precise_ip,
			char *name)
{
	int processor,processor_notfound=0;
	processor=detect_processor();
	switch(processor)
	{
	case PROCESSOR_SANDYBRIDGE:
	case PROCESSOR_SANDYBRIDGE_EP:
		*config=0x1cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:LATENCY_ABOVE_THRESHOLD");
		break;
	case PROCESSOR_IVYBRIDGE:
	case PROCESSOR_IVYBRIDGE_EP:
		*config=0x1cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:LATENCY_ABOVE_THRESHOLD");
		break;
	case PROCESSOR_HASWELL:
	case PROCESSOR_HASWELL_EP:
		*config=0x1cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:LATENCY_ABOVE_THRESHOLD");
		break;
	case PROCESSOR_BROADWELL:
		*config=0x1cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:LATENCY_ABOVE_THRESHOLD");
		break;
	case PROCESSOR_KNIGHTS_LANDING:
	/* KNL Performance Counters
	 * https://software.intel.com/en-us/articles/intel-xeon-phi-x200-family-processor-performance-monitoring-reference-manual
	 */
		*config=0x0404; /* Use 0x0204 for "MEM_UOPS_RETIRED:L2_HIT_LOADS; 0x0404 for "MEM_UOPS_RETIRED:L2_MISS_LOADS */
		*precise_ip=2;
		strcpy(name,"MEM_UOPS_RETIRED:L2_MISS_LOADS");
		break;
	default:
		*config=0x0;
		*precise_ip=0;
		strcpy(name,"UNKNOWN");
		processor_notfound=-1;
		break;
	}
	return processor_notfound;
}

static int get_latency_store_event(unsigned long long *config,
			int *precise_ip,
			char *name)
{
	int processor,processor_notfound=0;
	processor=detect_processor();
	switch(processor) {
	case PROCESSOR_SANDYBRIDGE:
	case PROCESSOR_SANDYBRIDGE_EP:
		*config=0x2cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:PRECISE_STORE");
		break;
	case PROCESSOR_IVYBRIDGE:
	case PROCESSOR_IVYBRIDGE_EP:
		*config=0x2cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:PRECISE_STORE");
		break;
	case PROCESSOR_HASWELL:
	case PROCESSOR_HASWELL_EP:
		*config=0x2cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:PRECISE_STORE");
		break;
	case PROCESSOR_BROADWELL:
		*config=0x2cd;
		*precise_ip=2;
		strcpy(name,"MEM_TRANS_RETIRED:PRECISE_STORE");
		break;
        case PROCESSOR_KNIGHTS_LANDING:
	default:
		*config=0x0;
		*precise_ip=0;
		strcpy(name,"UNKNOWN");
		processor_notfound=-1;
		break;
	}
	return processor_notfound;
}


static int PEBS_load_period  = 1000000;
static int PEBS_store_period = 1000000;
static int PEBS_load_enabled = FALSE;
static int PEBS_store_enabled = FALSE;
static int PEBS_minimumLoadLatency = 3;

void Extrae_IntelPEBS_setLoadPeriod(int period)
{ PEBS_load_period = period; }

void Extrae_IntelPEBS_setStorePeriod (int period)
{ PEBS_store_period = period; }

void Extrae_IntelPEBS_setLoadSampling (int enabled)
{ PEBS_load_enabled = enabled; }

void Extrae_IntelPEBS_setMinimumLoadLatency (int numcycles)
{ PEBS_minimumLoadLatency = numcycles; }

void Extrae_IntelPEBS_setStoreSampling (int enabled)
{ PEBS_store_enabled = enabled; }

#define MMAP_DATA_SIZE 8

static char **extrae_intel_pebs_mmap = NULL;
static long long prev_head;
static long long global_sample_type;
static int *perf_pebs_fd = NULL;
static int mmap_pages=1+MMAP_DATA_SIZE;

#define MALLOC_ONCE /* Define this or there's a free that gets instrumented */

#if defined(MALLOC_ONCE)
#define ALLOCATED_SIZE MMAP_DATA_SIZE*4096
static unsigned char **data_thread_buffer = NULL;
#endif

/* This function extracts PEBS entries from the previously allocated buffer 
   in data ptr. At this moment, we only should have 1 event in the buffer */
static long long extrae_perf_mmap_read_pebs (void *extrae_intel_pebs_mmap_thread,
	int mmap_size, long long prev_head, int sample_type, int *events_read,
	long long *ip, long long *addr, long long *weight,
	union perf_mem_data_src *data_src)
{

	struct perf_event_mmap_page *control_page = extrae_intel_pebs_mmap_thread;
	long long head,offset;
	int size;
	long long bytesize,prev_head_wrap;
	unsigned char *data;
	void *data_mmap=extrae_intel_pebs_mmap_thread+sysconf(_SC_PAGESIZE);

	if (mmap_size==0)
		return 0;

	if (control_page == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error mmap page NULL\n");
		return -1;
	}

	head = control_page->data_head;
	rmb(); /* Must always follow read of data_head */

	size = head-prev_head;

	bytesize = mmap_size*sysconf(_SC_PAGESIZE);

	if (size > bytesize)
		fprintf (stderr, PACKAGE_NAME": Error! overflowed the mmap buffer %d>%lld bytes\n", size, bytesize);

#if !defined(MALLOC_ONCE)
	data = malloc(bytesize);
	if (data == NULL)
		return -1;
#else
	data = data_thread_buffer[THREADID];
	if (bytesize > ALLOCATED_SIZE)
	{
		fprintf (stderr, PACKAGE_NAME": Error! overflow in the allocated size for PEBS buffer\n");
		return -1;
	}
	if (data == NULL)
		return -1;
#endif

	prev_head_wrap=prev_head%bytesize;

	//   printf("Copying %d bytes from %d to %d\n",
	//	  bytesize-prev_head_wrap,prev_head_wrap,0);
	memcpy (data,(unsigned char*)data_mmap + prev_head_wrap,
		bytesize-prev_head_wrap);

	//printf("Copying %d bytes from %d to %d\n",
	//	  prev_head_wrap,0,bytesize-prev_head_wrap);

	memcpy(data+(bytesize-prev_head_wrap),(unsigned char *)data_mmap,
		prev_head_wrap);

	struct perf_event_header *event;

	offset=0;
	if (events_read)
		*events_read=0;

	while (offset < size)
	{
#if defined(DEBUG)
		printf("Offset %d Size %d\n",offset,size);
#endif
		event = (struct perf_event_header *) & data[offset];

#if defined(DEBUG)
		{
			switch(event->type)
			{
				case PERF_RECORD_MMAP:
					printf("PERF_RECORD_MMAP"); break;
				case PERF_RECORD_LOST:
					printf("PERF_RECORD_LOST"); break;
				case PERF_RECORD_COMM:
					printf("PERF_RECORD_COMM"); break;
				case PERF_RECORD_EXIT:
					printf("PERF_RECORD_EXIT"); break;
				case PERF_RECORD_THROTTLE:
					printf("PERF_RECORD_THROTTLE"); break;
				case PERF_RECORD_UNTHROTTLE:
					printf("PERF_RECORD_UNTHROTTLE"); break;
				case PERF_RECORD_FORK:
					printf("PERF_RECORD_FORK"); break;
				case PERF_RECORD_READ:
					printf("PERF_RECORD_READ"); break;
				case PERF_RECORD_SAMPLE:
					printf("PERF_RECORD_SAMPLE [%x]",sample_type); break;
				case PERF_RECORD_MMAP2:
					printf("PERF_RECORD_MMAP2"); break;
				default: printf("UNKNOWN %d",event->type); break;
			}

			printf(", MISC=%d (",event->misc);
			switch(event->misc & PERF_RECORD_MISC_CPUMODE_MASK)
			{
				case PERF_RECORD_MISC_CPUMODE_UNKNOWN:
					printf("PERF_RECORD_MISC_CPUMODE_UNKNOWN"); break; 
				case PERF_RECORD_MISC_KERNEL:
					printf("PERF_RECORD_MISC_KERNEL"); break;
				case PERF_RECORD_MISC_USER:
					printf("PERF_RECORD_MISC_USER"); break;
				case PERF_RECORD_MISC_HYPERVISOR:
					printf("PERF_RECORD_MISC_HYPERVISOR"); break;
				case PERF_RECORD_MISC_GUEST_KERNEL:
					printf("PERF_RECORD_MISC_GUEST_KERNEL"); break;
				case PERF_RECORD_MISC_GUEST_USER:
					printf("PERF_RECORD_MISC_GUEST_USER"); break;
				default:
					printf("Unknown %d!\n",event->misc); break;
			}

			/* Both have the same value */
			if (event->misc & PERF_RECORD_MISC_MMAP_DATA) {
				printf(",PERF_RECORD_MISC_MMAP_DATA or PERF_RECORD_MISC_COMM_EXEC ");
			}

			if (event->misc & PERF_RECORD_MISC_EXACT_IP) {
				printf(",PERF_RECORD_MISC_EXACT_IP ");
			}

			if (event->misc & PERF_RECORD_MISC_EXT_RESERVED) {
				printf(",PERF_RECORD_MISC_EXT_RESERVED ");
			}

			printf("), Size=%d\n",event->size);
		}
#endif /* DEBUG */

		offset+=8; /* skip header */

		/***********************/
		/* Print event Details */
		/***********************/

		if (event->type == PERF_RECORD_SAMPLE)
		{

			if (sample_type & PERF_SAMPLE_IP)
			{
				if (ip != NULL)
					memcpy (ip,&data[offset],sizeof(long long));

#if defined(DEBUG)
				printf("\tPERF_SAMPLE_IP, IP: %llx\n",*ip);
#endif
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_ADDR)
			{
				if (addr != NULL)
					memcpy (addr,&data[offset],sizeof(long long));

#if defined(DEBUG)
				if (addr != NULL)
					printf ("\tPERF_SAMPLE_ADDR, addr: %llx\n",*addr);
#endif
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_WEIGHT)
			{
				if (weight != NULL)
					memcpy(weight,&data[offset],sizeof(long long));

#if defined(DEBUG)
				if (weight != NULL)
					printf ("\tPERF_SAMPLE_WEIGHT, Weight: %lld\n",*weight);
#endif
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_DATA_SRC)
			{
				if (data_src != NULL)
					memcpy (data_src, &data[offset],
					  sizeof(union perf_mem_data_src));

				offset += sizeof(union perf_mem_data_src);

#if defined(DEBUG)
				if (data_src != NULL)
				{
					printf("\t\t");

					if (data_src->mem_lvl & PERF_MEM_LVL_NA)
						printf("Level Not available ");
					if (data_src->mem_lvl & PERF_MEM_LVL_HIT)
						printf("Hit ");
					if (data_src->mem_lvl & PERF_MEM_LVL_MISS)
						printf("Miss ");
					if (data_src->mem_lvl & PERF_MEM_LVL_L1)
						printf("L1 cache ");
					if (data_src->mem_lvl & PERF_MEM_LVL_LFB)
						printf("Line fill buffer ");
					if (data_src->mem_lvl & PERF_MEM_LVL_L2)
						printf("L2 cache ");
					if (data_src->mem_lvl & PERF_MEM_LVL_L3)
						printf("L3 cache ");
					if (data_src->mem_lvl & PERF_MEM_LVL_LOC_RAM)
						printf("Local DRAM ");
					if (data_src->mem_lvl & PERF_MEM_LVL_REM_RAM1)
						printf("Remote DRAM 1 hop ");
					if (data_src->mem_lvl & PERF_MEM_LVL_REM_RAM2)
						printf("Remote DRAM 2 hops ");
					if (data_src->mem_lvl & PERF_MEM_LVL_REM_CCE1)
						printf("Remote cache 1 hop ");
					if (data_src->mem_lvl & PERF_MEM_LVL_REM_CCE2)
						printf("Remote cache 2 hops ");
					if (data_src->mem_lvl & PERF_MEM_LVL_IO)
						printf("I/O memory ");
					if (data_src->mem_lvl & PERF_MEM_LVL_UNC)
						printf("Uncached memory ");

					if (data_src->mem_dtlb & PERF_MEM_TLB_NA)
						printf("Not available ");
					if (data_src->mem_dtlb & PERF_MEM_TLB_HIT)
						printf("Hit ");
					if (data_src->mem_dtlb & PERF_MEM_TLB_MISS)
						printf("Miss ");
					if (data_src->mem_dtlb & PERF_MEM_TLB_L1)
						printf("Level 1 TLB ");
					if (data_src->mem_dtlb & PERF_MEM_TLB_L2)
						printf("Level 2 TLB ");
					if (data_src->mem_dtlb & PERF_MEM_TLB_WK)
						printf("Hardware walker ");
					if (data_src->mem_dtlb & PERF_MEM_TLB_OS)
						printf("OS fault handler ");
					printf("\n");
				}
#endif /* DEBUG */
			}

		}
		else
		{
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": Unhandled perf record type %d\n",
			  event->type);
#endif
		}
		if (events_read) (*events_read)++;
	}

	control_page->data_tail = head;

#if !defined(MALLOC_ONCE)
	free(data);
#endif

	return head;
}

/* This handler will deal with the PEBS buffer when it monitors LOADS
   and the buffer is full (1 entry only). It emits everythin Extrae needs
   timestamp, reference to memory, portion of the memory hierarchy that
   provides it, the access cost. */
static void extrae_intel_pebs_handler_load (int signum, siginfo_t *info,
	void *uc)
{
	UINT64 t;
	int ret;
	int fd = info->si_fd;
	long long ip, addr, weight;
	union perf_mem_data_src data_src;
	unsigned memlevel, memhitormiss;
	unsigned tlblevel, tlbhitormiss;

	UNREFERENCED_PARAMETER(signum);
	UNREFERENCED_PARAMETER(uc);

	if (extrae_intel_pebs_mmap[THREADID] == NULL) return;

	ret = ioctl (fd, PERF_EVENT_IOC_DISABLE, 0);

	prev_head = extrae_perf_mmap_read_pebs (extrae_intel_pebs_mmap[THREADID],
	  MMAP_DATA_SIZE, prev_head, global_sample_type, NULL,
	  &ip, &addr, &weight, &data_src);

	if (tracejant && Extrae_isSamplingEnabled() && !Backend_inInstrumentation(THREADID))
	{
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
	
		t = Clock_getCurrentTime_nstore();
	
		SAMPLE_EVENT_HWC_PARAM(t, SAMPLING_ADDRESS_LD_EV, ip, addr);
		SAMPLE_EVENT_NOHWC_PARAM(t, SAMPLING_ADDRESS_MEM_LEVEL_EV, memhitormiss,
		  memlevel);
		SAMPLE_EVENT_NOHWC_PARAM(t, SAMPLING_ADDRESS_TLB_LEVEL_EV, tlbhitormiss,
		  tlblevel);
		SAMPLE_EVENT_NOHWC(t, SAMPLING_ADDRESS_REFERENCE_COST_EV, weight);
	
		Extrae_trace_callers (t, 5, CALLER_SAMPLING); 
	}

	ret = ioctl (fd, PERF_EVENT_IOC_REFRESH, 1);

	(void) ret;
}

/* This handler will deal with the PEBS buffer when it monitors STORES
   and the buffer is full (1 entry only). It emits everythin Extrae needs
   timestamp  and reference to memory. */
static void extrae_intel_pebs_handler_store (int signum, siginfo_t *info,
	void *uc)
{
	UINT64 t;
	int ret;
	int fd = info->si_fd;
	long long ip, addr;

	UNREFERENCED_PARAMETER(signum);
	UNREFERENCED_PARAMETER(uc);

	if (extrae_intel_pebs_mmap[THREADID] == NULL) return;

	ret = ioctl (fd, PERF_EVENT_IOC_DISABLE, 0);

	prev_head = extrae_perf_mmap_read_pebs (extrae_intel_pebs_mmap[THREADID],
	  MMAP_DATA_SIZE, prev_head, global_sample_type, NULL,
	  &ip, &addr, NULL, NULL);

	if (tracejant && Extrae_isSamplingEnabled() && !Backend_inInstrumentation(THREADID))
	{
		t = Clock_getCurrentTime_nstore();
	
		SAMPLE_EVENT_HWC_PARAM(t, SAMPLING_ADDRESS_ST_EV, ip, addr);
	
		Extrae_trace_callers (t, 5, CALLER_SAMPLING); 
	}

	ret = ioctl (fd, PERF_EVENT_IOC_REFRESH, 1);

	(void) ret;
}

static unsigned int pebs_init_threads = 0;
static pthread_mutex_t pebs_init_lock;

/* Extrae_IntelPEBS_enable (int loads).
   initializes the sampling based on PEBS. If loads is TRUE, then the PEBS is
   setup to monitor LOAD instructions, otherwise it monitors STORE instructions.
*/
int Extrae_IntelPEBS_enable (int loads)
{
	int ret;
	int result,precise_ip;
	char event_name[BUFSIZ];
	struct perf_event_attr pe;
	struct sigaction sa;

	if (!PEBS_load_enabled && !PEBS_store_enabled)
		return 0;

	/* Need a lock as different threads may be initializing, thus, allocating structures simultaneously */
	pthread_mutex_lock(&pebs_init_lock);
	if (THREADID >= pebs_init_threads)
	{
		unsigned int i = 0;

		/* Extend the data structures to the maximum number of threads seen so far */
		extrae_intel_pebs_mmap = (char **)realloc(extrae_intel_pebs_mmap, (THREADID+1) * sizeof(char *));
		perf_pebs_fd = (int *)realloc(perf_pebs_fd, (THREADID+1) * sizeof(int));
		for (i=pebs_init_threads; i<(THREADID+1); i++)
		{
			extrae_intel_pebs_mmap[i] = NULL;
			perf_pebs_fd[i] = -1;
		}

#if defined(MALLOC_ONCE)
		data_thread_buffer = (unsigned char **)realloc(data_thread_buffer, (THREADID+1) * sizeof(unsigned char *));
		for (i=pebs_init_threads; i<(THREADID+1); i++)
		{
			data_thread_buffer[i] = malloc (ALLOCATED_SIZE);
			if (data_thread_buffer[i] == NULL)
				fprintf (stderr, PACKAGE_NAME": Error! overflow in the allocated size for PEBS buffer\n");
		}
#endif

		pebs_init_threads = THREADID+1;
	}
	pthread_mutex_unlock(&pebs_init_lock);

	memset(&sa, 0, sizeof(struct sigaction));
	sa.sa_sigaction = loads?
	  extrae_intel_pebs_handler_load:extrae_intel_pebs_handler_store;
	sa.sa_flags = SA_SIGINFO;

	if (sigaction( SIGIO, &sa, NULL) < 0)
	{
		fprintf (stderr, PACKAGE_NAME": Error setting up signal handler\n");
		return -1;
	}

	/* Set up Appropriate Event */
	memset (&pe,0,sizeof(struct perf_event_attr));

	if (loads)
	{
		result = get_latency_load_event (&pe.config, &precise_ip, event_name);
		pe.config1 = PEBS_minimumLoadLatency;
	}
	else
	{
		result = get_latency_store_event (&pe.config, &precise_ip, event_name);
		pe.config1 = 0;
	}

	if (result<0)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot get latency %s event for PEBS\n",
		  loads?"load":"store");
		return -1;
	}
	else
		pe.type = PERF_TYPE_RAW;

	pe.size = sizeof(struct perf_event_attr);
	pe.precise_ip = precise_ip;

	pe.size = sizeof(struct perf_event_attr);
	pe.sample_period = loads?PEBS_load_period:PEBS_store_period;
	pe.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_WEIGHT |
	  PERF_SAMPLE_DATA_SRC | PERF_SAMPLE_ADDR;

	global_sample_type = pe.sample_type;

	pe.read_format = 0;
	pe.disabled = 1;
	pe.pinned = 1;
	pe.exclude_kernel = 1;
	pe.exclude_hv = 1;
	pe.wakeup_events = 1;

	perf_pebs_fd[THREADID] = perf_event_open(&pe,0,-1,-1,0);
	if (perf_pebs_fd[THREADID] < 0)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot open the perf_event file descriptor\n");
		return -1;
	}

	extrae_intel_pebs_mmap[THREADID] = mmap (NULL, mmap_pages*sysconf(_SC_PAGESIZE),
	  PROT_READ|PROT_WRITE, MAP_SHARED, perf_pebs_fd[THREADID], 0);
	if (extrae_intel_pebs_mmap[THREADID] == MAP_FAILED)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot mmap to the perf_event\n");
		close (perf_pebs_fd[THREADID]);
		return -1;
	}

	struct f_owner_ex owner;
	owner.type = F_OWNER_TID;
	owner.pid = syscall(SYS_gettid);

	ret = fcntl(perf_pebs_fd[THREADID], F_SETFL, O_RDWR|O_NONBLOCK|O_ASYNC);
	ret = fcntl(perf_pebs_fd[THREADID], F_SETSIG, SIGIO);
	ret = fcntl(perf_pebs_fd[THREADID], F_SETOWN, getpid());
	ret = fcntl(perf_pebs_fd[THREADID], F_SETOWN_EX, &owner);
	ret = ioctl(perf_pebs_fd[THREADID], PERF_EVENT_IOC_RESET, 0);

	ret = ioctl(perf_pebs_fd[THREADID], PERF_EVENT_IOC_ENABLE,0);
	if (ret < 0)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot enable the PEBS sampling file descriptor\n");
		return -1;
	}

	return 0;
}

/*  Extrae_IntelPEBS_disable
    Stops using PEBS. It stops the sampling mechanism */
void Extrae_IntelPEBS_disable (void)
{
	unsigned int i = 0;

	for (i=0; i<pebs_init_threads; i++) {
		if (perf_pebs_fd[i] < 0) {
			ioctl(perf_pebs_fd[i], PERF_EVENT_IOC_REFRESH, 0);
			close (perf_pebs_fd[i]);
		}

		if (extrae_intel_pebs_mmap[i] != NULL) {
			munmap (extrae_intel_pebs_mmap[i], mmap_pages*sysconf(_SC_PAGESIZE));
			extrae_intel_pebs_mmap[i] = NULL;
		}
	}
}

/* Extrae_IntelPEBS_nextSampling
   Alternates between sampling loads and stores if multiplexing is requested
   by the user. It simply stops the sampling and starts it again. */

static int PEBS_current_sampling_load = TRUE;
void Extrae_IntelPEBS_nextSampling (void)
{
	/* Alternate if both are enabled */
	if (PEBS_load_enabled && PEBS_store_enabled)
	{
		Extrae_IntelPEBS_disable();
		if (PEBS_current_sampling_load)
			PEBS_current_sampling_load = FALSE;
		else
			PEBS_current_sampling_load = TRUE;
		Extrae_IntelPEBS_enable (PEBS_current_sampling_load);
	}
}
