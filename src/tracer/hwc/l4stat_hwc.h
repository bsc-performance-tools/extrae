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

#ifndef __L4STAT_HWC_H__
#define __L4STAT_HWC_H__

#include "num_hwc.h"

/*------------------------------------------------ Prototypes ---------------*/

void HWCBE_L4STAT_Initialize(int TRCOptions);
int HWCBE_L4STAT_Init_Thread(UINT64 time, int threadid, int forked);
int HWCBE_L4STAT_Allocate_eventsets_per_thread(int num_set, int old_thread_num, int new_thread_num);

int HWCBE_L4STAT_Start_Set(UINT64 countglops, UINT64 time, int numset, int threadid);
int HWCBE_L4STAT_Stop_Set(UINT64 time, int numset, int threadid);
int HWCBE_L4STAT_Add_Set(int pretended_set, int rank, int ncounters, char **counters, char *domain,
						 char *change_at_globalops, char *change_at_time, int num_overflows,
						 char **overflow_counters, unsigned long long *overflow_values);

int HWCBE_L4STAT_Read(unsigned int tid, long long *store_buffer);
int HWCBE_L4STAT_Read_Sampling(unsigned int tid, long long *store_buffer);
int HWCBE_L4STAT_Reset(unsigned int tid);
int HWCBE_L4STAT_Accum(unsigned int tid, long long *store_buffer);
void HWCBE_L4STAT_Update_Sampling_Cores(unsigned int tid);

void HWCBE_L4STAT_CleanUp(unsigned nthreads);

HWC_Definition_t *HWCBE_L4STAT_GetCounterDefinitions(unsigned *count);

#define L4STAT_BAD_CMD "N/A. Wrong event"

#define L4STAT_NULL -1

static const char *l4stat_event_names[] = {
	"Instruction cache miss",				 /* 0x00 */
	"Instruction MMU TLB miss",				 /* 0x01 */
	"Instruction cache hold",				 /* 0x02 */
	"Instruction MMU hold",					 /* 0x03 */
	L4STAT_BAD_CMD,							 /* 0x04 */
	L4STAT_BAD_CMD,							 /* 0x05 */
	L4STAT_BAD_CMD,							 /* 0x06 */
	L4STAT_BAD_CMD,							 /* 0x07 */
	"Data cache (read) miss",				 /* 0x08 */
	"Data MMU TLB miss",					 /* 0x09 */
	"Data cache hold",						 /* 0x0a */
	"Data MMU hold",						 /* 0x0b */
	L4STAT_BAD_CMD,							 /* 0x0c */
	L4STAT_BAD_CMD,							 /* 0x0d */
	L4STAT_BAD_CMD,							 /* 0x0e */
	L4STAT_BAD_CMD,							 /* 0x0f */
	"Data write buffer hold",				 /* 0x10 */
	"Total instruction count",				 /* 0x11 */
	"Integer instruction count",			 /* 0x12 */
	"Floating-point unit instruction count", /* 0x13 */
	"Branch prediction miss",				 /* 0x14 */
	"Execution time, exluding debug mode",   /* 0x15 */
	L4STAT_BAD_CMD,							 /* 0x16 */
	"AHB utilization (per AHB master)",		 /* 0x17 */
	"AHB utilization (total)",				 /* 0x18 */
	L4STAT_BAD_CMD,							 /* 0x19 */
	L4STAT_BAD_CMD,							 /* 0x1a */
	L4STAT_BAD_CMD,							 /* 0x1b */
	L4STAT_BAD_CMD,							 /* 0x1c */
	L4STAT_BAD_CMD,							 /* 0x1d */
	L4STAT_BAD_CMD,							 /* 0x1e */
	L4STAT_BAD_CMD,							 /* 0x1f */
	L4STAT_BAD_CMD,							 /* 0x20 */
	L4STAT_BAD_CMD,							 /* 0x21 */
	"Integer branches",						 /* 0x22 */
	L4STAT_BAD_CMD,							 /* 0x23 */
	L4STAT_BAD_CMD,							 /* 0x24 */
	L4STAT_BAD_CMD,							 /* 0x25 */
	L4STAT_BAD_CMD,							 /* 0x26 */
	L4STAT_BAD_CMD,							 /* 0x27 */
	"CALL instructions",					 /* 0x28 */
	L4STAT_BAD_CMD,							 /* 0x29 */
	L4STAT_BAD_CMD,							 /* 0x2a */
	L4STAT_BAD_CMD,							 /* 0x2b */
	L4STAT_BAD_CMD,							 /* 0x2c */
	L4STAT_BAD_CMD,							 /* 0x2d */
	L4STAT_BAD_CMD,							 /* 0x2e */
	L4STAT_BAD_CMD,							 /* 0x2f */
	"Regular type 2 instructions",			 /* 0x30 */
	L4STAT_BAD_CMD,							 /* 0x31 */
	L4STAT_BAD_CMD,							 /* 0x32 */
	L4STAT_BAD_CMD,							 /* 0x33 */
	L4STAT_BAD_CMD,							 /* 0x34 */
	L4STAT_BAD_CMD,							 /* 0x35 */
	L4STAT_BAD_CMD,							 /* 0x36 */
	L4STAT_BAD_CMD,							 /* 0x37 */
	"LOAD and STORE instructions",			 /* 0x38 */
	"LOAD instructions",					 /* 0x39 */
	"STORE instructions",					 /* 0x3a */
	L4STAT_BAD_CMD,							 /* 0x3b */
	L4STAT_BAD_CMD,							 /* 0x3c */
	L4STAT_BAD_CMD,							 /* 0x3d */
	L4STAT_BAD_CMD,							 /* 0x3e */
	L4STAT_BAD_CMD,							 /* 0x3f */
	"AHB IDLE cycles",						 /* 0x40 */
	"AHB BUSY cycles",						 /* 0x41 */
	"AHB Non-Seq. transfers",				 /* 0x42 */
	"AHB Seq. transfers",					 /* 0x43 */
	"AHB read accesses",					 /* 0x44 */
	"AHB write accesses",					 /* 0x45 */
	"AHB byte accesses",					 /* 0x46 */
	"AHB half-word accesses",				 /* 0x47 */
	"AHB word accesses",					 /* 0x48 */
	"AHB double word accesses",				 /* 0x49 */
	"AHB quad word accesses",				 /* 0x4A */
	"AHB eight word accesses",				 /* 0x4B */
	"AHB waitstates",						 /* 0x4C */
	"AHB RETRY responses",					 /* 0x4D */
	"AHB SPLIT responses",					 /* 0x4E */
	"AHB SPLIT delay",						 /* 0x4F */
	"AHB bus locked",						 /* 0x50 */
	L4STAT_BAD_CMD,							 /* 0x51 */
	L4STAT_BAD_CMD,							 /* 0x52 */
	L4STAT_BAD_CMD,							 /* 0x53 */
	L4STAT_BAD_CMD,							 /* 0x54 */
	L4STAT_BAD_CMD,							 /* 0x55 */
	L4STAT_BAD_CMD,							 /* 0x56 */
	L4STAT_BAD_CMD,							 /* 0x57 */
	L4STAT_BAD_CMD,							 /* 0x58 */
	L4STAT_BAD_CMD,							 /* 0x59 */
	L4STAT_BAD_CMD,							 /* 0x5a */
	L4STAT_BAD_CMD,							 /* 0x5b */
	L4STAT_BAD_CMD,							 /* 0x5c */
	L4STAT_BAD_CMD,							 /* 0x5d */
	L4STAT_BAD_CMD,							 /* 0x5e */
	L4STAT_BAD_CMD,							 /* 0x5f */
	"external event 0",						 /* 0x60 */
	"external event 1",						 /* 0x61 */
	"external event 2",						 /* 0x62 */
	"external event 3",						 /* 0x63 */
	"external event 4",						 /* 0x64 */
	"external event 5",						 /* 0x65 */
	"external event 6",						 /* 0x66 */
	"external event 7",						 /* 0x67 */
	"external event 8",						 /* 0x68 */
	"external event 9",						 /* 0x69 */
	"external event 10",					 /* 0x6A */
	"external event 11",					 /* 0x6B */
	"external event 12",					 /* 0x6C */
	"external event 13",					 /* 0x6D */
	"external event 14",					 /* 0x6E */
	"external event 15",					 /* 0x6F */
	"AHB IDLE cycles (2)",					 /* 0x70 */
	"AHB BUSY cycles (2)",					 /* 0x71 */
	"AHB Non-Seq. transfers (2)",			 /* 0x72 */
	"AHB Seq. transfers (2)",				 /* 0x73 */
	"AHB read accesses (2)",				 /* 0x74 */
	"AHB write accesses (2)",				 /* 0x75 */
	"AHB byte accesses (2)",				 /* 0x76 */
	"AHB half-word accesses (2)",			 /* 0x77 */
	"AHB word accesses (2)",				 /* 0x78 */
	"AHB double word accesses (2)",			 /* 0x79 */
	"AHB quad word accesses (2)",			 /* 0x7A */
	"AHB eight word accesses (2)",			 /* 0x7B */
	"AHB waitstates (2)",					 /* 0x7C */
	"AHB RETRY responses (2)",				 /* 0x7D */
	"AHB SPLIT responses (2)",				 /* 0x7E */
	"AHB SPLIT delay (2)",					 /* 0x7F */
	"PMC: master 0 has grant",				 /* 0x80 */
	"PMC: master 1 has grant",				 /* 0x81 */
	"PMC: master 2 has grant",				 /* 0x82 */
	"PMC: master 3 has grant",				 /* 0x83 */
	"PMC: master 4 has grant",				 /* 0x84 */
	"PMC: master 5 has grant",				 /* 0x85 */
	"PMC: master 6 has grant",				 /* 0x86 */
	"PMC: master 7 has grant",				 /* 0x87 */
	"PMC: master 8 has grant",				 /* 0x88 */
	"PMC: master 9 has grant",				 /* 0x89 */
	"PMC: master 10 has grant",				 /* 0x8A */
	"PMC: master 11 has grant",				 /* 0x8B */
	"PMC: master 12 has grant",				 /* 0x8C */
	"PMC: master 13 has grant",				 /* 0x8D */
	"PMC: master 14 has grant",				 /* 0x8E */
	"PMC: master 15 has grant",				 /* 0x8F */
	"PMC: master 0 lacks grant",			 /* 0x90 */
	"PMC: master 1 lacks grant",			 /* 0x91 */
	"PMC: master 2 lacks grant",			 /* 0x92 */
	"PMC: master 3 lacks grant",			 /* 0x93 */
	"PMC: master 4 lacks grant",			 /* 0x94 */
	"PMC: master 5 lacks grant",			 /* 0x95 */
	"PMC: master 6 lacks grant",			 /* 0x96 */
	"PMC: master 7 lacks grant",			 /* 0x97 */
	"PMC: master 8 lacks grant",			 /* 0x98 */
	"PMC: master 9 lacks grant",			 /* 0x99 */
	"PMC: master 10 lacks grant",			 /* 0x9A */
	"PMC: master 11 lacks grant",			 /* 0x9B */
	"PMC: master 12 lacks grant",			 /* 0x9C */
	"PMC: master 13 lacks grant",			 /* 0x9D */
	"PMC: master 14 lacks grant",			 /* 0x9E */
	"PMC: master 15 lacks grant",			 /* 0x9F */
	""};

/*------------------------------------------------ Useful Macros ------------*/

/**
 * Stores which counters did overflow in the given buffer (?).
 */
#define HARDWARE_COUNTERS_OVERFLOW(nc, counters, no, counters_ovf, values_ptr) \
	{                                                                          \
		int found, cc, co;                                                     \
                                                                               \
		for (cc = 0; cc < nc; cc++)                                            \
		{                                                                      \
			for (co = 0, found = 0; co < no; co++)                             \
				found |= counters[cc] == counters_ovf[co];                     \
			if (found)                                                         \
				values_ptr[cc] = (long long)(SAMPLE_COUNTER);                  \
			else                                                               \
				values_ptr[cc] = (long long)(NO_COUNTER);                      \
		}                                                                      \
		for (cc = nc; cc < MAX_HWC; cc++)                                      \
			values_ptr[cc] = (long long)(NO_COUNTER);                          \
	}

/**
 * Returns the EventSet of the given thread for the current set.
 */

#endif /* __PAPI_HWC_H__ */
