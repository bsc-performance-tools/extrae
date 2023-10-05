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

#ifndef MISC_PRV_EVENTS_H
#define MISC_PRV_EVENTS_H

#if HAVE_STDIO_H
# include <stdio.h>
#endif

void Enable_MISC_Operation (int type);
void MISCEvent_WriteEnabledOperations (FILE * fd, long long options);

unsigned MISC_event_GetValueForForkRelated (unsigned type);
unsigned MISC_event_GetValueForDynamicMemory (unsigned type);

#define BG_TORUS_A            "BG A Coordinate in Torus"
#define BG_TORUS_B            "BG B Coordinate in Torus"
#define BG_TORUS_C            "BG C Coordinate in Torus"
#define BG_TORUS_D            "BG D Coordinate in Torus"
#define BG_TORUS_E            "BG E Coordinate in Torus"

#define BG_PROCESSOR_ID       "BG Processor ID"

#if defined(PARALLEL_MERGE)
void Share_MISC_Operations (void);
#endif

#define DYNAMIC_MEM_LBL                 "Dynamic memory calls"
#define MALLOC_LBL                      "malloc()"
#define ADD_RESERVED_MEM_LBL            "Allocated usable memory size"
#define SUB_RESERVED_MEM_LBL            "Freed usable memory size"
#define CALLOC_LBL                      "calloc()"
#define REALLOC_LBL                     "realloc()"
#define FREE_LBL                        "free()"
#define POSIX_MEMALIGN_LBL              "posix_memalign()"
#define MEMKIND_MALLOC_LBL              "memkind_malloc()"
#define MEMKIND_CALLOC_LBL              "memkind_calloc()"
#define MEMKIND_REALLOC_LBL             "memkind_realloc()"
#define MEMKIND_POSIX_MEMALIGN_LBL      "memkind_posix_memalign()"
#define MEMKIND_FREE_LBL                "memkind_free()"
#define KMPC_MALLOC_LBL                 "kmpc_malloc()"
#define KMPC_CALLOC_LBL                 "kmpc_calloc()"
#define KMPC_REALLOC_LBL                "kmpc_realloc()"
#define KMPC_FREE_LBL                   "kmpc_free()"
#define KMPC_ALIGNED_MALLOC_LBL         "kmpc_aligned_malloc()"
#define DYNAMIC_MEM_REQUESTED_SIZE_LBL  "Requested size in dynamic memory call"
#define DYNAMIC_MEM_POINTER_IN_LBL      "In pointer (free, realloc)"
#define DYNAMIC_MEM_POINTER_OUT_LBL     "Out pointer (malloc, calloc, realloc)"

#define MEMKIND_PARTITION_LBL				"Memkind partition"
#define MEMKIND_PARTITION_OTHER_LBL			"Other"
#define MEMKIND_PARTITION_DEFAULT_LBL			"Default"
#define MEMKIND_PARTITION_HBW_LBL			"HBW"
#define MEMKIND_PARTITION_HBW_HUGETLB_LBL		"HBW Huge TLB"
#define MEMKIND_PARTITION_HBW_PREFERRED_LBL		"HBW Preferred"
#define MEMKIND_PARTITION_HBW_PREFERRED_HUGETLB_LBL	"HBW Preferred Huge TLB"
#define MEMKIND_PARTITION_HUGETLB_LBL			"Huge TLB"
#define MEMKIND_PARTITION_HBW_GBTLB_LBL			"HBW GBTLB"
#define MEMKIND_PARTITION_HBW_PREFERRED_GBTLB_LBL 	"HBW Preferred GBTLB"
#define MEMKIND_PARTITION_GBTLB_LBL			"GBTLB"
#define MEMKIND_PARTITION_HBW_INTERLEAVE_LBL		"HBW Interleave"
#define MEMKIND_PARTITION_INTERLEAVE_LBL		"Interleave"

#define SAMPLING_ADDRESS_LD_LBL             "Sampled address (load)"
#define SAMPLING_ADDRESS_ST_LBL             "Sampled address (store)"
#define SAMPLING_ADDRESS_MEM_LEVEL_LBL      "Memory hierarchy location for sampled address"
#define SAMPLING_ADDRESS_MEM_HITORMISS_LBL  "Memory hierarchy location for sampled address hit?"
#define SAMPLING_ADDRESS_TLB_LEVEL_LBL      "TLB hierarchy location for sampled address"
#define SAMPLING_ADDRESS_TLB_HITORMISS_LBL  "TLB hierarchy location for sampled address hit?"
#define SAMPLING_ADDRESS_REFERENCE_COST_LBL "Memory reference cost in core cycles"

#define IO_LBL                          "I/O calls"
#define OPEN_LBL                        "open()"
#define FOPEN_LBL                       "fopen()"
#define READ_LBL                        "read()"
#define WRITE_LBL                       "write()"
#define FREAD_LBL                       "fread()"
#define FWRITE_LBL                      "fwrite()"
#define PREAD_LBL                       "pread()"
#define PWRITE_LBL                      "pwrite()"
#define READV_LBL                       "readv()"
#define WRITEV_LBL                      "writev()"
#define PREADV_LBL                      "preadv()"
#define PWRITEV_LBL                     "pwritev()"
#define IOCTL_LBL                       "ioctl()"
#define CLOSE_LBL                       "close()"
#define FCLOSE_LBL                      "fclose()"
#define IO_DESCRIPTOR_LBL               "I/O descriptor"
#define IO_SIZE_LBL                     "I/O size"
#define IO_DESCRIPTOR_TYPE_LBL          "I/O descriptor type"
#define FILE_NAME_LBL                   "Filename"
#define IOCTL_REQUEST_LBL               "ioctl request code"

#define CLOCK_FROM_SYSTEM_LBL           "RAW clock() value from system"

#define CPU_EVENT_INTERVAL_LBL          "CPU-Event sampling interval"

#endif
