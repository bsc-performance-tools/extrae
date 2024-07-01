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

#include "common.h"

#include <config.h>

#if defined(HAVE_STDLIB_H)
# include <stdlib.h>
#endif

#include "misc_prv_events.h"
#include "misc_prv_semantics.h"
#include "intel-pebs-types.h"
#include "labels.h"
#include "addr2info.h"

#define PRV_FORK_VALUE          1
#define PRV_WAIT_VALUE          2
#define PRV_WAITPID_VALUE       3
#define PRV_EXEC_VALUE          4
#define PRV_SYSTEM_VALUE        5

#define PRV_MALLOC_VALUE                 1
#define PRV_FREE_VALUE                   2
#define PRV_REALLOC_VALUE                3
#define PRV_CALLOC_VALUE                 4
#define PRV_POSIX_MEMALIGN_VALUE         5
#define PRV_MEMKIND_MALLOC_VALUE         6
#define PRV_MEMKIND_CALLOC_VALUE         7
#define PRV_MEMKIND_REALLOC_VALUE        8
#define PRV_MEMKIND_POSIX_MEMALIGN_VALUE 9
#define PRV_MEMKIND_FREE_VALUE           10
#define PRV_KMPC_MALLOC_VALUE            11
#define PRV_KMPC_FREE_VALUE              12
#define PRV_KMPC_REALLOC_VALUE           13
#define PRV_KMPC_CALLOC_VALUE            14
#define PRV_KMPC_ALIGNED_MALLOC_VALUE    15

#define APPL_INDEX              0
#define FLUSH_INDEX             1
#define TRACING_INDEX           2
#define INOUT_INDEX             3
#define FORK_SYSCALL_INDEX      4
#define GETCPU_INDEX            5
#define TRACE_INIT_INDEX        6
#define DYNAMIC_MEM_INDEX       7
#define SAMPLING_MEM_INDEX      8

#define MAX_MISC_INDEX	       9

#define NUM_MISC_PRV_ELEMENTS  15

struct t_event_misc2prv
{
        int misc_type;
        int prv_value;
        int used;
};

struct t_prv_val_label
{
        int value;
        char *label;
};

static struct t_event_misc2prv event_misc2prv[NUM_MISC_PRV_ELEMENTS] = {
	{READ_EV, READ_VAL_EV, FALSE},
	{WRITE_EV, WRITE_VAL_EV, FALSE},
	{FREAD_EV, FREAD_VAL_EV, FALSE},
	{FWRITE_EV, FWRITE_VAL_EV, FALSE},
	{PREAD_EV, PREAD_VAL_EV, FALSE},
	{PWRITE_EV, PWRITE_VAL_EV, FALSE},
	{READV_EV, READV_VAL_EV, FALSE},
	{WRITEV_EV, WRITEV_VAL_EV, FALSE},
	{PREADV_EV, PREADV_VAL_EV, FALSE},
	{PWRITEV_EV, PWRITEV_VAL_EV, FALSE},
	{OPEN_EV, OPEN_VAL_EV, FALSE},
	{FOPEN_EV, FOPEN_VAL_EV, FALSE},
	{IOCTL_EV, IOCTL_VAL_EV, FALSE},
	{CLOSE_EV, CLOSE_VAL_EV, FALSE},
	{FCLOSE_EV, FCLOSE_VAL_EV, FALSE}
};

static struct t_prv_val_label misc_prv_val_label[NUM_MISC_PRV_ELEMENTS] = {
	{READ_VAL_EV, READ_LBL},
	{WRITE_VAL_EV, WRITE_LBL},
	{FREAD_VAL_EV, FREAD_LBL},
	{FWRITE_VAL_EV, FWRITE_LBL},
	{PREAD_VAL_EV, PREAD_LBL},
	{PWRITE_VAL_EV, PWRITE_LBL},
	{READV_VAL_EV, READV_LBL},
	{WRITEV_VAL_EV, WRITEV_LBL},
	{PREADV_VAL_EV, PREADV_LBL},
	{PWRITEV_VAL_EV, PWRITEV_LBL},
	{OPEN_VAL_EV, OPEN_LBL},
	{FOPEN_VAL_EV, FOPEN_LBL},
	{IOCTL_VAL_EV, IOCTL_LBL},
	{CLOSE_VAL_EV, CLOSE_LBL},
	{FCLOSE_VAL_EV, FCLOSE_LBL}
};

/******************************************************************************
 **      Function name : search_misc_event
 **      
 **      Description : 
 ******************************************************************************/

static int search_misc_event (int type)
{
        int i;

        for (i = 0; i < NUM_MISC_PRV_ELEMENTS; i++)
                if (event_misc2prv[i].misc_type == type)
                        break;
        if (i < NUM_MISC_PRV_ELEMENTS)
                return i;
        return -1;
}

/******************************************************************************
 **      Function name : Used_MISC_Operation
 **      
 **      Description : Mark I/O operation as used in event_misc2prv
 ******************************************************************************/

void Used_MISC_Operation (int Op)
{
	int index;

	index = search_misc_event (Op);
	if (index >= 0)
	{
		event_misc2prv[index].used = TRUE;
	}
}

/******************************************************************************
 **      Function name : get_misc_prv_val_label
 **      
 **      Description : 
 ******************************************************************************/

static char *get_misc_prv_val_label (int val)
{
        int i;

        for (i = 0; i < NUM_MISC_PRV_ELEMENTS; i++)
                if (misc_prv_val_label[i].value == val)
                        break;
        if (i < NUM_MISC_PRV_ELEMENTS)
	{
                return misc_prv_val_label[i].label;
	}
        return NULL;
}

static int inuse[MAX_MISC_INDEX] = { FALSE };

void Enable_MISC_Operation (int type)
{
	if (type == APPL_EV)
		inuse[APPL_INDEX] = TRUE;
	else if (type == FLUSH_EV)
		inuse[FLUSH_INDEX] = TRUE;
	else if (type == TRACING_EV)
		inuse[TRACING_INDEX] = TRUE;
	else if (type == READ_EV   || type == WRITE_EV   ||
	         type == FREAD_EV  || type == FWRITE_EV  ||
	         type == PREAD_EV  || type == PWRITE_EV  ||
	         type == READV_EV  || type == WRITEV_EV  ||
	         type == PREADV_EV || type == PWRITEV_EV ||
	         type == OPEN_EV   || type == FOPEN_EV   ||
	         type == IOCTL_EV  || type == CLOSE_EV   ||
	         type == FCLOSE_EV)
		{
			inuse[INOUT_INDEX] = TRUE;
			Used_MISC_Operation(type);
		}
	else if (type == FORK_EV || type == WAIT_EV || type == WAITPID_EV ||
	  type == EXEC_EV || type == SYSTEM_EV)
		inuse[FORK_SYSCALL_INDEX] = TRUE;
	else if (type == GETCPU_EV)
		inuse[GETCPU_INDEX] = TRUE;
	else if (type == TRACE_INIT_EV)
		inuse[TRACE_INIT_INDEX] = TRUE;
	else if (type == MALLOC_EV                 ||
			 type == ADD_RESERVED_MEM_EV       ||
			 type == SUB_RESERVED_MEM_EV       ||
	         type == REALLOC_EV                ||
	         type == FREE_EV                   ||
	         type == CALLOC_EV                 ||
	         type == POSIX_MEMALIGN_EV         ||
	         type == MEMKIND_MALLOC_EV         ||
	         type == MEMKIND_CALLOC_EV         ||
	         type == MEMKIND_REALLOC_EV        ||
	         type == MEMKIND_POSIX_MEMALIGN_EV ||
	         type == MEMKIND_FREE_EV           ||
	         type == KMPC_MALLOC_EV            ||
	         type == KMPC_CALLOC_EV            ||
	         type == KMPC_REALLOC_EV           ||
	         type == KMPC_FREE_EV              ||
	         type == KMPC_ALIGNED_MALLOC_EV)
		inuse[DYNAMIC_MEM_INDEX] = TRUE;
	else if (type == SAMPLING_ADDRESS_MEM_LEVEL_EV ||
	  type == SAMPLING_ADDRESS_TLB_LEVEL_EV ||
	  type == SAMPLING_ADDRESS_LD_EV || type == SAMPLING_ADDRESS_ST_EV ||
	  type == SAMPLING_ADDRESS_REFERENCE_COST_EV)
		inuse[SAMPLING_MEM_INDEX] = TRUE;
}

unsigned MISC_event_GetValueForForkRelated (unsigned type)
{
	switch (type)
	{
		case FORK_EV:
			return PRV_FORK_VALUE;
		case WAIT_EV:
			return PRV_WAIT_VALUE;
		case WAITPID_EV:
			return PRV_WAITPID_VALUE;
		case EXEC_EV:
			return PRV_EXEC_VALUE;
		case SYSTEM_EV:
			return PRV_SYSTEM_VALUE;
		default:
			return 0;
	}
}

unsigned MISC_event_GetValueForDynamicMemory (unsigned type)
{
	switch (type)
	{
		case MALLOC_EV:
			return PRV_MALLOC_VALUE;
		case FREE_EV:
			return PRV_FREE_VALUE;
		case REALLOC_EV:
			return PRV_REALLOC_VALUE;
		case CALLOC_EV:
			return PRV_CALLOC_VALUE;
		case POSIX_MEMALIGN_EV:
			return PRV_POSIX_MEMALIGN_VALUE;
		case MEMKIND_MALLOC_EV:
			return PRV_MEMKIND_MALLOC_VALUE;
		case MEMKIND_CALLOC_EV:
			return PRV_MEMKIND_CALLOC_VALUE;
		case MEMKIND_REALLOC_EV:
			return PRV_MEMKIND_REALLOC_VALUE;
		case MEMKIND_POSIX_MEMALIGN_EV:
			return PRV_MEMKIND_POSIX_MEMALIGN_VALUE;
		case MEMKIND_FREE_EV:
			return PRV_MEMKIND_FREE_VALUE;
		case KMPC_MALLOC_EV:
			return PRV_KMPC_MALLOC_VALUE;
		case KMPC_CALLOC_EV:
			return PRV_KMPC_CALLOC_VALUE;
		case KMPC_REALLOC_EV:
			return PRV_KMPC_REALLOC_VALUE;
		case KMPC_FREE_EV:
			return PRV_KMPC_FREE_VALUE;
		case KMPC_ALIGNED_MALLOC_EV:
			return PRV_KMPC_ALIGNED_MALLOC_VALUE;
		default:
			return 0;
	}
}

void MISCEvent_WriteEnabledOperations (FILE * fd, long long options)
{
	int jj;
	char *label;	

	if (options & TRACEOPTION_BG_ARCH)
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_PROCESSOR_ID, BG_PROCESSOR_ID);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_A, BG_TORUS_A);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_B, BG_TORUS_B);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_C, BG_TORUS_C);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_D, BG_TORUS_D);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_E, BG_TORUS_E);
		LET_SPACES (fd);
	}
	if (inuse[GETCPU_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, GETCPU_EV, GETCPU_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, CPU_EVENT_INTERVAL_EV, CPU_EVENT_INTERVAL_LBL);
		LET_SPACES(fd);
	}
	if (inuse[APPL_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, APPL_EV, APPL_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, EVT_BEGIN_LBL);
		LET_SPACES (fd);
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, CLOCK_FROM_SYSTEM_EV,
		  CLOCK_FROM_SYSTEM_LBL);
		LET_SPACES (fd);
	}
	if (inuse[FLUSH_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, FLUSH_EV, FLUSH_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, EVT_BEGIN_LBL);
		LET_SPACES (fd);
	}
	if (inuse[TRACING_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, TRACING_EV, TRACING_LBL);

		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, TRAC_DISABLED_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, TRAC_ENABLED_LBL);
		LET_SPACES (fd);
	}
	if (inuse[TRACE_INIT_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, TRACE_INIT_EV, TRACE_INIT_LBL);

		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, EVT_BEGIN_LBL);
		LET_SPACES (fd);
	}
	if (inuse[INOUT_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, IO_EV, IO_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		
		for (jj = 0; jj < NUM_MISC_PRV_ELEMENTS; jj++)
                {
                        if(event_misc2prv[jj].used)
                        {
                                label = get_misc_prv_val_label (event_misc2prv[jj].prv_value);
                                fprintf (fd, "%d   %s\n", event_misc2prv[jj].prv_value, label);
                        }
		}

		LET_SPACES (fd);
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, IO_SIZE_EV, IO_SIZE_LBL);
		LET_SPACES (fd);
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, IO_DESCRIPTOR_EV, IO_DESCRIPTOR_LBL);
		LET_SPACES (fd);
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, IO_DESCRIPTOR_TYPE_EV, IO_DESCRIPTOR_TYPE_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d    Unknown type\n", DESCRIPTOR_TYPE_UNKNOWN);
		fprintf (fd, "%d    Regular file\n", DESCRIPTOR_TYPE_REGULARFILE);
		fprintf (fd, "%d    Socket\n", DESCRIPTOR_TYPE_SOCKET);
		fprintf (fd, "%d    FIFO or PIPE\n", DESCRIPTOR_TYPE_FIFO_PIPE);
		fprintf (fd, "%d    Terminal\n", DESCRIPTOR_TYPE_ATTY);
		LET_SPACES(fd);
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, IOCTL_REQUEST_EV, IOCTL_REQUEST_LBL);
		LET_SPACES(fd);
	}
	if (inuse[FORK_SYSCALL_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, FORK_SYSCALL_EV, FORK_SYSCALL_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", PRV_FORK_VALUE, FORK_LBL);
		fprintf (fd, "%d      %s\n", PRV_WAIT_VALUE, WAIT_LBL);
		fprintf (fd, "%d      %s\n", PRV_WAITPID_VALUE, WAITPID_LBL);
		fprintf (fd, "%d      %s\n", PRV_EXEC_VALUE, EXEC_LBL);
		fprintf (fd, "%d      %s\n", PRV_SYSTEM_VALUE, SYSTEM_LBL);
		LET_SPACES (fd);
	}
	if (inuse[DYNAMIC_MEM_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, DYNAMIC_MEM_EV, DYNAMIC_MEM_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", PRV_MALLOC_VALUE, MALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_FREE_VALUE, FREE_LBL);
		fprintf (fd, "%d      %s\n", PRV_REALLOC_VALUE, REALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_CALLOC_VALUE, CALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_POSIX_MEMALIGN_VALUE, POSIX_MEMALIGN_LBL);
		fprintf (fd, "%d      %s\n", PRV_MEMKIND_MALLOC_VALUE, MEMKIND_MALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_MEMKIND_CALLOC_VALUE, MEMKIND_CALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_MEMKIND_REALLOC_VALUE, MEMKIND_REALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_MEMKIND_POSIX_MEMALIGN_VALUE, MEMKIND_POSIX_MEMALIGN_LBL);
		fprintf (fd, "%d      %s\n", PRV_MEMKIND_FREE_VALUE, MEMKIND_FREE_LBL);
		fprintf (fd, "%d      %s\n", PRV_KMPC_MALLOC_VALUE, KMPC_MALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_KMPC_FREE_VALUE, KMPC_FREE_LBL);
		fprintf (fd, "%d      %s\n", PRV_KMPC_REALLOC_VALUE, KMPC_REALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_KMPC_CALLOC_VALUE, KMPC_CALLOC_LBL);
		fprintf (fd, "%d      %s\n", PRV_KMPC_ALIGNED_MALLOC_VALUE, KMPC_ALIGNED_MALLOC_LBL);
		LET_SPACES (fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, ADD_RESERVED_MEM_EV, ADD_RESERVED_MEM_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SUB_RESERVED_MEM_EV, SUB_RESERVED_MEM_LBL);
		LET_SPACES (fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT,
		  DYNAMIC_MEM_REQUESTED_SIZE_EV,
		  DYNAMIC_MEM_REQUESTED_SIZE_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT,
		  DYNAMIC_MEM_POINTER_IN_EV,
		  DYNAMIC_MEM_POINTER_IN_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT,
		  DYNAMIC_MEM_POINTER_OUT_EV,
		  DYNAMIC_MEM_POINTER_OUT_LBL);
		LET_SPACES (fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
                fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, MEMKIND_PARTITION_EV, MEMKIND_PARTITION_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_DEFAULT_VAL, MEMKIND_PARTITION_DEFAULT_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HBW_VAL, MEMKIND_PARTITION_HBW_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HBW_HUGETLB_VAL, MEMKIND_PARTITION_HBW_HUGETLB_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HBW_PREFERRED_VAL, MEMKIND_PARTITION_HBW_PREFERRED_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HBW_PREFERRED_HUGETLB_VAL, MEMKIND_PARTITION_HBW_PREFERRED_HUGETLB_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HUGETLB_VAL, MEMKIND_PARTITION_HUGETLB_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HBW_GBTLB_VAL, MEMKIND_PARTITION_HBW_GBTLB_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HBW_PREFERRED_GBTLB_VAL, MEMKIND_PARTITION_HBW_PREFERRED_GBTLB_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_GBTLB_VAL, MEMKIND_PARTITION_GBTLB_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_HBW_INTERLEAVE_VAL, MEMKIND_PARTITION_HBW_INTERLEAVE_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_INTERLEAVE_VAL, MEMKIND_PARTITION_INTERLEAVE_LBL);
		fprintf (fd, "%d      %s\n", MEMKIND_PARTITION_OTHER_VAL, MEMKIND_PARTITION_OTHER_LBL);

		LET_SPACES (fd);

	}
	if (inuse[SAMPLING_MEM_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SAMPLING_ADDRESS_LD_EV,
		  SAMPLING_ADDRESS_LD_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SAMPLING_ADDRESS_ST_EV,
		  SAMPLING_ADDRESS_ST_LBL);
		LET_SPACES (fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SAMPLING_ADDRESS_MEM_LEVEL_EV,
		  SAMPLING_ADDRESS_MEM_LEVEL_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d other (uncacheable or I/O)\n", PEBS_MEMORYHIERARCHY_UNCACHEABLE_IO);
		fprintf (fd, "%d L1 cache\n", PEBS_MEMORYHIERARCHY_MEM_LVL_L1);
		fprintf (fd, "%d Line Fill Buffer (LFB)\n", PEBS_MEMORYHIERARCHY_MEM_LVL_LFB);
		fprintf (fd, "%d L2 cache\n", PEBS_MEMORYHIERARCHY_MEM_LVL_L2);
		fprintf (fd, "%d L3 cache\n", PEBS_MEMORYHIERARCHY_MEM_LVL_L3);
		fprintf (fd, "%d Remote cache (1 hop)\n", PEBS_MEMORYHIERARCHY_MEM_LVL_RCACHE_1HOP);
		fprintf (fd, "%d Remote cache (2 hops)\n", PEBS_MEMORYHIERARCHY_MEM_LVL_RCACHE_2HOP);
		fprintf (fd, "%d DRAM (local)\n", PEBS_MEMORYHIERARCHY_MEM_LVL_LOCAL_RAM);
		fprintf (fd, "%d DRAM (remote, 1 hop)\n", PEBS_MEMORYHIERARCHY_MEM_LVL_REMOTE_RAM_1HOP);
		fprintf (fd, "%d DRAM (remote, 2 hops)\n", PEBS_MEMORYHIERARCHY_MEM_LVL_REMOTE_RAM_2HOP);
		LET_SPACES(fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SAMPLING_ADDRESS_MEM_HITORMISS_EV,
		  SAMPLING_ADDRESS_MEM_HITORMISS_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d N/A\n", PEBS_MEMORYHIERARCHY_UNKNOWN);
		fprintf (fd, "%d hit\n", PEBS_MEMORYHIERARCHY_HIT);
		fprintf (fd, "%d miss\n", PEBS_MEMORYHIERARCHY_MISS);
		LET_SPACES (fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SAMPLING_ADDRESS_TLB_LEVEL_EV,
		  SAMPLING_ADDRESS_TLB_LEVEL_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d other (hw walker or OS fault handler)\n", PEBS_MEMORYHIERARCHY_TLB_OTHER);
		fprintf (fd, "%d L1 TLB\n", PEBS_MEMORYHIERARCHY_TLB_L1);
		fprintf (fd, "%d L2 TLB\n", PEBS_MEMORYHIERARCHY_TLB_L2);
		LET_SPACES (fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SAMPLING_ADDRESS_TLB_HITORMISS_EV,
		  SAMPLING_ADDRESS_TLB_HITORMISS_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d N/A\n", PEBS_MEMORYHIERARCHY_UNKNOWN);
		fprintf (fd, "%d hit\n", PEBS_MEMORYHIERARCHY_HIT);
		fprintf (fd, "%d miss\n", PEBS_MEMORYHIERARCHY_MISS);
		LET_SPACES (fd);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, SAMPLING_ADDRESS_REFERENCE_COST_EV,
		  SAMPLING_ADDRESS_REFERENCE_COST_LBL);
		LET_SPACES (fd);
	}

	if (inuse[DYNAMIC_MEM_INDEX] ||inuse[SAMPLING_MEM_INDEX])
		Address2Info_Write_MemReferenceCaller_Labels (fd);

	/* These events are always emitted */
	fprintf (fd, "%s\n", TYPE_LABEL);
	fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, PID_EV, PID_LBL);
	fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, PPID_EV, PPID_LBL);
	fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, FORK_DEPTH_EV, FORK_DEPTH_LBL);
	LET_SPACES (fd);
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"
#include "mpi-tags.h"

void Share_MISC_Operations (void)
{
	int res, i, max;
	int tmp2[3], tmp[3] = { Rusage_Events_Found, Memusage_Events_Found, Syscall_Events_Found };
	int tmp_in[RUSAGE_EVENTS_COUNT], tmp_out[RUSAGE_EVENTS_COUNT];
	int tmp3_in[MEMUSAGE_EVENTS_COUNT], tmp3_out[MEMUSAGE_EVENTS_COUNT];
	int tmp_misc[MAX_MISC_INDEX];

	res = MPI_Reduce (inuse, tmp_misc, MAX_MISC_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #1");
	for (i = 0; i < MAX_MISC_INDEX; i++)
		inuse[i] = tmp_misc[i];

	res = MPI_Reduce (tmp, tmp2, 3, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #2");
	Rusage_Events_Found = tmp2[0];
	Memusage_Events_Found = tmp2[1];
	Syscall_Events_Found = tmp2[2];

	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		tmp_in[i] = GetRusage_Labels_Used[i];
	res = MPI_Reduce (tmp_in, tmp_out, RUSAGE_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #3");
	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		GetRusage_Labels_Used[i] = tmp_out[i];

	for (i = 0; i < MEMUSAGE_EVENTS_COUNT; i++)
		tmp3_in[i] = Memusage_Labels_Used[i];
	res = MPI_Reduce (tmp3_in, tmp3_out, MEMUSAGE_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #6");
	for (i = 0; i < MEMUSAGE_EVENTS_COUNT; i++)
		Memusage_Labels_Used[i] = tmp3_out[i];

  for (i = 0; i < SYSCALL_EVENTS_COUNT; i++)                                   
    tmp3_in[i] = Syscall_Labels_Used[i];                                       
  res = MPI_Reduce (tmp3_in, tmp3_out, SYSCALL_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
  MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #7");                     
  for (i = 0; i < SYSCALL_EVENTS_COUNT; i++)                                   
    Syscall_Labels_Used[i] = tmp3_out[i];                                      

	res = MPI_Reduce (&MaxClusterId, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #8");
	MaxClusterId = max;
}

#endif
