#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "BurstInfo.h"

/* Allocate the structure to store the information extracted from the buffer */
BurstInfo_t * new_BurstInfo (int task_id, int thread_id, int num_bursts, int hwc_per_burst)
{
    BurstInfo_t *bi = NULL;

    bi = (BurstInfo_t *)malloc(sizeof(BurstInfo_t));
    if (bi != NULL)
    {
        bi->TaskID = task_id;
        bi->ThreadID = thread_id;

        bi->num_Bursts = 0;
        bi->num_HWCperBurst = hwc_per_burst;

        bi->Timestamp = (long long *)malloc(num_bursts * sizeof(long long));
        bi->Durations = (long long *)malloc(num_bursts * sizeof(long long));
        bi->HWCValues = (long long *)malloc(num_bursts * hwc_per_burst * sizeof(long long));
        bi->HWCSet    = (int *)malloc(num_bursts * sizeof(int));
    }
    return bi;
}

/* Free structures */
void BurstInfo_Free (BurstInfo_t * bi)
{
    if (bi != NULL)
    {
        xfree(bi->Timestamp);
        xfree(bi->Durations);
        xfree(bi->HWCValues);
		xfree(bi->HWCSet);
        xfree(bi);
    }
}

void BurstInfo_FreeArray (BurstInfo_t **bi_list, int num_be)
{
	int i;
	if ((num_be > 0) && (bi_list != NULL))
	{
		for (i=0; i<num_be; i++)
		{
			BurstInfo_Free(bi_list[i]);
		}
		xfree (bi_list);
	}
}


/* Split the struct into different variables
 */
void BurstInfo_Serialize(BurstInfo_t *bi, int *TaskID, int *ThreadID, int *num_Bursts, int *num_HWCperBurst, long long **Timestamp, long long **Durations, long long **HWCValues, int **HWCSet)
{
	if (bi != NULL)
	{
	    *TaskID = bi->TaskID;
		*ThreadID = bi->ThreadID;
	    *num_Bursts = bi->num_Bursts;
	    *num_HWCperBurst = bi->num_HWCperBurst;
	    *Timestamp = bi->Timestamp;
	    *Durations = bi->Durations;
	    *HWCValues = bi->HWCValues;
		*HWCSet = bi->HWCSet;
	}
}

/* Build the struct from multiple variables
 */
BurstInfo_t * BurstInfo_Assemble (int TaskID, int ThreadID, int num_Bursts, int num_HWCperBurst, long long *Timestamp, long long *Durations, long long *HWCValues, int *HWCSet)
{
    BurstInfo_t *bi = (BurstInfo_t *)malloc(sizeof(BurstInfo_t));
    if (bi != NULL)
    {
        bi->TaskID = TaskID;
		bi->ThreadID = ThreadID;
        bi->num_Bursts = num_Bursts;
        bi->num_HWCperBurst = num_HWCperBurst;
        bi->Timestamp = Timestamp;
        bi->Durations = Durations;
        bi->HWCValues = HWCValues;
		bi->HWCSet = HWCSet;
    }
    return bi;
}

