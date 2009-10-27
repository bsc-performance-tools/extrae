
#ifndef __BURST_INFO_H__
#define __BURST_INFO_H__

typedef struct
{
    int        TaskID;
	int        ThreadID;

    int        num_Bursts;
    int        num_HWCperBurst;

    long long *Timestamp;
    long long *Durations;
    long long *HWCValues;
    int       *HWCSet;

} BurstInfo_t;

BurstInfo_t * new_BurstInfo (int task_id, int thread_id, int num_bursts, int hwc_per_burst);
void BurstInfo_Free (BurstInfo_t *bi);
void BurstInfo_FreeArray (BurstInfo_t **bi_list, int num_be);

void BurstInfo_Serialize(BurstInfo_t *bi, int *TaskID, int *ThreadID, int *num_Bursts, int *num_HWCperBurst, long long **Timestamp, long long **Durations, long long **HWCValues, int **HWCSet);
BurstInfo_t * BurstInfo_Assemble (int TaskID, int ThreadID, int num_Bursts, int num_HWCperBurst, long long *Timestamp, long long *Durations, long long *HWCValues, int *HWCSet);


#endif /* __BURST_INFO_H__ */
