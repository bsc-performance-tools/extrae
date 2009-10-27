#ifndef __BURSTINFO_BE_H__
#define __BURSTINFO_BE_H__

#include "BurstInfo.h"
#include "signal_interface.h"
#include "record.h"

#define BYTES_PER_MB (1024 * 1024)

int Extract_BurstInfo (int task_id, int thread_id, unsigned long long min_time, unsigned long long max_time, unsigned long long min_burst_length, BurstInfo_t **bi_io);
int Event_IsBurstBegin (event_t *current);
int Event_IsBurstEnd (event_t *current);
void Filter_Periods (int task_id, int thread_id, int numPeriods, Period_t * listPeriods);

#endif /* __BURSTINFO_BE_H__ */
