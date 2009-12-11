#include <stdio.h>
#include <stdlib.h>
#include "BurstInfo_BE.h"
#include "events.h"
#include "record.h"
#include "timesync.h"
#include "utils.h"
#include "trace_buffers.h"
//#include "num_hwc.h"
#include "hwc.h"

/* Extract the information from the buffer */
int Extract_BurstInfo (int task_id, int thread_id, unsigned long long min_time, unsigned long long max_time, unsigned long long min_burst_length, BurstInfo_t **bi_io)
{
	int count_events = 0, skip_events = 0, total_events = 0, current_burst = 0;
	Buffer_t *buffer = TRACING_BUFFER(thread_id);

	BufferIterator_t *it2 = BIT_NewForward(buffer);
	BufferIterator_t *it = BIT_NewRange(buffer, min_time, max_time);
	long long fake_latency = 0;
	event_t *last_begin = NULL;
	int Mb = 0;
	BurstInfo_t *bi = NULL;


	total_events = Buffer_GetFillCount(buffer);

#if 0
	/* Skip already clustered bursts */
	while ((!BIT_OutOfBounds(it)) && (BIT_IsMaskSet(it, MASK_CLUSTERED)))
	{
		skip_events ++;
		BIT_Next(it);	
	}
#endif

	while ( (!BIT_OutOfBounds(it2)) && (Get_EvTime(BIT_GetEvent(it2)) <= min_time) )
	{
		skip_events ++;
		BIT_Next (it2);
	}
	/* fprintf(stderr, "[EXTRACTOR %d] skip_events=%d (remaining=%d)\n", task_id, skip_events, total_events-skip_events); */

	if (!BIT_OutOfBounds(it))
	{
		bi = new_BurstInfo (task_id, thread_id, (total_events - skip_events) / 2, MAX_HWC);

		fake_latency = Get_EvTime(BIT_GetEvent(it));

		while (!BIT_OutOfBounds(it))
		{
			event_t *current = BIT_GetEvent(it);

			if (Event_IsBurstBegin (current))
			{
				/* Save a pointer to the last "Burst Begin" event seen */
				last_begin = current;
			}
			else if ((Event_IsBurstEnd (current)) && (last_begin != NULL))
			{
//				long long ts = Get_EvTime(last_begin) - fake_latency;
				long long ts = TIMESYNC(task_id, Get_EvTime(last_begin));
//				long long ts = Get_EvTime(last_begin);
				long long dur = Get_EvTime(current) - Get_EvTime(last_begin);
				long long *hwc = Get_EvHWCVal(current);
				int set = Get_EvHWCSet(current);

				/* Filter bursts by duration */
				if (dur >= min_burst_length)
				{
					if (set != Get_EvHWCSet(last_begin))
					{
						/* fprintf(stderr, "[EXTRACTOR %d] Burst Begins with set %d and ends with set %d\n", 
							task_id, Get_EvHWCSet(last_begin), set); */
					}
					else 
					{
						bi->Timestamp[current_burst] = ts;
						bi->Durations[current_burst] = dur;
						bi->HWCSet[current_burst] = set;
						for (int i=0; i<bi->num_HWCperBurst; i++)
						{
							int hwc_idx = (current_burst * bi->num_HWCperBurst) + i;
							bi->HWCValues[hwc_idx] = hwc[i];
			
							if (!HWC_Resetting())
							{
								long long *ini_hwc = Get_EvHWCVal(last_begin);
								bi->HWCValues[hwc_idx] -= ini_hwc[i];
							}
						}
						bi->num_Bursts = ++current_burst;
					}
				}
				last_begin = NULL;
			}

#if 0
			/* Mask as clustered */
			BIT_MaskSet(it, MASK_CLUSTERED);
#endif

			count_events ++;
			BIT_Next(it);
		}

		/* Count how many MBytes of data were processed */
//		Mb = ((count_events * sizeof(event_t)) + BYTES_PER_MB - 1) / BYTES_PER_MB;
		Mb = (count_events * sizeof(event_t));
	}

	*bi_io = bi;
	/* fprintf(stderr, "[EXTRACTOR %d] count_events=%d mb=%d\n", task_id, count_events, Mb); */
	return Mb;
}

/* Check whether the given event represents the beginning of a burst */
int Event_IsBurstBegin (event_t *current)
{
    int type = Get_EvEvent (current);
    int value = Get_EvValue (current);

    return (((IsMPI(type)) && (value == EVT_END)) || ((IsBurst(type)) && (value == EVT_BEGIN)));
}

/* Check whether the given event represents the end of a burst */
int Event_IsBurstEnd (event_t *current)
{
    int type = Get_EvEvent (current);
    int value = Get_EvValue (current);

    return (((IsMPI(type)) && (value == EVT_BEGIN)) || ((IsBurst(type)) && (value == EVT_END)));
}

void Filter_Periods (int task_id, int thread_id, int numPeriods, Period_t *listPeriods)
{
	int curPeriod = 0;
	bool found = false;
	event_t *evt = NULL;
	Buffer_t *buffer = TRACING_BUFFER(thread_id);
	BufferIterator_t *it = BIT_NewForward(buffer);
	
    /* Discard all data */
    NewMask_SetRegion(buffer, Buffer_GetHead(buffer), Buffer_GetTail(buffer), MASK_NOFLUSH);

    curPeriod = 0;
    while (curPeriod < numPeriods)
    {
		unsigned long long bestIniTime, bestEndTime;
		event_t *periodBeginEvt = NULL, *periodEndEvt = NULL;

        if ((listPeriods[curPeriod].iters < 2) || (listPeriods[curPeriod].length == 0))
        {
            curPeriod ++;
            continue;
        }

        bestIniTime = TIMEDESYNC(task_id, listPeriods[curPeriod].best_ini);
        bestEndTime = TIMEDESYNC(task_id, listPeriods[curPeriod].best_end);

		/* Find the event whose time corresponds to the start of this period */
		periodBeginEvt = NULL;
		while ((!BIT_OutOfBounds(it)) && (periodBeginEvt == NULL))
		{
			evt = BIT_GetEvent(it);

			if (Get_EvTime(evt) < bestIniTime) BIT_Next(it);
			else periodBeginEvt = evt;
		}

		/* Find the end of the period */
		periodEndEvt = periodBeginEvt;
		found = false;
		while ((!BIT_OutOfBounds(it)) && (!found))
		{
			evt = BIT_GetEvent(it);

			periodEndEvt = evt;
			if (Get_EvTime(evt) <= bestEndTime) BIT_Next(it);
			else found = true;
		}

		if ((periodBeginEvt != NULL) && (periodEndEvt != periodBeginEvt))
		{
			NewMask_UnsetRegion (buffer, periodBeginEvt, periodEndEvt, MASK_NOFLUSH);
		}

        curPeriod ++;
    }
}

