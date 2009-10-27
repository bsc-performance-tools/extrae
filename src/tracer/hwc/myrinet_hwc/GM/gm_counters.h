#ifndef __GM_COUNTERS_H__
#define __GM_COUNTERS_H__

void MYRINET_start_counters();
void MYRINET_reset_counters();
int  MYRINET_read_counters(int num_events, u_int32_t * values);
int  MYRINET_num_counters();
int  MYRINET_counters_labels(char *** avail_counters);

#endif /* __GM_COUNTERS_H__ */
