#ifndef __MX_COUNTERS_H__
#define __MX_COUNTERS_H__

extern struct mxmpi_var mxmpi;

int MYRINET_num_counters ();
int MYRINET_start_counters ();
int MYRINET_reset_counters ();
int MYRINET_read_counters (int num_events, uint32_t * values);
int MYRINET_counters_labels (char *** avail_counters);

#ifdef MX_MARENOSTRUM_API
int MYRINET_num_routes ();
int MYRINET_read_routes (int mpi_rank, int num_routes, uint32_t * values);
int MYRINET_reset_routes ();
#endif

#endif
