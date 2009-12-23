#ifndef __CLUSTER_TOOL_H__
#define __CLUSTER_TOOL_H__

#include "MRNetClustering.h"
#include "BurstInfo_FE.h"
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include "extern.h" /* #define PARAVER_OFFSET */

//#define BOTH_REPRESENTATIVES_AND_RANDOM
#define CLUSTERING_FED_RANDOMLY
#define MAX_REPRESENTATIVES 8
#define PCT_BURSTS 15
#define MAX_EASILY_HANDLES 8000

#define MIN(a,b) ( a < b ? a : b)
#define timerclear(tvp) ((tvp)->tv_sec = (tvp)->tv_usec = 0)
#define diff_time(entry_time, exit_time) (long)(exit_time.tv_sec - entry_time.tv_sec)

#define CLUSTERING_BASENAME "CLUSTERING_STEP_"
#define MAX_BASENAME_LENGTH 256

typedef std::map< std::pair< int, int >, std::vector< int > > ClusterIDs_m;

typedef std::map< int, std::vector<int> > HWCSetIds_m;

class ClusterTool 
{
public:
	ClusterTool(string basename, HWCSetIds_m * hwc_set_ids);
	~ClusterTool();
	void feed(int num_be, BurstInfo_t **bi_list, int Max_Representatives = 0, int Pct_Bursts = 0);
	bool cluster();
	void classify (int num_be, BurstInfo_t **bi_list, ClusterIDs_m & cids);
	void print_plots(string infix);
	string getBasename();
	MRNetClustering * getClustering();


private:
	MRNetClustering *C;
	string OutputBasename;
	HWCSetIds_m * HWC_Set_Ids;

	void convert (BurstInfo_t *bi, int nb, INT32 *task_id, INT32 *thread_id, UINT64 *timestamp, UINT64 *duration, vector<INT64> *hwc_values);
};

#endif /* __CLUSTER_TOOL_H__ */
