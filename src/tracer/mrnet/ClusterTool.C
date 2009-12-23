#include "ClusterTool.h"

ClusterTool::ClusterTool(string basename, HWCSetIds_m * hwc_set_ids)
{
	C = new MRNetClustering();
	OutputBasename = basename;
	//StepID = step_id;
	HWC_Set_Ids = hwc_set_ids;
	C->InitClustering("./cl.I.IPC.xml", true, true);
}

ClusterTool::~ClusterTool()
{
	delete C;
}

void ClusterTool::feed(int num_be, BurstInfo_t **bi_list, int Max_Representatives, int Pct_Bursts)
{
    int i, j;
    int representatives = 0;
    int abs_total_bursts = 0;

    for (i=0; i<num_be; i++)
    {
        abs_total_bursts += bi_list[i]->num_Bursts;
    }
    int EASILY_HANDLES = MIN(MAX_EASILY_HANDLES, abs_total_bursts);

/* #if defined(BOTH_REPRESENTATIVES_AND_RANDOM) || !defined(CLUSTERING_FED_RANDOMLY) */
    if (Max_Representatives > 0) 
    {
		/* Select a few representative tasks to clusterize */
	    representatives = MIN(num_be, Max_Representatives);
	    fprintf(stderr, "[FE] Selecting all bursts from %d representative tasks\n", representatives);
	    for (i=0; i<representatives; i++)
	    {
	        BurstInfo_t *bi = bi_list[i];
	        for (j=0; j<bi->num_Bursts; j++)
	        {
	            INT32  TaskId, ThreadId;
	            UINT64 BeginTime, Duration;
	            vector<INT64> HWCValues;

	            convert (bi, j, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);
	            C->SetHWCountersGroup((*HWC_Set_Ids)[bi->HWCSet[j]]);
	            C->NewPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);
	        }
	        fprintf(stderr, "%d ", bi->TaskID);
	    }
	    fprintf(stderr, "\n");
	}
/* #endif */

/* #if defined(BOTH_REPRESENTATIVES_AND_RANDOM) || defined(CLUSTERING_FED_RANDOMLY) */
    /* Select random bursts from all remaining tasks to clusterize */
    for (i=representatives; i<num_be; i++)
    {
        BurstInfo_t *bi = bi_list[i];
        int total_bursts = bi->num_Bursts;
		int num_samples = 0;

		if (Pct_Bursts > 0)
		{
			/* Select the given % of bursts */
    	    num_samples = (total_bursts * Pct_Bursts) / 100;
		}
		else
		{
			/* Select the corresponding % of bursts defined by EASILY_HANDLES */
        	double task_contribution_pct = (double)(total_bursts * 100) / (double)abs_total_bursts;
	        num_samples = (int)((EASILY_HANDLES * task_contribution_pct) / 100);
        	fprintf(stderr, "task_contribution_pct=%f ", task_contribution_pct);
		}	
		/* Calculate the intervals to get the samples from */
        double range = total_bursts / num_samples;

        fprintf(stderr, "total_bursts=%d num_samples=%d range=%f abs_total_bursts=%d\n", 
			total_bursts, num_samples, range, abs_total_bursts);
/*
        fprintf(stderr, "[FE] RANDOM SAMPLES FROM: ");
*/
        srand(getpid());

        for (j=0; j<num_samples; j++)
        {
            int sample;

            sample = (int) (range*rand()/(RAND_MAX+1.0));
            sample += (int)(j * range);

            if ((sample >= 0) || (sample < total_bursts))
            {
                INT32  TaskId, ThreadId;
                UINT64 BeginTime, Duration;
                vector<INT64> HWCValues;

                convert (bi, sample, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);
                C->SetHWCountersGroup((*HWC_Set_Ids)[bi->HWCSet[sample]]);
                C->NewPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);
            }
            else
            {
                fprintf(stderr, "BAD_SAMPLE!!\n");
            }
        }
/*
        fprintf(stderr, "%d ", bi->TaskID);
        fprintf(stderr, "\n");
*/
    }
/* #endif */
}

bool ClusterTool::cluster()
{
    struct timeval entry_time;
    struct timeval exit_time;
    char cmd[2048];
    bool ok = false;

    /* Invoke the clustering tool */
    fprintf(stderr, "[FE] do_Clusters: Invoking the clustering tool...\n");
    fflush(stderr);
    timerclear(&entry_time);
    timerclear(&exit_time);
    gettimeofday(&entry_time, 0);

    ok = C->ExecuteClustering (true, OutputBasename.c_str());

    gettimeofday(&exit_time, 0);
    fprintf(stderr, "[FE] do_Clusters: Clustering tool returns %s after %ld seconds.\n", (ok ? "successfully" : "with errors"), diff_time(entry_time, exit_time));

    /* Compute CPI Stack statistics */
    snprintf(cmd, sizeof(cmd), "%s/bin/cpistack_stats.pl %s.clusters_info.csv", getenv("MPITRACE_HOME"), OutputBasename.c_str());
    fprintf(stderr, "[FE] Computing CPI Stack statistics: %s\n", cmd);
    system (cmd);

	return ok;
}


void ClusterTool::classify (int num_be, BurstInfo_t **bi_list, ClusterIDs_m & cids)
{
    struct timeval entry_time;
    struct timeval exit_time;

    timerclear(&entry_time);
    timerclear(&exit_time);

    gettimeofday(&entry_time, 0);

    /* Classify all points */
    for (int i=0; i<num_be; i++)
    {
        BurstInfo_t *bi = bi_list[i];

        std::pair< int, int > key = std::make_pair(bi->TaskID, bi->ThreadID);
        for (int j=0; j<bi->num_Bursts; j++)
        {
            INT32  TaskId, ThreadId;
            UINT64 BeginTime, Duration;
            vector<INT64> HWCValues;

            convert (bi, j, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);

            C->SetHWCountersGroup((*HWC_Set_Ids)[bi->HWCSet[j]]);
            cids[key].push_back( C->ClassifyPoint( TaskId, ThreadId, BeginTime, Duration, HWCValues ) + PARAVER_OFFSET );
        }
    }
    gettimeofday(&exit_time, 0);
    fprintf(stderr, "[FE] do_Clusters: Classification lasted %ld seconds.\n", diff_time(entry_time, exit_time));
}

void ClusterTool::print_plots(string infix)
{
	string basename = OutputBasename + infix;

	C->PrintGNUPlot (basename.c_str());
}

void ClusterTool::convert (BurstInfo_t *bi, int nb, INT32 *task_id, INT32 *thread_id, UINT64 *timestamp, UINT64 *duration, vector<INT64> *hwc_values)
{
    *task_id = bi->TaskID;
    *thread_id = bi->ThreadID;
    *timestamp = bi->Timestamp[nb];
    *duration = bi->Durations[nb];
    for (int i=0; i<bi->num_HWCperBurst; i++)
    {
        (*hwc_values).push_back(bi->HWCValues[(nb*bi->num_HWCperBurst)+i]);
    }
}

string ClusterTool::getBasename()
{
	return OutputBasename;
}

MRNetClustering * ClusterTool::getClustering()
{
	return C;
}

