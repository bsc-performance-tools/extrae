/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/clusterize.C,v $
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "BurstInfo_FE.h"
#include "MRNetClustering.h"

std::map< int, std::vector<int> > HWC_Sets_Ids;

#define MIN(a,b) ( a < b ? a : b)

#if 0
void FeedWithData (MRNetClustering *C, int num_be, BurstInfo_t **bi_list, int num_representatives)
{
    int i, j, k;
	int max;

	if (num_representatives > 0)
		max = MIN(num_be, num_representatives);
	else
		max = num_be;
	
	for (i=0; i<max; i++)
	{
		BurstInfo_t *bi = bi_list[i];

		for (j=0; j<bi->num_Bursts; j++)
		{
            INT32  TaskId, ThreadId;
            UINT64 BeginTime, Duration;
            vector<INT64> HWCValues;

			TaskId = bi->TaskID;
			ThreadId = bi->ThreadID;
			BeginTime = bi->Timestamp[j];
			Duration = bi->Durations[j];
			for (k=0; k<bi->num_HWCperBurst; k++)
			{
				HWCValues.push_back(bi->HWCValues[(j*bi->num_HWCperBurst)+k]);
			}			
			C->SetHWCountersGroup(HWC_Sets_Ids[bi->HWCSet[j]]);
			C->NewPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);
			fprintf(stderr, "[FeedWithData] calling NewPoint (%d, %d, %lld, %lld)\n", TaskId, ThreadId, BeginTime, Duration);
			for (k=0; k<bi->num_HWCperBurst; k++)
			{
				fprintf(stderr, "hwc%d=%lld ", k, HWCValues[k]); 
			}
			fprintf(stderr, "\n");
		}
	}
}
#endif

void Convert (BurstInfo_t *bi, int nb, INT32 *task_id, INT32 *thread_id, UINT64 *timestamp, UINT64 *duration, vector<INT64> *hwc_values)
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

void FeedWithData (MRNetClustering *C, int num_be, BurstInfo_t **bi_list, int Max_Representatives, int Pct_Bursts)
{
    int i, j, k;
    int representatives = 0;


	if (Max_Representatives > 0) 
	{
	    /* Select a few tasks to clusterize */

	    representatives = MIN(num_be, Max_Representatives);
	    fprintf(stderr, "[FE] REPRESENTATIVES: ");
	    for (i=0; i<representatives; i++)
	    {
			BurstInfo_t *bi = bi_list[i];
	        for (j=0; j<bi->num_Bursts; j++)
	        {
	            INT32  TaskId, ThreadId;
	            UINT64 BeginTime, Duration;
	            vector<INT64> HWCValues;

	            Convert (bi, j, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);
	            C->SetHWCountersGroup(HWC_Sets_Ids[bi->HWCSet[j]]);
	            C->NewPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);
	        }
	        fprintf(stderr, "%d ", bi->TaskID);
	    }
	    fprintf(stderr, "\n");
	}

	if (Pct_Bursts > 0)
	{
	    /* Select random bursts from all tasks to clusterize */

	    for (i=representatives; i<num_be; i++)
	    {
	        BurstInfo_t *bi = bi_list[i];
	        int total_bursts = bi->num_Bursts;
	        int num_samples = (total_bursts * Pct_Bursts) / 100;
	        double range = total_bursts / num_samples;

	        fprintf(stderr, "total_bursts=%d num_samples=%d range=%f\n", total_bursts, num_samples, range);
	        fprintf(stderr, "[FE] RANDOM SAMPLES FROM: ");
	
	        srand(getpid());

	        for (j=0; j<num_samples; j++)
	        {
	            int sample;

	            sample = (int) (range*rand()/(RAND_MAX+1.0));
	            sample += j * range;
	
	            if ((sample >= 0) || (sample < total_bursts))
	            {
					INT32  TaskId, ThreadId;
					UINT64 BeginTime, Duration;
					vector<INT64> HWCValues;

					Convert (bi, sample, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);
					C->SetHWCountersGroup(HWC_Sets_Ids[bi->HWCSet[sample]]);
	                C->NewPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);
	            }
	            else
	            {
	                fprintf(stderr, "BAD_SAMPLE!!\n");
	            }
	        }
	        fprintf(stderr, "%d ", bi->TaskID);
	        fprintf(stderr, "\n");
	    }
	}
}

void Dump_CIDS(MRNetClustering *C, int num_be, BurstInfo_t **bi_list, char *OutPrefix)
{
	FILE * outfile;
	char tmpfilename[256];

	snprintf(tmpfilename, sizeof(tmpfilename), "%s%s", OutPrefix, ".CIDLIST");
	outfile = fopen(tmpfilename, "w+");

    /* Classify all points */
    for (int i=0; i<num_be; i++)
    {
		BurstInfo_t *bi = bi_list[i];
		for (int j=0; j<bi->num_Bursts; j++)
		{
			INT32  TaskId, ThreadId;
			UINT64 BeginTime, Duration;
			vector<INT64> HWCValues;
			INT32 cid;

			Convert (bi, j, &TaskId, &ThreadId, &BeginTime, &Duration, &HWCValues);
			C->SetHWCountersGroup(HWC_Sets_Ids[bi->HWCSet[j]]);
			cid = C->ClassifyPoint(TaskId, ThreadId, BeginTime, Duration, HWCValues);

			fprintf(outfile, "%d\n", (int)cid);
		}
	}
	fclose(outfile);
}


int main(int argc, char **argv)
{
	MRNetClustering *C = new MRNetClustering();
	char *OutPrefix, *InFile;
    BurstInfo_t **bi_list = NULL;
    int NumBackends = 0;
	int Max_Representatives = 0;
	int Pct_Bursts = 15;

    if (argc != 5) 
    {
        fprintf(stderr, "Syntax error: %s <input_data_file.bbi> <output_file_name> <Max_Representatives> <Pct_Bursts>\n", basename(argv[0]));
        exit(1);
    }
	InFile = argv[1];
	OutPrefix = argv[2];
	Max_Representatives =  atoi(argv[3]);
	Pct_Bursts = atoi(argv[4]);

	NumBackends = BurstInfo_LoadArray(InFile, &bi_list, &HWC_Sets_Ids);

    if ((NumBackends > 0) && (bi_list != NULL))
    {
		bool ok;

		C->InitClustering("./cl.I.IPC.xml", true, false);
	
		/* Convert input and feed clustering tool */
		FeedWithData (C, NumBackends, bi_list, Max_Representatives, Pct_Bursts);

		/* Execute the analysis */
		ok = C->ExecuteClustering (true, OutPrefix);
		C->PrintGNUPlot (OutPrefix);

        fprintf(stderr, "%s: Clustering tool returned %s.\n", basename(argv[0]), (ok ? "successfully" : "with errors"));

		if (ok) Dump_CIDS(C, NumBackends, bi_list, OutPrefix);
	}
    else
    {
        fprintf(stderr, "%s: Invalid data in file '%s'.\n", basename(argv[0]), argv[1]);
    }

	
	BurstInfo_FreeArray (bi_list, NumBackends);
	return 0;
}
