#include "BurstsExtractor.h"
#include "OnlineControl.h"
#include "ClusteringWorker.h"
#if USE_HARDWARE_COUNTERS
# include "hwc.h"
#endif
#include "utils.h"
#include "online_buffers.h"

static void ClusteringDataExtractor(libDistributedClustering *libClustering)
{
  int i, num_bursts = 0;
  
  /* Extract all bursts since the last analysis */
  BurstsExtractor *extBursts = new BurstsExtractor(0, false);

  extBursts->Extract(
    Online_GetAppResumeTime(),
    Online_GetAppPauseTime()
  );

  Bursts *data = extBursts->GetBursts();

  /* Query the counters read in each counter set */ 
  map< int, vector<event_type_t> > HWCSetToIds;
  int num_sets = HWC_Get_Num_Sets();

  for (int set_id=0; set_id < num_sets; set_id++)
  {
    int *array_hwc_ids = NULL;
    int  num_hwcs  = HWC_Get_Set_Counters_ParaverIds(set_id, &array_hwc_ids);
    vector<event_type_t> vector_hwc_ids;
    for (int hwc=0; hwc < num_hwcs; hwc++)
    {
      vector_hwc_ids.push_back( (event_type_t)array_hwc_ids[hwc] );
    }
    xfree(array_hwc_ids);
    HWCSetToIds[set_id] = vector_hwc_ids;
  }

  /* Feed the bursts one by one to the clustering library */
  for (int current_burst=0; current_burst < data->GetNumberOfBursts(); current_burst++)
  {
    int set_id = data->GetBurstCountersSet(current_burst);

    timestamp_t BeginTime = data->GetBurstTime(current_burst);
    duration_t  Duration  = data->GetBurstDuration(current_burst);
    timestamp_t EndTime   = BeginTime + Duration;
    long long *hwcs = NULL;
    data->GetBurstCountersValues(current_burst, hwcs);
    map<event_type_t, event_value_t> Counters;

    for (int current_hwc=0; current_hwc < HWCSetToIds[set_id].size(); current_hwc++)
    {
      event_type_t hwc_id = (event_type_t) HWCSetToIds[set_id][current_hwc];
      event_value_t hwc_value = (event_value_t) hwcs[current_hwc];

      Counters[hwc_id] = hwc_value;
    }
    xfree(hwcs);

    /* DEBUG 
    fprintf(stderr, "[DEBUG] ClusteringDataExtractor: Burst #%d: BeginTime=%llu Duration=%llu EndTime=%llu Counters=", i, BeginTime, Duration, EndTime);
    map<event_type_t, event_value_t>::iterator it;
    for (it = Counters.begin(); it != Counters.end(); ++ it)
    {
      fprintf(stderr, "%d:%lld ", it->first, it->second);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "[DEBUG] Calling libClustering->NewBurst on burst #%d\n", current_burst); */

    libClustering->NewBurst( 0, 0, 0, BeginTime, EndTime, Duration, Counters );
  }

  delete extBursts;
}

static void ClustersFeedback(vector<timestamp_t> &BeginTimes, vector<timestamp_t> &EndTimes, vector<cluster_id_t> &ClusterIDs)
{
  for (int i=0; i<ClusterIDs.size(); i++)
  {
    TRACE_ONLINE_EVENT(BeginTimes[i], CLUSTER_ID_EV, ClusterIDs[i] + PARAVER_OFFSET);
    TRACE_ONLINE_EVENT(EndTimes[i], CLUSTER_ID_EV, 0);
  }
}


ClusteringWorker::ClusteringWorker() : TDBSCANWorkerOnline( &ClusteringDataExtractor, &ClustersFeedback )
{
}

