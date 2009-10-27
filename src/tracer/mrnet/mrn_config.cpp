#include "mrn_config.h"

static int AnalysisType = MRN_ANALYSIS_CLUSTER;
static int StartAfterSecs = 30; 
static int TargetTraceSize = DEFAULT_TARGET_TRACE_SIZE;

void MRNCfg_SetTargetTraceSize(int size)
{
	TargetTraceSize = size;
}

void MRNCfg_SetAnalysisType(int analysis, int start_after)
{
	AnalysisType = analysis;
	StartAfterSecs = start_after;
}

int MRNCfg_GetTargetTraceSize() { return TargetTraceSize; }
int MRNCfg_GetAnalysisType() { return AnalysisType; }
int MRNCfg_GetStartAfter() { return StartAfterSecs; }
