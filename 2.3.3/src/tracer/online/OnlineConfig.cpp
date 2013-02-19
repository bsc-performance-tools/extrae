#include "OnlineConfig.h"

static int OnlineUserEnabled = 0;

static int AnalysisType = ONLINE_DO_NOTHING;
static int AnalysisFreq = 30;

void Online_Enable() 
{
  OnlineUserEnabled = 1;
}

void Online_Disable()
{
  OnlineUserEnabled = 0;
}

int Online_isEnabled()
{
  return OnlineUserEnabled;
}

void Online_SetAnalysis(int analysis_type)
{
  AnalysisType = analysis_type;
}

void Online_SetFrequency(int seconds) 
{
  AnalysisFreq = seconds;
}

int Online_GetAnalysis()
{
  return AnalysisType;
}

int Online_GetFrequency()
{
  return AnalysisFreq;
}

