#ifndef __MRN_CONFIG_H__
#define __MRN_CONFIG_H__

#define DEFAULT_TARGET_TRACE_SIZE 100 /* Target trace default size in Mb's */

enum 
{
	MRN_ANALYSIS_CLUSTER,
	MRN_ANALYSIS_SPECTRAL
};

#if defined(__cplusplus)
extern "C" {
#endif
void MRNCfg_SetTargetTraceSize(int size);
void MRNCfg_SetAnalysisType(int analysis, int start_after);

int MRNCfg_GetTargetTraceSize();
int MRNCfg_GetAnalysisType();
int MRNCfg_GetStartAfter();
#if defined(__cplusplus)
}
#endif

#endif /* __MRN_CONFIG_H__ */

