#ifndef __ONLINE_CONFIG_H__
#define __ONLINE_CONFIG_H__

enum
{
  ONLINE_DO_NOTHING,
  ONLINE_DO_CLUSTERING,
  ONLINE_DO_SPECTRAL
};

#if defined(__cplusplus)
extern "C" {
#endif

void Online_Enable();
void Online_Disable();
int  Online_isEnabled();

void Online_SetAnalysis(int analyis_type);
void Online_SetFrequency(int seconds);

int Online_GetAnalysis();
int Online_GetFrequency();

#if defined(__cplusplus)
}
#endif

#endif /* __ONLINE_CONFIG_H__ */
