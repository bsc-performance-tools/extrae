#ifndef __BURSTINFO_FE_H__
#define __BURSTINFO_FE_H__

#include "BurstInfo.h"
#include <map>
#include <vector>

void BurstInfo_DumpArray (BurstInfo_t **bi_list, int count, char *out_suffix, std::map< int, std::vector<int> > & m);
int BurstInfo_LoadArray (char *FileBBI, BurstInfo_t ***bi_io, std::map< int, std::vector<int> > * m);

#endif /* __BURSTINFO_FE_H__ */
