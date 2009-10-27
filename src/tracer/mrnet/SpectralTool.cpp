#include "SpectralTool.h"
#include <stdlib.h>
#include <stdio.h>

SpectralTool::SpectralTool(BurstInfo_t **bi_list, int count, char * prefix)
{
	struct signal_elem ** signals = NULL;
	signalSize * sigsize = NULL;

	signals = (struct signal_elem **)malloc(count * sizeof(struct signal_elem *));
	sigsize = (int *)malloc(count * sizeof(signalSize));

	for (int i=0; i<count; i++)
	{
        BurstInfo_t *bi = bi_list[i]; 

		signalSize tmp  = generateSignal (bi->Timestamp, bi->Durations, bi->Durations, bi->num_Bursts, &(signals[i]));
		sigsize[i] = tmp;
	}

	sum_sigsize = addNSignals(signals, sigsize, count, &sum_signal);

	for (int i=0; i<count; i++)
	{
		destroySignal(signals[i]);
	}

	outFilePrefix = prefix;
	numPeriods = 0;
	listPeriods = NULL;
}

void SpectralTool::execute()
{
	numPeriods = ExecuteAnalysis(sum_signal, sum_sigsize, 2, outFilePrefix, &listPeriods);
}

int SpectralTool::get_NumPeriods()
{
	return numPeriods;
}

Period_t * SpectralTool::get_Period(int np)
{
	return listPeriods[np];
}

