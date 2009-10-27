#include "libextract/BurstInfo.h"
#include "signal_interface.h"

class SpectralTool
{
	struct signal_elem * sum_signal;
	int sum_sigsize;

    char * outFilePrefix;
    int numPeriods;
    Period_t ** listPeriods;

	public:
		SpectralTool(BurstInfo_t **, int, char *);
		void execute();

		int get_NumPeriods();
		Period_t * get_Period(int np);
};

