#include <mrnet/MRNet.h>
#include "mrnet_commands.h"

using MRN::Stream;

class Protocol 
{
	public:
		virtual int run(Stream * stream) = 0;
};

