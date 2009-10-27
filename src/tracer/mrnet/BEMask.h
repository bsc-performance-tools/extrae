#ifndef __BEMASK_H__
#define __BEMASK_H__

#include <mrnet/MRNet.h>

using namespace MRN;

class BEMask
{
	int * Mask;
	int TotalBEs;

	public:
		BEMask(MRN::Network * n);
		~BEMask();
		int Check(int be);
		void Set(int be);
		void Unset(int be);
		int * get_Selection ();
		int size();
};

#endif /* __BEMASK_H__ */
