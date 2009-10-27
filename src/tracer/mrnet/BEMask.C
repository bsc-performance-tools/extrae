#include "BEMask.h"

BEMask::BEMask (MRN::Network *n)
{
	TotalBEs = n->get_NetworkTopology()->get_BackEndNodes().size();

	Mask = (int *)malloc(sizeof(int) * TotalBEs);
	bzero (Mask, sizeof(int) * TotalBEs);
}

BEMask::~BEMask ()
{
	free (Mask);
}

int BEMask::Check (int be)
{
    if ((be >= 0) && (be < TotalBEs))
    {
        return Mask[be];
    }
	else
	{
		return 0;
	}
}

void BEMask::Set (int be)
{
	if ((be >= 0) && (be < TotalBEs))
	{
		Mask[be] = 1;
	}
}

void BEMask::Unset (int be)
{
	if ((be >= 0) && (be < TotalBEs))
	{
	    Mask[be] = 0;
	}
}

int * BEMask::get_Selection ()
{
	return Mask;
}

int BEMask::size()
{
	return TotalBEs;
}

