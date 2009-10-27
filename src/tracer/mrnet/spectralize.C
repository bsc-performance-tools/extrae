/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/spectralize.C,v $
 | 
 | @last_commit: $Date: 2009/05/25 16:12:54 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id: spectralize.C,v 1.4 2009/05/25 16:12:54 gllort Exp $";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "BurstInfo_FE.h"
#include "SpectralTool.h"

#include <map>
#include <vector>
std::map< int, std::vector<int> > HWC_Sets_Ids;

int main(int argc, char **argv)
{
    char *OutPrefix, *InFile;
    BurstInfo_t **bi_list = NULL;
    int NumBackends = 0;
	SpectralTool * st;

    if (argc != 3)
    {
        fprintf(stderr, "Syntax error: %s <input_data_file.bbi> <output_file_name>\n", basename(argv[0]));
        exit(1);
    }
    InFile = argv[1];
    OutPrefix = argv[2];

	NumBackends = BurstInfo_LoadArray(InFile, &bi_list, &HWC_Sets_Ids);
    if ((NumBackends > 0) && (bi_list != NULL))
    {
		st = new SpectralTool (bi_list, NumBackends, OutPrefix);

		st->execute();
	}
    else
    {
        fprintf(stderr, "%s: Invalid data in file '%s'.\n", basename(argv[0]), InFile);
    }

    BurstInfo_FreeArray (bi_list, NumBackends);
    return 0;
}

