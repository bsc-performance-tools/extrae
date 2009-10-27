/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/protocol.C,v $
 | 
 | @last_commit: $Date: 2009/04/21 10:40:40 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: protocol.C,v 1.4 2009/04/21 10:40:40 gllort Exp $";

#include <mrnet/MRNet.h>
#include "mrnet_commands.h"
#include "protocol.h"

int Execute_Command_Handler (cmd_handler_t handlersList[], int cmd_id, Stream * stream)
{
	int i, found, rc = 0;
	found = FALSE;

	i = 0;
	while ((handlersList[i].command != MRN_ALL_COMMANDS) && (!found))
	{
		if (handlersList[i].command == cmd_id)
		{
			found = TRUE;
		}
		else 
		{
			i ++;
		}
	}
	if (found)
	{
		rc = handlersList[i].handler(stream);
	}
	else
	{
		PRINT_WHERE;
		fprintf(stderr, "Handler not found for command '%d'.\n", cmd_id);
	}
	return rc;
}

#if 0
void * Get_Handler (cmd_handler_t handlersList[], int cmd_id)
{
    int i, found, rc = 0;
    found = FALSE;

    i = 0;
    while ((handlersList[i].command != MRN_ALL_COMMANDS) && (!found))
    {
        if (handlersList[i].command == cmd_id)
        {
            found = TRUE;
        }
        else
        {
            i ++;
        }
    }
    if (found)
    {
		return handlersList[i].handler;
    }
    else
    {
        PRINT_WHERE;
        fprintf(stderr, "Handler not found for command '%d'.\n", cmd_id);
    	return NULL;
    }
}
#endif

