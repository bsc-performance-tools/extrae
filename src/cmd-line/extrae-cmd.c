/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#include "common.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "extrae-cmd.h"
#include "extrae-cmd-init.h"
#include "extrae-cmd-emit.h"
#include "extrae-cmd-fini.h"

int main (int argc, char *argv[])
{
	int i = 1;

	while (i < argc)
	{
		int advance = 0;

		if (!strncasecmp (argv[i], CMD_INIT, strlen (CMD_INIT)))
			advance = Extrae_CMD_Init (i+1, argc, argv);
		else if (!strncasecmp (argv[i], CMD_EMIT, strlen(CMD_EMIT)))
			advance = Extrae_CMD_Emit (i+1, argc, argv);
		else if (!strncasecmp (argv[i], CMD_FINI, strlen(CMD_FINI)))
			advance = Extrae_CMD_Fini (i+1, argc, argv);
		else
			fprintf (stderr, "Unknown option %s\n", argv[i]);

		i += advance+1;
	}

	if (argc == 1)
	{
		fprintf (stdout, PACKAGE_NAME": extrae-cmd Extrae command line version\n"
		                 "Available commands are:\n"
		                 " - init TASKID numThreads\n"
		                 "   Initializes the instrumentation package.\n"
		                 " - emit THREADID TYPE VALUE\n"
		                 "   Emits into the thread THREADID an event with a given pair <TYPE,VALUE>\n"
		                 " - fini\n"
		                 "   Finalizes the instrumentation package.\n");
	}

	return 0;
}
