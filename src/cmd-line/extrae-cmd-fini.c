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
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif

#include "extrae-cmd.h"
#include "extrae-cmd-fini.h"

#include "wrapper.h"

int Extrae_CMD_Fini (int i, int argc, char *argv[])
{
	char HOST[1024];

	UNREFERENCED_PARAMETER (i);
	UNREFERENCED_PARAMETER (argc);
	UNREFERENCED_PARAMETER (argv);

	if (0 == gethostname (HOST, sizeof(HOST)))
	{
		char TMPFILE[2048];

		char CMDPREFIX[TMP_DIR_LEN];
		Extrae_get_cmd_prefix(CMDPREFIX);

		sprintf(TMPFILE, "%s"EXTRAE_CMD_FILE_PREFIX"%s", CMDPREFIX, HOST);
		if (unlink(TMPFILE) == -1)
		{
			int err = errno;
			fprintf(stderr, PACKAGE_NAME"("CMD_FINI"): %s (%s)\n", strerror(err), TMPFILE);
		}
	}

	return 0;
}

