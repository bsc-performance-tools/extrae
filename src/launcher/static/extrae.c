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

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif

int main (int argc, char *argv[], char *envp[])
{
	struct stat buf;
	char *preload;
	char *xml;
	char tmp[512];
	int i;

	if (argc < 4)
	{
		fprintf (stderr, "You must provide:\n"
		                 " - location of the .so to preload\n"
		                 " - xml configuration file\n"
		                 " - binary and its arguments\n");
		return -2;
	}

	if (getenv("EXTRAE_HOME") == NULL)
	{
		fprintf (stderr, "EXTRAE_HOME is not set!\n");
		return -4;
	}

	sprintf (tmp, "%s/lib/%s", getenv("EXTRAE_HOME"), argv[1]);
	i = stat (tmp, &buf);
	if (i != 0)
	{
		fprintf (stderr, "Cannot access file %s\n", tmp);
		return -3;
	}

	i = stat (argv[2], &buf);
	if (i != 0)
	{
		fprintf (stderr, "Cannot access file %s\n", argv[2]);
		return -3;
	}

	i = 0;
	while (envp[i] != NULL)
		i++;

	envp[i] = preload = (char *) malloc ((strlen("LD_PRELOAD=")+strlen(getenv("EXTRAE_HOME"))+strlen("/lib/")+strlen(argv[1])+1)*sizeof(char));
	if (preload == NULL)
	{
		fprintf (stderr, "Cannot allocate memory for preload env variable\n");
		return -1;
	}
	sprintf (preload, "LD_PRELOAD=%s/lib/%s", getenv("EXTRAE_HOME"), argv[1]);
	i++;

	envp[i] = xml = (char*) malloc ((strlen ("EXTRAE_CONFIG_FILE=")+strlen(argv[2])+1)*sizeof(char));
	if (xml == NULL)
	{
		fprintf (stderr, "Cannot allocate memory for xml env variable\n");
		return -1;
	}
	sprintf (xml, "EXTRAE_CONFIG_FILE=%s", argv[2]);
	i++;

	envp[i] = NULL;

	execve (argv[3], &argv[3], envp);

	/* This will never be reached */
	return 0;
}
