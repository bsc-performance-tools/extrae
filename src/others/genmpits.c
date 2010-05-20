/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#define MAX_MPIT_FILES 8192

char **fitxers = NULL;

void AddFile (char *file)
{
	int task = 0;
	char *tmp = (char*) malloc((strlen(file)+1)*sizeof(char));
	char *tmp_name;
	int i;

	strncpy (tmp, file, strlen(file));
	tmp_name = &(tmp[strlen(tmp)-strlen(EXT_MPIT)-DIGITS_TASK-DIGITS_THREAD]);
	for (i = 0; i < DIGITS_TASK; i++)
	{
		task = task * 10 + ((int) tmp_name[0] - ((int) '0'));
		tmp_name++;
	}
	fitxers[task] = tmp;
}

void AddFileOfFiles (char *file)
{
  FILE *fd = fopen (file, "r");
  char path[4096];

  if (fd == NULL)
  {
    fprintf (stderr, "Unable to open %s file.\n", file);
    return;
  }

  while (!feof (fd))
  {
    fscanf (fd, "%s\n", path);
    AddFile (path);
	}

  fclose (fd);
}

int main (int argc, char *argv[])
{
	int i;

	fitxers = (char**) malloc (MAX_MPIT_FILES*sizeof(char*));
	for (i = 0; i < MAX_MPIT_FILES; i++)
		fitxers[i] = NULL;

	i = 1;
	while (i < argc)
	{
		if (strncmp("-f",argv[i],2) == 0)
		{
			i++;
			if (i < argc)
				AddFileOfFiles(argv[i]);
			else
				fprintf (stderr, "You must give a parameter for the -f option\n");
		}
		else
			AddFile (argv[i]);
		i++;
	}

	for (i = 0; i < MAX_MPIT_FILES; i++)
	{
		if (fitxers[i] != NULL)
		{
			fprintf (stdout, "%s\n", fitxers[i]);
			free (fitxers[i]);
		}
	}
	free (fitxers);

	return 0;
}

