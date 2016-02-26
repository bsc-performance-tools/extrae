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

#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "record.h"

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
		if (fscanf (fd, "%s\n", path) == 1)
			AddFile (path);

	fclose (fd);
}

void ReduceFileTo (char *file, int numofevents)
{
	int res;

	res = truncate (file, numofevents * sizeof(event_t));
	if (res < 0)
	{
		perror ("truncate failed. Reported error:\n");
		exit (-1);
	}
}

int main (int argc, char *argv[])
{
	int numOfEvents;
	int i;

	fitxers = (char**) malloc (MAX_MPIT_FILES*sizeof(char*));
	for (i = 0; i < MAX_MPIT_FILES; i++)
		fitxers[i] = NULL;

	if (argc < 2)
	{
		printf ("reducempit N File1.mpit [*.mpit]\n");
		exit (-1);
	}

	numOfEvents = atoi (argv[1]);
	if (numOfEvents < 1)
	{
		printf ("N must be >= 1");
		exit (-1);
	}

	i = 2;
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
		if (fitxers[i] != NULL)
		{
			ReduceFileTo (fitxers[i], numOfEvents);
		}


	free (fitxers);

	return 0;
}

