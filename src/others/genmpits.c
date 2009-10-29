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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
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

