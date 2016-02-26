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
#ifdef HAVE_DIRENT_H
# include <dirent.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_LIMITS_H
# include <limits.h> /* defines NAME_MAX */
#endif

#if defined(__APPLE__)
# define NAME_MAX 512
#endif

#define MAX_MPIT_FILES 32768

char **files = NULL;
int lastfile = 0;

void DumpFiles (int pidlabel)
{
	int task, pid, i, j;
	char *tmp, *tmp_name;

	for (j = 0; j < lastfile; j++)
	{
		tmp = files[j];
		tmp_name = &(tmp[strlen(tmp)-strlen(EXT_MPIT)-DIGITS_TASK-DIGITS_THREAD]);
		for (task = 0, i = 0; i < DIGITS_TASK; i++, tmp_name++)
			task = task * 10 + ((int) tmp_name[0] - ((int) '0'));

		tmp_name = &(tmp[strlen(tmp)-strlen(EXT_MPIT)-DIGITS_PID-DIGITS_TASK-DIGITS_THREAD]);
		for (pid = 0, i = 0; i < DIGITS_PID; i++, tmp_name++)
			pid = pid * 10 + ((int) tmp_name[0] - ((int) '0'));

		/* If this is not the first file but we are in task id 0, generate a new app */
		if (task == 0 && j > 0)
			fprintf (stdout, "--\n");

		if (pidlabel)
			fprintf (stdout, "%s named process-%d\n", tmp, pid);
		else
			fprintf (stdout, "%s named\n", tmp);
	}
}

int SortFilesByTime_cbk (const void *p1, const void *p2)
{
	char *s1 = (*(char**) p1), *tmp1;
	char *s2 = (*(char**) p2), *tmp2;
	int pid1, pid2, i;

	tmp1 = &(s1[strlen(s1)-strlen(EXT_MPIT)-DIGITS_PID-DIGITS_TASK-DIGITS_THREAD]);
	for (pid1 = 0, i = 0; i < DIGITS_PID; i++, tmp1++)
		pid1 = pid1 * 10 + ((int) tmp1[0] - ((int) '0'));

	tmp2 = &(s2[strlen(s2)-strlen(EXT_MPIT)-DIGITS_PID-DIGITS_TASK-DIGITS_THREAD]);
	for (pid2 = 0, i = 0; i < DIGITS_PID; i++, tmp2++)
		pid2 = pid2 * 10 + ((int) tmp2[0] - ((int) '0'));

	if (pid1 == pid2)
		return 0;
	else if (pid1 > pid2)
		return 1;
	else
		return -1;
}

void SortFilesByTime(void)
{	
	qsort(files, lastfile, sizeof(char*), SortFilesByTime_cbk);
}

void AddFile (char *file)
{
	files[lastfile++] = strdup (file);
}

void AddFileOfFiles (char *file)
{
	FILE *fd = fopen (file, "r");
	char path[NAME_MAX];

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

int main (int argc, char *argv[])
{
	int pidlabel = FALSE;
	int i;

	files = (char**) malloc (MAX_MPIT_FILES*sizeof(char*));
	if (files == NULL)
	{
		fprintf (stderr, "Cannot allocate %ld bytes of memory to allocate file names\n", MAX_MPIT_FILES*sizeof(char*));
		return -2;
	}
	for (i = 0; i < MAX_MPIT_FILES; i++)
		files[i] = NULL;

	i = 1;
	while (i < argc)
	{
		if (strncmp ("-pidlabel", argv[i], 9) == 0)
		{
			pidlabel = TRUE;
		}
		else if (strncmp ("-f", argv[i], 2) == 0)
		{
			i++;
			if (i < argc)
				AddFileOfFiles(argv[i]);
			else
			{
				fprintf (stderr, "You must give a parameter for the -f option\n");
				return -2;
			}
		}
		else
		{
			struct stat sb;
			stat (argv[i], &sb);

			if ((sb.st_mode & S_IFMT) == S_IFDIR)
			{
				size_t len;
				struct dirent *de;
				DIR *d = opendir(argv[i]);

				if (d == NULL)
				{
					fprintf (stderr, "%s is not a directory!\n", argv[i]);
					return -1;
				}

				while ((de = readdir(d)) != NULL)
					if ((len = strlen(de->d_name)) > 5)
						if (strncmp (&de->d_name[len-5], ".mpit", 5) == 0)
						{
							char fullname[NAME_MAX];
							sprintf (fullname, "%s/%s", argv[i], de->d_name);
							AddFile (fullname);
						}
			}
			else if ((sb.st_mode & S_IFMT) == S_IFREG)
			{
				AddFile (argv[i]);
			}
			else
			{
				fprintf (stderr, "Parameter %s is neither a regular file nor a directory! Ignored...\n", argv[i]);
			}
		}
		i++;
	}

	SortFilesByTime ();
	DumpFiles (pidlabel);

	free (files);

	return 0;
}

