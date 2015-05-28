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
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif

#include <mpi.h>

#include "utils.h"
#include "utils_mpi.h"

static int ExtraeUtilsMPI_CheckSharedDisk_stat (const char *directory)
{
	int rank, size;
	PMPI_Comm_rank (MPI_COMM_WORLD, &rank);
	PMPI_Comm_size (MPI_COMM_WORLD, &size);

	if (size > 1)
	{
		int result;
		int howmany;
		char name[MPI_MAX_PROCESSOR_NAME];
		char name_master[MPI_MAX_PROCESSOR_NAME];
		int len;

		if (rank == 0)
		{
			PMPI_Get_processor_name(name, &len);
			PMPI_Bcast (name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
		}
		else
			PMPI_Bcast (name_master, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

		if (rank == 0)
		{
			struct stat s;
			int ret;
			unsigned res = 1;
			unsigned length = strlen(directory)+strlen("/shared-disk-testXXXXXX")+1;
			char *template = malloc (length*sizeof(char));
			if (!template)
			{
				fprintf (stderr, PACKAGE_NAME":Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
				exit (-1);
			}
			sprintf (template, "%s/shared-disk-testXXXXXX", directory);
			ret = mkstemp (template);
			if (ret < 0)
			{
				fprintf (stderr, PACKAGE_NAME":Error! cannot determine whether %s is a shared disk. Failed to create temporal file!\n", directory);
				exit (-1);
			}
			PMPI_Bcast (&length, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			PMPI_Bcast (template, length, MPI_CHAR, 0, MPI_COMM_WORLD);
			ret = stat (template, &s);
			PMPI_Bcast (&s, sizeof(struct stat), MPI_BYTE, 0, MPI_COMM_WORLD);
			PMPI_Reduce (&res, &howmany, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			unlink (template);
			free (template);
			result = howmany == size;
			PMPI_Bcast (&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		else
		{
			struct stat s, master_s;
			int ret, res;
			char *template;
			unsigned length;
			PMPI_Bcast (&length, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			template = malloc (length*sizeof(char));
			if (!template)
			{
				fprintf (stderr, PACKAGE_NAME":Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
				exit (-1);
			}
			PMPI_Bcast (template, length, MPI_CHAR, 0, MPI_COMM_WORLD);
			PMPI_Bcast (&master_s, sizeof(struct stat), MPI_BYTE, 0, MPI_COMM_WORLD);
			ret = stat (template, &s);
			res = ret == 0 &&
			  (master_s.st_uid == s.st_uid) &&
			  (master_s.st_gid == s.st_gid) &&
			  (master_s.st_ino == s.st_ino) &&
			  (master_s.st_mode == s.st_mode);
			PMPI_Reduce (&res, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			free (template);
			PMPI_Bcast (&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		return result;
	}
	else
		return directory_exists(directory);
}

/* A "slower" alternative? */
#if 0
static int ExtraeUtilsMPI_CheckSharedDisk_openread (const char *directory)
{
	int rank, size;
	PMPI_Comm_rank (MPI_COMM_WORLD, &rank);
	PMPI_Comm_size (MPI_COMM_WORLD, &size);

	if (size > 1)
	{
		int result;
		int howmany;
		char name[MPI_MAX_PROCESSOR_NAME];
		char name_master[MPI_MAX_PROCESSOR_NAME];
		int len;

		if (rank == 0)
		{
			PMPI_Get_processor_name(name, &len);
			PMPI_Bcast (name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
		}
		else
			PMPI_Bcast (name_master, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

		if (rank == 0)
		{
			unsigned res = 1;
			unsigned length = strlen(directory)+strlen("/shared-disk-testXXXXXX")+1;
			char *template = malloc (length*sizeof(char));
			if (!template)
			{
				fprintf (stderr, PACKAGE_NAME":Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
				exit (-1);
			}
			sprintf (template, "%s/shared-disk-testXXXXXX", directory);
			int ret = mkstemp (template);
			if (ret < 0)
			{
				fprintf (stderr, PACKAGE_NAME":Error! cannot determine whether %s is a shared disk. Failed to create temporal file!\n", directory);
				exit (-1);
			}
			ssize_t r = write (ret, name, strlen(name)+1);
			if (r != (ssize_t) (strlen(name)+1))
			{
				fprintf (stderr, PACKAGE_NAME":Error! cannot determine whether %s is a shared disk. Failed to write to temporal file!\n", directory);
				exit (-1);
			}
			PMPI_Bcast (&length, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			PMPI_Bcast (template, length, MPI_CHAR, 0, MPI_COMM_WORLD);
			PMPI_Reduce (&res, &howmany, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			unlink (template);
			free (template);
			result = howmany == size;
			PMPI_Bcast (&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		else
		{
			int fd;
			unsigned res = 0;
			char *template;
			unsigned length;
			PMPI_Bcast (&length, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			template = malloc (length*sizeof(char));
			if (!template)
			{
				fprintf (stderr, PACKAGE_NAME":Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
				exit (-1);
			}
			PMPI_Bcast (template, length, MPI_CHAR, 0, MPI_COMM_WORLD);
			fd = open(template, O_RDONLY);
			if (fd >= 0)
			{
				ssize_t r = read (fd, name, sizeof(name));
				if (r > 0 && r < (ssize_t) (sizeof(name)))
					if (strncmp (name, name_master, r) == 0)
						res = 1;
			}
			PMPI_Reduce (&res, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			free (template);
			PMPI_Bcast (&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		return result;
	}
	else
		return directory_exists(directory);
}
#endif

/* Check whether a given directory exists for every process in
	MPI_COMM_WORLD. All the processes receive the result */

int ExtraeUtilsMPI_CheckSharedDisk (const char *directory)
{
	return ExtraeUtilsMPI_CheckSharedDisk_stat (directory);
}

