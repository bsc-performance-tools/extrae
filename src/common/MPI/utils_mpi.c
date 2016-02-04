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
#ifdef HAVE_SIONLIB
# include "sion.h"
#endif

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
				fprintf (stderr, PACKAGE_NAME": Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
				exit (-1);
			}
			sprintf (template, "%s/shared-disk-testXXXXXX", directory);
			ret = mkstemp (template);
			if (ret < 0)
			{
				fprintf (stderr, PACKAGE_NAME": Error! cannot determine whether %s is a shared disk. Failed to create temporal file!\n", directory);
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
				fprintf (stderr, PACKAGE_NAME": Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
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
				fprintf (stderr, PACKAGE_NAME": Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
				exit (-1);
			}
			sprintf (template, "%s/shared-disk-testXXXXXX", directory);
			int ret = mkstemp (template);
			if (ret < 0)
			{
				fprintf (stderr, PACKAGE_NAME": Error! cannot determine whether %s is a shared disk. Failed to create temporal file!\n", directory);
				exit (-1);
			}
			ssize_t r = write (ret, name, strlen(name)+1);
			if (r != (ssize_t) (strlen(name)+1))
			{
				fprintf (stderr, PACKAGE_NAME": Error! cannot determine whether %s is a shared disk. Failed to write to temporal file!\n", directory);
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
				fprintf (stderr, PACKAGE_NAME": Error! cannot determine whether %s is a shared disk. Failed to allocate memory!\n", directory);
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

#ifdef HAVE_SIONLIB
#define FNAMELEN 255
#define BUFSIZE (1024*1024)

void rename_or_copy_sionlib (const char *origen, const char *desti)
{
	int rank, size, globalrank, sid, numFiles;
	char fname[FNAMELEN], *newfname=NULL;
	MPI_Comm gComm, lComm;
	sion_int64 chunksize,left;
	sion_int32 fsblksize;
	size_t btoread, bread, bwrote;
	char *localbuffer;
	FILE *fileptr;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* inital parameters */
	strcpy(fname, "data.mpit");

	numFiles   = 1;
	gComm      = lComm = MPI_COMM_WORLD;
 	chunksize  = 10*1024*1024;
	fsblksize  = 1*1024*1024;
	globalrank = rank;

	char buffer[65536];
	int fd_o, fd_d;
	ssize_t res;

	/* Open the files */
	fd_o = open (origen, O_RDONLY);
	if (fd_o == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error while trying to open %s \n", origen);
		fflush (stderr);
		return;
        }

	sid = sion_paropen_mpi(fname, "bw", &numFiles, gComm, &lComm,
	  &chunksize, &fsblksize, &globalrank, &fileptr, &newfname);  

	/* Copy the file */
	res = read (fd_o, buffer, sizeof (buffer));
	int elems_written;
	int tmp1 = 0;
	while (res != 0 && res != -1)
	{
		elems_written = fwrite(buffer, 1, res, fileptr);
		tmp1 += elems_written;
		if (elems_written == -1)
			break;
		res = read (fd_o, buffer, sizeof (buffer));
	}
                        
	/* If failed, just close!  */
	if (res == -1)
	{
		close (fd_o);
		fprintf (stderr, PACKAGE_NAME": Error while trying to move files %s to %s\n", origen, desti);
		fflush (stderr);
		return;
	}

	/* Close the files */
	close (fd_o);
	sion_parclose_mpi(sid);
	
	/* Remove the files */
	unlink (origen);
}

void append_from_to_file_sionlib (const char *source, const char *destination)
{
	int rank, size, globalrank, sid, i, numFiles;
	MPI_Comm gComm, lComm;
	sion_int64 chunksize, left;
	sion_int32 fsblksize;

	char fname[1000], *newfname = NULL;
	strcpy(fname, "parfile.sion");
	numFiles   = 1;
	gComm      = lComm = MPI_COMM_WORLD;
	chunksize  = 10*1024*1024;
	fsblksize  = 1*1024*1024;
	FILE *fileptr;

	PMPI_Comm_rank (MPI_COMM_WORLD, &rank);
	PMPI_Comm_size (MPI_COMM_WORLD, &size);
	globalrank = rank;

	char buffer[65536];
	int fd_o, fd_d;
	ssize_t res;

	/* Open the files */
	fd_o = open (source, O_RDONLY);
	if (fd_o == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error while trying to open %s \n", source);
		fflush (stderr);
		return;
	}

	sid = sion_paropen_mpi(fname, "bw", &numFiles, gComm, &lComm,
	  &chunksize, &fsblksize, &globalrank, &fileptr, &destination);

	/* Copy the file */
	int bytes_written;
	res = read (fd_o, buffer, sizeof (buffer));
	while (res > 0)
	{
		bytes_written = fwrite(buffer, 1, res, fileptr);
		if (bytes_written == -1)
			break;
		res = read (fd_o, buffer, sizeof (buffer)); 	
	}

	sion_parclose_mpi(sid);

	/* If failed, just close!  */
	if (res == -1)
	{
		close (fd_d);
		unlink (destination);
		fprintf (stderr, PACKAGE_NAME": Error while trying to move files %s to %s\n", source, destination);
		fflush (stderr);
		return;
	}

	/* Close the files */
	close (fd_o);
}
#endif

/* Check whether a given directory exists for every process in
	MPI_COMM_WORLD. All the processes receive the result */

int ExtraeUtilsMPI_CheckSharedDisk (const char *directory)
{
	return ExtraeUtilsMPI_CheckSharedDisk_stat (directory);
}

/* Share the XML configuration file across processes */
char * MPI_Distribute_XML_File (int rank, int world_size, const char *file)
{
	char hostname[1024];
	char *result_file = NULL;
	off_t file_size;
	int fd;
	char *storage;
	int has_hostname = FALSE;

	has_hostname = gethostname(hostname, 1024 - 1) == 0;

	/* If no other tasks are running, just return the same file */
	if (world_size == 1)
	{
		/* Copy the filename */
		result_file = strdup (file);
		if (result_file == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML file!\n");
			exit (0);
		}
		return result_file;
	}

	if (rank == 0)
	{
		/* Copy the filename */
		result_file = (char*) malloc ((strlen(file)+1)*sizeof(char));
		if (result_file == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML file!\n");
			exit (0);
		}
		memset (result_file, 0, (strlen(file)+1)*sizeof(char));
		strncpy (result_file, file, strlen(file));

		/* Open the file */
		fd = open (result_file, O_RDONLY);

		/* If open fails, just return the same fail... XML parsing will fail too! */
		if (fd < 0)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot open XML configuration file (%s)!\n", result_file);
			exit (0);
		}

		file_size = lseek (fd, 0, SEEK_END);
		lseek (fd, 0, SEEK_SET);

		/* Send the size */
		PMPI_Bcast (&file_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

		/* Allocate & Read the file */
		storage = (char*) malloc ((file_size)*sizeof(char));
		if (storage == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML distribution!\n");
			exit (0);
		}
		if (file_size != read (fd, storage, file_size))
		{
			fprintf (stderr, PACKAGE_NAME": Unable to read XML file for its distribution on host %s\n", has_hostname?hostname:"unknown");
			exit (0);
		}

		/* Send the file */
		PMPI_Bcast (storage, file_size, MPI_BYTE, 0, MPI_COMM_WORLD);

		/* Close the file */
		close (fd);
		free (storage);

		return result_file;
	}
	else
	{
		/* Receive the size */
		PMPI_Bcast (&file_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
		storage = (char*) malloc ((file_size)*sizeof(char));
		if (storage == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML distribution!\n");
			exit (0);
		}

		/* Build the temporal file pattern */
		if (getenv("TMPDIR"))
		{
			int len = 14 + strlen(getenv("TMPDIR")) + 1;
			/* If TMPDIR exists but points to non-existent directory, create it */
			if (!directory_exists (getenv("TMPDIR")))
				mkdir_recursive (getenv("TMPDIR"));

			/* 14 is the length from /XMLFileXXXXXX */
			result_file = (char*) malloc (len * sizeof(char));
			snprintf (result_file, len, "%s/XMLFileXXXXXX", getenv ("TMPDIR"));
		}
		else
		{
			/* 13 is the length from XMLFileXXXXXX */
			result_file = (char*) malloc ((13+1)*sizeof(char));
			sprintf (result_file, "XMLFileXXXXXX");
		}

		/* Create the temporal file */
		fd = mkstemp (result_file);

		/* Receive the file */
		PMPI_Bcast (storage, file_size, MPI_BYTE, 0, MPI_COMM_WORLD);

		if (file_size != write (fd, storage, file_size))
		{
			fprintf (stderr, PACKAGE_NAME": Unable to write XML file for its distribution (%s) - host %s\n", result_file, has_hostname?hostname:"unknown");
			perror("write");
			exit (0);
		}

		/* Close the file, free and return it! */
		close (fd);
		free (storage);

		return result_file;
	}
}
