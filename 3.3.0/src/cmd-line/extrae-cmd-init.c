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

#include "extrae_user_events.h"

#include "extrae-cmd.h"
#include "extrae-cmd-init.h"

static unsigned _TASKID = 0;
static unsigned _NTHREADS = 1;
static unsigned _NTASKS = 1;

static unsigned CMD_INIT_TASKID (void)
{
	return _TASKID;
}

static unsigned CMD_INIT_NUMTASKS (void)
{
	return _NTASKS;
}

static unsigned CMD_INIT_NUMTHREADS (void)
{
	return _NTHREADS;
}

static void Extrae_CMD_Init_dump_info (void)
{
	pid_t p = getpid();
	char HOST[1024];

	if (0 == gethostname (HOST, sizeof(HOST)))
	{
		char TMPFILE[2048];
		int fd;

		sprintf (TMPFILE, EXTRAE_CMD_FILE_PREFIX"%s", HOST);
		fd = creat (TMPFILE, S_IRUSR|S_IWUSR);
		if (fd >= 0)
		{
			char buffer[1024];
			sprintf (buffer, "%u\n%u\n%u\n", p, _TASKID, _NTHREADS);
			if (write (fd, buffer, strlen(buffer)) != (ssize_t) strlen(buffer))
				fprintf (stderr, CMD_INIT " Error! Failed to write on temporary file\n");
			close (fd);
		}
		else
			fprintf (stderr, CMD_INIT " Error! Failed to create temporary file\n");
	}
}

int Extrae_CMD_Init (int i, int argc, char *argv[])
{
	int taskid, nthreads;
	char *endptr;

	if (argc-i < 2)
	{
		fprintf (stderr, CMD_INIT" command requires 2 parameters TASKID and Number of Threads/Slots\n");
		return 0;
	}

	taskid = strtol (argv[i], &endptr, 10);
	if (endptr == &argv[i][strlen(argv[i])])
	{
		if (taskid < 0)
		{
			fprintf (stderr, CMD_INIT" command cannot handle negative TASKID\n");
			return 0;
		}
		else
			_TASKID = taskid;
	}

	nthreads = strtol (argv[i+1], &endptr, 10);
	if (endptr == &argv[i+1][strlen(argv[i+1])])
	{
		if (nthreads < 0)
		{
			fprintf (stderr, CMD_INIT" command cannot handle negative Number of Threads/Slots\n");
			return 0;
		}
		else
			_NTHREADS = nthreads;
	}

	Extrae_set_taskid_function (CMD_INIT_TASKID);
	Extrae_set_numthreads_function (CMD_INIT_NUMTHREADS);
	_NTASKS = _TASKID+1;
	Extrae_set_numtasks_function (CMD_INIT_NUMTASKS);
	putenv ("EXTRAE_ON=1");
	Extrae_init();

	Extrae_CMD_Init_dump_info();

	Extrae_fini();

	return 2;
}

