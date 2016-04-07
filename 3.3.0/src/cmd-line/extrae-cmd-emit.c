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

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif

#include "extrae_user_events.h"
#include "extrae-cmd.h"
#include "extrae-cmd-emit.h"

#include "wrapper.h"

static int _THREADID = 0;
static int _TASKID = 0;
static int _NTHREADS = 1;
static int _NTASKS = 1;
static pid_t pid;

static unsigned CMD_EMIT_TASKID (void)
{
	return _TASKID;
}

static unsigned CMD_EMIT_NUMTASKS (void)
{
	return _NTASKS;
}

static unsigned CMD_EMIT_NUMTHREADS (void)
{
	return _NTHREADS;
}

static unsigned CMD_EMIT_NUMTHREAD (void)
{
	return _THREADID;
}

static void Extrae_CMD_Emit_get_info (void)
{
	char HOST[1024];
	if (0 == gethostname (HOST, sizeof(HOST)))
	{
		char TMPFILE[2048];
		int fd;

		sprintf (TMPFILE, EXTRAE_CMD_FILE_PREFIX"%s", HOST);
		fd = open (TMPFILE, O_RDONLY);
		if (fd >= 0)
		{
			char buffer[1024];
			if (read (fd, buffer, sizeof(buffer)) > 0)
			{
				if (3 != sscanf (buffer, "%u\n%u\n%u\n", &pid, &_TASKID, &_NTHREADS))
					fprintf (stderr, CMD_EMIT " Error! Faild to parse temporary file (%s)\n", TMPFILE);
			}
			else
				fprintf (stderr, CMD_EMIT " Error! Failed to read from temporary file (%s)\n", TMPFILE);
			close (fd);
		}
		else
			fprintf (stderr, CMD_INIT " Error! Failed to open temporary file (%s)\n", TMPFILE);
	}
}

int Extrae_CMD_Emit (int i, int argc, char *argv[])
{
	int threadid;
	int type;
	long long value;
	extrae_type_t TYPE = 0;
	extrae_value_t VALUE = 0;
	char *endptr;
	char extrae_append_pid[128]; 

	if (argc-i < 3)
	{
		fprintf (stderr, CMD_EMIT" command requires 3 parameters SLOT, TYPE and VALUE\n");
		return 0;
	}

	Extrae_CMD_Emit_get_info();

	threadid = strtol (argv[i], &endptr, 10);
	if (endptr == &argv[i][strlen(argv[i])])
	{
		if (threadid < 0)
		{
			fprintf (stderr, CMD_EMIT" command cannot handle negative SLOT\n");
			return 0;
		}
		else
			_THREADID = threadid;
	}

	type = strtol (argv[i+1], &endptr, 10);
	if (endptr == &argv[i+1][strlen(argv[i+1])])
	{
		if (type < 0)
		{
			fprintf (stderr, CMD_EMIT" command cannot handle negative TYPE\n");
			return 0;
		}
		else
			TYPE = type;
	}

	value = strtoll (argv[i+2], &endptr, 10);
	if (endptr == &argv[i+2][strlen(argv[i+2])])
	{
		if (value < 0)
		{
			fprintf (stderr, CMD_EMIT" command cannot handle negative VALUE\n");
			return 0;
		}
		else
			VALUE = value;
	}

	Extrae_set_taskid_function (CMD_EMIT_TASKID);
	Extrae_set_numthreads_function (CMD_EMIT_NUMTHREADS);
	Extrae_set_threadid_function (CMD_EMIT_NUMTHREAD);
	_NTASKS = _TASKID+1;
	Extrae_set_numtasks_function (CMD_EMIT_NUMTASKS);

	putenv ("EXTRAE_ON=1");
	sprintf (extrae_append_pid, "EXTRAE_APPEND_PID=%u", pid);
	putenv (extrae_append_pid);

	Extrae_init ();

	Extrae_event (TYPE, VALUE);

	Extrae_fini ();

	return 3;
}

