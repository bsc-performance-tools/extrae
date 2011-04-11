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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#if defined(MPI_SUPPORT)
# ifdef HAVE_MPI_H
#  include <mpi.h>
# endif
#endif
#if defined(PACX_SUPPORT)
# include <pacx.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
# include <sys/resource.h>
#endif
#ifdef HAVE_SYS_UIO_H
#include <sys/uio.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_MATH_H
# include <math.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif
#if defined(HAVE_PTHREAD_H) && defined(PTHREAD_SUPPORT)
# include <pthread.h>
#endif

#if defined(IS_CELL_MACHINE)
# include "defaults.h"
# include "cell_wrapper.h"
#endif

#include "wrapper.h"
#if defined(MPI_SUPPORT)
# include "mpi_wrapper.h"
#endif
#if defined(PACX_SUPPORT)
# include "pacx_wrapper.h"
#endif
#include "misc_wrapper.h"
#include "clock.h"
#include "hwc.h"
#include "signals.h"
#include "utils.h"
#include "calltrace.h"
#include "xml-parse.h"
#if defined(DEAD_CODE)
# include "myrinet_hwc.h"
#endif
#include "UF_gcc_instrument.h"
#include "UF_xl_instrument.h"
#include "mode.h"
#include "events.h"
#if defined(OMP_SUPPORT)
# include "omp_probe.h"
# include "omp_wrapper.h"
#endif
#include "trace_buffers.h"
#include "timesync.h"
#if defined(HAVE_MRNET)
# include "mrn_config.h"
# include "mrnet_be.h"
#endif
#if defined(UPC_SUPPORT)
# include <external/upc.h>
#endif
#include "common_hwc.h"

#if defined(EMBED_MERGE_IN_TRACE)
# include "mpi2out.h"
# include "options.h"
#endif

int Extrae_Flush_Wrapper (Buffer_t *buffer);

#warning "Control variables below (tracejant, tracejant_mpi, tracejant_hwc_mpi...) should be moved to mode.c and indexed per mode"

/***** Variable global per saber si en un moment donat cal tracejar ******/
int tracejant = TRUE;

/***** Variable global per saber si MPI s'ha de tracejar *****************/
int tracejant_mpi = TRUE;

/***** Variable global per saber si MPI s'ha de tracejar amb hwc *********/
int tracejant_hwc_mpi = FALSE;

/***** Variable global per saber si OpenMP s'ha de tracejar **************/
int tracejant_omp = TRUE;

/***** Variable global per saber si OpenMP s'ha de tracejar amb hwc ******/
int tracejant_hwc_omp = FALSE;

/***** Variable global per saber si pthread s'ha de tracejar **************/
int tracejant_pthread = TRUE;

/***** Variable global per saber si pthread s'ha de tracejar amb hwc ******/
int tracejant_hwc_pthread = FALSE;

/***** Variable global per saber si UFs s'han de tracejar amb hwc ********/
int tracejant_hwc_uf = FALSE;

/*** Variable global per saber si hem d'obtenir comptador de la xarxa ****/
int tracejant_network_hwc = FALSE;

/** Store information about rusage?                                     **/
int tracejant_rusage = FALSE;

/** Store information about malloc?                                     **/
int tracejant_memusage = FALSE;

/**** Variable global que controla quin subset de les tasks generen o ****/
/**** no generen trasa ***************************************************/
int *TracingBitmap = NULL;

int mpit_gathering_enabled;

/*************************************************************************/

/** Variable global per saber si en general cal interceptar l'aplicacio **/
int mpitrace_on = FALSE;

/* Where is the tracing facility located                                 */
char trace_home[TMP_DIR];

/* Time of the first event (APPL_EV) */
unsigned long long ApplBegin_Time = 0;

/************** Variable global amb el nom de l'aplicacio ****************/
char PROGRAM_NAME[256];

/*************************************************************************/

unsigned long long last_mpi_exit_time = 0;
unsigned long long last_mpi_begin_time = 0;

/* Control del temps de traceig */
unsigned long long initTracingTime = 0;
unsigned long long MinimumTracingTime;
int hasMinimumTracingTime = FALSE;

unsigned long long WantedCheckControlPeriod = 0;

/******* Variable amb l'estructura que a SGI es guarda al PRDA *******/
//struct trace_prda *PRDAUSR;

/*********************************************************************/

/******************************************************************************
 **********************         V A R I A B L E S        **********************
 ******************************************************************************/

Buffer_t **TracingBuffer = NULL;
Buffer_t **SamplingBuffer = NULL;
unsigned int min_BufferSize = EVT_NUM;

//event_t **buffers;
unsigned int buffer_size = EVT_NUM;
unsigned file_size = 0;

#define MBytes							*1024*1024

unsigned int mptrace_IsMPI = FALSE;
//unsigned int TaskID = 0;
unsigned NumOfTasks = 0;
static unsigned current_NumOfThreads = 0;
static unsigned maximum_NumOfThreads = 0;

unsigned int mptrace_suspend_tracing = FALSE;
unsigned int mptrace_tracing_is_suspended = FALSE;

#define TMP_NAME_LENGTH     512
#define APPL_NAME_LENGTH    512
char appl_name[APPL_NAME_LENGTH];
char final_dir[TMP_DIR];
char tmp_dir[TMP_DIR];
#if defined(DEAD_CODE)
char base_dir[TMP_DIR];
char symbol[TRACE_FILE];
#endif

/* Know if the run is controlled by a creation of a file  */
char ControlFileName[TMP_DIR];
int CheckForControlFile = FALSE;
int CheckForGlobalOpsTracingIntervals = FALSE;

int circular_buffering = 0;

#if defined(EMBED_MERGE_IN_TRACE)
int MergeAfterTracing = FALSE;
#endif

static unsigned get_maximum_NumOfThreads (void)
{
	return maximum_NumOfThreads;
}

static unsigned get_current_NumOfThreads (void)
{
	return current_NumOfThreads;
}

/******************************************************************************
 **      Function name : VerifyLicenseExecution (void)
 **      Author : HSG
 **      Description : Checks whether the license is ok on this node
 ******************************************************************************/
/* Include license management */
#if defined (LICENSE) && !defined (LICENSE_IN_MERGE)
# include "license.c"
#endif

static void VerifyLicenseExecution (void)
{
#if defined(LICENSE) && !defined(LICENSE_IN_MERGE)
	int res = verify_execution ();

	lvalida = verify_execution ();
	if (!lvalida)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Invalid license!?\n");
		exit (-1);
	}
#endif
}


/******************************************************************************
 ***  read_environment_variables
 ***  Reads some environment variables. Returns 0 if the tracing was disabled,
 ***  otherwise, 1.
 ******************************************************************************/

static int read_environment_variables (int me)
{
#if defined(MPI_SUPPORT) || defined(PACX_SUPPORT)
	char *mpi_callers;
#endif
  char *dir, *str, *res_cwd;
	char *file;
	char cwd[TMP_DIR];

	/* Check if the tracing is enabled. If not, just exit from here */
	str = getenv ("EXTRAE_ON");
	mpitrace_on = (str != NULL && (strcmp (str, "1") == 0));
	if (me == 0 && !mpitrace_on)
	{
		fprintf (stdout, PACKAGE_NAME": Application has been linked or preloaded with Extrae, BUT EXTRAE_ON is NOT enabled!\n");
		return 0;
	}

	/* Define the tracing home */
	str = getenv ("EXTRAE_HOME");
	if (str != NULL)
	{
		strncpy (trace_home, str, TMP_DIR);
	}
	else
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Warning! EXTRAE_HOME has not been defined!.\n");
	}

#if USE_HARDWARE_COUNTERS
	if (getenv("EXTRAE_COUNTERS") != NULL)
	{
		HWC_Initialize (0);
		HWC_Parse_Env_Config (me);
	}
#endif

	/* Initial Tracing Mode? */
	if ((str = getenv("EXTRAE_INITIAL_MODE")) != NULL)
	{
		if (strcasecmp(str, "detail") == 0)
		{
			TMODE_setInitial (TRACE_MODE_DETAIL);
		}
		else if (strcasecmp(str, "bursts") == 0)
		{
			TMODE_setInitial (TRACE_MODE_BURSTS);
		}
	}

	/* Whether we have to generate PARAVER or DIMEMAS traces */
	if ((str = getenv ("EXTRAE_TRACE_TYPE")) != NULL)
	{
		if (strcasecmp (str, "DIMEMAS") == 0)
		{
			Clock_setType (USER_CLOCK);
			if (me == 0)
				fprintf (stdout, PACKAGE_NAME": Generating intermediate files for Dimemas traces.\n");
		}
		else
		{
			Clock_setType (REAL_CLOCK);
			if (me == 0)
				fprintf (stdout, PACKAGE_NAME": Generating intermediate files for Paraver traces.\n");
		}
	}
	else
	{
		Clock_setType (REAL_CLOCK);
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Generating intermediate files for Paraver traces.\n");
	}

	/* Minimum CPU Burst duration? */
	if ((str = getenv("EXTRAE_BURST_THRESHOLD")) != NULL) 
	{
		TMODE_setBurstsThreshold (getTimeFromStr (str, "EXTRAE_BURST_THRESHOLD", me));
	}

#if defined(MPI_SUPPORT)
	/* Collect MPI statistics in the library? */
	if ((str = getenv ("EXTRAE_MPI_STATISTICS")) != NULL)
	{
		if (strcmp(str, "1") == 0)
		{
			TMODE_setBurstsStatistics (ENABLED);
		}
		else
		{
			TMODE_setBurstsStatistics (DISABLED);
		}
	}
#elif defined(PACX_SUPPORT)
	/* Collect MPI statistics in the library? */
	if ((str = getenv ("EXTRAE_PACX_STATISTICS")) != NULL)
	{
		if (strcmp(str, "1") == 0)
		{
			TMODE_setBurstsStatistics (ENABLED);
		}
		else
		{
			TMODE_setBurstsStatistics (DISABLED);
		}
	}
#endif

	/*
	* EXTRAEDIR : Output directory for traces.
	*/
	res_cwd = getcwd (cwd, sizeof(cwd));

	if ( (dir = getenv ("EXTRAE_FINAL_DIR")) == NULL )
		if ( (dir = getenv ("EXTRAE_DIR")) == NULL )
			if ( (dir = res_cwd) == NULL )
 				dir = ".";

	if (strlen(dir) > 0)
	{
			if (dir[0] != '/')
				sprintf (final_dir, "%s/%s", res_cwd, dir);
			else
				strcpy (final_dir, dir);
	}
	else
		strcpy (final_dir, dir);

	if ( (dir = getenv ("EXTRAE_DIR")) == NULL )
		if ( (dir = res_cwd) == NULL )
			dir = ".";
	strcpy (tmp_dir, dir);

	if (me == 0)
	{
		if (strcmp (tmp_dir, final_dir) != 0)
		{
			fprintf (stdout, PACKAGE_NAME": Temporal directory for the intermediate traces is %s\n", tmp_dir);
			fprintf (stdout, PACKAGE_NAME": Final directory for the intermediate traces is %s\n", final_dir);
		}
		else
		{
			fprintf (stdout, PACKAGE_NAME": Intermediate files will be stored in %s\n", final_dir);
		}
	}

	/* EXTRAE_CONTROL_FILE, activates the tracing when this file is created */
	if ((file = getenv ("EXTRAE_CONTROL_FILE")) != NULL)
	{
		CheckForControlFile = TRUE;
		strcpy (ControlFileName, file);
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Control file is %s.\n          Tracing will be disabled until the file exists\n", ControlFileName);
	}
	else
		CheckForControlFile = FALSE;

	/* EXTRAE_CONTROL_GLOPS, activates the tracing on a global op series */
	if ((str = getenv ("EXTRAE_CONTROL_GLOPS")) != NULL)
	{
		CheckForGlobalOpsTracingIntervals = TRUE;
		Parse_GlobalOps_Tracing_Intervals (str);
	}

	/*
	 * EXTRAE_BUFFER_SIZE : Tells the buffer size for each thread.
	 */
	if ((str = getenv ("EXTRAE_BUFFER_SIZE")) != NULL)
	{
		buffer_size = atoi (str);
		if (buffer_size <= 0)
			buffer_size = EVT_NUM;
	}
	else
		buffer_size = EVT_NUM;
	if (me == 0)
		fprintf (stdout, PACKAGE_NAME": Tracing buffer can hold %d events\n", buffer_size);

	/*
	 * EXTRAE_FILE_SIZE: Limits the intermediate file size for each thread.
	 */
	if ((str = getenv ("EXTRAE_FILE_SIZE")) != NULL)
	{
		file_size = atoi (str);
		if (file_size <= 0 && me == 0)
			fprintf (stderr, PACKAGE_NAME": Invalid EXTRAE_FILE_SIZE environment variable value.\n");
		else if (file_size > 0 && me == 0)
			fprintf (stderr, PACKAGE_NAME": EXTRAE_FILE_SIZE set to %d Mbytes.\n", file_size);
	}

	/* 
	 * EXTRAE_MINIMUM_TIME : Set the minimum tracing time...
	 */
	MinimumTracingTime = getTimeFromStr (getenv("EXTRAE_MINIMUM_TIME"), "EXTRAE_MINIMUM_TIME", me);
	hasMinimumTracingTime = (MinimumTracingTime != 0);
	if (me == 0 && hasMinimumTracingTime)
	{
		if (MinimumTracingTime >= 1000000000)
			fprintf (stdout, PACKAGE_NAME": Minimum tracing time will be %llu seconds\n", MinimumTracingTime / 1000000000);
		else
			fprintf (stdout, PACKAGE_NAME": Minimum tracing time will be %llu nanoseconds\n", MinimumTracingTime);
	}

	/* 
	 * EXTRAE_CONTROL_TIME : Set the control tracing time...
	 */
	WantedCheckControlPeriod = getTimeFromStr (getenv("EXTRAE_CONTROL_TIME"), "EXTRAE_CONTROL_TIME", me);
	if (me == 0 && WantedCheckControlPeriod != 0)
	{
		if (WantedCheckControlPeriod >= 1000000000)
			fprintf (stdout, PACKAGE_NAME": Control file will be checked every %llu seconds\n", WantedCheckControlPeriod / 1000000000);
		else
			fprintf (stdout, PACKAGE_NAME": Control file will be checked every %llu nanoseconds\n", WantedCheckControlPeriod);
	}

#if defined(MPI_SUPPORT)
	/* Control if the user wants to add information about MPI caller routines */
	mpi_callers = getenv ("EXTRAE_MPI_CALLER");
	if (mpi_callers != NULL) Parse_Callers (me, mpi_callers, CALLER_MPI);
#elif defined(PACX_SUPPORT)
	/* Control if the user wants to add information about MPI caller routines */
	mpi_callers = getenv ("EXTRAE_PACX_CALLER");
	if (mpi_callers != NULL) Parse_Callers (me, mpi_callers, CALLER_MPI);
#endif

#if defined(MPI_SUPPORT)
	/* Check if we must gather all the MPIT files into one target (MASTER) node */
	str = getenv("EXTRAE_GATHER_MPITS");
	if ((str != NULL) && (strcmp(str, "1") == 0))
	{
		mpit_gathering_enabled = TRUE;
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": All MPIT files will be gathered at the end of the execution.\n");
	}
#endif

	/* Check if the buffer must be treated as a circular buffer instead a linear buffer with many flushes */
	str = getenv ("EXTRAE_CIRCULAR_BUFFER");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		circular_buffering = TRUE;
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Circular buffer enabled!\n");
	}

	/* Get the program name if available. It will be used to form the MPIT filenames */
	str = getenv ("EXTRAE_PROGRAM_NAME");
	if (!str)
		strncpy (PROGRAM_NAME, "TRACE", strlen("TRACE")+1);
	else
		strncpy (PROGRAM_NAME, str, sizeof(PROGRAM_NAME));
	PROGRAM_NAME[255] = '\0';

#if defined(MPI_SUPPORT)
	/* Check if the MPI must be disabled */
	str = getenv ("EXTRAE_DISABLE_MPI");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": MPI calls are NOT traced.\n");
  	tracejant_mpi = FALSE;
	}
#elif defined(PACX_SUPPORT)
	/* Check if the PACX must be disabled */
	str = getenv ("EXTRAE_DISABLE_PACX");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": PACX calls are NOT traced.\n");
  	tracejant_mpi = FALSE;
	}
#endif

#if defined(MPI_SUPPORT)
	/* HWC must be gathered at MPI? */
	str = getenv ("EXTRAE_MPI_COUNTERS_ON");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": HWC reported in the MPI calls.\n");
		tracejant_hwc_mpi = TRUE;
	}
	else
		tracejant_hwc_mpi = FALSE;
#elif defined(PACX_SUPPORT)
	/* HWC must be gathered at PACX? */
	str = getenv ("EXTRAE_PACX_COUNTERS_ON");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": HWC reported in the PACX calls.\n");
		tracejant_hwc_mpi = TRUE;
	}
#endif

	/* Enable rusage information? */
	str = getenv ("EXTRAE_RUSAGE");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Resource usage is enabled at flush buffer.\n");
		tracejant_rusage = TRUE;
	}
	else
		tracejant_rusage = FALSE;

	/* Enable memusage information? */
	str = getenv ("EXTRAE_MEMUSAGE");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Memory usage is enabled at flush buffer.\n");
		tracejant_memusage = TRUE;
	}
	else
		tracejant_memusage = FALSE;

#if defined(TEMPORARILY_DISABLED)
	/* Enable network counters? */
	str = getenv ("EXTRAE_NETWORK_COUNTERS");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Network counters are enabled.\n");
		tracejant_network_hwc = TRUE;
	}
	else
#endif
		tracejant_network_hwc = FALSE;
	
	/* Add UF routines to instrument under GCC -finstrument-function callback
	   routines */
	str = getenv ("EXTRAE_FUNCTIONS");
	if (str != NULL)
	{
		InstrumentUFroutines_XL (me, str);
		InstrumentUFroutines_GCC (me, str);
	}

	/* HWC must be gathered at UF? */
	str = getenv ("EXTRAE_FUNCTIONS_COUNTERS_ON");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": User Function routines will collect HW counters information.\n");
		tracejant_hwc_uf = TRUE;
	}
	else
		tracejant_hwc_uf = FALSE;

#if defined(OMP_SUPPORT)
	/* Check if the OpenMP tracing must be disabled */
	str = getenv ("EXTRAE_DISABLE_OMP");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": OpenMP runtime calls are NOT traced.\n");
  	tracejant_omp = FALSE;
	}

	/* HWC must be gathered at OpenMP? */
	str = getenv ("EXTRAE_OMP_COUNTERS_ON");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": HWC reported in the OpenMP calls.\n");
		tracejant_hwc_omp = TRUE;
	}
	else
		tracejant_hwc_omp = FALSE;

	/* Will we trace openmp-locks ? */
	str = getenv ("EXTRAE_OMP_LOCKS");
	setTrace_OMPLocks ((str != NULL && (strcmp (str, "1"))));
#endif

#if defined(PTHREAD_SUPPORT)
	/* Check if the pthread tracing must be disabled */
	str = getenv ("EXTRAE_DISABLE_PTHREAD");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": pthread runtime calls are NOT traced.\n");
  	tracejant_omp = FALSE;
	}

	/* HWC must be gathered at OpenMP? */
	str = getenv ("EXTRAE_PTHREAD_COUNTERS_ON");
	if (str != NULL && (strcmp (str, "1") == 0))
	{
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": HWC reported in the pthread calls.\n");
		tracejant_hwc_pthread = TRUE;
	}
	else
		tracejant_hwc_pthread = FALSE;

	/* Will we trace openmp-locks ? */
	str = getenv ("EXTRAE_PTHREAD_LOCKS");
	setTrace_PTHREADLocks ((str != NULL && (strcmp (str, "1"))));
#endif

	/* Should we configure a signal handler ? */
	str = getenv ("EXTRAE_SIGNAL_FLUSH_TERMINATE");
	if (str != NULL)
	{
		if (strcasecmp (str, "USR1") == 0)
		{
			if (me == 0)
				fprintf (stderr,"\n"PACKAGE_NAME": Signal USR1 will flush the buffers to the disk and stop further tracing\n");
			Signals_SetupFlushAndTerminate (SIGUSR1);
		}
		else if (strcasecmp (str, "USR2") == 0)
		{
			if (me == 0)
				fprintf (stderr,"\n"PACKAGE_NAME": Signal USR2 will flush the buffers to the disk and stop further tracing\n");
			Signals_SetupFlushAndTerminate (SIGUSR2);
		}
		else
		{
			if (me == 0)
				fprintf (stderr,"\nWARNING: Value '%s' for EXTRAE_SIGNAL_FLUSH is unrecognized\n", str);
		}
	}

#if defined(IS_CELL_MACHINE)

# ifndef SPU_USES_WRITE
	/* Configure DMA channel for the transferences */
	str = getenv("EXTRAE_SPU_DMA_CHANNEL");
	if (str == (char *)NULL) {
		spu_dma_channel = DEFAULT_DMA_CHANNEL;
	}
	else {
		spu_dma_channel = atoi(str);
	}
	if ((spu_dma_channel < 0) || (spu_dma_channel > 31))
	{
		if (TASKID == 0)
			fprintf (stderr, PACKAGE_NAME": Invalid DMA channel '%d'. Using default channel '%d'.\n", spu_dma_channel, DEFAULT_DMA_CHANNEL);
		spu_dma_channel = DEFAULT_DMA_CHANNEL;
	}
# else
	if (getenv("EXTRAE_SPU_DMA_CHANNEL") != NULL)
		if (TASKID == 0)
			fprintf (stdout, PACKAGE_NAME": SPUs will write directly to disk. Ignoring EXTRAE_SPU_DMA_CHANNEL\n");
# endif /* SPU_USES_WRITE */

	/* Configure the buffer size for each SPU */
	str = getenv("EXTRAE_SPU_BUFFER_SIZE");
	if (str == (char *)NULL) {
		spu_buffer_size = DEFAULT_SPU_BUFFER_SIZE;
	}
	else {
		spu_buffer_size = atoi(str);
	}
	if (spu_buffer_size < 10)
	{
		if (TASKID == 0)
			fprintf (stderr, PACKAGE_NAME": SPU tracing buffer size '%d' too small. Using default SPU buffer size '%d'.\n", spu_buffer_size, DEFAULT_SPU_BUFFER_SIZE);
		spu_buffer_size = DEFAULT_SPU_BUFFER_SIZE;
	}
	else
	{
		if (TASKID == 0)
			fprintf (stdout, PACKAGE_NAME": SPU tracing buffer size is %d events.\n", spu_buffer_size);
	}

	/* Limit the total size of tracing of each spu */
	str = getenv ("EXTRAE_SPU_FILE_SIZE");
	if (str == (char *)NULL) {
		spu_file_size = DEFAULT_SPU_FILE_SIZE;
	}
	else {
	spu_file_size = atoi(str);
	}
	if (spu_file_size < 1)
	{
		if (TASKID == 0)
			fprintf (stderr, PACKAGE_NAME": SPU tracing buffer size '%d' too small. Using default SPU buffer size '%d'.\n", spu_file_size, DEFAULT_SPU_FILE_SIZE);
		spu_file_size = DEFAULT_SPU_FILE_SIZE;
	}
	else
	{
		if (TASKID == 0)
			fprintf (stdout, PACKAGE_NAME": SPU tracing file size limit is %d mbytes.\n", spu_file_size);
	}

#endif /* IS_CELL_MACHINE */

	return 1;
}

/*
 * Inicializa el valor de las variables globales Trace_MPI_Caller, 
 * MPI_Caller_Deepness y MPI_Caller_Count en funcion de los parametros
 * seteados en la variable de entorno EXTRAE_MPI_CALLER
 * El formato de esta variable es una cadena separada por comas, donde
 * cada parametro se refiere a la profundidad en la pila de llamadas
 * de un MPI caller que queremos tracear. La cadena puede contener rangos:
 *            EXTRAE_MPI_CALLER = 1,2,3...7-9...
 */
void Parse_Callers (int me, char * mpi_callers, int type)
{
   char * callers, * caller, * error;
   int from, to, i, tmp;

   if (CALLER_MPI != type && CALLER_SAMPLING != type)
      return;

   callers = (char *)malloc(sizeof(char)*(strlen(mpi_callers)+1));
   strcpy(callers, mpi_callers);

   while ((caller = strtok(callers, (const char *)",")) != NULL) {
      callers = NULL;
      if (sscanf(caller, "%d-%d", &from, &to) != 2) {
         /* 
          * No es un rango => Intentamos convertir el string a un numero.  
          */
         from = to = strtol(caller, &error, 10);
         if ((!strcmp(caller,"\0")  || strcmp(error,"\0")) ||
            (((to == (int)LONG_MIN) || (to == (int)LONG_MAX)) && (errno == ERANGE)))
         {
					 if (me == 0)
            fprintf(stderr, PACKAGE_NAME": WARNING! Ignoring value '%s' in EXTRAE_MPI_CALLER"
								" environment variable.\n", caller);
            continue;
         }
      }

      if (from > to) {
         tmp  = to;
         to   = from;
         from = tmp;
      }

      /* Comprobamos que estamos en un rango valido */
      if ((from < 1) || (to < 1) || (from > MAX_CALLERS)) {
         if (me == 0)
            fprintf(stderr, PACKAGE_NAME": WARNING! Value(s) '%s' in EXTRAE_*_CALLER "
                            "out of bounds (Min 1, Max %d)\n", caller, MAX_CALLERS);
         continue;
      }
      if (to > MAX_CALLERS) {
         to = MAX_CALLERS;
         if (me == 0)
            fprintf(stderr, PACKAGE_NAME": WARNING! Value(s) '%s' in EXTRAE_*_CALLER out of bounds (Min 1, Max %d)\n"
                            PACKAGE_NAME": Reducing MPI callers range from %d to MAX value %d\n", caller, MAX_CALLERS, from, to);
      }
      fflush(stderr);
      fflush(stdout);

      /* Reservamos memoria suficiente para el vector */
      if (Trace_Caller[type] == NULL) {
         Trace_Caller[type] = (int *)malloc(sizeof(int) * to);
         for (i = 0; i < to; i++) Trace_Caller [type][i] = 0;
         Caller_Deepness[type] = i;
      }
      else if (to > Caller_Deepness[type]) {
         Trace_Caller[type] = (int *)realloc(Trace_Caller[type], sizeof(int) * to);
         for (i = Caller_Deepness[type]; i < to; i++) Trace_Caller [type][i] = 0;
         Caller_Deepness[type] = i;
      }

      for (i = from-1; i < to; i++) {
         /* Marcamos que el mpi caller a profundidad i lo queremos tracear */
         Trace_Caller[type][i] = 1;
         Caller_Count[type]++;
      }                                                                           
   }                                                                              
   if (Caller_Count[type] > 0 && me == 0) {
      fprintf(stdout, PACKAGE_NAME": Tracing %d level(s) of %s callers: [ ",
				 	Caller_Count[type], (CALLER_MPI==type)?"MPI":"Sampling");
      for (i=0; i<Caller_Deepness[type]; i++) {
         if (Trace_Caller[type][i]) fprintf(stdout, "%d ", i+1);
      }
      fprintf(stdout, "]\n");
   }
}

typedef struct {
   int glop_id;
   int trace_status;
} GlOp_t;

typedef struct {
   GlOp_t * glop_list;
   int n_glops;
   int next;
} GlOps_Intervals_t;

GlOps_Intervals_t glops_intervals = { NULL, 0, 0 };

static void Add_GlOp_Interval (int glop_id, int trace_status)
{
   int idx;
   idx = glops_intervals.n_glops ++;
   glops_intervals.glop_list = (GlOp_t *)realloc(glops_intervals.glop_list, glops_intervals.n_glops * sizeof(GlOp_t));
   glops_intervals.glop_list[idx].glop_id = glop_id;
   glops_intervals.glop_list[idx].trace_status = trace_status;
}

void Parse_GlobalOps_Tracing_Intervals(char * sequence) {
   int match, i, n_pairs;
   int start = 0, stop = 0, last_stop = 0;
   char ** tmp;

   if ((sequence == (char *)NULL) || (strlen(sequence) == 0)) return;

   n_pairs = explode(sequence, ",", &tmp);

   for (i=0; i<n_pairs; i++)
   {
      match = sscanf(tmp[i], "%d-%d", &start, &stop);
      if (match == 2) {
         if (start >= stop) {
            fprintf(stderr, PACKAGE_NAME": WARNING! Ignoring invalid pair '%s' (stopping before starting)\n", tmp[i]);
            continue;
         }
         if (start <= last_stop) {
            fprintf(stderr, PACKAGE_NAME": WARNING! Ignoring overlapped pair '%s' (starting at %d but previous interval stops at %d)\n", tmp[i], start, last_stop);
            continue;
         }
         Add_GlOp_Interval(start, RESTART);
         Add_GlOp_Interval(stop, SHUTDOWN);
      }
      else {
         start = atoi(tmp[i]);
         if (start == 0) {
            fprintf(stderr, PACKAGE_NAME": WARNING! Ignoring '%s'\n", tmp[i]);
            continue;
         }
         if (start <= last_stop) {
            fprintf(stderr, PACKAGE_NAME": WARNING! Ignoring '%s' (starting at %d but previous interval stops at %d)\n", tmp[i], start, last_stop);
            continue;
         }
         fprintf(stderr, "... started at global op #%d and won't stop until the application finishes\n", start);
         Add_GlOp_Interval(start, RESTART);
         break;
      }
      last_stop = stop;
   }
}

int GlobalOp_Changes_Trace_Status (int current_glop)
{
   if (glops_intervals.n_glops > 0)
   {
      if (glops_intervals.glop_list[glops_intervals.next].glop_id == current_glop)
      {
         glops_intervals.n_glops --;
         glops_intervals.next ++;

         return glops_intervals.glop_list[glops_intervals.next-1].trace_status;
      }
      else {
         return KEEP;
      }
   }
   else
   {
      return KEEP;
   }
}

/******************************************************************************
 ***  make_names
 ******************************************************************************/

static void make_names (void)
{
  char *fname;

  fname = PROGRAM_NAME + strlen (PROGRAM_NAME);
  while ((fname != PROGRAM_NAME) && (*fname != '/'))
    fname--;
  if (*fname == '/')
    fname++;

  strcpy (appl_name, fname);
}

/******************************************************************************
 ***  remove_temporal_files()
 ******************************************************************************/
int remove_temporal_files(void)
{
  unsigned int thread;
  char tmpname[TMP_NAME_LENGTH];

  for (thread = 0; thread < get_maximum_NumOfThreads(); thread++)
  {
		FileName_PTT(tmpname, Get_TemporalDir(TASKID), appl_name, getpid(), TASKID, thread, EXT_TMP_MPIT);
    if (unlink(tmpname) == -1)
      fprintf (stderr, PACKAGE_NAME": Error removing a temporal tracing file\n");

		FileName_PTT(tmpname, Get_TemporalDir(TASKID), appl_name, getpid(), TASKID, thread, EXT_TMP_SAMPLE);
    if (unlink(tmpname) == -1)
      fprintf (stderr, PACKAGE_NAME": Error removing a temporal sampling file\n");
  }
  return 0;
}

/**
 * Allocates a new tracing & sampling buffer for a given thread. 
 * \param thread_id The thread identifier.
 * \return 1 if success, 0 otherwise. 
 */
static int Allocate_buffer_and_file (int thread_id)
{
	int ret;
	int attempts = 100;
	char tmp_file[TMP_NAME_LENGTH];

	/* Some FS looks quite lazy and "needs time" to create directories?
	   This loop fixes this issue (seen in BGP) */
	ret = mkdir_recursive (Get_TemporalDir(TASKID));
	while (!ret && attempts > 0)
	{
		ret = mkdir_recursive (Get_TemporalDir(TASKID));
		attempts --;
	}
	if (!ret && attempts == 0)
	{
		fprintf (stderr, PACKAGE_NAME ": Error! Task %d was unable to create temporal directory %s\n", TASKID, Get_TemporalDir(TASKID));
	}

	FileName_PTT(tmp_file, Get_TemporalDir(TASKID), appl_name, getpid(), TASKID, thread_id, EXT_TMP_MPIT);

	TracingBuffer[thread_id] = new_Buffer (buffer_size, tmp_file);
	if (TracingBuffer[thread_id] == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error allocating tracing buffer for thread %d\n", thread_id);
		return 0;
	}
	if (circular_buffering)
		Buffer_SetFlushCallback (TracingBuffer[thread_id], Buffer_DiscardOldest);
	else
		Buffer_SetFlushCallback (TracingBuffer[thread_id], Extrae_Flush_Wrapper);

#if defined(SAMPLING_SUPPORT)
	FileName_PTT(tmp_file, Get_TemporalDir(TASKID), appl_name, getpid(), TASKID, thread_id, EXT_TMP_SAMPLE);
	SamplingBuffer[thread_id] = new_Buffer (buffer_size, tmp_file);
	if (SamplingBuffer[thread_id] == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error allocating sampling buffer for thread %d\n", thread_id);
		return 0;
	}
	Buffer_SetFlushCallback (SamplingBuffer[thread_id], NULL);
#endif

	return 1;
}

/**
 * Allocates as many tracing & sampling buffers as threads.
 * \param num_threads Total number of threads.
 * \return 1 if success, 0 otherwise.
 */
int Allocate_buffers_and_files (int world_size, int num_threads)
{
	int i;

#if !defined(HAVE_MRNET)
	UNREFERENCED_PARAMETER(world_size);
#else
#if 0
	/* FIXME: Temporarily disabled. new_buffer_size overflows when target_mbs > 1000 */
	if (MRNet_isEnabled())
	{
		int new_buffer_size = 0;
		int target_mbs = MRNCfg_GetTargetTraceSize();

		/* Override buffer_size depending on target trace size */
		new_buffer_size = ((target_mbs * 1024 * 1024 * 2) / (world_size * sizeof(event_t)));

		fprintf(stdout, PACKAGE_NAME": Overriding buffer size with MRNet configuration (target=%dMb, buffer_size=%d events)\n",
			target_mbs, new_buffer_size);

		buffer_size = new_buffer_size;
	}
#endif
#endif

	xmalloc(TracingBuffer, num_threads * sizeof(Buffer_t *));
#if defined(SAMPLING_SUPPORT)
	xmalloc(SamplingBuffer, num_threads * sizeof(Buffer_t *));
#endif

	for (i = 0; i < num_threads; i++)
		Allocate_buffer_and_file (i);

	return TRUE;
}

/**
 * Allocates more tracing & sampling buffers for extra threads.
 * \param new_num_threads Total new number of threads.
 * \return 1 if success, 0 otherwise.
 */
int Reallocate_buffers_and_files (int new_num_threads)
{
	int i;

	xrealloc(TracingBuffer, TracingBuffer, new_num_threads * sizeof(Buffer_t *));
#if defined(SAMPLING_SUPPORT)
	xrealloc(SamplingBuffer, SamplingBuffer, new_num_threads * sizeof(Buffer_t *));
#endif

	for (i = get_maximum_NumOfThreads(); i < new_num_threads; i++)
		Allocate_buffer_and_file (i);

	return TRUE;
}

/******************************************************************************
 **      Function name : Allocate_Task_Bitmap
 **      Author : HSG
 **      Description : Creates a bitmap mask just to know which ranks must 
 **        collect information or not.
 *****************************************************************************/
static int Allocate_Task_Bitmap (int size)
{
	int i;

	TracingBitmap = (int *) malloc (size * sizeof (int));
	if (TracingBitmap == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! Cannot obtain memory for tasks bitmap\n");
		exit (-1);
	}

	for (i = 0; i < size; i++)
		TracingBitmap[i] = TRUE;

	return 0;
}

#if defined(OMP_SUPPORT)
static int getnumProcessors (void)
{
	int numProcessors;

#if HAVE_SYSCONF
	numProcessors = (int) sysconf (_SC_NPROCESSORS_CONF);
	if (-1 == numProcessors)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot determine number of configured processors using sysconf\n");
		exit (-1);
	}
#else
# error "Cannot determine number of processors"
#endif

	return numProcessors;
}
#endif /* OMP_SUPPORT */

#if defined(PTHREAD_SUPPORT)


/*
  This 'pthread_key' will store the thread identifier of every pthread
  created and instrumented 
*/

static pthread_key_t pThreadIdentifier;
static pthread_mutex_t pThreadIdentifier_mtx;

void Backend_CreatepThreadIdentifier (void)
{
	pthread_key_create (&pThreadIdentifier, NULL);
	pthread_mutex_init (&pThreadIdentifier_mtx, NULL);
}

void Backend_SetpThreadIdentifier (int ID)
{
	pthread_setspecific (pThreadIdentifier, (void*) ID);
}

int Backend_GetpThreadIdentifier (void)
{
	return (int) pthread_getspecific (pThreadIdentifier);
}

void Backend_NotifyNewPthread (void)
{
	int numthreads;

	pthread_mutex_lock (&pThreadIdentifier_mtx);

	numthreads = Backend_getNumberOfThreads();
	Backend_SetpThreadIdentifier (numthreads);
	Backend_ChangeNumberOfThreads (numthreads+1);

	pthread_mutex_unlock (&pThreadIdentifier_mtx);
}

void Backend_setNumTentativeThreads (int numofthreads)
{
	int numthreads = get_current_NumOfThreads();

	/* These calls just allocate memory and files for the given num threads, but does not
	   modify the current number of threads */
	Backend_ChangeNumberOfThreads (numofthreads);
	Backend_ChangeNumberOfThreads (numthreads);
}

#endif

/******************************************************************************
 * Backend_preInitialize :
 ******************************************************************************/

int Backend_preInitialize (int me, int world_size, char *config_file)
{
	char *shell_name;
#if defined(OMP_SUPPORT)
	char *omp_value;
	char *new_num_omp_threads_clause;
	int numProcessors;
#endif
#if USE_HARDWARE_COUNTERS
	int set;
#endif

#if defined(PTHREAD_SUPPORT)
	Backend_CreatepThreadIdentifier ();
#endif

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: THID=%d Backend_preInitialize (rank=%d, size=%d, config_file=\n", THREADID, me, world_size, config_file);
#endif

	/* Allocate a bitmap to know which tasks are tracing */
	Allocate_Task_Bitmap (world_size);

	/* Check if the license is correct or not for this node/process */
	VerifyLicenseExecution();

	/* Just check we aren't running mpirun nor a shell! */
	if (!(strcmp (PROGRAM_NAME, "mpirun")))
		return FALSE;

	shell_name = getenv ("SHELL");
	if (shell_name != NULL && !(strcmp (PROGRAM_NAME, shell_name)))
		return FALSE;

	/* Obtain the number of runnable threads in this execution.
	   Just check for OMP_NUM_THREADS env var (if this compilation
	   allows instrumenting OpenMP */

#if defined(OMP_SUPPORT)

#if !defined(DYNINST_MODULE)
	openmp_tracing_init();
#endif

	numProcessors = getnumProcessors();

	new_num_omp_threads_clause = (char*) malloc ((strlen("OMP_NUM_THREADS=xxxx")+1)*sizeof(char));
	if (NULL == new_num_omp_threads_clause)
	{
		fprintf (stderr, PACKAGE_NAME": Unable to allocate memory for tentative OMP_NUM_THREADS\n");
		exit (-1);
	}
	if (numProcessors >= 10000) /* xxxx in new_omp_threads_clause -> max 9999 */
	{
		fprintf (stderr, PACKAGE_NAME": Insufficient memory allocated for tentative OMP_NUM_THREADS\n");
		exit (-1);
	}

	sprintf (new_num_omp_threads_clause, "OMP_NUM_THREADS=%d\n", numProcessors);
	omp_value = getenv ("OMP_NUM_THREADS");
	if (omp_value)
	{
		int num_of_threads = atoi (omp_value);
		if (num_of_threads != 0)
		{
			current_NumOfThreads = maximum_NumOfThreads = num_of_threads;
			if (me == 0)
				fprintf (stdout, PACKAGE_NAME": OMP_NUM_THREADS set to %d\n", num_of_threads);
		}
		else
		{
			if (me == 0)
				fprintf (stderr,
					PACKAGE_NAME": OMP_NUM_THREADS is mandatory for this tracing library!\n"\
					PACKAGE_NAME": Setting OMP_NUM_THREADS to %d\n", numProcessors);
			putenv (new_num_omp_threads_clause);
			current_NumOfThreads = maximum_NumOfThreads = numProcessors;
		}
	}
	else
	{
		if (me == 0)
			fprintf (stderr,
				PACKAGE_NAME": OMP_NUM_THREADS is mandatory for this tracing library!\n"\
				PACKAGE_NAME": Setting OMP_NUM_THREADS to %d\n", numProcessors);
		putenv (new_num_omp_threads_clause);
		current_NumOfThreads = maximum_NumOfThreads = numProcessors;
	}
#elif defined(SMPSS_SUPPORT)
	extern int css_get_max_threads(void);

	current_NumOfThreads = maximum_NumOfThreads = css_get_max_threads();
#elif defined(NANOS_SUPPORT)
	extern unsigned int nanos_extrae_get_max_threads(void);

	current_NumOfThreads = maximum_NumOfThreads = nanos_extrae_get_max_threads();

#elif defined(UPC_SUPPORT)
	/* Set the current number of threads that is the UPC app running - THREADS! */
	current_NumOfThreads = maximum_NumOfThreads = GetNumUPCthreads ();

	/* Set the actual task identifier for this process -- in a full
	   shared/distributed PGAS - UPC system --. */
	TaskID_Setup (GetUPCprocID());
#else
	/* If we don't support OpenMP we still have this running thread :) */
	current_NumOfThreads = maximum_NumOfThreads = 1;
	if (getenv("OMP_NUM_THREADS"))
	{
		if (me == 0)
			fprintf (stderr,
				PACKAGE_NAME": Warning! OMP_NUM_THREADS is set but OpenMP is not supported!\n");
	}
#endif

#if defined(IS_CELL_MACHINE)
	prepare_CELLTrace_init (maximum_NumOfThreads);
#endif

	/* Initialize the clock */
	CLOCK_INIT;

	/* Configure the tracing subsystem */
#if defined(HAVE_XML2)
	if (config_file != NULL)
	{
		Parse_XML_File  (me, world_size, config_file);
	}
	else
	{
		if (getenv ("EXTRAE_ON") != NULL)
			read_environment_variables (me);
		else
			fprintf (stdout, PACKAGE_NAME": Application has been linked or preloaded with Extrae, BUT neither EXTRAE_ON nor EXTRAE_CONFIG_FILE are set!\n");
	}
#else
	if (getenv("EXTRAE_ON") != NULL)
		read_environment_variables (me);
	else
		fprintf (stdout, PACKAGE_NAME": Application has been linked or preloaded with Extrae, BUT EXTRAE_ON is NOT set!\n");
#endif

	/* If we aren't tracing, just skip everything! */
	if (!mpitrace_on)
		return FALSE;

	make_names ();

	/* Allocate the buffers and trace files */
	Allocate_buffers_and_files (world_size, maximum_NumOfThreads);

	/* This has been moved a few lines above to make sure the APPL_EV is the first in the trace */
	ApplBegin_Time = TIME;
	TRACE_EVENT (ApplBegin_Time, APPL_EV, EVT_BEGIN);

#if USE_HARDWARE_COUNTERS
	/* Write hardware counters definitions */
	for (set=0; set<HWC_Get_Num_Sets(); set++)
	{
		int num_hwc, *HWCid;

		num_hwc = HWC_Get_Set_Counters_Ids (set, &HWCid); /* HWCid is allocated up to MAX_HWC and sets NO_COUNTER where appropriate */
		TRACE_EVENT_AND_GIVEN_COUNTERS (ApplBegin_Time, HWC_DEF_EV, set, MAX_HWC, HWCid);
	}

	/* Start reading counters */
	HWC_Start_Counters (maximum_NumOfThreads);
#endif

	/* Initialize Tracing Mode related variables */
	Trace_Mode_Initialize (maximum_NumOfThreads);

#if !defined(IS_BG_MACHINE) && defined(TEMPORARILY_DISABLED)
	Myrinet_HWC_Initialize();
#endif

	last_mpi_exit_time = ApplBegin_Time;

	return TRUE;
}

/******************************************************************************
 * unsigned Backend_getNumberOfThreads (void)
 ******************************************************************************/
unsigned Backend_getNumberOfThreads (void)
{
	return get_current_NumOfThreads();
}

/******************************************************************************
 * unsigned Backend_getMaximumOfThreads (void)
 ******************************************************************************/
unsigned Backend_getMaximumOfThreads (void)
{
	return get_maximum_NumOfThreads();
}

/******************************************************************************
 * Backend_ChangeNumberOfThreads (unsigned numberofthreads)
 ******************************************************************************/
int Backend_ChangeNumberOfThreads (unsigned numberofthreads)
{
	unsigned new_num_threads = numberofthreads;

	/* If we aren't tracing, just skip everything! */
	if (!mpitrace_on)
		return FALSE;

	/* Just modify things if there are more threads */
	if (new_num_threads > maximum_NumOfThreads)
	{
		/* Reallocate the buffers and trace files */
		Reallocate_buffers_and_files (new_num_threads);

		/* Reallocate trace mode */
		Trace_Mode_reInitialize (maximum_NumOfThreads, new_num_threads);

#if USE_HARDWARE_COUNTERS
		/* Reallocate and start reading counters for these threads */
		HWC_Restart_Counters (maximum_NumOfThreads, new_num_threads);
#endif

		maximum_NumOfThreads = current_NumOfThreads = new_num_threads;
	}
	else
		current_NumOfThreads = new_num_threads;

	return TRUE;
}

static int GetTraceOptions (void)
{
	/* Calculate the options */
	int options = TRACEOPTION_NONE;

	if (circular_buffering)
		options |= TRACEOPTION_CIRCULAR_BUFFER;

#if USE_HARDWARE_COUNTERS
	options |= TRACEOPTION_HWC;
#endif

#if defined(IS_BIG_ENDIAN)
	options |= TRACEOPTION_BIGENDIAN;
#endif

	options |= (Clock_getType() == REAL_CLOCK)?TRACEOPTION_PARAVER:TRACEOPTION_DIMEMAS;

#if defined(IS_BG_MACHINE)
	options |= TRACEOPTION_BG_ARCH;
#elif defined(IS_MN_MACHINE)
	options |= TRACEOPTION_MN_ARCH;
#else
	options |= TRACEOPTION_UNK_ARCH;
#endif

	return options;
}

int Backend_postInitialize (int rank, int world_size, unsigned long long SynchroInitTime, unsigned long long SynchroEndTime, char **node_list)
{
#warning "Aixo caldria separar-ho en mes events (mpi_init no hi fa res aqui!!) -- synchro + options"
	int i;
	unsigned long long *StartingTimes=NULL, *SynchronizationTimes=NULL;
#if defined(MPI_SUPPORT)
	int rc;
#endif

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: THID=%d Backend_postInitialize (rank=%d, size=%d, syn_init_time=%llu, syn_fini_time=%llu\n", THREADID, rank, world_size, SynchroInitTime, SynchroEndTime);
#endif

	TimeSync_Initialize (world_size);

	xmalloc(StartingTimes, world_size * sizeof(UINT64));
	bzero(StartingTimes, world_size * sizeof(UINT64));
	xmalloc(SynchronizationTimes, world_size * sizeof(UINT64));
	bzero(SynchronizationTimes, world_size * sizeof(UINT64));

#if defined(MPI_SUPPORT)
	rc = PMPI_Allgather (&ApplBegin_Time, 1, MPI_LONG_LONG, StartingTimes, 1, MPI_LONG_LONG, MPI_COMM_WORLD);
	rc = PMPI_Allgather (&SynchroEndTime, 1, MPI_LONG_LONG, SynchronizationTimes, 1, MPI_LONG_LONG, MPI_COMM_WORLD);
#else
	StartingTimes[0] = ApplBegin_Time;
	SynchronizationTimes[0] = SynchroEndTime;
#endif
	
	for (i=0; i<world_size; i++)
	{
		char *node = (node_list == NULL) ? "" : node_list[i];
		TimeSync_SetInitialTime (i, StartingTimes[i], SynchronizationTimes[i], node);
	}

	TimeSync_CalculateLatencies (TS_NODE);

	xfree(StartingTimes);
	xfree(SynchronizationTimes);

#if defined(HAVE_MRNET)
	if (MRNet_isEnabled())
	{
		int rc = Join_MRNet(rank);

		if (rc)
		{
			fprintf (stdout, PACKAGE_NAME": MRNet successfully set up.\n");
		}
		else
		{
			fprintf (stderr, PACKAGE_NAME": Error while setting up the MRNet.\n");
			exit(1);
		}
	}
#endif

	/* Add MPI_init begin and end events */
	TRACE_MPIINITEV (SynchroInitTime, MPI_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	TRACE_MPIINITEV (SynchroEndTime, MPI_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, GetTraceOptions());

	/* HSG force a write to disk! */
	Buffer_Flush(TracingBuffer[THREADID]);

	if (mpitrace_on && !CheckForControlFile && !CheckForGlobalOpsTracingIntervals)
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": Successfully initiated with %d tasks\n\n", world_size);
	}
	else if (mpitrace_on && CheckForControlFile && !CheckForGlobalOpsTracingIntervals)
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": Successfully initiated with %d tasks BUT disabled by EXTRAE_CONTROL_FILE\n\n", world_size);

		/* Just disable the tracing until the control file is created */
		Extrae_shutdown_Wrapper();
		mpitrace_on = 0;		/* Disable full tracing. It will allow us to know if files must be deleted or kept */
	}
	else if (mpitrace_on && !CheckForControlFile && CheckForGlobalOpsTracingIntervals)
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": Successfully initiated with %d tasks BUT disabled by EXTRAE_CONTROL_GLOPS\n\n", world_size);
		Extrae_shutdown_Wrapper();
	}

	return TRUE;
}


/* HSG

 MN GPFS optimal files per directories is 512.
 
 Why blocking in 128 sets? Because each task may produce 2 files (mpit and
 sample), and also the final directory and the temporal directory may be the
 same. So on the worst case, there are 512 files in the directory at a time.

*/

static char _get_finaldir[TMP_DIR];
char *Get_FinalDir (int task)
{
  sprintf (_get_finaldir, "%s/set-%d", final_dir, task/128);
  return _get_finaldir;
}

static char _get_temporaldir[TMP_DIR];
char *Get_TemporalDir (int task)
{
  sprintf (_get_temporaldir, "%s/set-%d", tmp_dir, task/128);
  return _get_temporaldir;
}

/******************************************************************************
 ***  Backend_Finalize_close_mpits
 ******************************************************************************/
static void Backend_Finalize_close_mpits (int thread)
{
	int attempts = 100;
	int ret;
	char trace[TRACE_FILE];
	char tmp_name[TRACE_FILE];

	if (Buffer_IsClosed(TRACING_BUFFER(thread))) return;

	/* Some FS looks quite lazy and "needs time" to create directories?
	   This loop fixes this issue (seen in BGP) */
	ret = mkdir_recursive (Get_FinalDir(TASKID));
	while (!ret && attempts > 0)
	{
		ret = mkdir_recursive (Get_FinalDir(TASKID));
		attempts --;
	}
	if (!ret && attempts == 0)
	{
		fprintf (stderr, PACKAGE_NAME ": Error! Task %d was unable to create final directory %s\n", TASKID, Get_TemporalDir(TASKID));
	}

	Buffer_Close(TRACING_BUFFER(thread));

	FileName_PTT(tmp_name, Get_TemporalDir(TASKID), appl_name, getpid(), TASKID, thread, EXT_TMP_MPIT);
  FileName_PTT(trace, Get_FinalDir(TASKID), appl_name, getpid(), TASKID, thread, EXT_MPIT);

	rename_or_copy (tmp_name, trace);
	fprintf (stdout, PACKAGE_NAME": Intermediate raw trace file created : %s\n", trace);

#if defined(SAMPLING_SUPPORT)
	FileName_PTT(tmp_name, Get_TemporalDir(TASKID), appl_name, getpid(), TASKID, thread, EXT_TMP_SAMPLE);

	if (Buffer_GetFillCount(SAMPLING_BUFFER(thread)) > 0) 
	{
		Buffer_Flush(SAMPLING_BUFFER(thread));
		Buffer_Close(SAMPLING_BUFFER(thread));

		FileName_PTT(trace, Get_FinalDir(TASKID), appl_name, getpid(), TASKID, thread, EXT_SAMPLE);

		rename_or_copy (tmp_name, trace);
		fprintf (stdout, PACKAGE_NAME": Intermediate raw sample file created : %s\n", trace);
	}
	else
	{
		/* Remove file if empty! */
		unlink (tmp_name);

		fprintf (stdout, PACKAGE_NAME": Intermediate raw sample file NOT created (%d) due to lack of information gathered.\n", TASKID);
	}
#endif

#if defined(HAVE_MRNET)
	if (MRNet_isEnabled())
	{
		MRN_CloseFiles();
	}
#endif
}

/**
 * Flushes the buffer to disk and marks this I/O in trace.
 * \param buffer The buffer to be flushed.
 * \return 1 on success, 0 otherwise.
 */ 
int Extrae_Flush_Wrapper (Buffer_t *buffer)
{
	event_t FlushEv_Begin, FlushEv_End;
	int check_size;
	unsigned long long current_size;

	if (!Buffer_IsClosed (buffer))
	{
		FlushEv_Begin.time = TIME;
		FlushEv_Begin.event = FLUSH_EV;
		FlushEv_Begin.value = EVT_BEGIN;
		HARDWARE_COUNTERS_READ (THREADID, FlushEv_Begin, FALSE);

		Buffer_Flush (buffer);

		FlushEv_End.time = TIME;
		FlushEv_End.event = FLUSH_EV;
		FlushEv_End.value = EVT_END;
		HARDWARE_COUNTERS_READ (THREADID, FlushEv_End, FALSE);

		BUFFER_INSERT (THREADID, buffer, FlushEv_Begin);
		BUFFER_INSERT (THREADID, buffer, FlushEv_End);

		check_size = !hasMinimumTracingTime || (hasMinimumTracingTime && (TIME > MinimumTracingTime+initTracingTime));
		if (file_size > 0 && check_size)
		{
			if ((current_size = Buffer_GetFileSize (buffer)) >= file_size MBytes)
			{
				if (THREADID == 0)
				{
					fprintf (stdout, PACKAGE_NAME": File size limit reached. File occupies %llu bytes.\n", current_size);
					fprintf (stdout, "Further tracing is disabled.\n");
				}
				Backend_Finalize_close_mpits (THREADID);
				mpitrace_on = FALSE;
			}
		}
	}
	return 1;
}

void Backend_Finalize (void)
{
	unsigned thread;

	/* Stop sampling right now */
	setSamplingEnabled (FALSE);

	for (thread = 0; thread < maximum_NumOfThreads; thread++) 
	{
		Buffer_ExecuteFlushCallback (TracingBuffer[thread]);
	}
	if (THREADID == 0) 
	{
		iotimer_t tmp_time = TIME;
		Extrae_getrusage_Wrapper (tmp_time);
		Extrae_memusage_Wrapper (tmp_time);
	}
	for (thread = 0; thread < maximum_NumOfThreads; thread++)
	{
		TRACE_EVENT (TIME, APPL_EV, EVT_END);
		Buffer_ExecuteFlushCallback (TracingBuffer[thread]);
	}

	for (thread = 0; thread < maximum_NumOfThreads; thread++)
		Backend_Finalize_close_mpits (thread);

	if (TASKID == 0)
		fprintf (stdout, PACKAGE_NAME": Application has ended. Tracing has been terminated.\n");

#if defined(EMBED_MERGE_IN_TRACE)
	/* Launch the merger */
	if (MergeAfterTracing)
	{
		int ptask = 1, cfile = 1;
		char tmp[1024];
		mpitrace_on = FALSE; /* Turn off tracing now */

		if (TaskID_Get() == 0)
			fprintf (stdout, PACKAGE_NAME ": Proceeding with the merge of the intermediate tracefiles.\n");

#if defined(MPI_SUPPORT)
		/* Synchronize all tasks at this point so none overtakes the master and
		   gets and invalid/blank trace file list (.mpits file) */
		if (TaskID_Get() == 0)
			fprintf (stdout, PACKAGE_NAME ": Waiting for all tasks to reach the checkpoint.\n");

		MPI_Barrier (MPI_COMM_WORLD);
#endif

		sprintf (tmp, "%s/%s.mpits", final_dir, appl_name);
		merger_pre (NumOfTasks);
		Read_MPITS_file (tmp, &ptask, &cfile, FileOpen_Default);

		if (TaskID_Get() == 0)
			fprintf (stdout, PACKAGE_NAME ": Executing the merge process (using %s).\n", tmp);

		merger_post (NumOfTasks, TaskID_Get());
	}
#endif /* EMBED_MERGE_IN_TRACE */
}

