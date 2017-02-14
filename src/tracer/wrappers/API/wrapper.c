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
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
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

#if defined(IS_BG_MACHINE)
# if defined(IS_BGL_MACHINE)
#  include <rts.h>
# endif
# if defined(IS_BGP_MACHINE)
#  include <spi/kernel_interface.h>
#  include <common/bgp_personality.h>
#  include <common/bgp_personality_inlines.h>
# endif
# if defined(IS_BGQ_MACHINE)
#  include <firmware/include/personality.h>
#  include <spi/include/kernel/process.h>
#  include <spi/include/kernel/location.h>
# endif
#endif /* IS_BG_MACHINE */

#if defined(HAVE_SCHED_GETCPU)
	/* Although man indicates that this should be in sched.h, it seems to be not there */
	extern int sched_getcpu(void);
#endif 

#include "wrapper.h"
#if defined(MPI_SUPPORT)
# include "mpi_wrapper.h"
#endif
#include "misc_wrapper.h"
#include "fork_wrapper.h"
#include "clock.h"
#include "hwc.h"
#include "signals.h"
#include "utils.h"
#include "calltrace.h"
#include "xml-parse.h"
#include "UF_gcc_instrument.h"
#include "UF_xl_instrument.h"
#include "mode.h"
#include "events.h"
#if defined(OMP_SUPPORT)
# include "omp_probe.h"
# include "omp_wrapper.h"
# if defined(OMPT_INSTRUMENTATION)
#  include "ompt-wrapper.h"
# endif
#endif
#include "trace_buffers.h"
#include "timesync.h"
#if defined(HAVE_ONLINE)
# include "OnlineControl.h"
#endif
#if defined(UPC_SUPPORT)
# include <external/upc.h>
#endif
#if defined(CUDA_SUPPORT)
# include "cuda_wrapper.h"
# include "cuda_common.h"
#endif
#if defined(OPENCL_SUPPORT)
# include "opencl_wrapper.h"
#endif
#include "common_hwc.h"

#if defined(EMBED_MERGE_IN_TRACE)
# include "mpi2out.h"
# include "options.h"
#endif

#include "sampling-common.h"
#include "sampling-timer.h"
#if defined(ENABLE_PEBS_SAMPLING)
# include "sampling-intel-pebs.h"
#endif
#include "debug.h"
#include "threadinfo.h"
#if defined(PTHREAD_SUPPORT)
# include "pthread_probe.h"
#endif

#include "malloc_probe.h"
#include "malloc_wrapper.h"
#include "io_probe.h"
#include "io_wrapper.h"
#include "syscall_probe.h"
#include "syscall_wrapper.h"
#include "taskid.h"

int Extrae_Flush_Wrapper (Buffer_t *buffer);

static int requestedDynamicMemoryInstrumentation = FALSE;


#if defined(STANDALONE)
module_t *Modules = NULL;
NumberOfModules   = 0;

void Extrae_RegisterModule(module_id_t id, void *init_ptr, void *fini_ptr)
{
  NumberOfModules ++;
  xrealloc(Modules, Modules, sizeof(module_t) * NumberOfModules);

  Modules[NumberOfModules-1].init_ptr = init_ptr;
  Modules[NumberOfModules-1].fini_ptr = fini_ptr;
}
#endif /* STANDALONE */



void setRequestedDynamicMemoryInstrumentation (int b)
{
	requestedDynamicMemoryInstrumentation = b;
}

static int requestedIOInstrumentation = FALSE;

void setRequestedIOInstrumentation (int b)
{
	requestedIOInstrumentation = b;
}

static int requestedSysCallInstrumentation = FALSE;

void setRequestedSysCallInstrumentation (int b)
{
	requestedSysCallInstrumentation = b;
}

static void Backend_Finalize_close_mpits (pid_t pid, int thread, int append);

/***** Variable global per saber si en un moment donat cal tracejar ******/
int tracejant = TRUE;

/***** Variable global per saber si MPI s'ha de tracejar *****************/
int tracejant_mpi = TRUE;

/***** Variable global per saber si MPI s'ha de tracejar amb hwc *********/
int tracejant_hwc_mpi = TRUE;

/***** Variable global per saber si OpenMP s'ha de tracejar **************/
int tracejant_omp = TRUE;

/***** Variable global per saber si OpenMP s'ha de tracejar amb hwc ******/
int tracejant_hwc_omp = TRUE;

/***** Variable global per saber si pthread s'ha de tracejar **************/
int tracejant_pthread = TRUE;

void Extrae_set_pthread_tracing (int b)
{ tracejant_pthread = b; }

int Extrae_get_pthread_tracing (void)
{ return tracejant_pthread; }

/***** Variable global per saber si pthread s'ha de tracejar amb hwc ******/
int tracejant_hwc_pthread = TRUE;

void Extrae_set_pthread_hwc_tracing (int b)
{ tracejant_hwc_pthread = b; }

int Extrae_get_pthread_hwc_tracing (void)
{ return tracejant_hwc_pthread; }

/***** Variable global per saber si UFs s'han de tracejar amb hwc ********/
int tracejant_hwc_uf = TRUE;

/*** Variable global per saber si hem d'obtenir comptador de la xarxa ****/
int tracejant_network_hwc = FALSE;

/** Store information about rusage?                                     **/
int tracejant_rusage = FALSE;

/** Store information about malloc?                                     **/
int tracejant_memusage = FALSE;


/**** Variable global que controla quin subset de les tasks generen o ****/
/**** no generen trasa ***************************************************/
int *TracingBitmap = NULL;

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

/* CPU events emission frequency */
unsigned long long MinimumCPUEventTime = 0;
unsigned short AlwaysEmitCPUEvent = 0;
iotimer_t *LastCPUEmissionTime = NULL;
int *LastCPUEvent = NULL;

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

static unsigned current_NumOfThreads = 1;
static unsigned maximum_NumOfThreads = 1;

#define TMP_NAME_LENGTH     512
#define APPL_NAME_LENGTH    512
char appl_name[APPL_NAME_LENGTH];
char final_dir[TMP_DIR];
char tmp_dir[TMP_DIR];


/* Checks if there is a CPU event waiting to be emitted */
int PENDING_TRACE_CPU_EVENT(int thread_id, iotimer_t current_time)
{
	if (MinimumCPUEventTime > 0)
	{
		if (LastCPUEmissionTime[thread_id] == 0)
		{
			LastCPUEmissionTime[thread_id] = current_time;
		}
		else if ((current_time - LastCPUEmissionTime[thread_id]) >  MinimumCPUEventTime)
		{
			LastCPUEmissionTime[thread_id] = current_time;
			return 1;
		}
	}

	return 0;
}

/* HSG

 MN GPFS optimal files per directories is 512.
 
 Why blocking in 128 sets? Because each task may produce 2 files (mpit and
 sample), and also the final directory and the temporal directory may be the
 same. So on the worst case, there are 512 files in the directory at a time.

*/

int Extrae_Get_FinalDir_BlockSize(void)
{ return 128; }
static char _get_finaldir[TMP_DIR];
char *Get_FinalDir (int task)
{
	sprintf (_get_finaldir, "%s/set-%d", final_dir,
	  task/Extrae_Get_FinalDir_BlockSize());
	return _get_finaldir;
}
#if defined(MPI_SUPPORT)
char *Extrae_Get_FinalDirNoTask (void)
{
	return final_dir;
}
#endif

int Extrae_Get_TemporalDir_BlockSize(void)
{ return 128; }
static char _get_temporaldir[TMP_DIR];
char *Get_TemporalDir (int task)
{
	sprintf (_get_temporaldir, "%s/set-%d", tmp_dir,
	  task/Extrae_Get_TemporalDir_BlockSize());
	return _get_temporaldir;
}
#if defined(MPI_SUPPORT)
char *Extrae_Get_TemporalDirNoTask (void)
{
	return tmp_dir;
}
#endif

/* Know if the run is controlled by a creation of a file  */
static char ControlFileName[TMP_DIR];
static int CheckForControlFile = FALSE;
static int CheckForGlobalOpsTracingIntervals = FALSE;

int Extrae_getCheckControlFile (void)
{
	return CheckForControlFile;
}
char *Extrae_getCheckControlFileName (void)
{
	return ControlFileName;
}
int Extrae_getCheckForGlobalOpsTracingIntervals (void)
{
	return CheckForGlobalOpsTracingIntervals;
}

void Extrae_setCheckControlFile (int b)
{
	CheckForControlFile = b;
}
void Extrae_setCheckControlFileName (const char *f)
{
	strcpy (ControlFileName, f);
}
void Extrae_setCheckForGlobalOpsTracingIntervals (int b)
{
	CheckForGlobalOpsTracingIntervals = b;
}

int circular_buffering = 0;
event_t *circular_HEAD;

static void Extrae_getExecutableInfo (void);

#if defined(EMBED_MERGE_IN_TRACE)
int MergeAfterTracing = FALSE;
#endif

static int AppendingEventsToGivenPID = FALSE;
static int AppendingEventsToGivenPID_PID = 0;

void Extrae_setAppendingEventsToGivenPID (int pid)
{
	AppendingEventsToGivenPID = TRUE;
	AppendingEventsToGivenPID_PID = pid;
}

int Extrae_getAppendingEventsToGivenPID (int *pid)
{
	if (pid != NULL)
		*pid = AppendingEventsToGivenPID_PID;
	return AppendingEventsToGivenPID;
}

static int extrae_initialized = FALSE;

int EXTRAE_ON (void) { return mpitrace_on; }

int EXTRAE_INITIALIZED(void)
{ return extrae_initialized; }

void EXTRAE_SET_INITIALIZED(int init)
{ extrae_initialized = init; }

static unsigned get_maximum_NumOfThreads (void)
{
	return maximum_NumOfThreads;
}

static unsigned get_current_NumOfThreads (void)
{
	return current_NumOfThreads;
}

#if defined(IS_BG_MACHINE)
static void Extrae_BG_gettopology (int enter, UINT64 timestamp)
{
#if defined(IS_BGL_MACHINE)
	BGLPersonality personality;
	unsigned personality_size = sizeof (personality);

	rts_get_personality (&personality, personality_size);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_X, enter?personality.xCoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_Y, enter?personality.yCoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_Z, enter?personality.zCoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_PROCESSOR_ID, enter?rts_get_processor_id():0);
#endif

#if defined(IS_BGP_MACHINE)
	_BGP_Personality_t personality;
	unsigned personality_size = sizeof (personality);
	
	Kernel_GetPersonality (&personality, personality_size);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_X, enter?BGP_Personality_xCoord(&personality):0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_Y, enter?BGP_Personality_yCoord(&personality):0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_Z, enter?BGP_Personality_zCoord(&personality):0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_PROCESSOR_ID, enter?BGP_Personality_rankInPset (&personality):0);
#endif

#if defined(IS_BGQ_MACHINE)
	Personality_t personality;
	unsigned personality_size = sizeof (personality);
	Kernel_GetPersonality (&personality, personality_size);

	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_A, enter?personality.Network_Config.Acoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_B, enter?personality.Network_Config.Bcoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_C, enter?personality.Network_Config.Ccoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_D, enter?personality.Network_Config.Dcoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_TORUS_E, enter?personality.Network_Config.Ecoord:0);
	TRACE_MISCEVENT (timestamp, USER_EV, BG_PERSONALITY_PROCESSOR_ID, enter?Kernel_PhysicalProcessorID():0);
#endif
}
#endif

void Extrae_AnnotateCPU (UINT64 timestamp)
{
#if defined(HAVE_SCHED_GETCPU)
    int cpu = sched_getcpu();

    if (cpu != LastCPUEvent[THREADID] || AlwaysEmitCPUEvent)
    {
        LastCPUEvent[THREADID] = cpu;
        TRACE_EVENT (timestamp, GETCPU_EV, cpu);
    }
#else
    UNREFERENCED_PARAMETER(timestamp);
#endif
}

static void Extrae_AnnotateTopology (int enter, UINT64 timestamp)
{
#if defined(IS_BG_MACHINE)
	Extrae_BG_gettopology (enter, timestamp);
#else
	UNREFERENCED_PARAMETER(enter);
	Extrae_AnnotateCPU (timestamp);
#endif
}

char * Get_ApplName (void)
{
  return appl_name;
}

/******************************************************************************
 *** Store whether the app is MPI
 ******************************************************************************/
static int Extrae_Application_isMPI = FALSE;
static int Extrae_Application_isSHMEM = FALSE;

int Extrae_get_ApplicationIsMPI (void)
{
	return Extrae_Application_isMPI;
}

void Extrae_set_ApplicationIsMPI (int b)
{
	Extrae_Application_isMPI = b;
}

int Extrae_get_ApplicationIsSHMEM (void)
{
        return Extrae_Application_isSHMEM;
}

void Extrae_set_ApplicationIsSHMEM (int b)
{
        Extrae_Application_isSHMEM = b;
}


/******************************************************************************
 *** Store the first mechanism to initialize the tracing 
 ******************************************************************************/
static extrae_init_type_t Extrae_Init_Type = EXTRAE_NOT_INITIALIZED;

extrae_init_type_t Extrae_is_initialized_Wrapper (void)
{
	return EXTRAE_INITIALIZED()?Extrae_Init_Type:EXTRAE_NOT_INITIALIZED;
}

void Extrae_set_is_initialized (extrae_init_type_t type)
{
	Extrae_Init_Type = type;
}


/******************************************************************************
 ***  Backend_createExtraeDirectory
 ***   creates extrae directories to store tracefiles
 ******************************************************************************/
void Backend_createExtraeDirectory (int taskid, int Temporal)
{
	int ret;
	int attempts = 100;
	char *dirname;

	dirname = (Temporal)?Get_TemporalDir(taskid):Get_FinalDir(taskid);

	ret = mkdir_recursive (dirname);
	while (!ret && attempts > 0)
	{
		ret = mkdir_recursive (dirname);
		attempts--;
	}
	
	if (!ret && attempts == 0 && Temporal)
		fprintf (stderr, PACKAGE_NAME ": Error! Task %d was unable to create temporal directory %s\n", taskid, dirname);
	else if (!ret && attempts == 0 && !Temporal)
		fprintf (stderr, PACKAGE_NAME ": Error! Task %d was unable to create final directory %s\n", taskid, dirname);
}

/******************************************************************************
 ***  read_environment_variables
 ***  Reads some environment variables. Returns 0 if the tracing was disabled,
 ***  otherwise, 1.
 ******************************************************************************/

static int read_environment_variables (int me)
{
#if defined(MPI_SUPPORT)
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
		Extrae_setCheckControlFile (TRUE);
		Extrae_setCheckControlFileName (file);
		if (me == 0)
			fprintf (stdout, PACKAGE_NAME": Control file is %s.\n          Tracing will be disabled until the file exists\n", file);
	}
	else
		Extrae_setCheckControlFile (FALSE);

	/* EXTRAE_CONTROL_GLOPS, activates the tracing on a global op series */
	if ((str = getenv ("EXTRAE_CONTROL_GLOPS")) != NULL)
	{
		Extrae_setCheckForGlobalOpsTracingIntervals(TRUE);
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
	Extrae_pthread_instrument_locks ((str != NULL && (strcmp (str, "1"))));
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

	/* Add sampling capabilities */
#if defined(SAMPLING_SUPPORT)
	str = getenv ("EXTRAE_SAMPLING_PERIOD");
	if (str != NULL)
	{
		unsigned long long sampling_period = getTimeFromStr (
		  getenv("EXTRAE_SAMPLING_PERIOD"), "EXTRAE_SAMPLING_PERIOD", me);
		unsigned long long sampling_variability = 0;
		char *str_var = getenv("EXTRAE_SAMPLING_VARIABILITY");
		if (str_var != NULL)
			sampling_variability = getTimeFromStr (
				getenv("EXTRAE_SAMPLING_VARIABILITY"), "EXTRAE_SAMPLING_VARIABILITY", me);

		if (sampling_period != 0)
		{
			char *str2;
			if ((str2 = getenv ("EXTRAE_SAMPLING_CLOCKTYPE")) != NULL)
			{
				if (strcmp (str2, "DEFAULT") == 0)
					setTimeSampling (sampling_period, sampling_variability, SAMPLING_TIMING_DEFAULT);
				else if (strcmp (str2, "REAL") == 0)
					setTimeSampling (sampling_period, sampling_variability, SAMPLING_TIMING_REAL);
				else if (strcmp (str2, "VIRTUAL") == 0)
					setTimeSampling (sampling_period, sampling_variability, SAMPLING_TIMING_VIRTUAL);
				else if (strcmp (str2, "PROF") == 0)
					setTimeSampling (sampling_period, sampling_variability, SAMPLING_TIMING_PROF);
				else
				{
					if (me == 0)
						fprintf (stderr, "Extrae: Warning! Value '%s' <sampling type=\"..\" /> is unrecognized. Using default.\n", str2);
				}
			}
			else	
				setTimeSampling (sampling_period, sampling_variability, SAMPLING_TIMING_DEFAULT);

			if (me == 0)
				fprintf (stdout, "Extrae: Sampling enabled with a period of %lld microseconds and a variability of %lld microseconds.\n", sampling_period/1000, sampling_variability/1000);
		}
		else
		{
			if (me == 0)
				fprintf (stderr, "Extrae: Warning! Value '%s' for EXTRAE_SAMPLING_PERIOD is unrecognized\n", str);
		}
	}
	if (getenv ("EXTRAE_SAMPLING_CALLER") != NULL)
		Parse_Callers (me, getenv("EXTRAE_SAMPLING_CALLER"), CALLER_SAMPLING);
#endif

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

   callers = (char *)malloc(sizeof(char)*(strlen(mpi_callers)+1));
   strcpy(callers, mpi_callers);

   while ((caller = strtok(callers, (const char *)",")) != NULL) {
      callers = NULL;
      if (sscanf(caller, "%d-%d", &from, &to) != 2){
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
	if (Caller_Count[type] > 0 && me == 0)
	{	
		const char *s;
		const char *s_mpi = "MPI";
		const char *s_sampling = "Sampling";
		const char *s_malloc = "Dynamic-Memory";
		const char *s_io = "Input/Output";
		const char *s_syscall = "System Calls";
		const char *s_unknown = "unknown?";

		if (CALLER_MPI == type)
			s = s_mpi;
		else if (CALLER_SAMPLING == type)
			s = s_sampling;
		else if (CALLER_DYNAMIC_MEMORY == type)
			s = s_malloc;
		else if (CALLER_IO == type)
			s = s_io;
		else if (CALLER_SYSCALL == type)
			s = s_syscall;
		else
			s = s_unknown;

		fprintf(stdout, PACKAGE_NAME": Tracing %d level(s) of %s callers: [ ",
		  Caller_Count[type], s);

		for (i=0; i<Caller_Deepness[type]; i++)
			if (Trace_Caller[type][i])
				fprintf(stdout, "%d ", i+1);
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
 ***  copy_program_name
 ******************************************************************************/

static void copy_program_name (void)
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
	unsigned initialTASKID = Extrae_get_initial_TASKID();
	char hostname[1024];

	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");

	for (thread = 0; thread < get_maximum_NumOfThreads(); thread++)
	{
		/* Remember .mpit and .sample intermediate files are allocated once and do
	     NOT change even if the TASK changes during runtime, thus use initialTASKID
		   instead of TASKID */
		FileName_PTT(tmpname, Get_TemporalDir(initialTASKID), appl_name,
		  hostname, getpid(), initialTASKID, thread, EXT_TMP_MPIT);
		if (file_exists(tmpname))
			if (unlink(tmpname) == -1)
				fprintf (stderr, PACKAGE_NAME": Error removing a temporal tracing file (%s)\n", tmpname);

#if defined(SAMPLING_SUPPORT)
		FileName_PTT(tmpname, Get_TemporalDir(initialTASKID), appl_name,
		  hostname, getpid(), initialTASKID, thread, EXT_TMP_SAMPLE);
		if (file_exists(tmpname))
			if (unlink(tmpname) == -1)
				fprintf (stderr, PACKAGE_NAME": Error removing a temporal sampling file (%s)\n", tmpname);
#endif

		FileName_PTT(tmpname, Get_TemporalDir(TASKID), appl_name, hostname,
		  getpid(), TASKID, thread, EXT_SYM);
		if (file_exists (tmpname))
			if (unlink(tmpname) == -1)
				fprintf (stderr, PACKAGE_NAME": Error removing symbol file (%s)\n", tmpname);
	}
#if defined(HAVE_ONLINE)
	Online_CleanTemporaries();
#endif
	return 0;
}

/**
 * Allocates a new tracing & sampling buffer for a given thread. 
 * \param thread_id The thread identifier.
 * \return 1 if success, 0 otherwise. 
 */
static int Allocate_buffer_and_file (int thread_id, int forked)
{
	char tmp_file[TMP_NAME_LENGTH];
	unsigned initialTASKID = Extrae_get_initial_TASKID();
	char hostname[1024];

	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");

	FileName_PTT(tmp_file, Get_TemporalDir(initialTASKID), appl_name,
	  hostname, getpid(), initialTASKID, thread_id, EXT_TMP_MPIT);

	if (forked)
		Buffer_Free (TracingBuffer[thread_id]);

	LastCPUEmissionTime[thread_id] = 0;
	LastCPUEvent[thread_id] = 0;
	TracingBuffer[thread_id] = new_Buffer (buffer_size, tmp_file, TRUE);
	if (TracingBuffer[thread_id] == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error allocating tracing buffer for thread %d\n", thread_id);
		return 0;
	}
	if (circular_buffering)
	{
		Buffer_AddCachedEvent (TracingBuffer[thread_id], MPI_INIT_EV);
		Buffer_AddCachedEvent (TracingBuffer[thread_id], MPI_RANK_CREACIO_COMM_EV);
		Buffer_AddCachedEvent (TracingBuffer[thread_id], MPI_ALIAS_COMM_CREATE_EV);
		Buffer_AddCachedEvent (TracingBuffer[thread_id], HWC_CHANGE_EV);
		Buffer_SetFlushCallback (TracingBuffer[thread_id], Buffer_DiscardOldest);
	}
	else
	{
		Buffer_SetFlushCallback (TracingBuffer[thread_id], Extrae_Flush_Wrapper);
	}

#if defined(SAMPLING_SUPPORT)
	FileName_PTT(tmp_file, Get_TemporalDir(initialTASKID), appl_name,
	  hostname, getpid(), initialTASKID, thread_id, EXT_TMP_SAMPLE);

	if (forked)
		Buffer_Free (SamplingBuffer[thread_id]);

	SamplingBuffer[thread_id] = new_Buffer (buffer_size, tmp_file, FALSE);
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
static int Allocate_buffers_and_files (int world_size, int num_threads, int forked)
{
	int i;

#if 0 /* MRNET */
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
#else
	UNREFERENCED_PARAMETER(world_size);
#endif

	if (!forked)
	{
		xmalloc(TracingBuffer, num_threads * sizeof(Buffer_t *));
		xmalloc(LastCPUEmissionTime, num_threads * sizeof(iotimer_t));
		xmalloc(LastCPUEvent, num_threads * sizeof(int));
#if defined(SAMPLING_SUPPORT)
		xmalloc(SamplingBuffer, num_threads * sizeof(Buffer_t *));
#endif
	}

	for (i = 0; i < num_threads; i++)
		Allocate_buffer_and_file (i, forked);

	return TRUE;
}

/**
 * Allocates more tracing & sampling buffers for extra threads.
 * \param new_num_threads Total new number of threads.
 * \return 1 if success, 0 otherwise.
 */
static int Reallocate_buffers_and_files (int new_num_threads)
{
	int i;

	xrealloc(TracingBuffer, TracingBuffer, new_num_threads * sizeof(Buffer_t *));
	xrealloc(LastCPUEmissionTime, LastCPUEmissionTime, new_num_threads * sizeof(iotimer_t));
	xrealloc(LastCPUEvent, LastCPUEvent, new_num_threads * sizeof(int));
#if defined(SAMPLING_SUPPORT)
	xrealloc(SamplingBuffer, SamplingBuffer, new_num_threads * sizeof(Buffer_t *));
#endif

	for (i = get_maximum_NumOfThreads(); i < new_num_threads; i++)
		Allocate_buffer_and_file (i, FALSE);

	return TRUE;
}

/******************************************************************************
 **      Function name : Extrae_Allocate_Task_Bitmap
 **      Author : HSG
 **      Description : Creates a bitmap mask just to know which ranks must 
 **        collect information or not.
 *****************************************************************************/
int Extrae_Allocate_Task_Bitmap (int size)
{
	int i;

	TracingBitmap = (int *) realloc (TracingBitmap, size*sizeof (int));
	if (TracingBitmap == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! Cannot obtain memory for tasks bitmap\n");
		exit (-1);
	}

	for (i = 0; i < size; i++)
		TracingBitmap[i] = TRUE;

	return 0;
}

#if defined(OMP_SUPPORT) && !defined(OMPT_INSTRUMENTATION)
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
#endif /* OMP_SUPPORT && !OMPT_INSTRUMENTATION */

#if defined(PTHREAD_SUPPORT)

/*
  This 'pthread_key' will store the thread identifier of every pthread
  created and instrumented 
*/

static pthread_t *pThreads = NULL;
static unsigned numpThreads = 0;
static pthread_key_t pThreadIdentifier;
static pthread_mutex_t pThreadIdentifier_mtx;

static void Extrae_reallocate_pthread_info (int new_num_threads)
{
	xrealloc(pThreads, pThreads, new_num_threads * sizeof(pthread_t));
	numpThreads = new_num_threads;
}

void Backend_SetpThreadID (pthread_t *t, int threadid)
{
	pThreads[threadid] = *t;
}

pthread_t Backend_GetpThreadID (int threadid)
{
	return pThreads[threadid];
}

int Backend_ispThreadFinished (int threadid)
{
	return Backend_GetpThreadID(threadid) == 0;
}

void Backend_Flush_pThread (pthread_t t)
{
	unsigned u;

	for (u = 0; u < get_maximum_NumOfThreads(); u++)
		if (pThreads[u] == t)
		{
			pThreads[u] = (pthread_t)0; // This slot won't be used in the future

			Buffer_Flush(TRACING_BUFFER(u));
			Backend_Finalize_close_mpits (getpid(), u, FALSE);

			Buffer_Free (TRACING_BUFFER(u));
			TRACING_BUFFER(u) = NULL;
#if defined(SAMPLING_SUPPORT)
			Buffer_Free (SAMPLING_BUFFER(u));
			SAMPLING_BUFFER(u) = NULL;
#endif
			break;
		}
}

void Backend_CreatepThreadIdentifier (void)
{
	pthread_key_create (&pThreadIdentifier, NULL);
	pthread_mutex_init (&pThreadIdentifier_mtx, NULL);
}

void Backend_SetpThreadIdentifier (int ID)
{
	pthread_setspecific (pThreadIdentifier, (void*)((long) ID));
}

int Backend_GetpThreadIdentifier (void)
{
	return (int) ((long)pthread_getspecific (pThreadIdentifier));
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
#endif

void Backend_setNumTentativeThreads (int numofthreads)
{
	int numthreads = get_current_NumOfThreads();

	/* These calls just allocate memory and files for the given num threads, but does not
	   modify the current number of threads */
	Backend_ChangeNumberOfThreads (numofthreads);
	Backend_ChangeNumberOfThreads (numthreads);
}


/******************************************************************************
 * Backend_preInitialize :
 ******************************************************************************/

int Backend_preInitialize (int me, int world_size, const char *config_file, int forked)
{
	unsigned u;
	int runningInDynInst = FALSE;
	char trace_sym[TMP_DIR];
#if defined(OMP_SUPPORT) && !defined(OMPT_INSTRUMENTATION)
	char *omp_value;
	char *new_num_omp_threads_clause;
	int numProcessors;
#endif
#if USE_HARDWARE_COUNTERS
	int set;
#endif
	char hostname[1024];

	if (getenv("EXTRAE_APPEND_PID") != NULL)
		Extrae_setAppendingEventsToGivenPID (atoi(getenv("EXTRAE_APPEND_PID")));

	/* Mark the initialization as if we're in instrumentation */
	Backend_setInInstrumentation (THREADID, TRUE);

	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");

#if defined(STANDALONE)
	fprintf(stderr, "[DEBUG] NumberOfModules=%d\n", NumberOfModules);
	int current_module = 0;
	for (current_module=0; current_module<NumberOfModules; current_module++)
	{
		void (*module_init_func)(int) = Modules[i].init_ptr;

		module_init_func(me);
	}
	fprintf(stderr, "[DEBUG] current_NumOfThreads=%d maximum_NumOfThreads=%d\n", current_NumOfThreads, maximum_NumOfThreads);
#else

#if defined(INSTRUMENT_DYNAMIC_MEMORY)
	Extrae_malloctrace_init();
#endif

#if defined(INSTRUMENT_IO)
	Extrae_iotrace_init();
#endif

#if defined(PTHREAD_SUPPORT)
	Backend_CreatepThreadIdentifier ();

	/* Annotate myself */
	{
		pthread_t self = pthread_self();
		Extrae_reallocate_pthread_info (1);
		Backend_SetpThreadID (&self, 0);
	}
#endif

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: THID=%d Backend_preInitialize (rank=%d, size=%d, config_file=%s)\n", THREADID, me, world_size, config_file);
#endif

	if (getenv("EXTRAE_DYNINST_RUN") != NULL)
	{
		if (strcmp (getenv("EXTRAE_DYNINST_RUN"), "yes") == 0)
		{
			if (me == 0 && !forked)
				fprintf (stdout, PACKAGE_NAME": Target application is being run.\n");
			runningInDynInst = TRUE;
		}
	}
	else
	{
		if (me == 0 && !forked)
			fprintf (stdout, "Welcome to %s\n", PACKAGE_STRING);
	}

	/* Allocate a bitmap to know which tasks are tracing */
	if (!forked)
		Extrae_Allocate_Task_Bitmap (world_size);

#if defined(CUDA_SUPPORT)
	Extrae_CUDA_init (me);
#endif

#if defined(OPENCL_SUPPORT)
	Extrae_OpenCL_init (me);
#endif

#if defined(OMP_SUPPORT)
# if !defined(OMPT_INSTRUMENTATION)

	Extrae_OpenMP_init(me);

	/* Obtain the number of runnable threads in this execution.
	   Just check for OMP_NUM_THREADS env var (if this compilation
	   allows instrumenting OpenMP */
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

	sprintf (new_num_omp_threads_clause, "OMP_NUM_THREADS=%d", numProcessors);
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

# endif /* OMPT_INSTRUMENTATION */

#elif defined(SMPSS_SUPPORT) || defined(NANOS_SUPPORT) || defined (UPC_SUPPORT)
	current_NumOfThreads = maximum_NumOfThreads = Extrae_get_num_threads();
#else
	/* If we don't support OpenMP we still have this running thread :) */
	current_NumOfThreads = maximum_NumOfThreads = Extrae_get_num_threads();

	if (getenv("OMP_NUM_THREADS"))
	{
		if (me == 0)
			fprintf (stderr,
				PACKAGE_NAME": Warning! OMP_NUM_THREADS is set but OpenMP is not supported!\n");
	}
#endif

#endif /* STANDALONE */

	/* Initialize the clock */
	if (!forked)
		CLOCK_INIT (get_maximum_NumOfThreads());

	/* Allocate thread info structure */
	if (!forked)
		Extrae_allocate_thread_info (get_maximum_NumOfThreads());

	/* Configure the tracing subsystem */
	if (!forked)
	{
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
	}

	/* If we aren't tracing, just skip everything! */
	if (!mpitrace_on)
		return FALSE;

	copy_program_name ();

	/* Remove the .sym file if the file has not been generated by DynInst */
	if (me == 0 && !runningInDynInst)
	{
		FileName_P(trace_sym, final_dir, appl_name, EXT_SYM);
		if (file_exists(trace_sym))
			unlink (trace_sym);
	}

	Backend_ChangeNumberOfThreads_InInstrumentation (get_maximum_NumOfThreads());

	/* Remove the locals .sym file */
	for (u = 0; u < get_maximum_NumOfThreads(); u++)
	{
		FileName_PTT(trace_sym, Get_TemporalDir(Extrae_get_initial_TASKID()),
		  appl_name, hostname, getpid(), Extrae_get_initial_TASKID(), u,
		  EXT_SYM);
		if (file_exists (trace_sym))
			unlink (trace_sym);
	}

#if defined(MPI_SUPPORT)
	Extrae_MPI_prepareDirectoryStructures (me, world_size);
#else
	/* Create temporal directory */
	Backend_createExtraeDirectory (me, TRUE);

	/* Create final directory */
	Backend_createExtraeDirectory (me, FALSE);
#endif

	/* Allocate the buffers and trace files */
	Allocate_buffers_and_files (world_size, get_maximum_NumOfThreads(), forked);

#if defined(CUDA_SUPPORT)
	/* Allocate thread info for CUDA execs */
	Extrae_reallocate_CUDA_info (get_maximum_NumOfThreads());
#endif

	if (!Extrae_getAppendingEventsToGivenPID(NULL))
	{
		ApplBegin_Time = TIME;
		TRACE_EVENT (ApplBegin_Time, APPL_EV, EVT_BEGIN);
#if !defined(IS_BG_MACHINE)
		Extrae_AnnotateTopology (TRUE, ApplBegin_Time);
#endif
		TRACE_EVENT (ApplBegin_Time, CPU_EVENT_INTERVAL_EV, MinimumCPUEventTime);

	/* Initialize Tracing Mode related variables */
	if (forked)
		Trace_Mode_CleanUp();

	Trace_Mode_Initialize (get_maximum_NumOfThreads());

	Trace_Mode_Change (0, ApplBegin_Time);

#if USE_HARDWARE_COUNTERS
		/* Write hardware counter definitions into the .sym file */
		if (me == 0 && !forked)
		{
			unsigned count;
			HWC_Definition_t *hwc_defs = HWCBE_GET_COUNTER_DEFINITIONS(&count);
			if (hwc_defs != NULL)
			{
				u = 0;
				while (u < count)
				{
					Extrae_AddTypeValuesEntryToGlobalSYM ('H', hwc_defs[u].event_code,
						hwc_defs[u].description, (char)0, 0, NULL, NULL);
					u++;
				}
				free (hwc_defs);
			}
		}
	
		/* Write hardware counters set definitions (i.e. those given at config time) into the .mpit files*/
		for (set=0; set<HWC_Get_Num_Sets(); set++)
		{
			int *HWCid;
	
			HWC_Get_Set_Counters_Ids (set, &HWCid); /* HWCid is allocated up to MAX_HWC and sets NO_COUNTER where appropriate */
			TRACE_EVENT_AND_GIVEN_COUNTERS (ApplBegin_Time, HWC_DEF_EV, set, MAX_HWC, HWCid);
			xfree (HWCid);
		}
	
		/* Start reading counters */
		HWC_Start_Counters (get_maximum_NumOfThreads(), ApplBegin_Time, forked);
#endif
	}
	else
	{
		Trace_Mode_Initialize (get_maximum_NumOfThreads());
	}

#if !defined(IS_BG_MACHINE) && defined(TEMPORARILY_DISABLED)
	Myrinet_HWC_Initialize();
#endif

#if defined(HAVE_ONLINE)
	if (Online_isEnabled())
	{
		int rc;

		rc = Online_Init(me, world_size);
		if (rc == -1)
		{
			Online_Disable();
			fprintf(stderr, "Backend_preInitialize:: ERROR: Online_Init() failed.\n");
		}	
	}
#endif

	Extrae_getExecutableInfo ();

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

	if (EXTRAE_INITIALIZED())
	{
		/* Just modify things if there are more threads */
		if (new_num_threads > get_maximum_NumOfThreads())
		{
			unsigned u;
	
			/* Reallocate InInstrumentation structures */
			Backend_ChangeNumberOfThreads_InInstrumentation (new_num_threads);
			/* We leave... so, we're no longer in instrumentatin from this point */
			for (u = get_maximum_NumOfThreads(); u < new_num_threads; u++)
				Backend_setInInstrumentation (u, FALSE);
	
			/* Reallocate clock structures */
			Clock_AllocateThreads (new_num_threads);
	
			/* Reallocate the buffers and trace files */
			Reallocate_buffers_and_files (new_num_threads);
	
			/* Reallocate trace mode */
			Trace_Mode_reInitialize (get_maximum_NumOfThreads(), new_num_threads);
	
#if USE_HARDWARE_COUNTERS
			/* Reallocate and start reading counters for these threads */
			HWC_Restart_Counters (get_maximum_NumOfThreads(), new_num_threads);
#endif
	
			/* Allocate thread info structure */
			Extrae_reallocate_thread_info (get_maximum_NumOfThreads(), new_num_threads);
	
#if defined(CUDA_SUPPORT)
			/* Allocate thread info for CUDA execs */
			Extrae_reallocate_CUDA_info (new_num_threads);
#endif
	
#if defined(PTHREAD_SUPPORT)
			/* Allocate thread info for pthread execs */
			Extrae_reallocate_pthread_info (new_num_threads);
#endif
	
			maximum_NumOfThreads = current_NumOfThreads = new_num_threads;
		}
		else
			current_NumOfThreads = new_num_threads;
	}
	else
	{
		if (new_num_threads > get_maximum_NumOfThreads())
			maximum_NumOfThreads = new_num_threads;

		current_NumOfThreads = new_num_threads;
	}

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
#else
	options |= TRACEOPTION_UNK_ARCH;
#endif

	return options;
}

int Backend_postInitialize (int rank, int world_size, unsigned init_event,
	unsigned long long InitTime, unsigned long long EndTime, char **node_list)
{
	unsigned u;
	int i;
	unsigned long long *StartingTimes=NULL, *SynchronizationTimes=NULL;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: THID=%d Backend_postInitialize (rank=%d, size=%d, syn_init_time=%llu, syn_fini_time=%llu\n", THREADID, rank, world_size, InitTime, EndTime);
#endif

	TimeSync_Initialize (1, &world_size);

	xmalloc(StartingTimes, world_size * sizeof(UINT64));
	memset (StartingTimes, 0, world_size * sizeof(UINT64));
	xmalloc(SynchronizationTimes, world_size * sizeof(UINT64));
	memset (SynchronizationTimes, 0, world_size * sizeof(UINT64));

#if defined(MPI_SUPPORT)
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_MPI_INIT &&
	    world_size > 1)
	{
		int rc;
		rc = PMPI_Allgather (&ApplBegin_Time, 1, MPI_LONG_LONG, StartingTimes,
		  1, MPI_LONG_LONG, MPI_COMM_WORLD);
		if (rc != MPI_SUCCESS)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Could not gather starting times!\n");
			exit(1);
		}
		rc = PMPI_Allgather (&EndTime, 1, MPI_LONG_LONG, SynchronizationTimes,
		  1, MPI_LONG_LONG, MPI_COMM_WORLD);
		if (rc != MPI_SUCCESS)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Could not gather synchronization times!\n");
			exit(1);
		}
	}
	else
	{
		StartingTimes[0] = ApplBegin_Time;
		SynchronizationTimes[0] = EndTime;
	}
#else
	StartingTimes[0] = ApplBegin_Time;
	SynchronizationTimes[0] = EndTime;
#endif
	
	for (i=0; i<world_size; i++)
	{
		char *node = (node_list == NULL) ? "" : node_list[i];
		TimeSync_SetInitialTime (0, i, StartingTimes[i], SynchronizationTimes[i], node);
	}

	TimeSync_CalculateLatencies (TS_NODE);

	xfree(StartingTimes);
	xfree(SynchronizationTimes);

#if defined(HAVE_ONLINE)
	if (Online_isEnabled())
	{
		Online_Start(node_list);
	}
#endif

	if ((!Extrae_getAppendingEventsToGivenPID(NULL)) && (init_event != 0))
	{
		/* Add initialization begin and end events */
		TRACE_MPIINITEV (InitTime, init_event, EVT_BEGIN,
		  getpid(),
		  Extrae_isProcessMaster()?0:getppid(),
		  Extrae_myDepthOfAllProcesses(),
		  EMPTY,
		  EMPTY);
		Extrae_AnnotateTopology (TRUE, InitTime);
		Extrae_getrusage_set_to_0_Wrapper (InitTime);

		TRACE_MPIINITEV (EndTime, init_event, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, GetTraceOptions());
		Extrae_AnnotateTopology (FALSE, EndTime);
	}

	/* HSG force a write to disk! */
	Buffer_Flush(TRACING_BUFFER(THREADID));

	if (mpitrace_on &&
	  !Extrae_getCheckControlFile() &&
	  !Extrae_getCheckForGlobalOpsTracingIntervals())
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": Successfully initiated with %d tasks and %d threads\n\n", world_size, Backend_getNumberOfThreads());
	}
	else if (mpitrace_on &&
	  Extrae_getCheckControlFile() &&
	  !Extrae_getCheckForGlobalOpsTracingIntervals())
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": Successfully initiated with %d tasks and %d threads BUT disabled by EXTRAE_CONTROL_FILE\n\n", world_size, Backend_getNumberOfThreads());

		/* Just disable the tracing until the control file is created */
		Extrae_shutdown_Wrapper();
		mpitrace_on = 0;		/* Disable full tracing. It will allow us to know if files must be deleted or kept */
	}
	else if (mpitrace_on &&
	  !Extrae_getCheckControlFile() &&
	  Extrae_getCheckForGlobalOpsTracingIntervals())
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": Successfully initiated with %d tasks and %d threads BUT disabled by EXTRAE_CONTROL_GLOPS\n\n", world_size, Backend_getNumberOfThreads());
		Extrae_shutdown_Wrapper();
	}

	/* Enable dynamic memory instrumentation if requested */
	if (requestedDynamicMemoryInstrumentation)
		Extrae_set_trace_malloc (TRUE);

	/* Enable dynamic memory instrumentation if requested */
	if (requestedIOInstrumentation)
		Extrae_set_trace_io (TRUE);

	/* Enable system call instrumentation if requested */
	if (requestedSysCallInstrumentation)
	  Extrae_set_trace_syscall (TRUE);

	/* Enable sampling capabilities */
	Extrae_setSamplingEnabled (TRUE);

#if defined(ENABLE_PEBS_SAMPLING)
	Extrae_IntelPEBS_enable(TRUE);
#endif

	/* We leave... so, we're no longer in instrumentatin from this point */
	for (u = 0; u < get_maximum_NumOfThreads(); u++)
		Backend_setInInstrumentation (u, FALSE);

	EXTRAE_SET_INITIALIZED(TRUE);

	/* Mark the finalization of init as if we're not in instrumentation */
	Backend_setInInstrumentation (THREADID, FALSE);

	return TRUE;
}


/******************************************************************************
 ***  Backend_Finalize_close_mpits
 ******************************************************************************/

void Backend_Finalize_close_files(void)
{
	unsigned thread;

	for (thread = 0; thread < get_maximum_NumOfThreads(); thread++)
		Backend_Finalize_close_mpits (getpid(), thread, FALSE);
}

static void Backend_Finalize_close_mpits (pid_t pid, int thread, int append)
{
	int r;
	char trace[TMP_DIR];
	char tmp_name[TMP_DIR];
	unsigned initialTASKID;
	char hostname[1024];

	if (Buffer_IsClosed(TRACING_BUFFER(thread)))
		return;

	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");

	/* Note! If the instrumentation was initialized by Extrae_init, the TASKID
	   as that moment was 0, independently if MPI has run */
	initialTASKID = Extrae_get_initial_TASKID();

	Buffer_Close(TRACING_BUFFER(thread));

	/* Rename MPIT file (PID from origin is getpid(), PID from destination
	   maybe different if append == TRUE */
	if (append)
	{
		/* When appending, we add information into an .mpit rather than into a .ttmp */
		/* In this case, we have to honor the original PID (given by param) */
		FileName_PTT(tmp_name, Get_TemporalDir(initialTASKID), appl_name, hostname,
		  getpid(), initialTASKID, thread, EXT_TMP_MPIT);
		FileName_PTT(trace, Get_FinalDir(TASKID), appl_name, hostname, pid,
		  TASKID, thread, EXT_MPIT);
	}
	else
	{
		/* When in default (not appending) we create a .ttmp and then rename into .mpit */
		FileName_PTT(tmp_name, Get_TemporalDir(initialTASKID), appl_name, hostname,
		  getpid(), initialTASKID, thread, EXT_TMP_MPIT);
		FileName_PTT(trace, Get_FinalDir(TASKID), appl_name, hostname, getpid(),
		  TASKID, thread, EXT_MPIT);
	}
	if (!append)
		r = rename_or_copy (tmp_name, trace); 
	else
		r = append_from_to_file (tmp_name, trace);

	if (r == 0)
		fprintf (stdout,
		  PACKAGE_NAME": Intermediate raw trace file created : %s\n", trace);
	else
		fprintf (stdout,
		  PACKAGE_NAME": Intermediate raw trace was NOT created : %s\n", trace);

	/* Rename SAMPLE file, if it exists */
#if defined(SAMPLING_SUPPORT)
	FileName_PTT(tmp_name, Get_TemporalDir(initialTASKID), appl_name, hostname,
	  pid, initialTASKID, thread, EXT_TMP_SAMPLE);
	if (Buffer_GetFillCount(SAMPLING_BUFFER(thread)) > 0) 
	{
		Buffer_Flush(SAMPLING_BUFFER(thread));
		Buffer_Close(SAMPLING_BUFFER(thread));

		FileName_PTT(trace, Get_FinalDir(TASKID), appl_name, hostname, pid,
		  TASKID, thread, EXT_SAMPLE);
		r = rename_or_copy (tmp_name, trace);

		if (r == 0)
			fprintf (stdout,
			  PACKAGE_NAME": Intermediate raw sample file created : %s\n", trace);
		else
			fprintf (stdout,
			  PACKAGE_NAME": Intermediate raw sample was NOT created : %s\n", trace);
	}
	else
	{
		/* Remove file if empty! */
		unlink (tmp_name);
	}
#endif
    /* Copy or move SYM file, if it exists */
    FileName_PTT(tmp_name, Get_TemporalDir(initialTASKID), appl_name, hostname,
	  pid, initialTASKID, thread, EXT_SYM);
    if (file_exists(tmp_name)){
        FileName_PTT(trace, Get_FinalDir(initialTASKID), appl_name, hostname,
		  pid, initialTASKID, thread, EXT_SYM);
        r = rename_or_copy(tmp_name, trace);

		if (r == 0)
	    	fprintf (stdout,
	    	  PACKAGE_NAME": Intermediate raw sym file created : %s\n", trace);
		else
  	  		fprintf (stdout,
   	 		  PACKAGE_NAME": Intermediate raw sym was NOT created : %s\n", trace);
    }
}

/**
 * Force the given thread to flush 
 */
void Flush_Thread(int thread_id)
{
  Extrae_Flush_Wrapper( TRACING_BUFFER(thread_id) );
}

static int __Extrae_Flush_Wrapper_getCounters = TRUE;

int Extrae_Flush_Wrapper_getCounters(void)
{ return __Extrae_Flush_Wrapper_getCounters; }

void Extrae_Flush_Wrapper_setCounters (int c)
{ __Extrae_Flush_Wrapper_getCounters = c; }

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
		HARDWARE_COUNTERS_READ (THREADID, FlushEv_Begin, Extrae_Flush_Wrapper_getCounters());

		Buffer_Flush (buffer);

		FlushEv_End.time = TIME;
		FlushEv_End.event = FLUSH_EV;
		FlushEv_End.value = EVT_END;
		HARDWARE_COUNTERS_READ (THREADID, FlushEv_End, Extrae_Flush_Wrapper_getCounters());

		BUFFER_INSERT (THREADID, buffer, FlushEv_Begin);
#if !defined(IS_BG_MACHINE)
		Extrae_AnnotateTopology (TRUE, FlushEv_Begin.time);
#endif
		BUFFER_INSERT (THREADID, buffer, FlushEv_End);
#if !defined(IS_BG_MACHINE)
		Extrae_AnnotateTopology (TRUE, FlushEv_End.time);
#endif

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
				Backend_Finalize_close_mpits (getpid(), THREADID, FALSE);
				mpitrace_on = FALSE;
			}
		}
	}
	return 1;
}

void Backend_Finalize (void)
{
	unsigned thread;

#if defined(ENABLE_PEBS_SAMPLING)
	Extrae_IntelPEBS_disable();
#endif

#if defined(CUDA_SUPPORT)
	Extrae_CUDA_fini ();
#endif

#if defined(OPENCL_SUPPORT)
	Extrae_OpenCL_fini ();
#endif

#if defined(OMP_SUPPORT) && defined(OMPT_INSTRUMENTATION)
        ompt_finalize ();
#endif

	if (!Extrae_getAppendingEventsToGivenPID(NULL))
	{
#if defined(HAVE_ONLINE)
		/* Stop the analysis threads and flush the online buffers */
		if (Online_isEnabled())
		{
			Online_Stop();
		}
#endif /* HAVE_ONLINE */

		/* Stop collecting information from dynamic memory instrumentation */
		Extrae_set_trace_io (FALSE);

		/* Stop collecting information from dynamic memory instrumentation */
		Extrae_set_trace_malloc (FALSE);

		/* Stop sampling right now */
		Extrae_setSamplingEnabled (FALSE);
		unsetTimeSampling ();

		if (THREADID == 0) 
		{
			Extrae_getrusage_Wrapper ();
			Extrae_memusage_Wrapper ();
		}

#if !defined(IS_BG_MACHINE)
		Extrae_AnnotateTopology (TRUE, TIME);
#endif

		/* Write files back to disk , 1st part will include flushing events*/
		for (thread = 0; thread < get_maximum_NumOfThreads(); thread++) 
		{
			/* Prevent writing performance counters from another thread */
			if (thread != THREADID)
				Extrae_Flush_Wrapper_setCounters (FALSE);

			if (TRACING_BUFFER(thread) != NULL)
				Buffer_ExecuteFlushCallback (TRACING_BUFFER(thread));

			Extrae_Flush_Wrapper_setCounters (TRUE);
		}

		/* Final write files to disk, include renaming of the filenames,
		   do not need counters here */
		Extrae_Flush_Wrapper_setCounters (FALSE);
		for (thread = 0; thread < get_maximum_NumOfThreads(); thread++)
		{
			if (TRACING_BUFFER(thread) != NULL)
			{
				TRACE_EVENT (TIME, APPL_EV, EVT_END);
				Buffer_ExecuteFlushCallback (TRACING_BUFFER(thread));
				Backend_Finalize_close_mpits (getpid(), thread, FALSE);
			}
		}
	
		/* Free allocated memory */
		{
			if (TASKID == 0)
				fprintf (stdout, PACKAGE_NAME": Deallocating memory.\n");

			for (thread = 0; thread < get_maximum_NumOfThreads(); thread++)
			{
#if defined(PTHREAD_SUPPORT)
				pThreads[thread] = (pthread_t)0;
#endif
				Buffer_Free (TRACING_BUFFER(thread));
				TRACING_BUFFER(thread) = NULL;
#if defined(SAMPLING_SUPPORT)
				Buffer_Free (SAMPLING_BUFFER(thread));
				SAMPLING_BUFFER(thread) = NULL;
#endif
			}
			xfree(LastCPUEmissionTime);
			xfree(LastCPUEvent);
			xfree(TracingBuffer);
#if defined(SAMPLING_SUPPORT)
			xfree(SamplingBuffer);
#endif
			xfree (TracingBitmap);
			Extrae_allocate_thread_CleanUp();
			TimeSync_CleanUp();
			Trace_Mode_CleanUp();
			Clock_CleanUp();
			InstrumentUFroutines_GCC_CleanUp();
			InstrumentUFroutines_XL_CleanUp();
#if USE_HARDWARE_COUNTERS
			HWC_CleanUp (get_maximum_NumOfThreads());
#endif
		}

		if (TASKID == 0 && Extrae_isProcessMaster())
			fprintf (stdout, PACKAGE_NAME": Application has ended. Tracing has been terminated.\n");

		mpitrace_on = FALSE; /* Turn off tracing now */

		/* Mark as already finalized, thus avoiding further deinits */
		Extrae_set_is_initialized (EXTRAE_NOT_INITIALIZED);

#if defined(EMBED_MERGE_IN_TRACE)
		/* Launch the merger */
		if (MergeAfterTracing)
		{
			int ptask = 1;
			char tmp[1024];

			if (TASKID == 0)
				fprintf (stdout, PACKAGE_NAME ": Proceeding with the merge of the intermediate tracefiles.\n");

#if defined(MPI_SUPPORT)
			/* Synchronize all tasks at this point so none overtakes the master and
			   gets and invalid/blank trace file list (.mpits file) */
			if (TASKID == 0)
				fprintf (stdout, PACKAGE_NAME ": Waiting for all tasks to reach the checkpoint.\n");

			MPI_Barrier (MPI_COMM_WORLD);
#endif

			merger_pre (Extrae_get_num_tasks());

#if defined(MPI_SUPPORT)
			sprintf (tmp, "%s", Extrae_core_get_mpits_file_name());
#else
			sprintf (tmp, "%s/%s%s", final_dir, appl_name, EXT_MPITS);
#endif

			Read_MPITS_file (tmp, &ptask, FileOpen_Default, TASKID);

			if (TASKID == 0)
				fprintf (stdout, PACKAGE_NAME ": Executing the merge process (using %s).\n", tmp);

			merger_post (Extrae_get_num_tasks(), TASKID);
		}
#endif /* EMBED_MERGE_IN_TRACE */
	}
	else
	{
		int pid;
		Extrae_getAppendingEventsToGivenPID (&pid);
		Buffer_Flush(TRACING_BUFFER(THREADID));
		for (thread = 0; thread < get_maximum_NumOfThreads(); thread++) 
			Backend_Finalize_close_mpits (pid, thread, TRUE);
		remove_temporal_files ();
	}
}

//static int Extrae_inInstrumentation = FALSE;
static int *Extrae_inInstrumentation = NULL;
static int *Extrae_inSampling = NULL;

int Backend_inInstrumentation (unsigned thread)
{
	if ((Extrae_inInstrumentation != NULL) && (Extrae_inSampling != NULL))
		return (Extrae_inInstrumentation[thread] || Extrae_inSampling[thread]);
	else
		return FALSE;
}

void Backend_setInSampling (unsigned thread, int insampling)      
{                                                                               
	  if (Extrae_inSampling != NULL)                                         
			    Extrae_inSampling[thread] = insampling;                       
}                                                                               

void Backend_setInInstrumentation (unsigned thread, int ininstrumentation)
{
	if (Extrae_inInstrumentation != NULL)
		Extrae_inInstrumentation[thread] = ininstrumentation;
}

void Backend_ChangeNumberOfThreads_InInstrumentation (unsigned nthreads)
{
	Extrae_inInstrumentation = (int*) realloc (Extrae_inInstrumentation, sizeof(int)*nthreads);
	if (Extrae_inInstrumentation == NULL)
	{
		fprintf (stderr, PACKAGE_NAME
		  ": Failed to allocate memory for inInstrumentation structure\n");
		exit (-1);
	}
	Extrae_inSampling = (int*) realloc (Extrae_inSampling, sizeof(int)*nthreads);
	if (Extrae_inSampling == NULL)
	{
		fprintf (stderr, PACKAGE_NAME
		  ": Failed to allocate memory for inSampling structure\n");
		exit (-1);
	}
}

void Backend_Enter_Instrumentation (int Nevents)
{
	unsigned thread = THREADID;
	UINT64 current_time;

	if (!mpitrace_on)
		return;

	Backend_setInInstrumentation (thread, TRUE);

	/* Check if we have to fill the sampling buffer */
#if defined(SAMPLING_SUPPORT)
	if (Extrae_get_DumpBuffersAtInstrumentation())
		if (Buffer_IsFull (SAMPLING_BUFFER(THREADID)))
		{
			event_t FlushEv_Begin, FlushEv_End;

			/* Disable sampling first, and then reestablish whether it was set */
			int prev = Extrae_isSamplingEnabled();
			Extrae_setSamplingEnabled (FALSE);

			/* Get time now, to mark in the instrumentation buffer the begin of the flush */
			FlushEv_Begin.time = TIME;
			FlushEv_Begin.event = FLUSH_EV;
			FlushEv_Begin.value = EVT_BEGIN;
			HARDWARE_COUNTERS_READ (THREADID, FlushEv_Begin,
			  Extrae_Flush_Wrapper_getCounters());

			/* Actually flush buffer */
			Buffer_Flush(SAMPLING_BUFFER(THREADID));

			/* Get time now, to mark in the instrumentation buffer the end of the flush */
			FlushEv_End.time = TIME;
			FlushEv_End.event = FLUSH_EV;
			FlushEv_End.value = EVT_END;
			HARDWARE_COUNTERS_READ (THREADID, FlushEv_End,
			  Extrae_Flush_Wrapper_getCounters());

			/* Add events into instrumentation buffer */
			BUFFER_INSERT (THREADID, TRACING_BUFFER(THREADID), FlushEv_Begin);
			BUFFER_INSERT (THREADID, TRACING_BUFFER(THREADID), FlushEv_End);

			/* Reestablish sampling */
			Extrae_setSamplingEnabled (prev);
		}
#endif

	/* Check whether we will fill the buffer soon (or now) */
	if (Buffer_RemainingEvents(TracingBuffer[thread]) <= Nevents)
		Buffer_ExecuteFlushCallback (TracingBuffer[thread]);

	/* Record the time when this is happening
	   we need this for subsequent calls to TIME, as this is being cached in
	   clock routines */
	current_time = TIME;

	if (Trace_Mode_FirstMode(thread))
		Trace_Mode_Change (thread, current_time);

#if defined(PAPI_COUNTERS) || defined(PMAPI_COUNTERS)
	/* Must change counters? check only at detail tracing, at bursty
     tracing it is leveraged to the mpi macros at BURSTS_MODE_TRACE_MPIEVENT */
	if (CURRENT_TRACE_MODE(thread) == TRACE_MODE_DETAIL)
		HARDWARE_COUNTERS_CHANGE(current_time, thread);
#else
	/* Otherwise, get rid of a warning about "set but unused" */
	UNREFERENCED_PARAMETER(current_time);
#endif
}

void Backend_Leave_Instrumentation (void)
{
	unsigned thread = THREADID;

	if (!mpitrace_on)
		return;

	if (PENDING_TRACE_CPU_EVENT(thread, LAST_READ_TIME))
	{
		Extrae_AnnotateCPU(LAST_READ_TIME);
	}

	/* Change trace mode? (issue from API) */
	if (PENDING_TRACE_MODE_CHANGE(thread) && MPI_Deepness[thread] == 0)
		Trace_Mode_Change(thread, LAST_READ_TIME);

	Backend_setInInstrumentation (thread, FALSE);
}

void Extrae_AddTypeValuesEntryToGlobalSYM (char code_type, int type, char *description,
	char code_values, unsigned nvalues, unsigned long long *values,
	char **description_values)
{
	#define LINE_SIZE 2048
	char line[LINE_SIZE];
	char trace_sym[TMP_DIR];
	int fd;

	ASSERT(strlen(description)<LINE_SIZE, "Description for type is too large");

	FileName_P(trace_sym, final_dir, appl_name, EXT_SYM);

	if ((fd = open(trace_sym, O_WRONLY | O_APPEND | O_CREAT, 0644)) >= 0)
	{
		unsigned j;

		snprintf (line, sizeof(line), "%c %d \"%s\"", code_type, type,
			description);

		/* Avoid cr in line */
		for (j = 0; j < strlen(line); j++)
			if (line[j] == '\n')
				line[j] = ' ';

		if (write (fd, line, strlen(line)) < 0)
			fprintf (stderr, PACKAGE_NAME": Error writing definition into global symbolic file");

		if (write (fd, "\n", strlen("\n")) < 0)
			fprintf (stderr, PACKAGE_NAME": Error writing definition into global symbolic file");
		if (nvalues > 0)
		{
			unsigned i;
			for (i = 0; i < nvalues; i++)
			{
				ASSERT(strlen(description_values[i])<LINE_SIZE, "Description for value is too large");

				snprintf (line, sizeof(line), "%c %llu \"%s\"", code_values,
					values[i], description_values[i]);

				/* Avoid cr in line */
				for (j = 0; j < strlen(line); j++)
					if (line[j] == '\n')
						line[j] = ' ';

				if (write (fd, line, strlen(line)) < 0)
					fprintf (stderr, PACKAGE_NAME": Error writing definition into global symbolic file");
				if (write (fd, "\n", strlen("\n")) < 0)
					fprintf (stderr, PACKAGE_NAME": Error writing definition into global symbolic file");
			}
		}
		close (fd);
	}
	#undef LINE_SIZE
}

void Extrae_AddTypeValuesEntryToLocalSYM (char code_type, int type, char *description,
	char code_values, unsigned nvalues, unsigned long long *values,
	char **description_values)
{
	#define LINE_SIZE 2048
	char line[LINE_SIZE];
	char trace_sym[TMP_DIR];
	int fd;
	char hostname[1024];

	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");

	ASSERT(strlen(description)<LINE_SIZE, "Description for type is too large");

	FileName_PTT(trace_sym, Get_TemporalDir(TASKID), appl_name, hostname,
	  getpid(), TASKID, THREADID, EXT_SYM);

	if ((fd = open(trace_sym, O_WRONLY | O_APPEND | O_CREAT, 0644)) >= 0)
	{
		unsigned j;

		snprintf (line, sizeof(line), "%c %d \"%s\"", code_type, type,
			description);

		/* Avoid cr in line */
		for (j = 0; j < strlen(line); j++)
			if (line[j] == '\n')
					line[j] = ' ';

		if (write (fd, line, strlen(line)) < 0)
			fprintf (stderr, PACKAGE_NAME": Error writing definition into local symbolic file");
		if (write (fd, "\n", strlen("\n")) < 0)
			fprintf (stderr, PACKAGE_NAME": Error writing definition into local symbolic file");

		if (nvalues > 0)
		{
			unsigned i;
			for (i = 0; i < nvalues; i++)
			{
				ASSERT(strlen(description_values[i])<LINE_SIZE, "Description for value is too large");

				snprintf (line, sizeof(line), "%c %llu \"%s\"", code_values,
					values[i], description_values[i]);

				/* Avoid cr in line */
				for (j = 0; j < strlen(line); j++)
					if (line[j] == '\n')
						line[j] = ' ';

				if (write (fd, line, strlen(line)) < 0)
					fprintf (stderr, PACKAGE_NAME": Error writing definition into local symbolic file");
				if (write (fd, "\n", strlen("\n")) < 0)
					fprintf (stderr, PACKAGE_NAME": Error writing definition into local symbolic file");
			}
		}
		close (fd);
	}
	#undef LINE_SIZE
}

void Extrae_AddFunctionDefinitionEntryToLocalSYM (char code_type, void *address,
	char *functionname, char *modulename, unsigned fileline)
{
	#define LINE_SIZE 2048
	char line[LINE_SIZE];
	char trace_sym[TMP_DIR];
	int fd;
	char hostname[1024];

	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");

	ASSERT(strlen(functionname)+strlen(modulename)<LINE_SIZE, "Function name and module name are too large!");

	FileName_PTT(trace_sym, Get_TemporalDir(TASKID), appl_name, hostname,
	  getpid(), TASKID, THREADID, EXT_SYM);

	if ((fd = open(trace_sym, O_WRONLY | O_APPEND | O_CREAT, 0644)) >= 0)
	{
		unsigned j;

		/* Example of format: U 0x100016d4 fA mpi_test.c 0 */
		snprintf (line, sizeof(line), "%c %p \"%s\" \"%s\" %u", code_type,
			address, functionname, modulename, fileline);

		/* Avoid cr in line */
		for (j = 0; j < strlen(line); j++)
			if (line[j] == '\n')
				line[j] = ' ';

		if (write (fd, line, strlen(line)) < 0)
			fprintf (stderr, PACKAGE_NAME": Error writing function definition into local symbolic file");
		if (write (fd, "\n", strlen("\n")) < 0)
			fprintf (stderr, PACKAGE_NAME": Error writing function definition into local symbolic file");

		close (fd);
	}
	#undef LINE_SIZE
}

static void Extrae_getExecutableInfo (void)
{
#if defined(OS_LINUX) || defined(OS_ANDROID) || defined(HAVE_PROC_MAPS)

	FILE *mapsfile = fopen("/proc/self/maps", "r");

	if (mapsfile == NULL)
		return;

	#define LINE_SIZE 2048
	char line[LINE_SIZE];
	while (!feof(mapsfile))
		if (fgets (line, LINE_SIZE, mapsfile) != NULL)
		{
			int match;
			unsigned long start, end, offset;
			char module[LINE_SIZE];
			char perms[5];
			module[0] = 0;
		
			match = sscanf (line, "%lx-%lx %s %lx %*s %*u %[^\n]", &start, &end, perms,
			  &offset, module);

			if (match == 5)
				if (strcmp(perms, "r-xp") == 0 || strcmp(perms, "rwxp") == 0)
					if (strlen(module) > 0 && module[0] != '[') // avoid [vdso], [syscall] and others
					{
						char nline[LINE_SIZE];
						sprintf (nline, "%lx-%lx %lx %s", start, end, offset, module);
						Extrae_AddTypeValuesEntryToLocalSYM ('B', 0, nline, (char) 0, 0, NULL, NULL);
					}
		}
	#undef LINE_SIZE

	fclose(mapsfile);
#else
    if (TASKID == 0)
    {
        static Extrae_getExecutableInfo_first_time_call = 0;
        if (!Extrae_getExecutableInfo_first_time_call)
        {
            fprintf (stderr, PACKAGE_NAME": Warning! File /proc/self/maps doesn't exist. Address translation may be limited.\n");
        }
    }
#endif
}

/***
 * Backend_updateTaskID
 * what shall be done when changing the task ID, for instancen when executing 
 * first Extrae_init and then MPI_Init
 */
void Backend_updateTaskID (void)
{
	char file1[TMP_DIR], file2[TMP_DIR];
	unsigned thread;
	char hostname[1024];

	if (gethostname (hostname, sizeof(hostname)) != 0)
		sprintf (hostname, "localhost");

	if (Extrae_get_initial_TASKID() == TASKID)
		return;

	/* Rename SYM file, if it exists, per thread */
	for (thread = 0; thread < get_maximum_NumOfThreads(); thread++)
	{
		int r;

		FileName_PTT(file1, Get_TemporalDir(Extrae_get_initial_TASKID()),
		  appl_name, hostname, getpid(), Extrae_get_initial_TASKID(), thread,
		  EXT_SYM);
		if (file_exists (file1))
		{
			FileName_PTT(file2, Get_TemporalDir(TASKID), appl_name, hostname, 
			  getpid(), TASKID, thread, EXT_SYM);

			/* Remove file if it already exists, and then copy the "new" version */
			if (file_exists (file2))
				if (!unlink (file2) == 0)
					fprintf (stderr, PACKAGE_NAME": Cannot unlink symbolic file: %s, symbols will be corrupted!\n", file2);

			r = rename_or_copy (file1, file2); 
			if (r < 0)
				fprintf (stderr, PACKAGE_NAME": Error copying symbolicfile %s into %s!\n", file1, file2);
		}
	}
	/* NOTE: we skip renaming the MPIT and the SAMPLE files because there are
	   open handlers referring to this, and renaming the file would require
	   closing the handler and reopening it again.
	   The ACTUAL renaming will happen at final flush time!
	*/
}

unsigned long long getApplBeginTime()
{
  return ApplBegin_Time;
}

#if defined(STANDALONE)
void Extrae_core_set_current_threads(int current_threads)
{
  current_NumOfThreads = current_threads;
}

void Extrae_core_set_maximum_threads(int maximum_threads)
{
  maximum_NumOfThreads = maximum_threads;
}
#endif

