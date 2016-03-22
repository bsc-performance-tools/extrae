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

#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#include "clock.h"
#include "threadid.h"
#include "record.h"
#include "trace_macros.h"
#include "events.h"
#include "common.h"
#include "buffers.h"
#include "calltrace.h" 

#include "extrae_types.h"

#define EVT_NUM 500000

#define u_llong unsigned long long
#define LONG_PID 10
#define MASTER_ID 0

#define MAX_FUNCTION_NAME 450
#define MAX_FUNCTIONS 300

extern unsigned long long last_mpi_exit_time;
extern unsigned long long last_mpi_begin_time;
extern unsigned long long CPU_Burst_Threshold;
extern unsigned long long initTracingTime;

/*******************************************************************************
 *     
 ******************************************************************************/

#define trace_error(message) \
{ \
    write(2,message,strlen(message)); \
}

/* Es defineix el numero de caracters '_' que afegeix el compilador de fortran */
#include "defines.h"

extern unsigned int buffer_size;
extern unsigned file_size;

#include "taskid.h"

/************ Variable global per saber si cal tracejar **************/
// Serveix per deixar de tracejar un troc, de l'aplicacio
extern int tracejant;

// Serveix per tracejar una aplicacio sense contar res de MPI
extern int tracejant_mpi;

// Serveix per tracejar una aplicacio sense contar res de OpenMP
extern int tracejant_omp;

void Extrae_set_pthread_tracing (int b);
int Extrae_get_pthread_tracing (void);

void Extrae_set_pthread_hwc_tracing (int b);
int Extrae_get_pthread_hwc_tracing (void);

// Serveix per tracejar una subconjunt de tasks
extern int *TracingBitmap;

/****** Variable global per saber si cal tracejar l'aplicacio ********/
// Serveix per fer com si no hi hagues MPITRACE durant TOTA l'execucio
extern int mpitrace_on;

int EXTRAE_ON (void);

int  EXTRAE_INITIALIZED (void);
void EXTRAE_SET_INITIALIZED (int);

/****** Variable global per coneixer el nom del l'aplicacio *******/
// Serveix per poder donar als fitxers generats el nom del programa
#define TMP_DIR 1024
extern char PROGRAM_NAME[256];
extern char tmp_dir[TMP_DIR];
extern char final_dir[TMP_DIR];
extern char appl_name[512];
extern char trace_home[TMP_DIR];

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

char *Get_ApplName (void);
char *Get_FinalDir (int task);
char *Get_TemporalDir (int task);

unsigned Backend_getNumberOfThreads (void);
unsigned Backend_getMaximumOfThreads (void);

void Backend_Finalize (void);
void Backend_Finalize_close_files (void);

void Extrae_setAppendingEventsToGivenPID (int pid);
int Extrae_getAppendingEventsToGivenPID (int *pid);

#ifdef __cplusplus
}
#endif /* __cplusplus */

// Know if the run is controlled by a creation of a file 
int Extrae_getCheckControlFile (void);
char *Extrae_getCheckControlFileName (void);
int Extrae_getCheckForGlobalOpsTracingIntervals (void);
void Extrae_setCheckControlFile (int b);
void Extrae_setCheckControlFileName (const char *f);
void Extrae_setCheckForGlobalOpsTracingIntervals (int b);

/* Are HWC enabled? */
extern int Trace_HWC_Enabled;  
#define TRACING_HWC (Trace_HWC_Enabled)

/* Must we collect HWC on the MPI calls */
extern int tracejant_hwc_mpi;
#define TRACING_HWC_MPI (tracejant_hwc_mpi)

/* Must we collect HWC on the OpenMP runtime calls */
extern int tracejant_hwc_omp;
#define TRACING_HWC_OMP (tracejant_hwc_omp)

/* Must we collect HWC on the UF calls */
extern int tracejant_hwc_uf;
#define TRACING_HWC_UF (tracejant_hwc_uf)

/* Must we collect information about the network NIC */
extern int tracejant_network_hwc;
#define TRACING_NETWORK_HWC (tracejant_network_hwc)

/* Obtain information about RUSAGE ? */
extern int tracejant_rusage;
#define TRACING_RUSAGE (tracejant_rusage)

/* Obtain information about MALLOC ? */
extern int tracejant_memusage;
#define TRACING_MEMUSAGE (tracejant_memusage)

extern unsigned long long MinimumTracingTime;
extern int hasMinimumTracingTime;

extern unsigned long long WantedCheckControlPeriod;

extrae_init_type_t Extrae_is_initialized_Wrapper (void);
void Extrae_set_is_initialized (extrae_init_type_t type);

int Extrae_get_ApplicationIsMPI (void);
int Extrae_get_ApplicationIsSHMEM (void);
void Extrae_set_ApplicationIsMPI (int isMPI);
void Extrae_set_ApplicationIsSHMEM (int isSHMEM);

void Extrae_AnnotateCPU (UINT64 timestamp);

int Extrae_Allocate_Task_Bitmap (int size);

void Extrae_AddTypeValuesEntryToGlobalSYM (char code_type, int type, char *description,
	char code_values, unsigned nvalues, unsigned long long *values,
	char **description_values);
void Extrae_AddTypeValuesEntryToLocalSYM (char code_type, int type, char *description,
	char code_values, unsigned nvalues, unsigned long long *values,
	char **description_values);
void Extrae_AddFunctionDefinitionEntryToLocalSYM (char code_type, void *address,
	char *functionname, char *modulename, unsigned fileline);

void setRequestedDynamicMemoryInstrumentation (int b);
void setRequestedIOInstrumentation (int b);

int Backend_preInitialize (int rank, int world_size, const char *config_file, int forked);
int Backend_postInitialize (int rank, int world_size, unsigned init_event, unsigned long long InitTime, unsigned long long EndTime, char **node_list);
void Backend_updateTaskID (void);

int Backend_ChangeNumberOfThreads (unsigned numberofthreads);
void Backend_setNumTentativeThreads (int numofthreads);

#if defined(PTHREAD_SUPPORT)
void Backend_SetpThreadIdentifier (int ID);
int Backend_ispThreadFinished (int threadid);
pthread_t Backend_GetpThreadID (int threadid);
int Backend_GetpThreadIdentifier (void);
void Backend_SetpThreadID (pthread_t *t, int threadid);
void Backend_NotifyNewPthread (void);
void Backend_CreatepThreadIdentifier (void);
void Backend_Flush_pThread (pthread_t t);
#endif

iotimer_t Backend_Get_Last_Enter_Time (void);
iotimer_t Backend_Get_Last_Leave_Time (void);
void Backend_Enter_Instrumentation (int Nevents);
void Backend_Leave_Instrumentation (void);
int Backend_inInstrumentation (unsigned thread);
void Backend_setInInstrumentation (unsigned thread, int ininstrumentation);
void Backend_ChangeNumberOfThreads_InInstrumentation (unsigned nthreads);
void Backend_createExtraeDirectory (int taskid, int Temporal);
int Extrae_Get_FinalDir_BlockSize(void);
int Extrae_Get_TemporalDir_BlockSize(void);
char *Extrae_Get_FinalDirNoTask (void);
char *Extrae_Get_TemporalDirNoTask (void);

void advance_current(int);
extern int circular_buffering, circular_OVERFLOW;
extern event_t *circular_HEAD;

void Parse_Callers (int, char *, int);

int remove_temporal_files (void);

enum {
   KEEP,
   RESTART,
   SHUTDOWN
};

int GlobalOp_Changes_Trace_Status (int current_glop);
void Parse_GlobalOps_Tracing_Intervals(char * sequence);

void Flush_Thread(int thread_id);

#if defined(EMBED_MERGE_IN_TRACE)
extern int MergeAfterTracing;
#endif

unsigned long long getApplBeginTime();

#if defined(STANDALONE)

typedef enum 
{
  CORE_MODULE = 0,
  MPI_MODULE,
  OPENMP_MODULE
} module_id_t;

typedef struct
{
  module_id_t id;
  void (*init_ptr)(void);
  void (*fini_ptr)(void);
} module_t;

void Extrae_RegisterModule(module_id_t id, void *init_ptr, void *fini_ptr);
void Extrae_core_set_current_threads(int current_threads);
void Extrae_core_set_maximum_threads(int maximum_threads);

#endif /* STANDALONE */

#endif /* __WRAPPER_H__ */
