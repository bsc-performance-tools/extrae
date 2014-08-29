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

#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef OS_ANDROID
#include <pthread.h>
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

extern unsigned long long last_pacx_exit_time;
extern unsigned long long last_pacx_begin_time;
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

// Serveix per tracejar una aplicacio sense contar res de PACX
extern int tracejant_pacx;

// Serveix per tracejar una aplicacio sense contar res de OpenMP
extern int tracejant_omp;

// Serveix per tracejar una aplicacio sense contar res de pthread
extern int tracejant_pthread;

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
#endif

char *Get_FinalDir (int task);
char *Get_TemporalDir (int task);

#ifdef __cplusplus
}
#endif

// Know if the run is controlled by a creation of a file 
extern char ControlFileName[TMP_DIR];
extern int CheckForControlFile;
int remove_temporal_files(void);
extern int CheckForGlobalOpsTracingIntervals;

/* Are HWC enabled? */
extern int Trace_HWC_Enabled;  
#define TRACING_HWC (Trace_HWC_Enabled)

/* Must we collect HWC on the MPI calls */
extern int tracejant_hwc_mpi;
#define TRACING_HWC_MPI (tracejant_hwc_mpi)

/* Must we collect HWC on the PACX calls */
extern int tracejant_hwc_pacx;
#define TRACING_HWC_PACX (tracejant_hwc_pacx)

/* Must we collect HWC on the OpenMP runtime calls */
extern int tracejant_hwc_omp;
#define TRACING_HWC_OMP (tracejant_hwc_omp)

/* Must we collect HWC on the pthread runtime calls */
extern int tracejant_hwc_pthread;
#define TRACING_HWC_PTHREAD (tracejant_hwc_pthread)

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
unsigned Extrae_get_initial_TASKID (void);
void Extrae_set_initial_TASKID (unsigned u);

int Extrae_get_ApplicationIsMPI (void);
int Extrae_get_ApplicationIsPACX (void);
void Extrae_set_ApplicationIsMPI (int isMPI);
void Extrae_set_ApplicationIsPACX (int isPACX);

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

int Backend_preInitialize (int rank, int world_size, char *config_file, int forked);
int Backend_postInitialize (int rank, int world_size, unsigned init_event, unsigned long long InitTime, unsigned long long EndTime, char **node_list);
void Backend_updateTaskID (void);

unsigned Backend_getNumberOfThreads (void);
unsigned Backend_getMaximumOfThreads (void);

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
void Backend_Enter_Instrumentation (unsigned Nevents);
void Backend_Leave_Instrumentation (void);
int Backend_inInstrumentation (unsigned thread);
void Backend_setInInstrumentation (unsigned thread, int ininstrumentation);
void Backend_ChangeNumberOfThreads_InInstrumentation (unsigned nthreads);

void advance_current(int);
extern int circular_buffering, circular_OVERFLOW;
extern event_t *circular_HEAD;

void Parse_Callers (int, char *, int);

int file_exists (char *fitxer);

void Backend_Finalize (void);

enum {
   KEEP,
   RESTART,
   SHUTDOWN
};

int GlobalOp_Changes_Trace_Status (int current_glop);
void Parse_GlobalOps_Tracing_Intervals(char * sequence);

#if defined(HAVE_MRNET)
void clustering_filter (int thread, int *out_CountBursts, int **out_HWCIds, int *out_CountHWC, long long **out_Timestamp, int *len_Timestamp, long long **out_Durations, int *len_Durations, long long **out_HWCValues, int *len_HWCValues);
#endif /* HAVE_MRNET */

void Flush_Thread(int thread_id);

#if defined(EMBED_MERGE_IN_TRACE)
extern int MergeAfterTracing;
#endif

#endif /* __WRAPPER_H__ */
