/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
#include "clock.h"
#include "threadid.h"
#include "record.h"
#include "trace_macros.h"
#include "events.h"
#include "common.h"
#include "buffers.h"
#include "calltrace.h" 

#define EVT_NUM 100000
#define CACHE_LINE 128

#if !defined(TRUE)
# define TRUE  (1==1)
#endif

#if !defined(FALSE)
# define FALSE (1!=1)
#endif

#define TRACE_FILE 500

#define u_llong     unsigned long long
#define LONG_PID 10
#define MASTER_ID 0

#define MAX_FUNCTION_NAME 450
#define MAX_FUNCTIONS 300

extern unsigned long long last_mpi_exit_time;
extern unsigned long long last_mpi_begin_time;
extern unsigned long long CPU_Burst_Threshold;
extern unsigned long long initTracingTime;

#if 0
#define Tracefile_Name(name,tmpdir,appl,pid,rank,vpid) \
	sprintf((name),TRACE_FILE_NAME_LINE,(tmpdir),(appl),(pid),(rank),(vpid))

#define Samplefile_Name(name,tmpdir,appl,pid,rank,vpid) \
	sprintf((name),SAMPLE_FILE_NAME_LINE,(tmpdir),(appl),(pid),(rank),(vpid))

/*
 * Tracefile PRV format is :  applicationnamePPPPPPPPPP.prv
 *
 * where  PPPPPPPPPP is the parent pid (10 digits)
 */

#define BASE_TRACE_PRVNAME_LINE     "%s.%.10d.prv"
#define TRACEFILE_PRVNAME_LINE   "%s/%s.%.10d.prv"

#define Tracefile_PrvName(name,tmpdir,appl,pid) \
	sprintf((name),TRACEFILE_PRVNAME_LINE,(tmpdir),(appl),(pid))

/* 
 * Temporal file format is :  application_namePPPPPPPPPPRRRRRRVVVVVV.tmp
 *
 * where VVV is the virtual processor identifier 
 *       PPPPPPPPPP  pid  (10 digits) thread pid which creates 
 *                   the temporal files
 */

#define TEMPORAL_TRACE_NAME_LINE   "%s/%s.%.10d%.6d%.6u.ttmp"
#define TEMPORAL_SAMPLE_NAME_LINE  "%s/%s.%.10d%.6d%.6u.stmp"

#define Temporal_Trace_Name(name,tmpdir,appl,pid,rank,vpid) \
	   sprintf((name),TEMPORAL_TRACE_NAME_LINE,(tmpdir),(appl),(pid),(rank),(vpid))

#define Temporal_Sample_Name(name,tmpdir,appl,pid,rank,vpid) \
	   sprintf((name),TEMPORAL_SAMPLE_NAME_LINE,(tmpdir),(appl),(pid),(rank),(vpid))

#define CALLBACK_FILE        "%s/%s.%.10d.cbk"

#define Callback_Name(name,tmpdir,appl,pid) \
	   sprintf((name),CALLBACK_FILE,(tmpdir),(appl),(pid))

#define SYMBOL_FILE        "%s/%s.%.10d.sym"

#define Symbol_Name(name,tmpdir,appl,pid) \
	   sprintf((name),SYMBOL_FILE,(tmpdir),(appl),(pid))
#endif

/*******************************************************************************
 *     
 ******************************************************************************/

#define trace_error(message) \
{ \
    write(2,message,strlen(message)); \
}

/* Es defineix el numero de caracters '_' que afegeix el compilador de fortran */
#include "defines.h"

//extern event_t **buffers;
extern int *fd;
extern unsigned int buffer_size;
extern unsigned int hw_counters, event0, event1;

extern unsigned int mptrace_suspend_tracing;
extern unsigned int mptrace_tracing_is_suspended;
extern unsigned int mptrace_IsMPI;

#include "taskid.h"
//#define TASKID TaskID
//extern unsigned int TaskID;
extern int NumOfTasks;

/************ Variable global per saber si cal tracejar **************/
// Serveix per deixar de tracejar un troc, de l'aplicacio
extern int tracejant;

// Serveix per tracejar una aplicacio sense contar res de MPI
extern int tracejant_mpi;

// Serveix per tracejar una aplicacio sense contar res de OpenMP
extern int tracejant_omp;

// Serveix per tracejar una subconjunt de tasks
extern int *TracingBitmap;

/****** Variable global per saber si cal tracejar l'aplicacio ********/
// Serveix per fer com si no hi hagues MPITRACE durant TOTA l'execucio
extern int mpitrace_on;

/****** Variable global per coneixer el nom del l'aplicacio *******/
// Serveix per poder donar als fitxers generats el nom del programa
#define TMP_DIR 1024
extern char PROGRAM_NAME[256];
extern char tmp_dir[TMP_DIR];
extern char final_dir[TMP_DIR];
extern char appl_name[512];
extern char trace_home[TMP_DIR];

char *Get_FinalDir (int task);
char *Get_TemporalDir (int task);

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

int Backend_preInitialize (int rank, int world_size, char *config_file);
int Backend_postInitialize (int rank, int world_size, unsigned long long SynchroInitTime, unsigned long long SynchroEndTime, char **node_list);
unsigned Backend_getNumberOfThreads (void);
int Backend_ChangeNumberOfThreads (unsigned numberofthreads);
void Backend_SetpThreadIdentifier (int ID);
int Backend_GetpThreadIdentifier (void);
void Backend_NotifyNewPthread (void);
void Backend_setNumTentativeThreads (int numofthreads);

unsigned get_current_NumOfThreads (void);
unsigned get_maximum_NumOfThreads (void);

void advance_current(int);
extern int circular_buffering, circular_OVERFLOW;
event_t *circular_HEAD;

extern unsigned int buffer_size;
extern int file_size;

void Parse_Callers (int, char *, int);

int file_exists (char *fitxer);

void Thread_Finalization (void);

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

#endif /* __WRAPPER_H__ */
