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
#include "options.h"

#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "paraver_state.h" /* for joint states */

static int option_merge_Dump = FALSE;
int get_option_merge_Dump (void) { return option_merge_Dump; }
void set_option_merge_Dump (int b) { option_merge_Dump = b; }

static int option_merge_SincronitzaTasks = FALSE;
int get_option_merge_SincronitzaTasks (void) { return option_merge_SincronitzaTasks; }
void set_option_merge_SincronitzaTasks (int b) { option_merge_SincronitzaTasks = b; }

static int option_merge_SincronitzaTasks_byNode = TRUE;
int get_option_merge_SincronitzaTasks_byNode (void) { return option_merge_SincronitzaTasks_byNode; }
void set_option_merge_SincronitzaTasks_byNode (int b) { option_merge_SincronitzaTasks_byNode = b; }

static int option_merge_SincronitzaApps = FALSE;
int get_option_merge_SincronitzaApps (void) { return option_merge_SincronitzaApps; }
void set_option_merge_SincronitzaApps (int b) { option_merge_SincronitzaApps = b; }

static int option_merge_UseDiskForComms = FALSE;
int get_option_merge_UseDiskForComms (void) { return option_merge_UseDiskForComms; }
void set_option_merge_UseDiskForComms (int b) { option_merge_UseDiskForComms = b; }

static int option_merge_SkipSendRecvComms = FALSE;
int get_option_merge_SkipSendRecvComms (void) { return option_merge_SkipSendRecvComms; }
void set_option_merge_SkipSendRecvComms (int b) { option_merge_SkipSendRecvComms = b; }

static int option_merge_UniqueCallerID = TRUE;
int get_option_merge_UniqueCallerID (void) { return option_merge_UniqueCallerID; }
void set_option_merge_UniqueCallerID (int b) { option_merge_UniqueCallerID = b; }

static int option_merge_VerboseLevel = 0;
int get_option_merge_VerboseLevel (void) { return option_merge_VerboseLevel; }
void set_option_merge_VerboseLevel (int l) { option_merge_VerboseLevel = l; }

static char callback_file[1024] = "";
char * get_merge_CallbackFileName (void) { return callback_file; }
void set_merge_CallbackFileName (const char* s) { strcpy (callback_file, s); }

static char executable_file[1024] = ""; 
char * get_merge_ExecutableFileName (void) { return executable_file; }
void set_merge_ExecutableFileName (const char* s) { strcpy (executable_file, s); }

static int option_merge_TreeFanOut = 0;
int get_option_merge_TreeFanOut (void) { return option_merge_TreeFanOut; }
void set_option_merge_TreeFanOut (int tfo) { option_merge_TreeFanOut = tfo; }

static int option_merge_MaxMem = 512;
int get_option_merge_MaxMem (void) { return option_merge_MaxMem; }
void set_option_merge_MaxMem (int mm) { option_merge_MaxMem = mm; }

static int option_merge_ForceFormat = FALSE;
int get_option_merge_ForceFormat (void) { return option_merge_ForceFormat; }
void set_option_merge_ForceFormat (int b) { option_merge_ForceFormat = b; }

static int option_merge_NumApplications = 1;
int get_option_merge_NumApplications (void) { return option_merge_NumApplications; }
void set_option_merge_NumApplications (int n) { option_merge_NumApplications = n; }

static int option_merge_JointStates = TRUE;
int get_option_merge_JointStates (void) { return option_merge_JointStates; }
void set_option_merge_JointStates (int b) { option_merge_JointStates = b; }

static int option_merge_ParaverFormat = TRUE;
int get_option_merge_ParaverFormat (void) { return option_merge_ParaverFormat; }
void set_option_merge_ParaverFormat (int b) { option_merge_ParaverFormat = b; }

static int option_merge_SortAddresses = TRUE;
int get_option_merge_SortAddresses (void) { return option_merge_SortAddresses; }
void set_option_merge_SortAddresses (int b) { option_merge_SortAddresses = b; }

static int option_merge_NanosTaskView = FALSE;
int get_option_merge_NanosTaskView (void) { return option_merge_NanosTaskView; }
void set_option_merge_NanosTaskView (int b) {option_merge_NanosTaskView = b; }

static int option_merge_RemoveFiles = FALSE;
int get_option_merge_RemoveFiles (void) { return option_merge_RemoveFiles; }
void set_option_merge_RemoveFiles (int b) { option_merge_RemoveFiles = b; }

static int option_merge_DumpTime = TRUE;
int get_option_merge_DumpTime (void) { return option_merge_DumpTime; }
void set_option_merge_DumpTime (int b) { option_merge_DumpTime = b; }

static int option_merge_DumpSymtab = FALSE;
int get_option_merge_DumpSymtab (void) { return option_merge_DumpSymtab; }
void set_option_merge_DumpSymtab (int b) { option_merge_DumpSymtab = b; }

#if defined(IS_BG_MACHINE)
static int option_merge_BG_XYZT = FALSE;
int get_option_merge_BG_XYZT (void) { return option_merge_BG_XYZT; }
void set_option_merge_BG_XYZT (int b) { option_merge_BG_XYZT = b; }
#endif

static int option_merge_AbsoluteCounters = FALSE;
int get_option_merge_AbsoluteCounters (void) { return option_merge_AbsoluteCounters; }
void set_option_merge_AbsoluteCounters (int b) { option_merge_AbsoluteCounters = b; }

static long option_merge_StopAtPercentage = 0;
long get_option_merge_StopAtPercentage(void) { return option_merge_StopAtPercentage; }
void set_option_merge_StopAtPercentage(long b) { option_merge_StopAtPercentage = b; }

static int option_merge_TraceOverwrite = TRUE;
int get_option_merge_TraceOverwrite (void) { return option_merge_TraceOverwrite; }
void set_option_merge_TraceOverwrite (int b) { option_merge_TraceOverwrite = b; }

static int option_merge_TranslateAddresses = TRUE;
int get_option_merge_TranslateAddresses (void) { return option_merge_TranslateAddresses; }
void set_option_merge_TranslateAddresses (int b) { option_merge_TranslateAddresses = b; }

static int option_merge_EmitLibraryEvents = FALSE;
int get_option_merge_EmitLibraryEvents (void) { return option_merge_EmitLibraryEvents; }
void set_option_merge_EmitLibraryEvents (int b) { option_merge_EmitLibraryEvents = b; }

static int option_merge_TranslateDataAddresses = TRUE;
int get_option_merge_TranslateDataAddresses(void) { return option_merge_TranslateDataAddresses; }
void set_option_merge_TranslateDataAddresses(int b) { option_merge_TranslateDataAddresses = b; }

static unsigned short merge_OutputIsGzip = 0;
void set_option_merge_OutputIsGzip (unsigned short v) { merge_OutputIsGzip = v; }
unsigned short get_option_merge_OutputIsGzip (void) { return merge_OutputIsGzip; }

static char OutputTraceName[1024] = "";
static char OutputPCFName[1024]   = "";
static char OutputROWName[1024]   = "";
#if defined(IS_BG_MACHINE)
#if defined(DEAD_CODE)
static char OutputCRDName[1024]   = "";
#endif 
#endif

const char *get_merge_OutputFileName (outputFileName_t t){
    switch (t){
        case TRACE_FILENAME: return OutputTraceName;
        case PCF_FILENAME:   return OutputPCFName;
        case ROW_FILENAME:   return OutputROWName;
#if defined(IS_BG_MACHINE)
#if defined(DEAD_CODE)
        case CRD_FILENAME:   return OutputCRDName;
#endif 
#endif
        default:           return "";
    }
}

void set_merge_OutputFileName (outputFileName_t type, const char *name)
{
    switch (type) {
        case TRACE_FILENAME:
            strncpy(OutputTraceName, name, sizeof(OutputTraceName)-1);
            OutputTraceName[sizeof(OutputTraceName)-1] = '\0';
            break;
        case PCF_FILENAME:
            strncpy(OutputPCFName, name, sizeof(OutputPCFName)-1);
            OutputPCFName[sizeof(OutputPCFName)-1] = '\0';
            break;
        case ROW_FILENAME:
            strncpy(OutputROWName, name, sizeof(OutputROWName)-1);
            OutputROWName[sizeof(OutputROWName)-1] = '\0';
            break;  
#if defined(IS_BG_MACHINE)
#if defined(DEAD_CODE)
        case CRD_FILENAME:
            strncpy(OutputCRDName, name, sizeof(OutputCRDName)-1);
            OutputCRDName[sizeof(OutputCRDName)-1] = '\0';
            break;
#endif 
#endif

    }
}
