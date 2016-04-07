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

#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "paraver_state.h" /* for joint states */

static int option_merge_dump = FALSE;
int get_option_merge_dump (void) { return option_merge_dump; }
void set_option_merge_dump (int b) { option_merge_dump = b; }

static int option_merge_SincronitzaTasks = FALSE;
int get_option_merge_SincronitzaTasks (void) { return option_merge_SincronitzaTasks; }
void set_option_merge_SincronitzaTasks (int b) { option_merge_SincronitzaTasks = b; }

static int option_merge_SincronitzaTasks_byNode = FALSE;
int get_option_merge_SincronitzaTasks_byNode (void) { return option_merge_SincronitzaTasks_byNode; }
void set_option_merge_SincronitzaTasks_byNode (int b) { option_merge_SincronitzaTasks_byNode = b; }

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

static char OutputTraceName[1024] = "";
char * get_merge_OutputTraceName (void) { return OutputTraceName; }
void set_merge_OutputTraceName (const char* s) { strcpy (OutputTraceName, s); }

static int option_merge_GivenTraceName = FALSE;
int get_merge_GivenTraceName (void) { return option_merge_GivenTraceName; }
void set_merge_GivenTraceName (int b) { option_merge_GivenTraceName = b; }

static char callback_file[1024] = "";
char * get_merge_CallbackFileName (void) { return callback_file; }
void set_merge_CallbackFileName (const char* s) { strcpy (callback_file, s); }

static char symbol_file[1024] = "";
char * get_merge_SymbolFileName (void) { return symbol_file; }
void set_merge_SymbolFileName (const char* s) { strcpy (symbol_file, s); }

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

static int option_dump_Time = TRUE;
int get_option_dump_Time (void) { return option_dump_Time; }
void set_option_dump_Time (int b) { option_dump_Time = b; }

static int option_dump_Address = TRUE;
int get_option_dump_Addresses (void) { return option_dump_Address; }
void set_option_dump_Addresses (int b) { option_dump_Address = b; }

#if defined(IS_BG_MACHINE)
static int option_merge_BG_XYZT = FALSE;
int get_option_merge_BG_XYZT (void) { return option_merge_BG_XYZT; }
void set_option_merge_BG_XYZT (int b) { option_merge_BG_XYZT = b; }
#endif

static int option_merge_AbsoluteCounters = FALSE;
int get_option_merge_AbsoluteCounters (void) { return option_merge_AbsoluteCounters; }
void set_option_merge_AbsoluteCounters (int b) { option_merge_AbsoluteCounters = b; }

static int option_merge_TraceOverwrite = TRUE;
int get_option_merge_TraceOverwrite (void) { return option_merge_TraceOverwrite; }
void set_option_merge_TraceOverwrite (int b) { option_merge_TraceOverwrite = b; }

static int option_merge_TranslateAddresses = TRUE;
int get_option_merge_TranslateAddresses (void) { return option_merge_TranslateAddresses; }
void set_option_merge_TranslateAddresses (int b) { option_merge_TranslateAddresses = b; }

static int option_merge_EmitLibraryEvents = FALSE;
int get_option_merge_EmitLibraryEvents (void) { return option_merge_EmitLibraryEvents; }
void set_option_merge_EmitLibraryEvents (int b) { option_merge_EmitLibraryEvents = b; }

