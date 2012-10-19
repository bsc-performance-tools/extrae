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

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <list>
#include <string>
#include <iostream>
#include <fstream>

#include "debug.h"

using namespace std; 

#include <BPatch_point.h>
#include "commonSnippets.h"


BPatch_Vector<BPatch_function *> getRoutines (string &routine, BPatch_image *appImage, bool warn)
{
	BPatch_Vector<BPatch_function *> found_funcs;

	if (appImage->findFunction (routine.c_str(), found_funcs) == NULL)
	{
		if (warn)
		{
			string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find function ")+routine;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
		}
	}

	return found_funcs;
}

BPatch_function * getRoutine (string &routine, BPatch_image *appImage, bool warn)
{
	BPatch_Vector<BPatch_function *> found_funcs = getRoutines (routine, appImage, warn);

	if (found_funcs.size() < 1)
	{
		if (warn)
		{
			string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find function ")+routine;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
		}
		return NULL;
	}
	if (found_funcs[0] == NULL)
	{
		if (warn)
		{
			string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find function ")+routine;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
		}
		return NULL;
	}

	return found_funcs[0];
}

void wrapRoutine (BPatch_image *appImage, BPatch_process *appProcess,
	string routine, string wrap_begin, string wrap_end, unsigned nparams)
{
	BPatch_function *function = getRoutine (routine, appImage, false);

	if (function == NULL)
	{
		string error = string(PACKAGE_NAME": getRoutine: Failed to find function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_Vector<BPatch_point *> *entry_point = function->findPoint(BPatch_entry);
	if (!entry_point || (entry_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find entry point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
	}

	BPatch_Vector<BPatch_point *> *exit_point = function->findPoint(BPatch_exit);
	if (!exit_point || (exit_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find exit point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
	}

	if (wrap_begin.length() > 0)
	{
		BPatch_function *snippet_begin = getRoutine (wrap_begin, appImage, true);
		if (snippet_begin == NULL)
		{
			string error = string (PACKAGE_NAME": getRoutine: Failed to find wrap_begin ")+wrap_begin;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
			return;
		}

		BPatch_Vector<BPatch_snippet *> args_entry;
		for (unsigned u = 0; u < nparams; u++)
			args_entry.push_back (new BPatch_paramExpr (u));

		BPatch_funcCallExpr callExpr_entry (*snippet_begin, args_entry);

		if (appProcess->insertSnippet (callExpr_entry, *entry_point) == NULL)
			cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at entry point" << endl;
	}

	if (wrap_end.length() > 0)
	{
		BPatch_function *snippet_end = getRoutine (wrap_end, appImage, true);
		if (snippet_end == NULL)
		{
			string error = string (PACKAGE_NAME": getRoutine: Failed to find wrap_end ")+wrap_begin;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
			return;
		}
		BPatch_Vector<BPatch_snippet *> args_exit;
		BPatch_funcCallExpr callExpr_exit (*snippet_end, args_exit);

		if (appProcess->insertSnippet (callExpr_exit, *exit_point) == NULL)
			cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at entry point" << endl;
	}
}

void wrapTypeRoutine (BPatch_function *function, string routine, int type,
	BPatch_image *appImage, BPatch_process *appProcess)
{
	string snippet_name = "Extrae_function_from_address";

	BPatch_Vector<BPatch_point *> *entry_point = function->findPoint(BPatch_entry);
	if (!entry_point || (entry_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find entry point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_Vector<BPatch_point *> *exit_point = function->findPoint(BPatch_exit);
	if (!exit_point || (exit_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find exit point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_function *snippet = getRoutine (snippet_name, appImage, false);
	if (snippet == NULL)
	{
		string error = string (PACKAGE_NAME": getRoutine: Failed to find wrap_end ")+snippet_name;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_Vector<BPatch_snippet *> args_entry;
	BPatch_constExpr entry_param0(type);
	BPatch_constExpr entry_param1(function->getBaseAddr());
	args_entry.push_back(&entry_param0);
	args_entry.push_back(&entry_param1);

	BPatch_Vector<BPatch_snippet *> args_exit;
	BPatch_constExpr exit_param0(type);
	BPatch_constExpr exit_param1(0);
	args_exit.push_back(&exit_param0);
	args_exit.push_back(&exit_param1);

	BPatch_funcCallExpr callExpr_entry (*snippet, args_entry);
	BPatch_funcCallExpr callExpr_exit (*snippet, args_exit);

	if (appProcess->insertSnippet (callExpr_entry, *entry_point) == NULL)
		cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at entry point" << endl;

	if (appProcess->insertSnippet (callExpr_exit, *exit_point) == NULL)
		cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at exit point" << endl;
}

