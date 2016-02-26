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

#include <iostream>
#include <sstream>

using std::cout;
using std::cerr;
using std::endl;
using std::stringstream;

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "Messaging.h"

Messaging::Messaging()
{
  char *env_debug = getenv("EXTRAE_ONLINE_DEBUG");
  DebugEnabled = (env_debug != NULL);

  ProcessLabel = "<ROOT>";

  I_am_FE        = true;
  I_am_BE        = false;
  I_am_master_BE = false;
}

Messaging::Messaging(int be_rank, bool is_master)
{
  char *env_debug = getenv("EXTRAE_ONLINE_DEBUG");
  DebugEnabled = (env_debug != NULL);

  stringstream ss;
  ss << be_rank;
  ProcessLabel = "<BE #" + ss.str();
  if (is_master) 
  {
    ProcessLabel += "M";
  }
  ProcessLabel += ">";

  I_am_FE        = false;
  I_am_BE        = true;
  I_am_master_BE = is_master;
}

void Messaging::error(const char *fmt, ...)
{
  char buffer[4096];

  va_list va;
  va_start(va, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, va);
  va_end(va);

  buffer[ sizeof(buffer)-1 ] = '\0';
  buffer[ sizeof(buffer)-2 ] = '.';
  buffer[ sizeof(buffer)-3 ] = '.';
  buffer[ sizeof(buffer)-4 ] = '.';

  cerr << ProcessLabel << " ERROR: " << buffer << endl;
  cerr.flush();
}

void Messaging::say(ostream &out, const char *fmt, ...)
{
  char buffer[4096];

  va_list va;
  va_start(va, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, va);
  va_end(va);

  buffer[ sizeof(buffer)-1 ] = '\0';
  buffer[ sizeof(buffer)-2 ] = '.';
  buffer[ sizeof(buffer)-3 ] = '.';
  buffer[ sizeof(buffer)-4 ] = '.';

  out << ProcessLabel << " " << buffer << endl; 
  out.flush();
}

void Messaging::say_one(ostream &out, const char *fmt, ...)
{
  char    buffer[4096];
  va_list va;

  if ((I_am_FE) || (I_am_master_BE))
  {
    va_start(va, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, va);
    va_end(va);

    buffer[ sizeof(buffer)-1 ] = '\0';
    buffer[ sizeof(buffer)-2 ] = '.';
    buffer[ sizeof(buffer)-3 ] = '.';
    buffer[ sizeof(buffer)-4 ] = '.';

    out << ProcessLabel << " " << buffer << endl;
    out.flush();
  }
}

void Messaging::debug(ostream &out, const char *fmt, ...)
{
  char    buffer[4096];
  va_list va;

  if (DebugEnabled)
  {
    va_start(va, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, va);
    va_end(va);

    buffer[ sizeof(buffer)-1 ] = '\0';
    buffer[ sizeof(buffer)-2 ] = '.';
    buffer[ sizeof(buffer)-3 ] = '.';
    buffer[ sizeof(buffer)-4 ] = '.';

    out << "[DEBUG] " << ProcessLabel << " " << buffer << endl;
    out.flush();
  }
}

void Messaging::debug_one(ostream &out, const char *fmt, ...)
{
  char    buffer[4096];
  va_list va;
 
  if ((DebugEnabled) && ((I_am_FE) || (I_am_master_BE)))
  {
    va_start(va, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, va);
    va_end(va);
  
    buffer[ sizeof(buffer)-1 ] = '\0';
    buffer[ sizeof(buffer)-2 ] = '.';
    buffer[ sizeof(buffer)-3 ] = '.';
    buffer[ sizeof(buffer)-4 ] = '.';

    out << "[DEBUG] " << ProcessLabel << " " << buffer << endl;
    out.flush();
  }
}

bool Messaging::debugging()
{
  return DebugEnabled;
}

