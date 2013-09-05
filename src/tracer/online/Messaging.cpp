#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>

using std::cout;
using std::cerr;
using std::endl;
using std::stringstream;

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
  }
}

bool Messaging::debugging()
{
  return DebugEnabled;
}

