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

#include <fstream>

using std::ifstream;

#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif 
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "OnlineUtils.h"


/**
 * Adds a suffix to the given hostname to select an specific NIC. 
 * If the environment variable EXTRAE_ONLINE_NIC is set, its value is appended 
 * to the hostname (typically, "-ib0", "-myrinet1", etc.). If the variable
 * is not set, it looks automatically for the device /dev/infiniband. 
 * If found, it directly appends "-ib0" to the host (compatibility tested for MN and MT). 
 * Otherwise, it leaves the hostname as it is, which will probably make the connections
 * go through an Ethernet, so remember to define EXTRAE_ONLINE_NIC in machines where
 * there's no /dev/infiniband and an ib0 interface! 
 */
string Select_NIC(char *hostname)
{
  char  *hostname_suffix = getenv("EXTRAE_ONLINE_NIC");
  string Hostname(hostname);

  if (hostname_suffix != NULL)
  {
    string HostnameSuffix(hostname_suffix);
    Hostname += HostnameSuffix;
  }
  else 
  {
    ifstream device("/dev/infiniband");
    if (device.good())
    {
      string HostnameSuffix("-ib0");
      Hostname += HostnameSuffix;
    }
  }
  return Hostname;
}


/**
 * Check if the given directory exists and we have read/write permissions
 */
bool dir_exists(char *pathname)
{
#ifdef HAVE_SYS_STAT_H
  struct stat sb;

  if ((stat(pathname, &sb) == 0) && (S_ISDIR(sb.st_mode)) && (access(pathname, R_OK|W_OK) == 0))
  {
    return true;
  }
#else
# error "stat() is not supported. Please provide an alternate implementation for routine 'dir_exists'."
#endif
  return false;
}

