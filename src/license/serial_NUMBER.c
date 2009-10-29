/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

#if 0
static char rcsid[] = "$Id$";
#endif

#if defined(linux)
# ifdef HAVE_UNISTD_H
#  include <unistd.h>
# endif
# ifdef HAVE_ERRNO_H
#  include <errno.h>
# endif
# ifdef HAVE_STRING_H
#  include <string.h>
# endif
# ifdef HAVE_SYS_WAIT_H
#  include <sys/wait.h>
# endif
#elif defined(sun)
# ifdef HAVE_UNISTD_H
#  include <unistd.h>
# endif
# ifdef HAVE_STRING_H
#  include <string.h>
# endif
# ifdef HAVE_ERRNO_H
#  include <errno.h>
# endif
# ifdef HAVE_SYS_SYSTEMINFO_H
#  include <sys/systeminfo.h>
# endif
#elif defined(hpux)
# ifdef HAVE_SYS_UTSNAME_H
#  include <sys/utsname.h>
# endif
#elif defined(__alpha)
# ifdef HAVE_STRING_H
#  include <string.h>
# endif
# ifdef HAVE_ERRNO_H
#  include <errno.h>
# endif
#elif defined(sgi)
# ifdef HAVE_SYS_SYSTEMINFO_H
#  include <sys/systeminfo.h>
# endif
#endif

#if defined(linux)

unsigned long SerialNumberFromIFCONFIG (char *Addr1)
{
  char *piece = NULL, *middel = NULL;
  unsigned long long seriall;
  unsigned long serial;

  piece = (char *) malloc (1024);
  middel = strtok (Addr1, ":");
  sprintf (piece, "0x");
  while (middel != NULL)
  {
    sprintf (piece, "%s%s", piece, middel);
    middel = strtok (NULL, ":");
  }

  sscanf (piece, "%llx", &(seriall));
  seriall = seriall & 0x00000000FFFFFFFF;
  serial = (unsigned long) seriall;

  free (piece);
  return serial;
}


static unsigned long get_hardware_serial_number_Linux ()
{
  int pipefd[2];
  unsigned long serial;
  pid_t pidChild;
  int statptr;

  if (pipe (pipefd) == -1)
  {
    fprintf (stderr, "ERROR at pipe: %s\n", strerror (errno));
    return 1;
  }

  switch (pidChild = vfork ())
  {
    case 0:                    /* child process */
      close (1);                /* close standard output */

                    /** Redirect standar output to the pipe */
      if (dup (pipefd[1]) == -1)
      {
        fprintf (stderr, "ERROR at dup: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);
        exit (-1);                 /** Error */
      }

      if (system ("/sbin/ifconfig -a | grep eth0 | awk '{print $5}'") == -1)
      {
        fprintf (stderr, "ERROR at execlp: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);

        exit (-1);                /** Error */
      }
      exit (0);
      break;

    case -1:                   /* some error forking */
      fprintf (stderr, "ERROR at vfork: %s\n", strerror (errno));
      close (pipefd[0]);
      close (pipefd[1]);
      return 1;

    default:                   /* parent process, everything ok */
      close (pipefd[1]);                      /** Close the writing pipe */

      waitpid (pidChild, &(statptr), 0);        /* wait for the child */

      if (statptr != -1)
      {
        char *piece;
        int bytes;

        piece = (char *) malloc (1024);

        bytes = read (pipefd[0], piece, 1024);

        /*
         * The child process ended due to an exit 
         */

        serial = SerialNumberFromIFCONFIG (piece);

        free (piece);
        close (pipefd[0]);
        return serial;
      }
      else
      {                  /** Some error assume the process is alive */
        fprintf (stderr, "ERROR in child\n");
        close (pipefd[0]);
        exit (1);
      }
  }
  return 1;
}
#endif

#if defined(hpux)
static unsigned long get_hardware_serial_number_HPUX ()
{
  int pipefd[2];
  unsigned long long seriall;
  unsigned long serial;
  pid_t pidChild;
  int statptr;

  if (pipe (pipefd) == -1)
  {
    fprintf (stderr, "ERROR at pipe: %s\n", strerror (errno));
    return 1;
  }

  switch (pidChild = vfork ())
  {
    case 0:                    /* child process */
      close (1);                /* close standard output */

                    /** Redirect standar output to the pipe */
      if (dup (pipefd[1]) == -1)
      {
        fprintf (stderr, "ERROR at dup: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);
        exit (-1);                 /** Error */
      }

      if (execlp ("/usr/sbin/lanscan", "/usr/sbin/lanscan", "-a", NULL) == -1)
      {
        fprintf (stderr, "ERROR at execlp: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);

        exit (-1);                /** Error */
      }
      exit (0);
      break;

    case -1:                   /* some error forking */
      fprintf (stderr, "ERROR at vfork: %s\n", strerror (errno));
      close (pipefd[0]);
      close (pipefd[1]);
      return 1;

    default:                   /* parent process, everything ok */
      close (pipefd[1]);                      /** Close the writing pipe */

      waitpid (pidChild, &(statptr), 0);        /* wait for the child */

      if (WEXITSTATUS (statptr) != -1)
      {
        char *piece;
        int bytes;

        piece = (char *) malloc (1024);

        bytes = read (pipefd[0], piece, 1024);

        /*
         * The child process ended due to an exit 
         */
        sscanf (piece, "0x%llx", &(seriall));

        seriall = seriall & 0x00000000FFFFFFFF;

        serial = (unsigned long) seriall;

        free (piece);
        close (pipefd[0]);
        return serial;
      }
      else
      {                  /** Some error assume the process is alive */
        fprintf (stderr, "ERROR in child\n");
        close (pipefd[0]);
        exit (1);
      }
  }
  return 1;
}
#endif

#if defined(sun)

#define BUFFER_NAME 1024

char device[300], IPaddr[300], mask[300], Addr1[300], Addr2[300];

unsigned long SerialNumberFromNETSTAT (int fd)
{
  char *hostname;
  int bytes, ret;
  char *pointer = NULL;
  char *piece = NULL, *middel = NULL;
  char *bigpointer3 = NULL;
  int old = 0, found;
  unsigned long long seriall;
  unsigned long serial;

  piece = (char *) malloc (BUFFER_NAME);
  hostname = (char *) malloc (BUFFER_NAME);

  if (gethostname (hostname, BUFFER_NAME))
  {
    fprintf (stderr, "ERROR in gethostname %s\n", strerror (errno));
    exit (0);
  }

  while ((bytes = read (fd, piece, BUFFER_NAME)) != 0)
  {

printf("%s\n", piece);

    if ((bytes == -1) && (errno == EINTR))
      continue;
    piece[bytes] = 0;
    if (old != 0)
    {
      middel = (char *) malloc (old + 1);
      strcpy (middel, bigpointer3);
      free (bigpointer3);
    }
    bigpointer3 = (char *) malloc (old + bytes + 1);
    if (old != 0)
    {
      strcpy (bigpointer3, middel);
      strcat (bigpointer3, piece);
      free (middel);
    }
    else
      strcpy (bigpointer3, piece);
    old = old + bytes;
  }

  free (piece);

  found = 0;
  if (old)
  {
    piece = strtok (bigpointer3, "\n");
    while (piece != NULL)
    {
      ret =
        sscanf (piece, "%s %s %s %s %s\n", device, IPaddr, mask, Addr1,
                Addr2);
      if (ret == 5)
      {
        strcpy (Addr1, Addr2);
        ret = 4;
      }
      if (ret == 4)
      {
        if (strncmp (IPaddr, hostname, strlen (hostname)) == 0)
        {
          found = 1;
          break;
        }
      }
      piece = strtok (NULL, "\n");
    }
  }

  piece = (char *) malloc (BUFFER_NAME);
  if (found)
  {
    middel = strtok (Addr1, ":");
    sprintf (piece, "0x");
    while (middel != NULL)
    {
      sprintf (piece, "%s%s", piece, middel);
      middel = strtok (NULL, ":");
    }

    sscanf (piece, "%llx", &(seriall));
    seriall = seriall & 0x00000000FFFFFFFF;
    serial = (unsigned long) seriall;
  }

  free (piece);
  return serial;
}


static unsigned long get_hardware_serial_number_SUN ()
{
  int pipefd[2];
  unsigned long serial;
  pid_t pidChild;
  int statptr;

  if (pipe (pipefd) == -1)
  {
    fprintf (stderr, "ERROR at pipe: %s\n", strerror (errno));
    return 1;
  }

  switch (pidChild = vfork ())
  {
    case 0:                    /* child process */
      close (1);                /* close standard output */

                    /** Redirect standar output to the pipe */
      if (dup (pipefd[1]) == -1)
      {
        fprintf (stderr, "ERROR at dup: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);
        exit (-1);                 /** Error */
      }

      if (execlp ("/usr/bin/netstat", "/usr/bin/netstat", "-p", NULL) == -1)
      {
        fprintf (stderr, "ERROR at execlp: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);

        exit (-1);                /** Error */
      }
      exit (0);
      break;

    case -1:                   /* some error forking */
      fprintf (stderr, "ERROR at vfork: %s\n", strerror (errno));
      close (pipefd[0]);
      close (pipefd[1]);
      return 1;

    default:                   /* parent process, everything ok */
      close (pipefd[1]);                      /** Close the writing pipe */

      waitpid (pidChild, &(statptr), 0);        /* wait for the child */

      if (statptr != -1)
      {
        serial = SerialNumberFromNETSTAT (pipefd[0]);
        close (pipefd[0]);
        return serial;
      }
      else
      {                  /** Some error assume the process is alive */
        fprintf (stderr, "ERROR in child\n");
        close (pipefd[0]);
        exit (1);
      }
  }
  return 1;
}
#endif


#if defined(_AIX)

#ifdef HAVE_SYS_UTSNAME_H
# include <sys/utsname.h>
#endif

unsigned long get_hardware_serial_number_IBM ()
{
	struct utsname name;
	unsigned long serialid;

	uname (&name);
	serialid = strtoul (name.machine, (char **) 0, 10);

	return serialid;
}
#endif


#if defined(__alpha)

#define BUFFER_NAME 1024

char device[300], IPaddr[300], mask[300], Addr1[300], Rest[700];

unsigned long SerialNumberFromNETSTAT (int fd)
{
  int bytes, ret;
  char *pointer = NULL;
  char *piece = NULL, *middel = NULL;
  char *bigpointer3 = NULL;
  int old = 0, found;
  unsigned long long seriall;
  unsigned long serial;

  piece = (char *) malloc (BUFFER_NAME);

  while ((bytes = read (fd, piece, BUFFER_NAME)) != 0)
  {
    if ((bytes == -1) && (errno == EINTR))
      continue;
    piece[bytes] = 0;
    if (old != 0)
    {
      middel = (char *) malloc (old + 1);
      strcpy (middel, bigpointer3);
      free (bigpointer3);
    }
    bigpointer3 = (char *) malloc (old + bytes + 1);
    if (old != 0)
    {
      strcpy (bigpointer3, middel);
      strcat (bigpointer3, piece);
      free (middel);
    }
    else
      strcpy (bigpointer3, piece);
    old = old + bytes;
  }

  free (piece);

  found = 0;
  if (old)
  {
    piece = strtok (bigpointer3, "\n");
    if (piece != NULL)
    {
      piece = strtok (NULL, "\n");
      if (piece != NULL)
      {
        sscanf (piece, "%s %s %s %s [^\n]\n", device, IPaddr, mask, Addr1,
                Rest);


        piece = (char *) malloc (BUFFER_NAME);
        middel = strtok (Addr1, ":");
        sprintf (piece, "0x");
        while (middel != NULL)
        {
          sprintf (piece, "%s%s", piece, middel);

          middel = strtok (NULL, ":");
        }

        sscanf (piece, "%lx", &(seriall));
        seriall = seriall & 0x00000000FFFFFFFF;
        serial = (unsigned long) seriall;

        free (piece);
      }
    }
  }
  return serial;
}


static unsigned long get_hardware_serial_number_ALPHA ()
{
  int pipefd[2];
  unsigned long serial;
  pid_t pidChild;
  int statptr;

  if (pipe (pipefd) == -1)
  {
    fprintf (stderr, "ERROR at pipe: %s\n", strerror (errno));
    return 1;
  }
/* GLS: Antes VFORK. La ejecucion tardaba eones. En el man comenta que fork y vfork
 * son iguales en implementacion, pero que si se usan libc, libpthreads... se emplee
 * fork 
 */ 
  switch (pidChild = fork ())
  {
    case 0:                    /* child process */
      close (1);                /* close standard output */

                    /** Redirect standar output to the pipe */
      if (dup (pipefd[1]) == -1)
      {
        fprintf (stderr, "ERROR at dup: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);
        exit (-1);                 /** Error */
      }

      if (execlp ("/usr/sbin/netstat", "/usr/sbin/netstat", "-ia", NULL) ==
          -1)
      {
        fprintf (stderr, "ERROR at execlp: %s\n", strerror (errno));
        close (pipefd[0]);
        close (pipefd[1]);

        exit (-1);                /** Error */
      }
      exit (0);
      break;

    case -1:                   /* some error forking */
      fprintf (stderr, "ERROR at vfork: %s\n", strerror (errno));
      close (pipefd[0]);
      close (pipefd[1]);
      return 1;

    default:                   /* parent process, everything ok */
      close (pipefd[1]);                      /** Close the writing pipe */

      waitpid (pidChild, &(statptr), 0);        /* wait for the child */

      if (statptr != -1)
      {
        serial = SerialNumberFromNETSTAT (pipefd[0]);
        close (pipefd[0]);
        return serial;
      }
      else
      {                  /** Some error assume the process is alive */
        fprintf (stderr, "ERROR in child\n");
        close (pipefd[0]);
        exit (1);
      }
  }
  return 1;
}
#endif
