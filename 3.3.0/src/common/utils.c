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
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
#ifdef HAVE_ASSERT_H
# include <assert.h>
#endif

#include "utils.h"

int is_Whitespace(char c)
{
  /* Avoid using isspace() and iscntrl() to remove internal dependency with ctype_b/ctype_b_loc.
   * This symbol name depends on the glibc version; newer versions define ctype_b_loc and compat 
   * symbols have been removed. This dependency may end up in undefined references when porting
   * binaries to machines with different glibc versions.
   */

   return c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\n';
}

int is_Alphabetic(char c)
{
	/* Avoid using isspace() and iscntrl() to remove internal dependency with ctype_b/ctype_b_loc.
	 * This symbol name depends on the glibc version; newer versions define ctype_b_loc and compat 
	 * symbols have been removed. This dependency may end up in undefined references when porting
	 * binaries to machines with different glibc versions.
	 */

	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

/* Supress the spaces at both sides of the string sourceStr */
char *trim (char *sourceStr)
{ 
  int sourceLen = 0;
  int left = 0, right = 0;
  char *retStr = NULL;
  int retLen = 0;

  if (sourceStr == NULL) return NULL;
  
  sourceLen = strlen (sourceStr);

  left  = 0;
  right = sourceLen - 1;

  /* Find first non-whitespace character */
  while ((left < sourceLen) && (is_Whitespace(sourceStr[left])))
    left ++;

  /* Find last character before whitespaces */
  while ((right > left) && (is_Whitespace(sourceStr[right])))
    right --;

  /* Create a new string */
  retLen = (right - left + 1) + 1; // Extra 1 for the final '\0' 
  xmalloc(retStr, retLen * sizeof(char));
  retStr = strncpy (retStr, &sourceStr[left], retLen-1);
  retStr[retLen-1] = '\0';

  return retStr;
}

/**
 * Builds an array where each element is a token from 'sourceStr' separated by 'delimiter'
 * @param[in]     sourceStr  The string to explode
 * @param[in]     delimiter  The delimiting character 
 * @param[in,out] tokenArray The resulting vector
 *
 * @return Returns the number of tokens in the resulting array 
 */

int explode (char *sourceStr, const char *delimiter, char ***tokenArray)
{
   int num_tokens = 0;
   char **retArray = NULL;
   char *backupStr, *backupStr_ptr;
   char *token, *trimmed_token;
 
   if ((sourceStr != NULL) && (strlen(sourceStr) > 0))
   {
      /* Copy the original string to a local buffer, because strtok modifies it */
      backupStr = strdup (sourceStr);
      if (backupStr != NULL)
      {
         /* Save the original pointer for freeing later */
         backupStr_ptr = backupStr;

         /* Separate tokens by delimiter */
         while ((token = strtok(backupStr, delimiter)) != NULL)
         {
            backupStr = NULL;
            trimmed_token = trim (token);
            if (trimmed_token != NULL)
            {
               /* Save the token in a new position of the resulting vector */
               num_tokens ++;
               xrealloc(retArray, retArray, num_tokens * sizeof(char *));
               retArray[num_tokens-1] = strdup(trimmed_token);
               xfree (trimmed_token);
            }
         }
         xfree (backupStr_ptr);
      }
   }

   *tokenArray = retArray;
   return num_tokens;
}

/******************************************************************************
 **  Function name : append_from_to_file
 **  Author : HSG
 **  Description : Appends contents of source into destination
 ******************************************************************************/

int append_from_to_file (const char *source, const char *destination)
{
	char buffer[65536];
	int fd_o, fd_d;
	ssize_t res;

	/* Open the files */
	fd_o = open (source, O_RDONLY);
	if (fd_o == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error while trying to open %s \n", source);
		fflush (stderr);
		return -1;
	}
	fd_d = open (destination, O_WRONLY | O_APPEND, 0644);
	if (fd_d == -1)
	{
		close (fd_d);
		fprintf (stderr, PACKAGE_NAME": Error while trying to open %s \n", destination);
		fflush (stderr);
		return -1;
	}

	/* Copy the file */
	res = read (fd_o, buffer, sizeof (buffer));
	while (res != 0 && res != -1)
	{
		res = write (fd_d, buffer, res);
		if (res == -1)
			break;
		res = read (fd_o, buffer, sizeof (buffer));
	}

	/* If failed, just close!  */
	if (res == -1)
	{
		close (fd_d);
		close (fd_o);
		unlink (destination);
		fprintf (stderr, PACKAGE_NAME": Error while trying to move files %s to %s\n", source, destination);
		fflush (stderr);
		return -1;
	}

	/* Close the files */
	close (fd_d);
	close (fd_o);

	/* Remove the files */
	unlink (source);

	return 0;
}

/******************************************************************************
 **  Function name : rename_or_copy (char *, char *)
 **  Author : HSG
 **  Description : Tries to rename (if in the same /dev/) or moves the file.
 ******************************************************************************/

int rename_or_copy (char *origen, char *desti)
{
	if (rename (origen, desti) == -1)
	{
		if (errno == EXDEV)
		{
			char buffer[65536];
			int fd_o, fd_d;
			ssize_t res;

			/* Open the files */
			fd_o = open (origen, O_RDONLY);
			if (fd_o == -1)
			{
				fprintf (stderr, PACKAGE_NAME": Error while trying to open %s \n", origen);
				fflush (stderr);
				return -1;
			}
			fd_d = open (desti, O_WRONLY | O_CREAT | O_TRUNC, 0644);
			if (fd_d == -1)
			{
				close (fd_d);
				fprintf (stderr, PACKAGE_NAME": Error while trying to open %s \n", desti);
				fflush (stderr);
				return -1;
			}

			/* Copy the file */
			res = read (fd_o, buffer, sizeof (buffer));
			while (res != 0 && res != -1)
			{
				res = write (fd_d, buffer, res);
				if (res == -1)
					break;
				res = read (fd_o, buffer, sizeof (buffer));
			}

			/* If failed, just close!  */
			if (res == -1)
			{
				close (fd_d);
				close (fd_o);
				unlink (desti);
				fprintf (stderr, PACKAGE_NAME": Error while trying to move files %s to %s\n", origen, desti);
				fflush (stderr);
				return -1;
			}

			/* Close the files */
			close (fd_d);
			close (fd_o);

			/* Remove the files */
			unlink (origen);
		}
		else
		{
			perror("rename");
			fprintf (stderr, PACKAGE_NAME": Error while trying to move %s to %s\n", origen, desti);
			fflush (stderr);

			return -1;
		}
	}

	return 0;
}

unsigned long long getFactorValue (const char *value, const char *ref, int rank)
{
	unsigned long long Factor;
	char tmp_buff[256];

	if (value == NULL)
		return 0;

	strncpy (tmp_buff, value, sizeof(tmp_buff));

	switch (tmp_buff[strlen(tmp_buff)-1])
	{
		case 'K': /* Kilo */
		case 'k': /* Kilo */
			Factor = 1000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'M': /* Mega */
		case 'm': /* Mega */
			Factor  = 1000;
			Factor *= 1000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'G': /* Giga */
		case 'g': /* Giga */
			Factor  = 1000;
			Factor *= 1000;
			Factor *= 1000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'T': /* Tera */
		case 't': /* Tera */
			Factor  = 1000;
			Factor *= 1000;
			Factor *= 1000;
			Factor *= 1000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		default :
			Factor = 1;
			/* If the last char is a number, then units are omitted! */
			if (!(tmp_buff[strlen(tmp_buff)-1] >= '0' && tmp_buff[strlen(tmp_buff)-1] <= '9'))
			{
				if (rank == 0)
					fprintf (stdout, PACKAGE_NAME": Warning! %s time units unkown! Using seconds\n", ref);
			}
			break;
		}

		return atoll (tmp_buff) * Factor;
}

unsigned long long getTimeFromStr (const char *time, const char *envvar, int rank)
{
	unsigned long long MinTimeFactor;
	char tmp_buff[256];
	size_t strl;

	if (time == NULL)
		return 0;

    strncpy (tmp_buff, time, sizeof(tmp_buff));

	strl = strlen(tmp_buff);

	if (strl > 2 && is_Alphabetic(tmp_buff[strl-2]) && tmp_buff[strl-1] == 's')
	{
		tmp_buff[strl-1] = 0x0; // Strip the last 's' of ms/ns/us
	}

	switch (tmp_buff[strlen(tmp_buff)-1])
	{
		case 'D': /* Days */
			MinTimeFactor = 24*60*60;
			MinTimeFactor *= 1000000000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'M': /* Minutes */
			MinTimeFactor = 60;
			MinTimeFactor *= 1000000000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'H': /* Hours */
			MinTimeFactor = 60*60;
			MinTimeFactor *= 1000000000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 's': /* Seconds */
		case 'S':
			MinTimeFactor = 1;
			MinTimeFactor *= 1000000000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'm': /* Milliseconds */
			MinTimeFactor = 1000000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'u': /* Microseconds */
			MinTimeFactor = 1000;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		case 'n': /* Nanoseconds */
			MinTimeFactor = 1;
			tmp_buff[strlen(tmp_buff)-1] = (char) 0;
			break;

		default :
			MinTimeFactor = 1;
			MinTimeFactor *= 1000000000;
			/* If the last char is a number, then the time units are omitted! */
			if (tmp_buff[strlen(tmp_buff)-1] >= '0'
			&& tmp_buff[strlen(tmp_buff)-1] <= '9'
			&& rank == 0)
				fprintf (stdout,
					PACKAGE_NAME": Warning! %s time units not specified. Using seconds\n", envvar);
			else
			{
				if (rank == 0)
					fprintf (stdout, PACKAGE_NAME": Warning! %s time units unknown! Using seconds\n", envvar);
			}
			break;
		}

		return atoll (tmp_buff) * MinTimeFactor;
}

/******************************************************************************
 **      Function name : file_exists (char*)
 **      Author : HSG
 **      Description : Checks whether a file exists
 ******************************************************************************/
int file_exists (const char *fname)
{
#if defined(HAVE_ACCESS)
	return access (fname, F_OK) == 0;
#elif defined(HAVE_STAT64)
	struct stat64 sb;
	stat64 (fname, &sb);
	return (sb.st_mode & S_IFMT) == S_IFREG;
#elif defined(HAVE_STAT)
	struct stat sb;
	stat (fname, &sb);
	return (sb.st_mode & S_IFMT) == S_IFREG;
#else
	int fd = open (fname, O_RDONLY);
	if (fd >= 0)
	{
		close (fd);
		return TRUE;
	}
	else
		return FALSE;
#endif
}

/******************************************************************************
 **      Function name : directory_exists (char*)
 **      Author : HSG
 **      Description : Checks whether a directory exists
 ******************************************************************************/
int directory_exists (const char *fname)
{
#if defined(HAVE_STAT)
	struct stat sb;
	stat (fname, &sb);
	return S_ISDIR(sb.st_mode);
#else
# error "Don't know how to check whether a directory exists"
#endif
}

/******************************************************************************
 **      Function name : mkdir_recursive (char*)
 **      Author : HSG
 **      Description : make a recursive recursively
 ******************************************************************************/
int mkdir_recursive (const char *path)
{
	struct stat sb;

	if (stat (path, &sb) == -1)
	{
		char *original_path;
		char *dir;
		int result;

		/* dirname may modify its parameter */
		original_path = strdup (path);

		dir = dirname (original_path);

		if ((strcmp (dir, ".") != 0) && (strcmp (dir, "/") != 0))
			result = mkdir_recursive(dir)?mkdir (path, 0755) == 0 : 0;
		else
			result = mkdir (path, 0755) == 0;

		free (original_path);

		return result;
	}
	else
		return S_ISDIR(sb.st_mode);
}

int ExtraeUtils_shorten_string (unsigned nprefix, unsigned nsufix, const char *infix,
	unsigned buffersize, char *buffer, const char *string)
{
	assert (buffersize >= nprefix+nsufix+strlen(infix)+1);

	memset (buffer, 0, buffersize);

	/* Split if it does not fit */
	if (strlen(string) >= nprefix+nsufix+strlen(infix))
	{
		strncpy (buffer, string, nprefix);
		strncpy (&buffer[nprefix], infix, strlen(infix));
		strncpy (&buffer[nprefix+strlen(infix)], &string[strlen(string)-nsufix], nsufix);
		return TRUE;
	}
	else
	{
		/* Copy if it fits */
		strncpy (buffer, string, strlen(string));
		return 0;
	}
}

