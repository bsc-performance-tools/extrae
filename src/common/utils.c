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
static char *trim (char *sourceStr)
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
 **  Function name : rename_or_copy (char *, char *)
 **  Author : HSG
 **  Description : Tries to rename (if in the same /dev/) or moves the file.
 ******************************************************************************/

void rename_or_copy (char *origen, char *desti)
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
        return;
      }
      fd_d = open (desti, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd_d == -1)
      {
        close (fd_d);
        fprintf (stderr, PACKAGE_NAME": Error while trying to open %s \n", desti);
        fflush (stderr);
        return;
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
        return;
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
    }
  }
}

unsigned long long getFactorValue (char *value, char *ref, int rank)
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

unsigned long long getTimeFromStr (char *time, char *envvar, int rank)
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
int file_exists (char *fname)
{
#if defined(HAVE_ACCESS)
	return access (fname, F_OK) == 0;
#elif defined(HAVE_STAT)
	struct stat sb;
	stat (fname, &sb);
	return S_ISREG(sb.st_mode);
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
int directory_exists (char *fname)
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
int mkdir_recursive (char *path)
{
	struct stat sb;

	if (stat (path, &sb) == -1)
	{
		char *original_path;
		char *dir;
		int result;

		/* dirname may modify its parameter */
		original_path = strdup (path);

#if defined(OS_FREEBSD)
		dir = dirname ((const char*) original_path);
#else
		dir = dirname (original_path);
#endif

		if ((strcmp (dir, ".") != 0) && (strcmp (dir, "/") != 0))
			result = mkdir_recursive(dir)?mkdir (path, 0744) == 0 : 0;
		else
			result = mkdir (path, 0744) == 0;

		free (original_path);

		return result;
	}
	else
		return S_ISDIR(sb.st_mode);
}

#if defined(DEAD_CODE)
/* Concatenates two strings, reallocating the size of the first string as required */
static char * concat(char **prefix_ptr, char *suffix)
{
   char *prefix = *prefix_ptr;
   char *ret    = NULL;

   if ((prefix == NULL) && (suffix == NULL))
   {
      ret = NULL;
   }
   else if (suffix == NULL)
   {
      ret = prefix;
   }
   else if (prefix == NULL)
   {
      xmalloc(prefix, strlen(suffix) + 1);
      strcpy (prefix, suffix);
      ret = prefix;
   }
   else
   {
      xrealloc(prefix, prefix, strlen(prefix)+strlen(suffix)+1);
      strcat(prefix, suffix);
      ret = prefix;
   }
   return ret;
}

/* Replaces the text between start and end delimiters for the return value of the callback 'function' */
static char * expand(char *input, const char *start_delim, const char *end_delim, char *(*function)(const char *))
{
   char *str = NULL;
   char *token = NULL;
   char *begin = NULL;
   char *end = NULL;
   char *expanded_str = NULL;

   if ( (input == NULL) || ((input != NULL) && (strlen(input) <= 0)) )
   {
      return NULL;
   }

   /* Copy the original string, since we're going to modify it */
   xmalloc(str, strlen(input)+1);
   strcpy(str, input);

   token = str;
   /* Search for starting delimiters */
   while ((begin = strstr(token, start_delim)) != NULL)
   {
      /* Search for ending delimiter */
      if ((end = strstr (begin, end_delim)) != NULL)
      {
         int var_name_length = 0;

         /* Append to the output everything between the last delimiter and the current one */
         *begin = '\0';
         expanded_str = concat(&expanded_str, token);

         /* Get the variable name, between begin and end */
         var_name_length = end - begin - strlen(start_delim);
         if (var_name_length > 0)
         {
            char *var_name = NULL;

            xmalloc(var_name, var_name_length+1);
            memset(var_name, '\0', var_name_length+1);
            strncpy(var_name, begin + strlen(start_delim), var_name_length);

            if (function != NULL)
            {
               char *var_replace = NULL;

               var_replace = function((const char *)var_name);
               if ((var_replace != NULL) && (strlen(var_replace) > 0))
               {
                  expanded_str = concat(&expanded_str, var_replace);
               }
               /* FIXME: var_replace is not freed, because the return value of getenv can't be touched.
                * This is a small memory leak. When invoking a callback such as getenv we should 
                * write a wrapper which copies the result to our own memory and free always.
                */
            }
            xfree(var_name);
         }
         token = end + strlen(end_delim);
      }
      else
      {
         /* Bad syntax. Throw error? */
         break;
      }
   }
   /* Append the last part of the string */
   expanded_str = concat(&expanded_str, token);

   xfree(str);
   return expanded_str;
}
#endif

