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

#if !defined(HAVE_XML2)

# error "You need libxml2 to compile this file!"

#else

#include <libxml/xmlmemory.h>
#include <libxml/parser.h>

#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif
#if defined(MPI_SUPPORT)
# ifdef HAVE_MPI_H
# include <mpi.h>
# endif
#endif

#include "utils.h"
#include "mini-xml-parse.h"

static const xmlChar *xmlYES = (xmlChar*) "yes";
static const xmlChar *xmlCOMMENT = (xmlChar*) "COMMENT";
static const xmlChar *xmlTEXT = (xmlChar*) "text";
static const xmlChar XML_ENVVAR_CHARACTER = (xmlChar) '$';

/* Some global (but local in the module) variables */
static char *temporal_d = NULL, *final_d = NULL;
static int TraceEnabled = FALSE;
static int TracePrefixFound = FALSE;
static char TracePrefix[1024] = "TRACE";
static int TraceMPI = FALSE;
static int TraceOpenMP = FALSE;
static int TraceOpenMP_locks = FALSE;
static char *UFlist = NULL;
static int ExcludeAutomaticFunctions = FALSE;

int XML_CheckTraceEnabled (void)
{ return TraceEnabled; }

char * XML_GetFinalDirectory (void)
{ return final_d; }

char * XML_GetTracePrefix (void)
{ return TracePrefix; }

int XML_GetTraceMPI (void)
{ return TraceMPI; }

int XML_GetTraceOMP (void)
{ return TraceOpenMP; }

int XML_GetTraceOMP_locks (void)
{ return TraceOpenMP_locks; }

char* XML_UFlist (void)
{ return UFlist; }

int XML_have_UFlist (void)
{ return UFlist != NULL; }

int XML_excludeAutomaticFunctions (void)
{ return ExcludeAutomaticFunctions; }

#define is_Whitespace(c) \
   ((c) == ' ' || (c) == '\t' || (c) == '\v' || (c) == '\f' || (c) == '\n')

/* Free memory if not null */
#define XML_FREE(ptr) \
	if (ptr != NULL) xmlFree(ptr);

static xmlChar * deal_xmlChar_env (xmlChar *str)
{
	xmlChar *tmp;
	int i;
	int initial = 0;
	int sublen = 0;
	int length = xmlStrlen (str);
	int final = length;

	/* First get rid of the leading and trailing white spaces */
	for (i = 0; i < length; i++)
		if (!is_Whitespace (str[i]))
			break;
	initial = i;
	for (; final-1 >= i; final--)
		if (!is_Whitespace (str[final-1]))
			break;

	sublen = final - initial;

	tmp = xmlStrsub (str, initial, sublen);

	if (sublen > 0)
	{
		/* If the string is wrapped by XML_ENVVAR_CHARACTER, perform a getenv and
		   return its result */
		if (sublen > 1 && tmp[0] == XML_ENVVAR_CHARACTER && tmp[sublen-1] == XML_ENVVAR_CHARACTER)
		{
			char tmp2[sublen];
			memset (tmp2, 0, sublen);
			strncpy (tmp2, &tmp[1], sublen-2);

			if (getenv (tmp2) == NULL)
			{
				fprintf (stderr, PACKAGE_NAME": Environment variable %s is not defined!\n", tmp2);
				return NULL;
			}
			else
			{
				if (strlen(getenv(tmp2)) == 0)
				{
					fprintf (stderr, PACKAGE_NAME": Environment variable %s is set but empty!\n", tmp2);
					return NULL;
				}
				else
					return xmlCharStrdup (getenv(tmp2));
			}
		}
		else
			return tmp;
	}
	else
		return tmp;
}

static xmlChar * xmlNodeListGetString_env (xmlDocPtr doc, xmlNodePtr list, int inLine)
{
	xmlChar *tmp;

	tmp = xmlNodeListGetString (doc, list, inLine);
	if (tmp != NULL)
	{
		xmlChar *tmp2;
		tmp2 = deal_xmlChar_env (tmp);
		XML_FREE(tmp);
		return tmp2;
	}
	else
		return NULL;
}

static xmlChar* xmlGetProp_env (xmlNodePtr node, xmlChar *attribute)
{
	xmlChar *tmp;

	tmp = xmlGetProp (node, attribute);
	if (tmp != NULL)
	{
		xmlChar *tmp2;
		tmp2 = deal_xmlChar_env (tmp);
		XML_FREE(tmp);
		return tmp2;
	}
	else
		return NULL;	
}

/* Configure OpenMP related parameters */
static void Parse_XML_OMP (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	UNREFERENCED_PARAMETER(rank);
	UNREFERENCED_PARAMETER(xmldoc);

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Shall we instrument openmp lock routines? */
		else if (!xmlStrcmp (tag->name, TRACE_OMP_LOCKS))
		{
			xmlChar *enabled = xmlGetProp_env (tag, TRACE_ENABLED);
			TraceOpenMP_locks = ((enabled != NULL && !xmlStrcmp (enabled, xmlYES)));
			XML_FREE(enabled);
		}

		tag = tag->next;
	}
}

/* Configure storage related parameters */
static void Parse_XML_Storage (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	UNREFERENCED_PARAMETER(rank);

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Where must we store the intermediate files? DON'T FREE it's used below */
		else if (!xmlStrcmp (tag->name, TRACE_DIR))
		{
			xmlChar *enabled = xmlGetProp_env (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				temporal_d = (char*) xmlNodeListGetString_env (xmldoc, tag->xmlChildrenNode, 1);
			XML_FREE(enabled);
		}
		/* Where must we store the final intermediate files?  DON'T FREE it's used below */
		else if (!xmlStrcmp (tag->name, TRACE_FINAL_DIR))
		{
			xmlChar *enabled = xmlGetProp_env (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				final_d = (char*) xmlNodeListGetString_env (xmldoc, tag->xmlChildrenNode, 1);
			XML_FREE(enabled);
		}
		/* Obtain the MPIT prefix */
		else if (!xmlStrcmp (tag->name, TRACE_PREFIX))
		{
			xmlChar *enabled = xmlGetProp_env (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *p_name = (char*)xmlNodeListGetString_env (xmldoc, tag->xmlChildrenNode, 1);
				strncpy (TracePrefix, p_name, sizeof(TracePrefix));
				TracePrefixFound = TRUE;
				XML_FREE(p_name);
			}
			else
			{
				/* If not enabled, just put TRACE as the program name */
				strncpy (TracePrefix, "TRACE", sizeof(TracePrefix));
				TracePrefixFound = TRUE;
			}
			XML_FREE(enabled);
		}

		tag = tag->next;
	}
}

void Parse_XML_File (int rank, int world_size, char *filename)
{
	xmlNodePtr current_tag;
	xmlDocPtr  xmldoc;
	xmlNodePtr root_tag;
	char cwd[1024];

	UNREFERENCED_PARAMETER(world_size);

  /*
  * This initialize the library and check potential ABI mismatches
  * between the version it was compiled for and the actual shared
  * library used.
  */
	LIBXML_TEST_VERSION;
   
	xmldoc = xmlParseFile (filename);
	if (xmldoc != NULL)
	{
		root_tag = xmlDocGetRootElement (xmldoc);
		if (root_tag != NULL)
		{
			if (xmlStrcmp(root_tag->name, TRACE_TAG))
			{	
				fprintf (stderr, PACKAGE_NAME": Invalid configuration file\n");
			}
			else
			{
				/*
					 Are the MPI & OpenMP tracing enabled? Is the tracing fully enabled? 
					 try to remember if these values have been changed previously by the
					 API routines.
				*/

				/* Full tracing control */
				char *tracehome = (char*) xmlGetProp_env (root_tag, TRACE_HOME);
				xmlChar *traceenabled = xmlGetProp_env (root_tag, TRACE_ENABLED);
				TraceEnabled = (traceenabled != NULL) && !xmlStrcmp (traceenabled, xmlYES);

				XML_FREE(traceenabled);
				XML_FREE(tracehome);

				current_tag = root_tag->xmlChildrenNode;
				while (current_tag != NULL && TraceEnabled)
				{
					/* Skip coments */
					if (!xmlStrcmp (current_tag->name, xmlTEXT) || !xmlStrcmp (current_tag->name, xmlCOMMENT))
					{
					}
					/* UF related information instrumentation */
					else if (!xmlStrcmp (current_tag->name, TRACE_USERFUNCTION))
					{
						xmlChar *enabled = xmlGetProp_env (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
						{
							xmlChar *excludeautofunctions;
							UFlist = (char*) xmlGetProp_env (current_tag, TRACE_LIST);
							excludeautofunctions = xmlGetProp_env (current_tag, TRACE_EXCLUDE_AUTOMATIC_FUNCTIONS);
							if (excludeautofunctions != NULL)
								ExcludeAutomaticFunctions = !xmlStrcmp (excludeautofunctions, xmlYES);
							XML_FREE(excludeautofunctions);
						}
						XML_FREE(enabled);
					}
					/* MPI related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_MPI))
					{
						xmlChar *enabled = xmlGetProp_env (current_tag, TRACE_ENABLED);
						TraceMPI = (enabled != NULL && !xmlStrcmp (enabled, xmlYES));
						XML_FREE(enabled);
					}
					/* OpenMP related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_OMP))
					{
						xmlChar *enabled = xmlGetProp_env (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
						{
							Parse_XML_OMP (rank, xmldoc, current_tag);
							TraceOpenMP = TRUE;
						}
						else
							TraceOpenMP = FALSE;
						XML_FREE(enabled);
					}
					/* Storage related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_STORAGE))
					{
						xmlChar *enabled = xmlGetProp_env (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_Storage (rank, xmldoc, current_tag);
						XML_FREE(enabled);
					}

					current_tag = current_tag->next;
				}
			}
		}
		else
		{
			fprintf (stderr, PACKAGE_NAME": Error! Empty mpitrace configuration file\n");
		}

		xmlFreeDoc (xmldoc);
	}

	/* If the tracing has been enabled, continue with the configuration */
	if (TraceEnabled)
	{
		char *res;

		/* Have the user proposed any program name? */
		if (!TracePrefixFound)
			strncpy (TracePrefix, "TRACE", sizeof(TracePrefix));

		/* Just gather now the variables related to the directories! */

		/* Temporal directory must be checked against the configuration of the XML,
    	  and the current directory */
		if (temporal_d == NULL)
		{
			if ((res = getcwd(cwd, sizeof(cwd))) != NULL)
			{
				temporal_d = ".";
			}
			else
			{
				temporal_d = malloc ((strlen(res)+1)*sizeof(char));
				strcpy (temporal_d, res);
			}
		}

		/* Final directory must be checked against the configuration of the XML, 
  	    the temporal_directory and, finally, the current directory */
		if (final_d == NULL)
			final_d = temporal_d;
	}
}

#endif /* HAVE_XML2 */
