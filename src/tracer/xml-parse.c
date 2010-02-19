/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

#if defined(HAVE_XML2)

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
#  include <mpi.h>
# endif
#endif
#if defined(PACX_SUPPORT)
# include <pacx.h>
#endif

#include "utils.h"
#include "hwc.h"
#include "xml-parse.h"
#include "wrapper.h"
#include "signals.h"
#if defined(MPI_SUPPORT)
# include "mpi_wrapper.h"
#endif
#if defined(OMP_SUPPORT)
# include "omp_wrapper.h"
#endif
#if defined(IS_CELL_MACHINE)
# include "cell_wrapper.h"
#endif
#include "UF_gcc_instrument.h"
#if defined(HAVE_MRNET)
# include "mrn_config.h"
# include "mrnet_be.h"
#endif
#if defined(OMP_SUPPORT)
# include "omp_probe.h"
#endif

/* Some global (but local in the module) variables */
static char *temporal_d = NULL, *final_d = NULL;
#if defined(DEAD_CODE)
static int temporal_d_mkdir = TRUE, final_d_mkdir = TRUE;
#endif
static int TracePrefixFound = FALSE;

static const xmlChar *xmlYES = (xmlChar*) "yes";
static const xmlChar *xmlCOMMENT = (xmlChar*) "COMMENT";
static const xmlChar *xmlTEXT = (xmlChar*) "text";

/* Free memory if not null */
#define XML_FREE(ptr) \
	if (ptr != NULL) xmlFree(ptr);

/* master fprintf :) */
#define mfprintf \
	if (rank == 0) fprintf 

#if defined(MPI_SUPPORT)
/* Configure MPI related parameters */
static void Parse_XML_MPI (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlCOMMENT) || !xmlStrcmp (tag->name, xmlTEXT))
		{
		}
		/* Shall we gather counters in the MPI calls? */
		else if (!xmlStrcmp (tag->name, TRACE_COUNTERS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			tracejant_hwc_mpi = ((enabled != NULL && !xmlStrcmp (enabled, xmlYES))) || tracejant_hwc_mpi; /* PACX may have initialized it */
#if USE_HARDWARE_COUNTERS
			mfprintf (stdout, "mpitrace: MPI routines will %scollect HW counters information.\n", tracejant_hwc_mpi?"":"NOT ");
#else
			mfprintf (stdout, "mpitrace: <%s> tag at <MPI> level will be ignored. This library does not support CPU HW.\n", TRACE_COUNTERS);
			tracejant_hwc_mpi = FALSE;
#endif
			XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <MPI> level\n", tag->name);
		}

		tag = tag->next;
	}
}
#endif

#if defined(PACX_SUPPORT)
/* Configure PACX related parameters */
static void Parse_XML_PACX (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlCOMMENT) || !xmlStrcmp (tag->name, xmlTEXT))
		{
		}
		/* Shall we gather counters in the PACX calls? */
		else if (!xmlStrcmp (tag->name, TRACE_COUNTERS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			tracejant_hwc_mpi = ((enabled != NULL && !xmlStrcmp (enabled, xmlYES))) || tracejant_hwc_mpi; /* MPI may have initialized it */
#if USE_HARDWARE_COUNTERS
			mfprintf (stdout, "mpitrace: PACX routines will %scollect HW counters information.\n", tracejant_hwc_mpi?"":"NOT ");
#else
			mfprintf (stdout, "mpitrace: <%s> tag at <PACX> level will be ignored. This library does not support CPU HW.\n", TRACE_COUNTERS);
			tracejant_hwc_mpi = FALSE;
#endif
			XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <PACX> level\n", tag->name);
		}

		tag = tag->next;
	}
}
#endif

/* Configure Callers related parameters */
static void Parse_XML_Callers (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Must the tracing facility obtain information about MPI/PACX callers? */
		else if (!xmlStrcmp (tag->name, TRACE_MPI))
		{
#if defined(MPI_SUPPORT)
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *callers = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (callers != NULL)
					Parse_Callers (rank, callers, CALLER_MPI);
				XML_FREE(callers);
			}
			XML_FREE(enabled);
#else
			mfprintf (stdout, "mpitrace: <%s> tag at <Callers> level will be ignored. This library does not support MPI.\n", TRACE_MPI);
#endif
		}
		else if (!xmlStrcmp (tag->name, TRACE_PACX))
		{
#if defined(PACX_SUPPORT)
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *callers = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (callers != NULL)
					Parse_Callers (rank, callers, CALLER_MPI);
				XML_FREE(callers);
			}
			XML_FREE(enabled);
#else
			mfprintf (stdout, "mpitrace: <%s> tag at <Callers> level will be ignored. This library does not support PACX.\n", TRACE_PACX);
#endif
		}
		/* Must the tracing facility obtain information about callers at sample points? */
		else if (!xmlStrcmp (tag->name, TRACE_SAMPLING))
		{
#if defined(SAMPLING_SUPPORT)
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *callers = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (callers != NULL)
					Parse_Callers (rank, callers, CALLER_SAMPLING);
				XML_FREE(callers);
			}
			XML_FREE(enabled);
#else
			mfprintf (stdout, "mpitrace: <%s> tag at <Callers> level will be ignored. This library does not support SAMPLING.\n", TRACE_SAMPLING);
#endif
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <callers> level\n", tag->name);
		}

		tag = tag->next;
	}
}

#if defined(IS_CELL_MACHINE)
/* Configure SPU related parameters */
static void Parse_XML_CELL (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Buffer size of the SPU tracing unit  */
		else if (!xmlStrcmp (tag->name, TRACE_SPU_BUFFERSIZE))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *str = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				spu_buffer_size = (str!=NULL)?atoi (str):-1;
				if (spu_buffer_size < 10)
				{
					mfprintf (stderr, "CELLtrace: SPU tracing buffer size '%d' too small. Using default SPU buffer size '%d'.\n", spu_buffer_size, DEFAULT_SPU_BUFFER_SIZE);
					spu_buffer_size = DEFAULT_SPU_BUFFER_SIZE;
				}
				else
				{
					mfprintf (stdout, "CELLtrace: SPU tracing buffer size is %d events.\n", spu_buffer_size);
				}
				XML_FREE(str);
			}
			XML_FREE(enabled); 
		}
		/* DMA tag configuration for bulk transferences */
		else if (!xmlStrcmp (tag->name, TRACE_SPU_DMATAG))
		{
#ifndef SPU_USES_WRITE
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *str = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				spu_dma_channel = (str!=NULL)?atoi (str):-1;
				if ((spu_dma_channel < 0) || (spu_dma_channel > 31))
				{
					mfprintf (stderr, "CELLtrace: Invalid DMA channel '%s'. Using default channel.\n", str);
					spu_dma_channel = DEFAULT_DMA_CHANNEL;
				}
				else
				{
					mfprintf (stdout, "CELLtrace: Using DMA channel %d for memory transferences.\n", spu_dma_channel);
				}
				XML_FREE(str);
			}
			XML_FREE(enabled); 
#else
			mfprintf (stdout, "CELLtrace: SPUs will write directly to disk. Ignoring tag %s\n", TRACE_SPU_DMATAG);
#endif
		}
		/* SPU hosted file size limit */
		else if (!xmlStrcmp (tag->name, TRACE_SPU_FILESIZE))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *str = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				spu_file_size = (str!=NULL)?atoi (str):-1;
				if (spu_file_size < 1)
				{
					mfprintf (stderr, "CELLtrace: SPU tracing buffer size '%d' too small. Using default SPU buffer size '%d' mbytes.\n", spu_file_size, DEFAULT_SPU_FILE_SIZE); 
					spu_file_size = DEFAULT_SPU_FILE_SIZE;
				}
				else
				{
					mfprintf (stdout, "CELLtrace: SPU tracing file size limit is %d mbytes.\n", spu_file_size);
				}
				XML_FREE(str);
			}
			XML_FREE(enabled); 
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <CELL> level\n", tag->name);
		}

		tag = tag->next;
	}
}
#endif /* IS_CELL_MACHINE */

/* Configure Bursts related parameters */
static void Parse_XML_Bursts (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}

		/* Which is the threshold for the Bursts? */
		else if (!xmlStrcmp (tag->name, TRACE_THRESHOLD))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *str = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (str != NULL)
				{
					TMODE_setBurstsThreshold (getTimeFromStr (str, TRACE_THRESHOLD, rank));
				}
				XML_FREE(str);
			}
			XML_FREE(enabled);
		}
		else if (!xmlStrcmp (tag->name, TRACE_MPI_STATISTICS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			TMODE_setBurstsStatistics (enabled != NULL && !xmlStrcmp (enabled, xmlYES));
		}
		else if (!xmlStrcmp (tag->name, TRACE_PACX_STATISTICS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			TMODE_setBurstsStatistics (enabled != NULL && !xmlStrcmp (enabled, xmlYES));
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <Bursts> level\n", tag->name);
		}

		tag = tag->next;
	}
}


/* Configure UserFunction related parameters */
static void Parse_XML_UF (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;
	char *list = (char*) xmlGetProp (current_tag, TRACE_LIST);
	if (list == NULL)
		return;

	InstrumentUFroutines (rank, list);

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Shall we gather counters in the UF calls? */
		else if (!xmlStrcmp (tag->name, TRACE_COUNTERS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			tracejant_hwc_uf = ((enabled != NULL && !xmlStrcmp (enabled, xmlYES)));
#if USE_HARDWARE_COUNTERS
			mfprintf (stdout, "mpitrace: User Function routines will %scollect HW counters information.\n", tracejant_hwc_uf?"":"NOT ");
#else
			mfprintf (stdout, "mpitrace: <%s> tag at <user-functions> level will be ignored. This library does not support CPU HW.\n", TRACE_COUNTERS);
			tracejant_hwc_uf = FALSE;
#endif
			XML_FREE(enabled);
		}
		/* Will we limit the depth of the UF calls? */
		else if (!xmlStrcmp (tag->name, TRACE_MAX_DEPTH))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if ((enabled != NULL && !xmlStrcmp (enabled, xmlYES)))
			{
				char *str = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				int depth = (str != NULL)? atoi (str): 0;
				if (depth > 0)
				{
					mfprintf (stdout, "mpitrace: Limit depth for the user functions tracing set to %u\n", depth);
					setUFMaxDepth ((unsigned int)depth);
				}
				else
					mfprintf (stdout, "mpitrace: Warning! Invalid max-depth value\n");
			}
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <UserFunctions> level\n", tag->name);
		}

		tag = tag->next;
	}
}

#if defined(OMP_SUPPORT) || defined(SMPSS_SUPPORT)
/* Configure OpenMP related parameters */
static void Parse_XML_OMP (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

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
#if defined(OMP_SUPPORT)
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			setTrace_OMPLocks ((enabled != NULL && !xmlStrcmp (enabled, xmlYES)));
			XML_FREE(enabled);
#endif
		}
		/* Shall we gather counters in the UF calls? */
		else if (!xmlStrcmp (tag->name, TRACE_COUNTERS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			tracejant_hwc_omp = ((enabled != NULL && !xmlStrcmp (enabled, xmlYES)));
#if USE_HARDWARE_COUNTERS
			mfprintf (stdout, "mpitrace: OpenMP routines will %scollect HW counters information.\n", tracejant_hwc_omp?"":"NOT");
#else
			mfprintf (stdout, "mpitrace: <%s> tag at <OpenMP> level will be ignored. This library does not support CPU HW.\n", TRACE_COUNTERS);
			tracejant_hwc_omp = FALSE;
#endif
			XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <OpenMP> level\n", tag->name);
		}

		tag = tag->next;
	}
}
#endif

/* Configure storage related parameters */
static void Parse_XML_Storage (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Does the user want to change the file size? */
		else if (!xmlStrcmp (tag->name, TRACE_SIZE))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *fsize = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (fsize != NULL)
				{
					file_size = atoi(fsize);
					if (file_size <= 0)
					{
						mfprintf (stderr, "mpitrace: Invalid file size value.\n");
					}
					else if (file_size > 0)
					{
						mfprintf (stdout, "mpitrace: Intermediate file size set to %d Mbytes.\n", file_size);
					}
				}
				XML_FREE(fsize);
			}
			XML_FREE(enabled);
		}
		/* Where must we store the intermediate files? DON'T FREE it's used below */
		else if (!xmlStrcmp (tag->name, TRACE_DIR))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				temporal_d = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
#if defined(DEAD_CODE)
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				temporal_d_mkdir = !xmlStrcmp (xmlGetProp (tag, TRACE_MKDIR), xmlYES);
#endif
			XML_FREE(enabled);
		}
		/* Where must we store the final intermediate files?  DON'T FREE it's used below */
		else if (!xmlStrcmp (tag->name, TRACE_FINAL_DIR))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				final_d = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
#if defined(DEAD_CODE)
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				final_d_mkdir = !xmlStrcmp (xmlGetProp (tag, TRACE_MKDIR), xmlYES);
#endif
			XML_FREE(enabled);
		}
#if defined(MPI_SUPPORT)
		/* Must the tracing gather the MPITs into one process? */
		else if (!xmlStrcmp (tag->name, TRACE_GATHER_MPITS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			mpit_gathering_enabled = ((enabled != NULL && !xmlStrcmp (enabled, xmlYES)));
			mfprintf (stdout, "mpitrace: All MPIT files will %s be gathered at the end of the execution!\n", mpit_gathering_enabled?"":"NOT");
			XML_FREE(enabled);
		}
#endif
		/* Obtain the MPIT prefix */
		else if (!xmlStrcmp (tag->name, TRACE_PREFIX))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *p_name = xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				strncpy (PROGRAM_NAME, p_name, sizeof(PROGRAM_NAME));
				TracePrefixFound = TRUE;
				XML_FREE(p_name);
			}
			else
			{
				/* If not enabled, just put TRACE as the program name */
				strncpy (PROGRAM_NAME, "TRACE", strlen("TRACE")+1);
				TracePrefixFound = TRUE;
			}
			XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <Storage> level\n", tag->name);
		}

		tag = tag->next;
	}
}

/* Configure buffering related parameters */
static void Parse_XML_Buffer (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Must we limit the buffer size? */
		else if (!xmlStrcmp (tag->name, TRACE_SIZE))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *bsize = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (bsize != NULL)
				{
					int size = atoi(bsize);
					buffer_size = (size<=0)?EVT_NUM:size;
					mfprintf (stdout, "mpitrace: Tracing buffer can hold %d events\n", buffer_size);
				}
				XML_FREE(bsize);
			}
			XML_FREE(enabled);
		}
		/* Do we activate the circular buffering ? */
		else if (!xmlStrcmp (tag->name, TRACE_CIRCULAR))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				mfprintf (stdout, "mpitrace: Circular buffer %s.\n", circular_buffering?"enabled":"disabled");
				circular_buffering = 1;
			}
			XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <Buffer> level\n", tag->name);
		}

		tag = tag->next;
	}
}

#if USE_HARDWARE_COUNTERS
static void Parse_XML_Counters_CPU_Sampling (int rank, xmlDocPtr xmldoc, xmlNodePtr current, int *num, char ***counters, unsigned long long **frequencies)
{
	xmlNodePtr set_tag;
	xmlChar *enabled;
	int num_sampling_hwc, i;
	unsigned long long *t_frequencies = NULL;
	char **t_counters = NULL;

	/* Parse all HWC sets, and annotate them to use them later */
	set_tag = current->xmlChildrenNode;
	num_sampling_hwc = 0;
	while (set_tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (set_tag->name, xmlTEXT) || !xmlStrcmp (set_tag->name, xmlCOMMENT))
		{
		}
		else if (!xmlStrcmp (set_tag->name, TRACE_SAMPLING))
		{
			enabled = xmlGetProp (set_tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				if (atoll ((char*) xmlGetProp (set_tag, TRACE_FREQUENCY)) > 0)
					num_sampling_hwc++;
		}
		set_tag = set_tag->next;
	}

	if (num_sampling_hwc > 0)
	{
		t_counters = (char **) malloc (sizeof(char*) * num_sampling_hwc);
		t_frequencies = (unsigned long long *) malloc (sizeof(unsigned long long) * num_sampling_hwc);
	
		/* Parse all HWC sets, and annotate them to use them later */
		set_tag = current->xmlChildrenNode;
		i = 0;
		while (set_tag != NULL && i < num_sampling_hwc)
		{
			/* Skip coments */
			if (!xmlStrcmp (set_tag->name, xmlTEXT) || !xmlStrcmp (set_tag->name, xmlCOMMENT))
			{
			}
			else if (!xmlStrcmp (set_tag->name, TRACE_SAMPLING))
			{
				enabled = xmlGetProp (set_tag, TRACE_ENABLED);
				if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
				{
					t_counters[i] = (char*) xmlNodeListGetString (xmldoc, set_tag->xmlChildrenNode, 1);
					/* t_frequencies[i] = atoll ((char*) xmlGetProp (set_tag, TRACE_FREQUENCY)); */
					t_frequencies[i] = getFactorValue (((char*) xmlGetProp (set_tag, TRACE_FREQUENCY)), "XML:: sampling frequency property>", rank);

					if (t_frequencies[i] <= 0)
					{
						mfprintf (stderr, "mpitrace: Error invalid sampling frequency (%s) for counter %s\n", (char*) xmlGetProp (set_tag, TRACE_FREQUENCY), t_counters[i]);
					}
					else
						i++;
				}
			}
			set_tag = set_tag->next;
		}
	}
	
	*num = num_sampling_hwc;
	*frequencies = t_frequencies;
	*counters = t_counters;
}

/* Configure CPU HWC information */
static void Parse_XML_Counters_CPU (int rank, xmlDocPtr xmldoc, xmlNodePtr current)
{
	xmlNodePtr set_tag;
	char **setofcounters;
	xmlChar *enabled;
	int numofcounters, res;
	int numofsets = 0;

	/* Parse all HWC sets, and annotate them to use them later */
	set_tag = current->xmlChildrenNode;
	while (set_tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (set_tag->name, xmlTEXT) || !xmlStrcmp (set_tag->name, xmlCOMMENT))
		{
		}
		else if (!xmlStrcmp (set_tag->name, TRACE_HWCSET))
		{
			/* This 'numofsets' is the pretended number of set in the XML line. 
			It will help debugging the XML when multiple sets are defined */
			numofsets++;

			enabled = xmlGetProp (set_tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				int OvfNum;
				char **OvfCounters;
				unsigned long long *OvfFrequencies;

				char *counters, *domain, *changeat_glops, *changeat_time;
				
				counters = (char*) xmlNodeListGetString (xmldoc, set_tag->xmlChildrenNode, 1);
				domain = (char*) xmlGetProp (set_tag, TRACE_HWCSET_DOMAIN);
				changeat_glops = (char*) xmlGetProp (set_tag, TRACE_HWCSET_CHANGEAT_GLOBALOPS);
				changeat_time = (char*) xmlGetProp (set_tag, TRACE_HWCSET_CHANGEAT_TIME);

				numofcounters = explode (counters, ",", &setofcounters);

				Parse_XML_Counters_CPU_Sampling (rank, xmldoc, set_tag, &OvfNum, &OvfCounters, &OvfFrequencies);

				res = HWC_Add_Set (numofsets, rank, numofcounters, setofcounters, domain, changeat_glops, changeat_time, OvfNum, OvfCounters, OvfFrequencies);

				XML_FREE(counters);
				XML_FREE(changeat_glops);
				XML_FREE(changeat_time);
				XML_FREE(domain);
			}
			XML_FREE(enabled);
		}
		set_tag = set_tag->next;
	}
}
#endif /* USE_HARDWARE_COUNTERS */

/* Configure Counters related parameters */
static void Parse_XML_Counters (int rank, int world_size, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Here we will check all the options for <counters tag>. Nowadays the only
		   available subtag is <cpu> which depends on the availability of PAPI. */
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Check if the HWC are being configured at the XML. If so, init them
	   and gather all the sets so as to usem them later. */
		else if (!xmlStrcmp (tag->name, TRACE_CPU))
		{
			xmlChar *hwc_enabled = xmlGetProp (tag, TRACE_ENABLED);
			char *hwc_startset = (char*) xmlGetProp (tag, TRACE_STARTSET);
			if (hwc_enabled != NULL && !xmlStrcmp(hwc_enabled, xmlYES))
			{
#if USE_HARDWARE_COUNTERS
				HWC_Initialize (0);

				Parse_XML_Counters_CPU (rank, xmldoc, tag);
				if (hwc_startset != NULL)
					HWC_Parse_XML_Config (rank, world_size, hwc_startset);
#else
				mfprintf (stdout, "mpitrace: <%s> tag at <%s> level will be ignored. This library does not support CPU HW.\n", TRACE_CPU, TRACE_COUNTERS);
#endif
			}
			XML_FREE(hwc_startset);
			XML_FREE(hwc_enabled);
		}
		else if (!xmlStrcmp (tag->name, TRACE_NETWORK))
		{
#if defined(TEMPORARILY_DISABLED)
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			tracejant_network_hwc = (enabled != NULL && !xmlStrcmp (enabled, xmlYES));
			mfprintf (stdout, "mpitrace: Network counters are %s.\n", tracejant_network_hwc?"enabled":"disabled");
			XML_FREE(enabled);
#endif
		}
		else if (!xmlStrcmp (tag->name, TRACE_RUSAGE))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			tracejant_rusage = (enabled != NULL && !xmlStrcmp (enabled, xmlYES));
			mfprintf (stdout, "mpitrace: Resource usage is %s at flush buffer.\n", tracejant_rusage?"enabled":"disabled");
			XML_FREE(enabled);
		}
		else if (!xmlStrcmp (tag->name, TRACE_MEMUSAGE))
		{
            xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
            tracejant_memusage = (enabled != NULL && !xmlStrcmp (enabled, xmlYES));
            mfprintf (stdout, "mpitrace: Memory usage is %s at flush buffer.\n", tracejant_memusage?"enabled":"disabled");
            XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <Counters> level\n", tag->name);
		}

		tag = tag->next;
	}
}

/* Configure <remote-control> related parameters */
static void Parse_XML_RemoteControl (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;
	int countRemotesEnabled = 0;

	/* Parse all TAGs, and annotate them to use them later */
    tag = current_tag->xmlChildrenNode;
    while (tag != NULL)
    {
        /* Skip coments */
        if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
        {
        }
        else if (!xmlStrcmp (tag->name, REMOTE_CONTROL_METHOD_MRNET))
		{
            xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
            if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
            {
				xmlChar *target = xmlGetProp (tag, RC_MRNET_TARGET);
				xmlChar *analysis = xmlGetProp (tag, RC_MRNET_ANALYSIS);
				xmlChar *start_after = xmlGetProp (tag, RC_MRNET_START_AFTER);

				countRemotesEnabled++;

#if defined(HAVE_MRNET)
				/* Configure the target trace size */
				if (target != NULL)
				{
					MRNCfg_SetTargetTraceSize(atoi(target));
				}
				/* Configure the analysis type */
				if ((analysis != NULL) && (start_after != NULL))
				{
					if (xmlStrcmp (analysis, (xmlChar*) "clustering") == 0)
					{
						MRNCfg_SetAnalysisType(MRN_ANALYSIS_CLUSTER, atoi(start_after));
					}
					else if (xmlStrcmp (analysis, (xmlChar*) "spectral") == 0)
					{
						MRNCfg_SetAnalysisType(MRN_ANALYSIS_SPECTRAL, atoi(start_after));
					}
					else
					{
						mfprintf(stderr, "mpitrace: XML Error: Value '%s' is not valid for property '<%s>%s'\n",
							analysis, REMOTE_CONTROL_METHOD_MRNET, RC_MRNET_ANALYSIS);
						exit(-1);
					}
				}
				else
				{
					mfprintf(stderr, "mpitrace: XML error: Properties %s and %s are required for tag <%s>\n",
						RC_MRNET_ANALYSIS, RC_MRNET_START_AFTER, REMOTE_CONTROL_METHOD_MRNET);
					exit(-1);
				}
				/* Setup signals to pause/resume the application */
				Signals_SetupPauseAndResume (SIGUSR1, SIGUSR2);
				/* Activate the MRNet */
				Enable_MRNet();
#else
				mfprintf(stdout, "mpitrace: XML Warning: Remote control mechanism set to \"MRNet\" but this library does not support it.\n");
#endif 
				XML_FREE(target);
				XML_FREE(analysis);
				XML_FREE(start_after);
			}
			XML_FREE(enabled);
		}
        else if (!xmlStrcmp (tag->name, REMOTE_CONTROL_METHOD_SIGNAL))
		{
            xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
            if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
            {
				fprintf(stderr, "Parsing SIGNAL XML SUBSEC\n");
				countRemotesEnabled++;

				/* Which SIGNAL will we use to interrupt the tracing */
				xmlChar *which = xmlGetProp (tag,  RC_SIGNAL_WHICH);
				if (which != NULL)
				{
					if ((xmlStrcmp (which, (xmlChar*) "USR1") == 0) || (xmlStrcmp (which, (xmlChar*) "") == 0))
					{
						mfprintf (stdout, "mpitrace: Signal USR1 will flush buffers to disk and stop further tracing\n");
						Signals_SetupFlushAndTerminate (SIGUSR1);
					}
					else if (xmlStrcmp (which, (xmlChar *) "USR2") == 0)
					{
						mfprintf (stdout, "mpitrace: Signal USR2 will flush buffers to disk and stop further tracing\n");
						Signals_SetupFlushAndTerminate (SIGUSR2);
					}
					else
					{
						mfprintf (stderr, "mpitrace: XML Error: Value '%s' is not valid for property '<%s>%s'\n", 
							which, REMOTE_CONTROL_METHOD_SIGNAL, RC_SIGNAL_WHICH);
					}
				}
				XML_FREE(which);
			}
			XML_FREE(enabled);
		}
		tag = tag->next;
	}
	if (countRemotesEnabled > 1)
	{
		mfprintf (stderr, "mpitrace: XML error: Only 1 remote control mechanism can be active at a time at <%s>\n", TRACE_REMOTE_CONTROL);
		exit(-1);
	}
}

/* Configure <others> related parameters */
static void Parse_XML_TraceControl (int rank, int world_size, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Must we check for a control file? */
		else if (!xmlStrcmp (tag->name, TRACE_CONTROL_FILE))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *c_file = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (c_file != NULL)
				{
					char *tmp;

					CheckForControlFile = TRUE;
					strcpy (ControlFileName, c_file);
					mfprintf (stdout, "mpitrace: Control file is '%s'. Tracing will be disabled until the file exists.\n", c_file);

					/* Let the user tune how often will be checked the existence of the control file */
					tmp = (char*) xmlGetProp (tag, TRACE_FREQUENCY);
					if (tmp != NULL)
					{
						WantedCheckControlPeriod = getTimeFromStr (tmp, TRACE_FREQUENCY, rank);
						if (WantedCheckControlPeriod >= 1000000000)
						{
							mfprintf (stdout, "mpitrace: Control file will be checked every %llu seconds\n", WantedCheckControlPeriod / 1000000000);
						}
						else if (WantedCheckControlPeriod < 1000000000 && WantedCheckControlPeriod > 0)
						{
							mfprintf (stdout, "mpitrace: Control file will be checked every %llu nanoseconds\n", WantedCheckControlPeriod);
						}
					}
					XML_FREE(tmp);
				}
				XML_FREE(c_file);
			}
			XML_FREE(enabled);
		}
		/* Must we check for global-ops counters? */
		else if (!xmlStrcmp (tag->name, TRACE_CONTROL_GLOPS))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
#if defined(MPI_SUPPORT)
				char *trace_intervals = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (trace_intervals != NULL)
				{
					CheckForGlobalOpsTracingIntervals = TRUE;
					Parse_GlobalOps_Tracing_Intervals (trace_intervals);
					XML_FREE(trace_intervals);
				}
#else
				mfprintf (stdout, "mpitrace: Warning! <%s> tag will be ignored. This library does not support MPI.\n", TRACE_CONTROL_GLOPS);
#endif
			}
			XML_FREE(enabled);
		}
		else if (!xmlStrcmp (tag->name, TRACE_REMOTE_CONTROL))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				Parse_XML_RemoteControl (rank, xmldoc, tag);

#if defined(DEAD_CODE)
				xmlChar *method = xmlGetProp (tag, TRACE_REMOTE_CONTROL_METHOD);
				if (method != NULL && !xmlStrcmp (method, TRACE_REMOTE_CONTROL_MRNET))
				{
#if defined(HAVE_MRNET)
					Signals_SetupPauseAndResume (SIGUSR1, SIGUSR2);
					Enable_MRNet();
#else
					mfprintf(stdout, "mpitrace: Remote control set to \"mrnet\" but MRNet is not supported.\n");
#endif 
				}
				else if (method != NULL && !xmlStrcmp (method, TRACE_REMOTE_CONTROL_SIGNAL))
				{
					/* Which SIGNAL will we use to interrupt the tracing */
					xmlChar *str = xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
					if (str != NULL)
					{
						if ((xmlStrcmp (str, (xmlChar*) "USR1") == 0) || (xmlStrcmp (str, (xmlChar*) "") == 0))
						{
							mfprintf (stdout, "mpitrace: Signal USR1 will flush the buffers to the disk and stop further tracing\n");
							Signals_SetupFlushAndTerminate (SIGUSR1);
						}
						else if (xmlStrcmp (str, (xmlChar *) "USR2") == 0)
						{
							mfprintf (stdout, "mpitrace: Signal USR2 will flush the buffers to the disk and stop further tracing\n");
							Signals_SetupFlushAndTerminate (SIGUSR2);
						}
						else
						{
							mfprintf (stderr, "mpitrace: Error value '%s' is not a valid one for %s tag\n", str, TRACE_REMOTE_CONTROL);
						}
					}
					XML_FREE(str);
				}
				XML_FREE(method);
#endif
			}
			XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <%s> level\n", tag->name, TRACE_CONTROL);
		}

		tag = tag->next;
	}
}

/* Configure <others> related parameters */
static void Parse_XML_Others (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
	xmlNodePtr tag;

	/* Parse all TAGs, and annotate them to use them later */
	tag = current_tag->xmlChildrenNode;
	while (tag != NULL)
	{
		/* Skip coments */
		if (!xmlStrcmp (tag->name, xmlTEXT) || !xmlStrcmp (tag->name, xmlCOMMENT))
		{
		}
		/* Must the trace run for at least some time? */
		else if (!xmlStrcmp (tag->name, TRACE_MINIMUM_TIME))
		{
			xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
			if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
			{
				char *str = (char*) xmlNodeListGetString (xmldoc, tag->xmlChildrenNode, 1);
				if (str != NULL)
				{
					MinimumTracingTime = getTimeFromStr (str, TRACE_MINIMUM_TIME, rank);
					hasMinimumTracingTime = ( MinimumTracingTime != 0);
					if (MinimumTracingTime >= 1000000000)
					{
						mfprintf (stdout, "mpitrace: Minimum tracing time will be %llu seconds\n", MinimumTracingTime / 1000000000);
					}
					else if (MinimumTracingTime < 1000000000 && MinimumTracingTime > 0)
					{
						mfprintf (stdout, "mpitrace: Minimum tracing time will be %llu nanoseconds\n", MinimumTracingTime);
					}
				}
				XML_FREE(str);
			}
			XML_FREE(enabled);
		}
		else
		{
			mfprintf (stderr, "mpitrace: XML unknown tag '%s' at <Others> level\n", tag->name);
		}

		tag = tag->next;
	}
}

void Parse_XML_File (int rank, int world_size, char *filename)
{
	xmlNodePtr current_tag;
	xmlDocPtr  xmldoc;
	xmlNodePtr root_tag;
	char cwd[TMP_DIR];

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
				mfprintf (stderr, "mpitrace: Invalid configuration file\n");
			}
			else
			{
				/*
					 Are the MPI & OpenMP tracing enabled? Is the tracing fully enabled? 
					 try to remember if these values have been changed previously by the
					 API routines.
				*/

				/* Full tracing control */
				char *tracehome = (char*) xmlGetProp (root_tag, TRACE_HOME);
				xmlChar *xmlparserid = xmlGetProp (root_tag, TRACE_PARSER_ID);
				xmlChar *traceenabled = xmlGetProp (root_tag, TRACE_ENABLED);
				xmlChar *traceinitialmode = xmlGetProp (root_tag, TRACE_INITIAL_MODE);
				xmlChar *tracetype = xmlGetProp (root_tag, TRACE_TYPE);
				mpitrace_on = (traceenabled != NULL) && !xmlStrcmp (traceenabled, xmlYES);

				if (!mpitrace_on)
				{
					mfprintf (stdout, "mpitrace: Application has been linked or preloaded with mpitrace, BUT tracing is NOT set!\n");
				}
				else
				{
					/* Where is the tracing located? If defined, copy to the correct buffer! */
					if (xmlStrcmp (&rcsid[1], xmlparserid)) /* Skip first $ char */
					{
						mfprintf (stderr, "mpitrace: WARNING!\n");
						mfprintf (stderr, "mpitrace: WARNING! XML parser version and property '%s' do not match. Check the XML file. Trying to proceed...\n", TRACE_PARSER_ID);
						mfprintf (stderr, "mpitrace: WARNING!\n");
					}

					if (tracehome != NULL)
					{
						strncpy (trace_home, tracehome, TMP_DIR);
						mfprintf (stdout, "mpitrace: Tracing package is located on %s\n", trace_home);
					}
					else
					{
						mfprintf (stdout, "mpitrace: Warning! <%s> tag has no <%s> property defined.\n", TRACE_TAG, TRACE_HOME);
					}

					if (traceinitialmode != NULL)
					{
						if (!xmlStrcmp (traceinitialmode, TRACE_INITIAL_MODE_DETAIL))
						{
							TMODE_setInitial (TRACE_MODE_DETAIL);
						}
						else if (!xmlStrcmp (traceinitialmode, TRACE_INITIAL_MODE_BURSTS))
						{
							TMODE_setInitial (TRACE_MODE_BURSTS);
						}
						else
						{
							mfprintf (stdout, "mpitrace: Warning! Invalid value '%s' for property <%s> in tag <%s>.\n", traceinitialmode, TRACE_INITIAL_MODE, TRACE_TAG);
							TMODE_setInitial (TRACE_MODE_DETAIL);
						}
					}
					else
					{
						mfprintf (stdout, "mpitrace: Warning! Not given value for property <%s> in tag <%s>.\n", TRACE_INITIAL_MODE, TRACE_TAG);
						TMODE_setInitial (TRACE_MODE_DETAIL);
					}

					if (tracetype != NULL)
					{
						if (!xmlStrcmp (tracetype, TRACE_TYPE_PARAVER))
						{
							mfprintf (stdout, "mpitrace: Generating intermediate files for Paraver traces.\n");
							Clock_setType (REAL_CLOCK);
						}
						else if (!xmlStrcmp (tracetype, TRACE_TYPE_DIMEMAS))
						{
							mfprintf (stdout, "mpitrace: Generating intermediate files for Dimemas traces.\n");
							Clock_setType (USER_CLOCK);
						}
						else
						{
							mfprintf (stdout, "mpitrace: Warning! Invalid value '%s' for property <%s> in tag <%s>.\n", tracetype, TRACE_TYPE, TRACE_TAG);
							Clock_setType (REAL_CLOCK);
						}
					}
					else
					{
						mfprintf (stdout, "mpitrace: Warning! Not given value for property <%s> in tag <%s>.\n", TRACE_TYPE, TRACE_TAG);
						Clock_setType (REAL_CLOCK);
					}
				}
				XML_FREE(xmlparserid);
				XML_FREE(tracetype);
				XML_FREE(traceinitialmode);
				XML_FREE(traceenabled);
				XML_FREE(tracehome);

				current_tag = root_tag->xmlChildrenNode;
				while (current_tag != NULL && mpitrace_on)
				{
					/* Skip coments */
					if (!xmlStrcmp (current_tag->name, xmlTEXT) || !xmlStrcmp (current_tag->name, xmlCOMMENT))
					{
					}
					/* UF related information instrumentation */
					else if (!xmlStrcmp (current_tag->name, TRACE_USERFUNCTION))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_UF (rank, xmldoc, current_tag);
						XML_FREE(enabled);
					}
					/* Callers related information instrumentation */
					else if (!xmlStrcmp (current_tag->name, TRACE_CALLERS))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_Callers (rank, xmldoc, current_tag);
						XML_FREE(enabled);
					}
					/* MPI related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_MPI))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
						{
#if defined(MPI_SUPPORT)
							tracejant_mpi = TRUE;
							Parse_XML_MPI (rank, xmldoc, current_tag);
#else
							mfprintf (stdout, "mpitrace: Warning! <%s> tag will be ignored. This library does not support MPI.\n", TRACE_MPI);
							tracejant_mpi = FALSE || tracejant_mpi; /* May be initialized at PACX */
#endif
						}
						else
							tracejant_mpi = FALSE || tracejant_mpi; /* May be initialized at PACX */
						XML_FREE(enabled);
					}
					/* PACX related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_PACX))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
						{
#if defined(PACX_SUPPORT)
							tracejant_mpi = TRUE;
							Parse_XML_PACX (rank, xmldoc, current_tag);
#else
							mfprintf (stdout, "mpitrace: Warning! <%s> tag will be ignored. This library does not support PACX.\n", TRACE_PACX);
							tracejant_mpi = FALSE || tracejant_mpi; /* May be initialized at MPI */
#endif
						}
						else
							tracejant_mpi = FALSE || tracejant_mpi; /* May be initialized at MPI */
						XML_FREE(enabled);
					}
					/* Bursts related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_BURSTS))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
						{
							Parse_XML_Bursts (rank, xmldoc, current_tag);
						}
						XML_FREE(enabled);
					}
					/* OpenMP related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_OMP))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
						{
#if defined(OMP_SUPPORT) || defined(SMPSS_SUPPORT)
							tracejant_omp = TRUE;
							Parse_XML_OMP (rank, xmldoc, current_tag);
#else
							mfprintf (stdout, "mpitrace: Warning! <%s> tag will be ignored. This library does not support OpenMP.\n", TRACE_OMP);
							tracejant_omp = FALSE;
#endif
						}
						else
							tracejant_omp = FALSE;
						XML_FREE(enabled);
					}
					/* SPU related configuration*/
					else if (!xmlStrcmp (current_tag->name, TRACE_CELL))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
						{
#if defined(IS_CELL_MACHINE)
							cell_tracing_enabled = TRUE;
#if 0 /* unimplemented right now */
							Parse_XML_SPU (rank, xmldoc, current_tag);
#endif
#else
							mfprintf (stdout, "mpitrace: Warning! <%s> tag will be ignored. This library does not support Cell BE processors.\n", TRACE_CELL);
#endif
						}
#if defined(IS_CELL_MACHINE)
						else
							cell_tracing_enabled = FALSE;
#endif
						XML_FREE(enabled);
					}
					/* Storage related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_STORAGE))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_Storage (rank, xmldoc, current_tag);
						XML_FREE(enabled);
					}
					/* Buffering related configuration */
					else if (!xmlStrcmp (current_tag->name, TRACE_BUFFER))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_Buffer (rank, xmldoc, current_tag);
						XML_FREE(enabled);
					}
					/* Check if some other configuration info must be gathered */
					else if (!xmlStrcmp (current_tag->name, TRACE_OTHERS))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_Others (rank, xmldoc, current_tag);
						XML_FREE(enabled);
					}
					/* Check if some kind of counters must be gathered */
					else if (!xmlStrcmp (current_tag->name, TRACE_COUNTERS))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_Counters (rank, world_size, xmldoc, current_tag);
						XML_FREE(enabled);
					}
					/* Check for tracing control */
					else if (!xmlStrcmp (current_tag->name, TRACE_CONTROL))
					{
						xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
						if (enabled != NULL && !xmlStrcmp (enabled, xmlYES))
							Parse_XML_TraceControl (rank, world_size, xmldoc, current_tag);
						XML_FREE(enabled);
					}
					else
					{
						mfprintf (stderr, "mpitrace: Warning! XML unknown tag '%s'\n", current_tag->name);
					}

					current_tag = current_tag->next;
				}
			}
		}
		else
		{
			mfprintf (stderr, "mpitrace: Error! Empty mpitrace configuration file\n");
		}

		xmlFreeDoc (xmldoc);
	}

	/* If the tracing has been enabled, continue with the configuration */
	if (mpitrace_on)
	{
		char *res_cwd;

		/* Have the user proposed any program name? */
		if (!TracePrefixFound)
			strncpy (PROGRAM_NAME, "TRACE", strlen("TRACE")+1);

		/* Just gather now the variables related to the directories! */

		/* Start obtaining the current directory */
		res_cwd = getcwd (cwd, sizeof(cwd));

		/* Temporal directory must be checked against the configuration of the XML,
    	  and the current directory */
		if (temporal_d == NULL)
			if ((temporal_d = res_cwd) == NULL)
				temporal_d = ".";
		strcpy (tmp_dir, temporal_d);
		/* Force mkdir */
#if defined(DEAD_CODE)
		if (temporal_d_mkdir)
#endif
		mkdir_recursive (tmp_dir);

		/* Final directory must be checked against the configuration of the XML, 
  	    the temporal_directory and, finally, the current directory */
		if (final_d == NULL)
			final_d = temporal_d;

		if (strlen(final_d) > 0)
		{
			if (final_d[0] != '/')
				sprintf (final_dir, "%s/%s", res_cwd, final_d);
			else
				strcpy (final_dir, final_d);
		}
		else
			strcpy (final_dir, final_d);

		/* Force mkdir */
#if defined(DEAD_CODE)
		if (final_d_mkdir)
#endif
		mkdir_recursive (final_dir);

		if (strcmp (final_dir, tmp_dir) != 0)
		{
			mfprintf (stdout, "mpitrace: Temporal directory for the intermediate traces is %s\n", tmp_dir);
			mfprintf (stdout, "mpitrace: Final directory for the intermediate traces is %s\n", final_dir);
		}
		else
		{
			mfprintf (stdout, "mpitrace: Intermediate traces will be stored in %s\n", tmp_dir);
		}
	}
}

#endif /* HAVE_XML2 */
