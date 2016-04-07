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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "OnlineConfig.h"
#include "xml-parse-online.h"
#include "utils.h"


/**
 * Skip all the way up to the on-line section 
 */
void Parse_XML_Online_From_File (char *filename)
{
  xmlDocPtr  xmldoc;
  xmlNodePtr current_tag;
  xmlNodePtr root_tag;

  int online_config_parsed = 0;

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

      current_tag = root_tag->xmlChildrenNode;
      while ((current_tag != NULL) && (!online_config_parsed))
      {
        if (!xmlStrcasecmp (current_tag->name, TRACE_CONTROL))
        {
          xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
          if (enabled != NULL && !xmlStrcasecmp (enabled, xmlYES))
          {
            current_tag = current_tag->xmlChildrenNode;
          }
          XML_FREE(enabled);
        }
        else if (!xmlStrcasecmp (current_tag->name, TRACE_REMOTE_CONTROL))
        {
          xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
          if (enabled != NULL && !xmlStrcasecmp (enabled, xmlYES))
          {
            current_tag = current_tag->xmlChildrenNode;
          }
          XML_FREE(enabled);
        }
        else if (!xmlStrcasecmp (current_tag->name, REMOTE_CONTROL_METHOD_ONLINE))
        {
          xmlChar *enabled = xmlGetProp (current_tag, TRACE_ENABLED);
          if (enabled != NULL && !xmlStrcasecmp (enabled, xmlYES))
          {
            /* Enable the on-line analysis */
            Online_Enable();

            /* Parse the on-line analysis configuration */
            Parse_XML_Online(0, xmldoc, current_tag);
            online_config_parsed = 1;
          }
          else
          {
            return;
          }
        }
        else
        {
          current_tag = current_tag->next;
        }
      }
    }
  }
}


void Parse_XML_Online (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
  xmlNodePtr tag;

  tag = current_tag;

  xmlChar *analysis  = xmlGetProp (tag, RC_ONLINE_TYPE);
  xmlChar *frequency = xmlGetProp (tag, RC_ONLINE_FREQ);
  xmlChar *topology  = xmlGetProp (tag, RC_ONLINE_TOPO);

  /* Configure the type of analysis */
  if (analysis != NULL)
  {
    if (xmlStrcasecmp(analysis, RC_ONLINE_CLUSTERING) == 0)
    {
      Online_SetAnalysis(ONLINE_DO_CLUSTERING);
    }
    else if (xmlStrcasecmp(analysis, RC_ONLINE_SPECTRAL) == 0)
    {
      Online_SetAnalysis(ONLINE_DO_SPECTRAL);
    }
    else if (xmlStrcasecmp(analysis, RC_ONLINE_GREMLINS) == 0)
    {
      Online_SetAnalysis(ONLINE_DO_GREMLINS);
    }
    else
    {
      mfprintf(stderr, PACKAGE_NAME": XML Error: Value '%s' is not valid for property '<%s>'%s'\n",
        analysis, REMOTE_CONTROL_METHOD_ONLINE, RC_ONLINE_TYPE);
      exit(-1);
    }
  }

  /* Configure the frequency of the analysis */
  if (frequency != NULL)
  {
    Online_SetFrequencyString(frequency);
  }

  /* Configure the topology */
  if (topology != NULL)
  {
    Online_SetTopology((char *)topology);
  }
  XML_FREE(analysis);
  XML_FREE(frequency);

  tag = current_tag->xmlChildrenNode;
  while (tag != NULL)
  {
    if (!xmlStrcasecmp (tag->name, xmlTEXT) || !xmlStrcasecmp (tag->name, xmlCOMMENT)) { /* Skip coments */ }
#if defined(HAVE_CLUSTERING)
    else if (!xmlStrcasecmp (tag->name, RC_ONLINE_CLUSTERING))
    {
      xmlChar *clustering_config_str = xmlGetProp (tag, CLUSTERING_CONFIG);

      Online_SetClusteringConfig( (char *)clustering_config_str );

      XML_FREE(clustering_config_str);
    }
#endif
#if defined(HAVE_SPECTRAL)
    else if (!xmlStrcasecmp (tag->name, RC_ONLINE_SPECTRAL))
    {
      xmlChar *max_periods_str  = xmlGetProp (tag, SPECTRAL_MAX_PERIODS);
      xmlChar *num_iters_str    = xmlGetProp (tag, SPECTRAL_NUM_ITERS);
      xmlChar *min_seen_str     = xmlGetProp (tag, SPECTRAL_MIN_SEEN);
      xmlChar *min_likeness_str = xmlGetProp (tag, SPECTRAL_MIN_LIKENESS);
      int max_periods  = 0;

      if (max_periods_str != NULL)
      {
        if (strcmp((const char *)max_periods_str, "all") == 0) max_periods = 0;
        else max_periods = atoi((const char *)max_periods_str);
        Online_SetSpectralMaxPeriods ( max_periods );
      }
      if (num_iters_str    != NULL) 
      {
        Online_SetSpectralNumIters   ( atoi((const char *)num_iters_str) );
      }
      if (min_seen_str     != NULL) 
      {
        Online_SetSpectralMinSeen    ( atoi((const char *)min_seen_str) );
      }
      if (min_likeness_str != NULL) 
      {
        Online_SetSpectralMinLikeness( (atof((const char *)min_likeness_str) / 100.0) );
      }

      XML_FREE(max_periods_str);
      XML_FREE(num_iters_str);
      XML_FREE(min_seen_str);
      XML_FREE(min_likeness_str);

      Parse_XML_SpectralAdvanced(rank, xmldoc, tag->xmlChildrenNode);
    }
#endif
    else if (!xmlStrcasecmp (tag->name, RC_ONLINE_GREMLINS))
    {
      xmlChar *start_str     = xmlGetProp(tag, GREMLINS_START);
      xmlChar *increment_str = xmlGetProp(tag, GREMLINS_INCREMENT);
      xmlChar *roundtrip_str = xmlGetProp(tag, GREMLINS_ROUNDTRIP);
      xmlChar *loop_str      = xmlGetProp(tag, GREMLINS_LOOP);

      if (start_str != NULL)
      {
        Online_SetGremlinsStartCount ( atoi((const char *)start_str) );
      }
      if (increment_str != NULL)
      {
        Online_SetGremlinsIncrement ( atoi((const char *)increment_str) );
      }
      if (roundtrip_str != NULL)
      {
        if (strcmp((const char *)roundtrip_str, "yes") == 0)
        {
          Online_SetGremlinsRoundtrip ( 1 );
        }
      }
      if (loop_str != NULL)
      {
        if (strcmp((const char *)loop_str, "yes") == 0)
        {
          Online_SetGremlinsLoop( 1 );
        }
      }
    }
    tag = tag->next;
  }
}

#if defined(HAVE_SPECTRAL)
void Parse_XML_SpectralAdvanced (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag)
{
  xmlNodePtr tag, tag2;

  tag = current_tag;

  while (tag != NULL)
  {
    if (!xmlStrcasecmp (tag->name, xmlTEXT) || !xmlStrcasecmp (tag->name, xmlCOMMENT)) { /* Skip coments */ }

    else if (!xmlStrcasecmp (tag->name, RC_ONLINE_SPECTRAL_ADVANCED))
    {
      xmlChar *enabled = xmlGetProp (tag, TRACE_ENABLED);
      if (enabled != NULL && !xmlStrcasecmp (enabled, xmlYES))
      {
        xmlChar *burst_threshold = xmlGetProp (tag, SPECTRAL_BURST_THRESHOLD);
        Online_SetSpectralBurstThreshold( atof((const char *)burst_threshold) );
        XML_FREE(burst_threshold);

        tag2 = tag->xmlChildrenNode;

        while (tag2 != NULL)
        {
          if (!xmlStrcasecmp (tag2->name, xmlTEXT) || !xmlStrcasecmp (tag2->name, xmlCOMMENT)) { /* Skip coments */ }
          else if (!xmlStrcasecmp (tag2->name, RC_ONLINE_SPECTRAL_ADVANCED_PERIODIC_ZONE))
          {
            xmlChar *detail_level = xmlGetProp (tag2, SPECTRAL_DETAIL_LEVEL);

            Online_SetSpectralPeriodZoneLevel( detail_level );

            XML_FREE(detail_level);
          }
          else if (!xmlStrcasecmp (tag2->name, RC_ONLINE_SPECTRAL_ADVANCED_NON_PERIODIC_ZONE))
          {
            xmlChar *detail_level = xmlGetProp (tag2, SPECTRAL_DETAIL_LEVEL);
            xmlChar *min_duration = xmlGetProp (tag2, SPECTRAL_MIN_DURATION);

            Online_SetSpectralNonPeriodZoneLevel( detail_level );
            Online_SetSpectralNonPeriodZoneMinDuration( getTimeFromStr( min_duration, "<non_periodic_zone min_duration=\"..\" />", rank ) );

            XML_FREE(detail_level);
            XML_FREE(min_duration);
          }
          tag2 = tag2->next;
        }
      }
      XML_FREE(enabled);
    }
    tag = tag->next;
  }
}
#endif

