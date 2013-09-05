#include "common.h"

#include <stdlib.h>
#include <string.h>

#include "OnlineConfig.h"

#include "xml-parse-online.h"


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
    Online_SetFrequency(atoi((const char *)frequency));
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
    }
#endif
    tag = tag->next;
  }
}

