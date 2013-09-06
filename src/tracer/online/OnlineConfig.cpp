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

#include "OnlineConfig.h"

static int   OnlineUserEnabled = 0;                                     /* Controls if the online module is enabled     */

static int         OnlineType = ONLINE_DO_NOTHING;                      /* Controls the analysis that will be performed */
static int         OnlineFreq = DEFAULT_ANALYSIS_FREQUENCY;             /* Controls how often the analysis triggers     */
static const char *OnlineTopo = DEFAULT_TOPOLOGY;                       /* Specify the network topology                 */

#if defined(HAVE_SPECTRAL)
static int    cfgSpectralMaxPeriods  = DEFAULT_SPECTRAL_MAX_PERIODS;    /* How many (different) types of periodic behavior to look for      */
static int    cfgSpectralMinSeen     = DEFAULT_SPECTRAL_MIN_SEEN;       /* How many times a given behavior has to be seen before tracing it */
static int    cfgSpectralNumIters    = DEFAULT_SPECTRAL_NUM_ITERS;      /* How many iterations to trace for a given behavior                */
static double cfgSpectralMinLikeness = DEFAULT_SPECTRAL_MIN_LIKENESS;   /* Minimum similarity to consider to periods equivalent             */
#endif /* HAVE_SPECTRAL */

#if defined(HAVE_CLUSTERING)
static char *cfgClusteringConfig     = (char *)DEFAULT_CLUSTERING_CONFIG;       /* The clustering configuration xml file */
#endif /* HAVE_CLUSTERING */

/**
 * Enables the online module.
 */
void Online_Enable() 
{
  OnlineUserEnabled = 1;
}

/**
 * Disables the online module.
 */
void Online_Disable()
{
  OnlineUserEnabled = 0;
}

/**
 * Checks whether the online module is enabled.
 * @return 1 if enabled; 0 otherwise.
 */
int Online_isEnabled()
{
  return OnlineUserEnabled;
}

/**
 * Set the type of analysis to perform.
 */
void Online_SetAnalysis(int analysis_type)
{
  OnlineType = analysis_type;
}

/**
 * Check the type of analyisis that is being performed.
 * @return The type of analysis.
 */
int Online_GetAnalysis()
{
  return OnlineType;
}

/**
 * Set the frequency to repeat the analysis.
 */
void Online_SetFrequency(int seconds) 
{
  OnlineFreq = seconds;
}

/**
 * Check the frequency of analysis.
 * @return The analysis frequency.
 */
int Online_GetFrequency()
{
  return OnlineFreq;
}

/**
 * Set the topology for the reduction network.
 * @param topology The topology in mrnet_topgen format.
 */
void Online_SetTopology(char *topology)
{
  OnlineTopo = topology;
}

/**
 * Returns the reduction network topology.
 * @return The selected topology in mrnet_topgen format.
 */
char * Online_GetTopology()
{
  return (char *)OnlineTopo;
} 

/*****************************************************************\
|***                  SPECTRAL CONFIGURATION                   ***|
\*****************************************************************/
#if defined(HAVE_SPECTRAL)

void Online_SetSpectralMaxPeriods( int max_periods )
{
  if (max_periods >= 0)
    cfgSpectralMaxPeriods = max_periods;
}

void Online_SetSpectralMinSeen( int min_seen )
{
  if (min_seen >= 0)
    cfgSpectralMinSeen = min_seen;
}

void Online_SetSpectralNumIters( int num_iters )
{
  cfgSpectralNumIters = MAX(2, num_iters);
}

void Online_SetSpectralMinLikeness( double min_likeness )
{
  cfgSpectralMinLikeness = MAX(0, min_likeness);
  cfgSpectralMinLikeness = MIN(cfgSpectralMinLikeness, 1);
}

int Online_GetSpectralMaxPeriods (void)
{
  return cfgSpectralMaxPeriods;
}

int Online_GetSpectralMinSeen    (void)
{
  return cfgSpectralMinSeen;
}

int Online_GetSpectralNumIters   (void)
{
  return cfgSpectralNumIters;
}

double Online_GetSpectralMinLikeness(void)
{
  return cfgSpectralMinLikeness;
}

#endif /* HAVE_SPECTRAL */

#if defined(HAVE_CLUSTERING)

void Online_SetClusteringConfig( char *clustering_config_xml )
{
  cfgClusteringConfig = strdup(clustering_config_xml);
}

char * Online_GetClusteringConfig()
{
  return cfgClusteringConfig;
}

#endif /* HAVE_CLUSTERING */

