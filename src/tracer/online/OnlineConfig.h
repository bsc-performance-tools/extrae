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

#ifndef __ONLINE_CONFIG_H__
#define __ONLINE_CONFIG_H__

#include <config.h>

#define DEFAULT_ANALYSIS_FREQUENCY 60
#define DEFAULT_TOPOLOGY           "auto"

enum
{
  ONLINE_DO_NOTHING,
  ONLINE_DO_CLUSTERING,
  ONLINE_DO_SPECTRAL,
  ONLINE_DO_GREMLINS
};

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

void   Online_Enable( void );
void   Online_Disable( void );
int    Online_isEnabled( void );
int    Online_GetAnalysis( void );
void   Online_SetAnalysis( int analyis_type );
int    Online_GetFrequency( void );
void   Online_SetFrequency( int seconds );
void   Online_SetFrequencyString( char *seconds_str );
void   Online_UpdateFrequency( int pct );
char * Online_GetTopology( void );
void   Online_SetTopology( char *topology );

#if defined(HAVE_SPECTRAL)
#define DEFAULT_SPECTRAL_MAX_PERIODS          1
#define DEFAULT_SPECTRAL_MIN_SEEN             0
#define DEFAULT_SPECTRAL_NUM_ITERS            3
#define DEFAULT_SPECTRAL_MIN_LIKENESS         0.80
#define DEFAULT_SPECTRAL_NP_ZONE_MIN_DURATION 1000000000
#define DEFAULT_SPECTRAL_BURST_THRESHOLD      80

int                Online_GetSpectralMaxPeriods( void );
int                Online_GetSpectralMinSeen( void );
int                Online_GetSpectralNumIters( void );
double             Online_GetSpectralMinLikeness( void );
int                Online_GetSpectralPeriodZoneLevel( void );
int                Online_GetSpectralNonPeriodZoneLevel( void );
unsigned long long Online_GetSpectralNonPeriodZoneMinDuration( void );
double             Online_GetSpectralBurstThreshold( void );
void               Online_SetSpectralMaxPeriods( int max_periods );
void               Online_SetSpectralMinSeen( int min_seen );
void               Online_SetSpectralNumIters( int num_iters );
void               Online_SetSpectralMinLikeness( double min_likeness );
void               Online_SetSpectralPeriodZoneLevel( char *detail_level );
void               Online_SetSpectralNonPeriodZoneLevel( char *detail_level );
void               Online_SetSpectralNonPeriodZoneMinDuration( unsigned long long min_duration );
void               Online_SetSpectralBurstThreshold( double burst_threshold );


#endif /* HAVE_SPECTRAL */

#if defined(HAVE_CLUSTERING)
#define DEFAULT_CLUSTERING_CONFIG "cl.I.IPC.xml"

char * Online_GetClusteringConfig( void );
void   Online_SetClusteringConfig( char *clustering_config_xml );
#endif /* HAVE_CLUSTERING */

#define DEFAULT_GREMLINS_START_COUNT 0
#define DEFAULT_GREMLINS_INCREMENT   1
#define DEFAULT_GREMLINS_ROUNDTRIP   0
#define DEFAULT_GREMLINS_LOOP        0

int                Online_GetGremlinsStartCount( void );
int                Online_GetGremlinsIncrement( void );
int                Online_GetGremlinsRoundtrip( void );
int                Online_GetGremlinsLoop( void );
void               Online_SetGremlinsStartCount( int start_count );
void               Online_SetGremlinsIncrement( int increment );
void               Online_SetGremlinsRoundtrip( int enabled );
void               Online_SetGremlinsLoop( int enabled );

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* __ONLINE_CONFIG_H__ */
