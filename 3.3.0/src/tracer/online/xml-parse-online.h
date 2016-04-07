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

#ifndef __XML_PARSE_ONLINE_H__
#define __XML_PARSE_ONLINE_H__

#include <libxml/xmlmemory.h>
#include <libxml/parser.h>

#include "xml-parse.h" 

#if defined(__cplusplus)
extern "C" {
#endif

void Parse_XML_Online (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag);

void Parse_XML_Online_From_File (char *filename);

#if defined(HAVE_SPECTRAL)
void Parse_XML_SpectralAdvanced (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag);
#endif

#if defined(__cplusplus)
}
#endif

#endif /* __XML_PARSE_ONLINE_H__ */
