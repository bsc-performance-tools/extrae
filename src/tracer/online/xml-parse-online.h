#ifndef __ONLINE_XML_H__
#define __ONLINE_XML_H__

#include <libxml/xmlmemory.h>
#include <libxml/parser.h>

#include "xml-parse.h" 

#if defined(__cplusplus)
extern "C" {
#endif

void Parse_XML_Online (int rank, xmlDocPtr xmldoc, xmlNodePtr current_tag);

void Parse_XML_Online_From_File (char *filename);

#if defined(__cplusplus)
}
#endif

#endif /* __ONLINE_XML_H__ */
