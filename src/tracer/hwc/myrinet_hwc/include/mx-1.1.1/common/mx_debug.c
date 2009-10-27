/*************************************************************************
 * The contents of this file are subject to the MYRICOM MYRINET          *
 * EXPRESS (MX) NETWORKING SOFTWARE AND DOCUMENTATION LICENSE (the       *
 * "License"); User may not use this file except in compliance with the  *
 * License.  The full text of the License can found in LICENSE.TXT       *
 *                                                                       *
 * Software distributed under the License is distributed on an "AS IS"   *
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied.  See  *
 * the License for the specific language governing rights and            *
 * limitations under the License.                                        *
 *                                                                       *
 * Copyright 2003 - 2004 by Myricom, Inc.  All rights reserved.          *
 *************************************************************************/

#ifdef MX_KERNEL
#include "mx_arch.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include "mx_auto_config.h"
#include "mx_int.h"
#include "mx_debug.h"
#endif

uint32_t mx_debug_mask = 0;

#ifndef MX_KERNEL
void 
mx_assertion_failed (const char *assertion, int line, const char *file)
{
  printf("MX: assertion: <<%s>>  failed at line %d, file %s\n",
	 assertion, line, file);
  mx__abort();
}

#endif
