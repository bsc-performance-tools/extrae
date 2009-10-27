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

#ifndef MX_UTIL_H
#define MX_UTIL_H

#define MX_ROUND_UP(x,y) (((x) + (y)-1) & ~((y)-1))
#define MX_DIV_UP(x,y) (((x) + (y)-1) / (y))
#define MX_ROUND_DOWN(x,y) ((x) & ~((y)-1))

#define MX_MIN(x,y) ((x) < (y) ? (x) : (y))
#define MX_MAX(x,y) ((x) > (y) ? (x) : (y))

#define mx_mem_check(a) do { if (!(a)) goto handle_enomem; } while (0)

#endif
