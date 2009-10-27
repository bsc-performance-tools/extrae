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

#include "mx_auto_config.h"
#include "mx_stbar.h"

/* Emit a store barrier. */
void 
mx__stbar(void)
{
#ifdef __MX_STBAR 
  __MX_STBAR ();
#else
#error should not call mx__stbar.
#endif

  return;
}

/* Emit a read barrier. */
void 
mx__readbar(void)
{
#ifdef __MX_READBAR
  __MX_READBAR ();
#else
#error should not call mx__readbar.
#endif

  return;
}

/* Emit a write barrier. */
void 
mx__writebar(void)
{
#ifdef __MX_STBAR
  __MX_STBAR ();
#else
#error should not call mx__writebar.
#endif

  return;
}
