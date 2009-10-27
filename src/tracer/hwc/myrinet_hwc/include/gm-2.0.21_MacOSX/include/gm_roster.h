/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_roster_h_
#define _gm_roster_h_

/************
 * Roster
 ************/

/* Make sure all the GM developer usernames are defined.  These must
   be exactly the usernames used from the user's USER or LOGNAME
   environmental variable prefixed with GM_USER_.
   The Makefile causes -DGM_USER_${USER}=1 to be defined for if
   configured with --enable-maintainer-mode.

   These can be used for handy developer-specific debugging like
   
   > GM_PRINT (GM_USER_glenn, "Glenn, you goofed!\n");

   or in macros like
   
   > #if GM_USER_glenn | GM_USER_nelson
   > #error Glenn and Nelson, one of you goofed!
   > #endif

   */

#ifndef GM_USER_eugene
#define GM_USER_eugene 0
#endif

#ifndef GM_USER_feldy
#define GM_USER_feldy 0
#endif

#ifndef GM_USER_finucane
#define GM_USER_finucane 0
#endif

#ifndef GM_USER_gallatin
#define GM_USER_gallatin 0
#endif

#ifndef GM_USER_glenn
#define GM_USER_glenn 0
#endif

#ifndef GM_USER_karen
#define GM_USER_karen 0
#endif

#ifndef GM_USER_loic
#define GM_USER_loic 0
#endif

#ifndef GM_USER_lyan
#define GM_USER_lyan 0
#endif

#ifndef GM_USER_maxstern
#define GM_USER_maxstern 0
#endif

#ifndef GM_USER_nelson
#define GM_USER_nelson 0
#endif

#ifndef GM_USER_patrick
#define GM_USER_patrick 0
#endif

#ifndef GM_USER_reese
#define GM_USER_reese 0
#endif

#ifndef GM_USER_ruth
#define GM_USER_ruth 0
#endif

#ifndef GM_USER_susan
#define GM_USER_susan 0
#endif

#define gm_developer (GM_USER_eugene					\
		      | GM_USER_feldy					\
		      | GM_USER_finucane				\
		      | GM_USER_gallatin				\
		      | GM_USER_glenn					\
		      | GM_USER_karen					\
		      | GM_USER_loic					\
		      | GM_USER_lyan					\
		      | GM_USER_maxstern				\
		      | GM_USER_nelson					\
		      | GM_USER_patrick					\
		      | GM_USER_ruth					\
		      | GM_USER_susan)

#endif /* _gm_roster_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
