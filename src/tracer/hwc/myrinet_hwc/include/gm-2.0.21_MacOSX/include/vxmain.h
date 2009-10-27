/*
 * Include: vxmain.h
 * Author:  Joe Fogler
 * Date:    22-Jun-90
 *
 * Description:
 *
 * This include file permits command
 * line argument passing under vxworks.
 *
 * Example:
 *
 * #ifdef VXWORKS
 * #define MAINNAME main_a
 * #define VXMAIN yourprog
 * #define VXNAME "yourprog"
 * #include "vxmain.h"
 * #endif
 * main(argc, argv)
 * int argc;
 * char **argv;
 * {
 */
#ifdef MAINNAME			/* an optional name for main */
static int MAINNAME (int argc, char **argv);
#else
static int main();
#endif
static char prog[] = { VXNAME };
int VXMAIN (char *args)
{
  int argc = 0;
  char *argv[256];
  int rv;

  argv[argc++] = prog;
  if (args) {
  do {
   while (*args == ' ') args++;
   if (*args == '\0') break;
   argv[argc++] = args;
   while ((*args != ' ') && (*args != '\0')) args++;
   if (*args == ' ') *args++ = '\0';
  } while((*args != '\0') && (argc < 20));
  }
#ifdef MAINNAME
  rv = MAINNAME (argc, argv);
#else
  rv = main(argc, argv);
#endif
  return rv;
}
