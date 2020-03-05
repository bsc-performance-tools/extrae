#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <GASPI.h>
#include <GASPI_Ext.h>

#define _4GB 4294967296
#define _2GB 2147483648
#define _1GB 1073741824
#define _1MB 1048576
#define _2MB 2097152
#define _4MB 4194304
#define _8MB 8388608
#define _128MB 134217728
#define _500MB 524288000


void tsuite_init (int argc, char *argv[]);
void success_or_exit (const char *file, const int line,
                      const gaspi_return_t ec);
void must_fail (const char *file, const int line, const gaspi_return_t ec);
void must_timeout (const char *file, const int line, const gaspi_return_t ec);
void must_return_err (const char *file, const int line,
                      const gaspi_return_t ec, const gaspi_return_t err);

#define TSUITE_INIT(argc, argv) tsuite_init(argc, argv)
#define ASSERT(ec) success_or_exit (__FILE__, __LINE__, ec)
#define EXPECT_FAIL(ec) must_fail(__FILE__, __LINE__, ec)
#define EXPECT_TIMEOUT(ec) must_return_err(__FILE__, __LINE__, ec, GASPI_TIMEOUT)
#define EXPECT_FAIL_WITH(ec,err) must_return_err(__FILE__, __LINE__, ec, err)

void exit_safely (void);

#endif //_TEST_UTILS_H_
