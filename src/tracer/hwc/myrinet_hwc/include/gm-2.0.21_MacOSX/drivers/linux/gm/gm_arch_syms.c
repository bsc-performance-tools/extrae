
extern int smp_call_function __attribute__((weak));
void *smp_call_function_symbol = &smp_call_function;
extern unsigned long get_user_pages __attribute__((weak));
void *gm_linux_get_user_pages = &get_user_pages;
extern int kmap_high __attribute__((weak));
void *kmap_high_symbol = &kmap_high;
extern unsigned long sys_call_table[] __attribute__((weak));
unsigned long *sys_call_table_symbol = sys_call_table;
