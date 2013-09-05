#ifndef __ONLINE_ROOT_H__
#define __ONLINE_ROOT_H__

#define MAX_RETRIES 10

#define DEFAULT_FANOUT 32                          /* Default fan-out for the MRNet-tree             */

void Stop_FE();
void FE_main_loop(int frequency);


#endif /* __ONLINE_ROOT_H__ */
