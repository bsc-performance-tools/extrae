#ifndef __ONLINE_DEBUG_H__
#define __ONLINE_DEBUG_H__

#include <ostream>
#include <string>

using std::ostream;
using std::string;

class Messaging
{
  public:
    Messaging();
    Messaging(int be_rank, bool is_master);

    void say    (ostream &out, const char *fmt, ...);
    void say_one(ostream &out, const char *fmt, ...);

    void debug    (ostream &out, const char *fmt, ...);
    void debug_one(ostream &out, const char *fmt, ...);

    void error(const char *fmt, ...);

    bool debugging();

  private:
    bool   I_am_FE;
    bool   I_am_BE;
    bool   I_am_master_BE;
    string ProcessLabel;
    bool   DebugEnabled;
};

#endif /* __ONLINE_DEBUG_H__ */
