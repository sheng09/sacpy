#ifndef TOMO_CROSSTERM_______
#define TOMO_CROSSTERM_______

#include <string>
#include <utility>

class body_wave_pair : std::pair<int , int >
{};


class crossterm
{
public:
    crossterm(int cc_id, int body_wave_id1, int body_wave_id2, const char * tag) { init(cc_id, body_wave_id1, body_wave_id2, tag); }
    ~crossterm() {}
    inline int init(int cc_id, int body_wave_id1, int body_wave_id2, const char * tag)
    {
        d_ccid = cc_id;
        d_id1 = body_wave_id1;
        d_id2 = body_wave_id2;
        d_tag = tag;
        return 0;
    }
    inline int ccid() { return d_ccid; }
    inline int id1() { return d_id1; }
    inline int id2() { return d_id2; }
    inline std::string & tag() { return d_tag; }
private:
    int d_ccid;       // id of this cross-term
    int d_id1, d_id2; // body wave id
    std::string d_tag;
};

#endif

