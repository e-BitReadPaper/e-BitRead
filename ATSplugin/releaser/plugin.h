
#include <sstream>
#include <iomanip>
#include <string.h>
#include "stdio.h"

#include "ts/ts.h" /* ATS API */

#include "ts/remap.h" /* TSRemapInterface, TSRemapStatus, apiInfo */

#include "common.h"
#include "headers.h"
#include <fstream>

class PluginState
{
public:
    PluginState();
    double _totalMThroughput;
    double _maxMThtoughput;
    int _requestCount;
    TSMutex _lock;

    void update(double mThroughput);

    void reset();
};
