
/*
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.    See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.    The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

/**
 * @file plugin.cc
 * @brief traffic server plugin entry points.
 */

#include "plugin.h"
#include "timer.h"
#define BITRATE_COUNT 5

static const char *
getEventName(TSEvent event)
{
    switch (event) {
        case TS_EVENT_HTTP_CONTINUE:
            return "TS_EVENT_HTTP_CONTINUE";
        case TS_EVENT_HTTP_ERROR:
            return "TS_EVENT_HTTP_ERROR";
        case TS_EVENT_HTTP_READ_REQUEST_HDR:
            return "TS_EVENT_HTTP_READ_REQUEST_HDR";
        case TS_EVENT_HTTP_OS_DNS:
            return "TS_EVENT_HTTP_OS_DNS";
        case TS_EVENT_HTTP_SEND_REQUEST_HDR:
            return "TS_EVENT_HTTP_SEND_REQUEST_HDR";
        case TS_EVENT_HTTP_READ_CACHE_HDR:
            return "TS_EVENT_HTTP_READ_CACHE_HDR";
        case TS_EVENT_HTTP_READ_RESPONSE_HDR:
            return "TS_EVENT_HTTP_READ_RESPONSE_HDR";
        case TS_EVENT_HTTP_SEND_RESPONSE_HDR:
            return "TS_EVENT_HTTP_SEND_RESPONSE_HDR";
        case TS_EVENT_HTTP_REQUEST_TRANSFORM:
            return "TS_EVENT_HTTP_REQUEST_TRANSFORM";
        case TS_EVENT_HTTP_RESPONSE_TRANSFORM:
            return "TS_EVENT_HTTP_RESPONSE_TRANSFORM";
        case TS_EVENT_HTTP_SELECT_ALT:
            return "TS_EVENT_HTTP_SELECT_ALT";
        case TS_EVENT_HTTP_TXN_START:
            return "TS_EVENT_HTTP_TXN_START";
        case TS_EVENT_HTTP_TXN_CLOSE:
            return "TS_EVENT_HTTP_TXN_CLOSE";
        case TS_EVENT_HTTP_SSN_START:
            return "TS_EVENT_HTTP_SSN_START";
        case TS_EVENT_HTTP_SSN_CLOSE:
            return "TS_EVENT_HTTP_SSN_CLOSE";
        case TS_EVENT_HTTP_CACHE_LOOKUP_COMPLETE:
            return "TS_EVENT_HTTP_CACHE_LOOKUP_COMPLETE";
        case TS_EVENT_HTTP_PRE_REMAP:
            return "TS_EVENT_HTTP_PRE_REMAP";
        case TS_EVENT_HTTP_POST_REMAP:
            return "TS_EVENT_HTTP_POST_REMAP";
        default:
            return "UNHANDLED";
    }
    return "UNHANDLED";
}

PluginState::PluginState(): _totalMThroughput(0), _maxMThtoughput(0), _requestCount(0){
    _lock = TSMutexCreate();
    int period_in_ms           = 4000; //4s
    TimerEventReceiver *timer = new TimerEventReceiver(AsyncTimer::TYPE_PERIODIC, period_in_ms, this);
    ReleaserDebug("Created periodic timer %p with initial period 0, regular period %d and max instances 0", timer, period_in_ms);
}


time_t getTime(){
    time_t t;
    time(&t);
    return t;
}


void PluginState::reset() {
    TSMutexLock(_lock);
    std::ofstream fw;
    fw.open("./throughputLog.txt", std::ios::app);
    time_t t = getTime();
    if (_requestCount > 0){
        ReleaserDebug("avgMThroughput=%f, _totalMThroughput=%f,_maxMThtoughput=%f, _requestCount=%s, time=%ld",
                _totalMThroughput/_requestCount, _totalMThroughput,_maxMThtoughput, std::to_string(_requestCount).c_str(), t);
        fw <<   std::to_string(_totalMThroughput/_requestCount) + " " +
                std::to_string(_totalMThroughput) + " " +
                std::to_string(_maxMThtoughput) + " " +
                std::to_string(_requestCount) + " " +
                std::to_string(t)<< std::endl;
    }
    fw.close();
    _totalMThroughput = 0;
    _maxMThtoughput = 0;
    _requestCount = 0;
    TSMutexUnlock(_lock);
}


void PluginState::update(double mThroughput) {
    TSMutexLock(_lock);
    if(mThroughput>0){
        _totalMThroughput += mThroughput;
        if(mThroughput>_maxMThtoughput)
            _maxMThtoughput = mThroughput;
        _requestCount+=1;
    }
    ReleaserDebug("_totalMThroughput=%f, _maxMThtoughput=%f, _requestCount=%d",_totalMThroughput, _maxMThtoughput, _requestCount);
    TSMutexUnlock(_lock);
}

PluginState *pState;

/**
 * @brief Plugin initialization.
 * @param apiInfo remap interface info pointer
 * @param errBuf error message buffer
 * @param errBufSize error message buffer size
 * @return always TS_SUCCESS.
 */
TSReturnCode
TSRemapInit(TSRemapInterface *apiInfo, char *errBuf, int erroBufSize)
{
    pState = new PluginState;
    ReleaserDebug("TSRemapInit");
    return TS_SUCCESS;
}


/**
 * @brief Plugin instance data.
 */

struct ReleaserInstance {
    ReleaserInstance() {};

private:
    ReleaserInstance(ReleaserInstance const &);
};

/**
 * brief Plugin transaction data.
 */
class ReleaserTxnData
{
public:
    ReleaserTxnData(ReleaserInstance *inst)
            : _inst(inst), _action(5)
    {
    }
    ReleaserInstance *_inst; /* Pointer to the plugin instance */

    /* saves state between hooks */
    String _sHeader; // RL state
    int _action;
};


/**
 * @brief Callback function that handles RL operations.
 *
 * @param contp continuation associated with this function.
 * @param event corresponding event triggered at different hooks.
 * @param edata HTTP transaction structures.
 * @return always 0
 */
int
contHandleFetch(const TSCont contp, TSEvent event, void *edata) {
    ReleaserTxnData *data = static_cast<ReleaserTxnData *>(TSContDataGet(contp));
    TSHttpTxn txnp = static_cast<TSHttpTxn>(edata);
    TSMBuffer reqBuffer;
    TSMLoc reqHdrLoc;

    if (TS_SUCCESS != TSHttpTxnClientReqGet(txnp, &reqBuffer, &reqHdrLoc)) {
        ReleaserError("failed to get client request");
        TSHttpTxnReenable(txnp, TS_EVENT_HTTP_ERROR);
        return 0;
    }

    TSEvent retEvent = TS_EVENT_HTTP_CONTINUE;
    ReleaserDebug("event: %s (%d)", getEventName(event), event);

    switch (event) {
        case TS_EVENT_HTTP_POST_REMAP: {
            // RL ---------------------------------------------
            String s = "sudo /opt/releaserRL/build/releaserRL " + data->_sHeader + " " + std::to_string(pState->_requestCount/40);
            ReleaserDebug("%s", s.c_str());
            FILE *fp = popen(s.c_str(), "r");
            if (!fp) {
                ReleaserDebug("fp==0");
                break;
            }
            char buf[1000];
            memset(buf, 0, sizeof(buf));
            if (fgets(buf, sizeof(buf) - 1, fp) != 0) {
                data->_action = atoi(buf);
                ReleaserDebug("action=%d", data->_action);
            }
            pclose(fp);
            if(data->_action!=5) { //modify request url
                String path;
                TSMLoc url_loc;
                if (TS_SUCCESS == TSHttpHdrUrlGet(reqBuffer, reqHdrLoc, &url_loc)) {
                    int pathLen            = 0;
                    // get url -----------------------------------------
                    const char *path_ = TSUrlPathGet(reqBuffer, url_loc, &pathLen);
                    if (nullptr != path_) {
                        ReleaserDebug("path: '%.*s'", pathLen, path_);
                        path.assign(path_, pathLen);
                    } else {
                        ReleaserError("failed to get pristine URL path");
                    }
                    // get url -----------------------------------------
                    size_t start = 0;
                    size_t end = 0;
                    start = path.find("avc1");
                    if(start != String::npos){
                        start += 5;
                        end = path.find("seg");
                        if(end != String::npos){
                            end -= 2;
                        } else {
                            ReleaserError("failed to find avc1");
                            return TSREMAP_NO_REMAP;
                        }
                    } else {
                        ReleaserError("failed to find seg");
                        return TSREMAP_NO_REMAP;
                    }
                    std::string newUrl = path.replace(start, end-start+1, std::to_string(data->_action + 1));
                    // set url -----------------------------------------
                    if (TS_SUCCESS == TSUrlPathSet(reqBuffer, url_loc, newUrl.c_str(), (int)newUrl.length())) {
                        ReleaserDebug("setting URL path to '%.*s'", (int)newUrl.length(), newUrl.c_str());
                    } else {
                        ReleaserError("failed to set a URL path '%.*s'", (int)newUrl.length(), newUrl.c_str());
                    }
                    // set url -----------------------------------------

                    TSHandleMLocRelease(reqBuffer, TS_NULL_MLOC, url_loc);
                } else {
                    ReleaserError("failed to get pristine URL");
                }
            }
        } break;
        case TS_EVENT_HTTP_SEND_RESPONSE_HDR: {
            TSMBuffer bufp;
            TSMLoc hdrLoc;
            String header = "ActualBI";
            String value = std::to_string(data->_action) + std::to_string();
            if (TS_SUCCESS == TSHttpTxnClientRespGet(txnp, &bufp, &hdrLoc)) {
                setHeader(bufp, hdrLoc, header.c_str(), (int)header.length(), value.c_str(),
                          (int)value.length());
                TSHandleMLocRelease(bufp, TS_NULL_MLOC, hdrLoc);
                ReleaserDebug("setting response header ActualBI to '%s'", value.c_str());
            } else {
                ReleaserError("failed to retrieve client response header");
            }
        } break;
        case TS_EVENT_HTTP_TXN_CLOSE: {
            delete data;
            TSContDestroy(contp);
        } break;

        default: {
            ReleaserError("unhandled event: %s(%d)", getEventName(event), event);
        } break;

    }
    /* Release the request MLoc */
    TSHandleMLocRelease(reqBuffer, TS_NULL_MLOC, reqHdrLoc);

    /* Reenable and continue with the state machine. */
    TSHttpTxnReenable(txnp, retEvent);
    return 0;
}


/**
 * @brief Plugin new instance entry point.
 *
 * Processes the configuration and initializes the plugin instance.
 * @param argc plugin arguments number
 * @param argv plugin arguments
 * @param instance new plugin instance pointer (initialized in this function)
 * @param errBuf error message buffer
 * @param errBufSize error message buffer size
 * @return TS_SUCCES if success or TS_ERROR if failure
 */
TSReturnCode
TSRemapNewInstance(int argc, char *argv[], void **instance, char *errBuf, int errBufSize)
{
    bool failed = false;

    ReleaserInstance *inst = new ReleaserInstance();
    if (nullptr == inst) {
        failed = true;
    }

    if (failed) {
        ReleaserError("failed to initialize the plugin");
        delete inst;
        *instance = nullptr;
        return TS_ERROR;
    }

    *instance = inst;
    return TS_SUCCESS;
}

/**
 * @brief Plugin instance deletion clean-up entry point.
 * @param plugin instance pointer.
 */
void
TSRemapDeleteInstance(void *instance)
{
    ReleaserInstance *inst = (ReleaserInstance *)instance;
    delete inst;
}

/**
 * @brief Organizes the background fetch by registering necessary hooks, by identifying front-end vs back-end, first vs second
 * pass.
 *
 * Remap is never done, continue with next in chain.
 * @param instance plugin instance pointer
 * @param txnp transaction handle
 * @param rri remap request info pointer
 * @return always TSREMAP_NO_REMAP
 */
TSRemapStatus
TSRemapDoRemap(void *instance, TSHttpTxn txnp, TSRemapRequestInfo *rri)
{
    ReleaserDebug("TSRemapDoRemap");

    ReleaserInstance *inst = (ReleaserInstance *)instance;

    if (nullptr != inst) {
        // State header-----------------------------
        const String sName = "State";
        char sHeader[2000];
        memset(sHeader, 0, sizeof(sHeader));
        int sHeaderLen = 1999;
        // miss throughput header-------------------
        const String mName= "MThroughput";
        char mHeader[2000];
        memset(mHeader, 0, sizeof(mHeader));
        int mHeaderLen = 1999;
        // ------------------------------------------
        int methodLen                = 0;
        const char *method     = TSHttpHdrMethodGet(rri->requestBufp, rri->requestHdrp, &methodLen);
        if (nullptr != method && methodLen == TS_HTTP_LEN_GET && 0 == memcmp(TS_HTTP_METHOD_GET, method, TS_HTTP_LEN_GET)) {
            bool handleFetch = true;
            // State header-----------------------------
            if (headerExist(rri->requestBufp, rri->requestHdrp, sName.c_str(), sName.length())) {
                getHeader(rri->requestBufp, rri->requestHdrp, sName.c_str(), sName.length(), sHeader, &sHeaderLen);
            } else {
                handleFetch = false;
            }
             //miss throughput header--------------------
            if (headerExist(rri->requestBufp, rri->requestHdrp, mName.c_str(), mName.length())) {
                getHeader(rri->requestBufp, rri->requestHdrp, mName.c_str(), mName.length(), mHeader, &mHeaderLen);
                double mThroughput = atof(mHeader);
                if (mThroughput != -1) {
                    pState->update(mThroughput);
                }
            }
            ReleaserTxnData *data = new ReleaserTxnData(inst);
            if (nullptr != data and handleFetch) {
                data->_sHeader = sHeader;

                TSCont cont = TSContCreate(contHandleFetch, TSMutexCreate());
                TSContDataSet(cont, static_cast<void *>(data));

                TSHttpTxnHookAdd(txnp, TS_HTTP_POST_REMAP_HOOK, cont);
                TSHttpTxnHookAdd(txnp, TS_HTTP_SEND_RESPONSE_HDR_HOOK, cont);
                TSHttpTxnHookAdd(txnp, TS_HTTP_TXN_CLOSE_HOOK, cont);
            } else {
                ReleaserError("failed to allocate transaction data object");
            }
        } else {
            ReleaserDebug("not a GET method (%.*s), skipping", methodLen, method);
        }

    } else {
        ReleaserError("could not get Releaser instance");
    }

    return TSREMAP_NO_REMAP;
}
