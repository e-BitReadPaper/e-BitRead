originServers=["local","amazon"]
originServer = originServers[0]

if originServer == "local":
    URLPrefix = "http://219.223.189.147:80/video/"
elif originServer == "amazon":
    URLPrefix = "52.64.170.92:80/video/"


def getURLPrefix(videoName,bitrateIndex):
    url = URLPrefix + videoName + "/video/avc1/" + bitrateIndex+"/.*"
    # print('URL: %s' %url)
    return url


def keepList():
    videoL = open("../file/cachedFile/cachedFile_20190612_pure.txt").readlines() # videoName version chunk_count
    videoL = [[i.split(" ")[0], i.split(" ")[1]] for i in videoL]
    for videoTu in videoL:
        videoName = videoTu[0]
        bitrate = videoTu[1]
        #url_regex=example.com/game/.* pin-in-cache=1h
        print("url_regex=" + getURLPrefix(videoName, bitrate)+" pin-in-cache=365d")


def main():
    keepList()

main()