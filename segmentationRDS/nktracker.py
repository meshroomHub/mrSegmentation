import re

def extractNode(nkFileContent, nodeType):
    """
    Find the first occurence of nodeType in nkFileContent and return
    the part of the content from the first'{' after the nodeType to the
    corresponding '}

    Args:
    nkFileContent: Content to process.
    nodeType: Node type to find.

    Returns:
    the part of the content from the first'{' after the nodeType to the
    corresponding '}' or 'None' if the nodeType cannot be found.
    """

    idxNode = nkFileContent.find(nodeType)
    if idxNode == -1:
        return None

    idxStart = nkFileContent.find('{', idxNode + len(nodeType))
    if idxStart == -1:
        return None

    result = nkFileContent[idxStart:]
    nb = 0
    for i, c in enumerate(result):
        if c == '{':
            nb = nb + 1
        elif c == '}':
            nb = nb - 1
            if nb == 0:
                idxEnd = i
                break
    return result[:idxEnd+1]


def cleanNodeContent(content):
    """
    Clean nuke node content
    """

    # Remove eol and special caracters
    cleanContent = re.sub(r'\n', r' ', content)
    cleanContent = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleanContent)

    # Remove spaces before and after '{' and '}'
    cleanContent = re.sub(r'\s*\{\s*', '{', cleanContent)
    cleanContent = re.sub(r'\s*\}\s*', '}', cleanContent)
    return cleanContent

def getTracks(trackerContent):
    """
    Return the content between { } juste after the first keyword 'tracks' found in trackerContent
    """

    idxNode = trackerContent.find('tracks')
    if idxNode == -1:
        return None

    idxStart = trackerContent.find('{', idxNode + len('tracks'))
    if idxStart == -1:
        return None

    result = trackerContent[idxStart:]
    nb = 0
    for i, c in enumerate(result):
        if c == '{':
            nb = nb + 1
        elif c == '}':
            nb = nb - 1
            if nb == 0:
                idxEnd = i
                break
    return result[1:idxEnd]

def getTrackNumber(tracks:str):
    return int(tracks[1:tracks.find('}')].split(' ')[2])

def getTrackKeysNumber(tracks:str):
    return int(tracks[1:tracks.find('}')].split(' ')[1])

def fillTrack(tracks:str, startIdx:int, fields):
    """
    Return the track starting at index startIdx as a dictionnary
    """
    nbInfoPerTrack = getTrackKeysNumber(tracks)
    idx = startIdx + 1
    track = {}
    for i in range(0, nbInfoPerTrack):
        if tracks[idx] == '{' or i == nbInfoPerTrack - 1:
            idxEndInfo = tracks.find('}', idx)
            info = tracks[idx+1:idxEndInfo]
            idx = idxEndInfo + 1
        elif tracks[idx] == '"':
            idxEndInfo = tracks.find('"', idx+1)
            info = tracks[idx+1:idxEndInfo]
            idx = idxEndInfo + 1
        elif tracks[idx] == ' ':
            idxEndInfo = min(tracks.find(' ', idx+1), tracks.find('{', idx+1))
            info = tracks[idx+1:idxEndInfo]
            idx = idxEndInfo
        else:
            idxEndInfo = min(tracks.find(' ', idx), tracks.find('{', idx))
            info = tracks[idx:idxEndInfo]
            idx = idxEndInfo
        track[fields[i]] = info
    return track

def getTrackKeys(tracks:str):
    """
    Return a list of keys available for a track
    """
    nbInfoPerTrack = getTrackKeysNumber(tracks)
    idxFirstInfo = tracks.find('{{') + 1
    idx = idxFirstInfo
    keys = []
    for i in range(0, nbInfoPerTrack):
        idxEndKey = tracks.find('}', idx)
        key = tracks[idx+1:idxEndKey].split(' ')[3]
        keys.append(key)
        idx = idxEndKey + 1
    return keys

def getTrackIndexes(tracks:str):
    """
    Return a list of indexes corresponding to the beginning of each individual track
    """
    nbTracks = getTrackNumber(tracks)
    idxFirstTrack = tracks.find('}}{') + 3
    trackIdxs = [idxFirstTrack]
    idx = idxFirstTrack
    for i in range(1, nbTracks):
        idxNextTrack = tracks.find('}{{', idx) + 1
        trackIdxs.append(idxNextTrack)
        idx = idxNextTrack
    return trackIdxs

def getCurve(track:str):
    """
    Args:
        A piece of string starting with 'curve' and containing a track info
    Returns:
        the interpolation type if any or 'None'
        A dictionnary with the frameIds as keys and the values as float numbers
    """
    if len(track) < 6 or track[:5] != "curve":
        return (None, {})

    idxInterpolationType = 6
    if track[idxInterpolationType] == 'x':
        interpolationType = None
        idxData = idxInterpolationType
    else:
        interpolationType = track[6:track.find(' ', 6)]
        idxData = track.find('x', 6)

    idxEndData = track.find(' ', idxData)
    frameId = int(track[idxData+1:idxEndData])
    step = 1
    prevFrameId = frameId - step
    idxData = idxEndData + 1
    curve = {}

    while idxEndData != len(track):
        idxEndData = track.find(' ', idxData)
        if idxEndData == -1:
            idxEndData = len(track)
        if track[idxData] == 'x':
            frameId = int(track[idxData+1:idxEndData])
            step = frameId - prevFrameId
            prevFrameId = frameId
        else:
            value = track[idxData:idxEndData]
            curve[frameId] = float(value)
            frameId = frameId + step
        if idxEndData != len(track):
            idxData = idxEndData + 1

    return (interpolationType, curve)

def getValueAtFrame(curve, frameId:int):

    if curve == None:
        return None

    if frameId in curve.keys():
        return curve[frameId]
    else:
        prevFrameId = None
        nextFrameId = None
        for f in curve:
            if f < frameId and (prevFrameId is None or f > prevFrameId):
                prevFrameId = f
            if f > frameId and (nextFrameId is None or f < nextFrameId):
                nextFrameId = f
        if prevFrameId != None and nextFrameId != None:
            return curve[prevFrameId] + (curve[nextFrameId] - curve[prevFrameId]) * ((frameId - prevFrameId) / (nextFrameId - prevFrameId))
        else:
            return None

class nkTracker:
    def __init__(self, nkFile:str):
        self.tracks = {}
        try:
            with open(nkFile, 'r') as f:
                nkFileContent = f.read()
                Tracker4 = extractNode(nkFileContent, 'Tracker4')
                Tracker4 = cleanNodeContent(Tracker4)
                tracksStr = getTracks(Tracker4)
                fields = getTrackKeys(tracksStr)
                nbTracks = getTrackNumber(tracksStr)
                trackIdxs = getTrackIndexes(tracksStr)
                trs = [fillTrack(tracksStr, trackIdxs[i], fields) for i in range(0, nbTracks)]
                curveKeys = ['track_x', 'track_y', 'key_search_x', 'key_search_y', 'key_search_r', 'key_search_t', 'key_track_x', 'key_track_y', 'key_track_r', 'key_track_t']
                scalarKeys = ['pattern_x', 'pattern_y', 'pattern_r', 'pattern_t', 'search_x', 'search_y', 'search_r', 'search_t']

                for i in range(0, nbTracks):
                    trackName = trs[i]["name"]
                    track = {}
                    for key in curveKeys:
                        interpolationType, curve = getCurve(trs[i][key])
                        track[key] = {"interpolationType": interpolationType, "curve": curve}
                    for key in scalarKeys:
                        track[key] = float(trs[i][key])
                    self.tracks[trackName] = track

        except FileNotFoundError:
            print('Nuke tracker file cannot be found')

    def getTrackNames(self):
        return list(self.tracks.keys())

    def getDataAtFrame(self, frameId:int, height:int = -1, par:float = 1.0):
        """
        Returns track center and bounding box and search bounding box at a given frame.
        Because tracking infos are not linked with the size of the frames in the sequence on which the tracker is applied,
        and because for Nuke, the row at index 0 is the bottom one, to get the data with respect to the top row, the height
        of the frames in the sequence can be given as an argument. 

        Args:
        frameId: frame idx.
        height: frame height.
        par: pixel aspect ratio.

        Returns:
        A tuple containing the track center (x, y), the tracking and the search bounding boxes in the following format:
        [x_left, y_top, x_right, y_bottom].
        """
        data = {}
        for key in self.tracks.keys():
            bbT = None
            bbS = None
            x = getValueAtFrame(self.tracks[key]['track_x']['curve'], frameId)
            y = getValueAtFrame(self.tracks[key]['track_y']['curve'], frameId) / par
            if x is not None:
                bbT_TL_x = x + self.tracks[key]['pattern_x']
                bbT_TL_y = y + self.tracks[key]['pattern_t']
                bbT_BR_x = x + self.tracks[key]['pattern_r']
                bbT_BR_y = y + self.tracks[key]['pattern_y']
                bbS_TL_x = bbT_TL_x + self.tracks[key]['search_x']
                bbS_TL_y = bbT_TL_y + self.tracks[key]['search_t']
                bbS_BR_x = bbT_BR_x + self.tracks[key]['search_r']
                bbS_BR_y = bbT_BR_y + self.tracks[key]['search_y']
                if height > 0:
                    y = height - y
                    bbT_TL_y = height - bbT_TL_y
                    bbT_BR_y = height - bbT_BR_y
                    bbS_TL_y = height - bbS_TL_y
                    bbS_BR_y = height - bbS_BR_y
                bbT = [bbT_TL_x, bbT_TL_y, bbT_BR_x, bbT_BR_y]
                bbS = [bbS_TL_x, bbS_TL_y, bbS_BR_x, bbS_BR_y]
            data[key] = ((x, y), bbT, bbS)
        return data
