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


def fillTrack(tracks:str, startIdx:int, fields):
    """
    Return the track starting at index startIdx as a dictionnary
    """
    nbInfoPerTrack = int(tracks[1:tracks.find('}')].split(' ')[1])
    idx = startIdx + 1
    track = {}
    for i in range(0, nbInfoPerTrack):
        if tracks[idx] == '{' or i == nbInfoPerTrack - 1:
            idxEndInfo = tracks.find('}', idx)
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
