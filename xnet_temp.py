import re
import igraph as ig


def __readXnetVerticesHeader(fp, currentLineIndex, lastLine=""):
    headerLine = lastLine
    if len(headerLine) == 0:
        for lineData in fp:
            headerLine = lineData.rstrip()
            currentLineIndex += 1
            if len(headerLine) > 0:
                break

    headerEntries = headerLine.split()
    nodeCount = 0

    if len(headerEntries) == 0 or headerEntries[0].lower() != "#vertices":
        raise ValueError("Malformed xnet file [%s:%d]\n\t>%s" % (fp.name, currentLineIndex, headerLine))

    try:
        nodeCount = int(headerEntries[1])
    except ValueError:
        raise ValueError("Malformed xnet file [%s:%d]\n\t>%s" % (fp.name, currentLineIndex, headerLine))

    return nodeCount, currentLineIndex, ""


def __readXnetNames(fp, currentLineIndex, lastLine=""):
    names = []
    fileEnded = True
    for lineData in fp:
        lineString = lineData.rstrip()
        currentLineIndex += 1

        if len(lineString) == 0:
            continue

        lastLine = lineString

        if len(lineString) > 0 and lineString[0] == "#":
            fileEnded = False
            break

        if len(lineString) > 1 and lineString[0] == "\"" and lineString[-1] == "\"":
            lineString = lineString[1:-1]

        names.append(lineString)

    if fileEnded:
        lastLine = ""
    return names, currentLineIndex, lastLine


def __readXnetEdgesHeader(fp, currentLineIndex, lastLine=""):
    headerLine = lastLine
    if len(headerLine) == 0:
        for lineData in fp:
            headerLine = lineData.decode("utf-8").rstrip()
            currentLineIndex += 1
            if len(headerLine) > 0:
                break

    headerEntries = headerLine.split()

    weighted = False
    directed = False

    if len(headerEntries) == 0 or headerEntries[0].lower() != "#edges":
        raise ValueError("Malformed xnet file [%s:%d]\n\t>%s" % (fp.name, currentLineIndex, headerLine))

    for headerEntry in headerEntries:
        if headerEntry == "weighted":
            weighted = True
        if headerEntry == "nonweighted":
            weighted = False
        if headerEntry == "directed":
            directed = True
        if headerEntry == "undirected":
            directed = False

    return (weighted, directed), currentLineIndex, ""


def __readXnetEdges(fp, currentLineIndex, lastLine=""):
    edges = []
    weights = []
    fileEnded = True
    for lineData in fp:
        lineString = lineData.rstrip()
        currentLineIndex += 1

        if len(lineString) == 0:
            continue

        lastLine = lineString

        if len(lineString) > 0 and lineString[0] == "#":
            fileEnded = False
            break

        entries = lineString.split()

        if len(entries) < 2:
            raise ValueError("Malformed xnet file [%s:%d]\n\t>%s" % (fp.name, currentLineIndex, headerLine))
        try:
            edge = (int(entries[0]), int(entries[1]))
            weight = 1.0
            if len(entries) > 2:
                weight = float(entries[2])
            edges.append(edge)
            weights.append(weight)
        except ValueError:
            raise ValueError("Malformed xnet file [%s:%d]\n\t>%s" % (fp.name, currentLineIndex, headerLine))

    if fileEnded:
        lastLine = ""
    return (edges, weights), currentLineIndex, lastLine


def __readXnetProperty(fp, propertyFormat, currentLineIndex, lastLine=""):
    global __propertyFunctions
    properties = []
    propertyFunction = __propertyFunctions[propertyFormat]
    fileEnded = True
    for lineData in fp:
        lineString = lineData.rstrip()
        currentLineIndex += 1

        if len(lineString) == 0:
            continue

        lastLine = lineString

        if len(lineString) > 0 and lineString[0] == "#":
            fileEnded = False
            break

        if len(lineString) > 1 and lineString[0] == "\"" and lineString[-1] == "\"":
            lineString = lineString[1:-1]

        try:
            properties.append(propertyFunction(lineString))
        except ValueError:
            raise ValueError("Malformed xnet file [%s:%d]\n\t>%s" % (fp.name, currentLineIndex, lineString))

    if fileEnded:
        lastLine = ""

    return properties, currentLineIndex, lastLine


def __textSplit2(text):
    entries = text.split()
    if len(entries) < 2:
        raise ValueError();
    return float(entries[0]), float(entries[1]);


def __textSplit3(text):
    entries = text.split()
    if len(entries) < 3:
        raise ValueError()
    return float(entries[0]), float(entries[1]), float(entries[2]);


def __readNumberIgnoringNone(value):
    if value.lower() == "none":
        return 0
    else:
        return float(value)


__propertyHeaderRegular = re.compile("#([ve]) \"(.+)\" ([sn]|v2|v3)")
__propertyFunctions = {
    "s": str,
    "n": __readNumberIgnoringNone,
    "v2": __textSplit2,
    "v3": __textSplit3
}


def __readXnetPropertyHeader(fp, currentLineIndex, lastLine=""):
    global __propertyHeaderRegular
    headerLine = lastLine
    if len(headerLine) == 0:
        for lineData in fp:
            headerLine = lineData.rstrip()
            currentLineIndex += 1
            if len(headerLine) > 0:
                break

    headerEntries = __propertyHeaderRegular.findall(headerLine)

    if len(headerEntries) == 0 or (len(headerEntries) == 1 and len(headerEntries[0]) != 3):
        raise ValueError("Malformed xnet file [%s:%d]\n\t>%s" % (fp.name, currentLineIndex, headerLine))

    (propertyType, propertyName, propertyFormat) = headerEntries[0]
    return (propertyType, propertyName, propertyFormat), currentLineIndex, ""


def from_xnet_to_igraph(fp):
    network = None

    currentLineIndex = 0
    lastLine = ""
    (nodeCount, currentLineIndex, lastLine) = __readXnetVerticesHeader(fp, currentLineIndex, lastLine)
    (names, currentLineIndex, lastLine) = __readXnetNames(fp, currentLineIndex, lastLine)
    if 0 < len(names) < nodeCount:
        raise ValueError("Malformed xnet file [%s:%d]\n\t>%s [%d entries expected but only %d found]" % (
            fp.name, currentLineIndex, headerLine, nodeCount, len(names)))
    ((weighted, directed), currentLineIndex, lastLine) = __readXnetEdgesHeader(fp, currentLineIndex, lastLine)
    ((edges, weights), currentLineIndex, lastLine) = __readXnetEdges(fp, currentLineIndex, lastLine)
    network = ig.Graph(nodeCount, edges=edges, directed=directed)
    if len(names) > 0:
        network.vs["name"] = names
    if weighted:
        network.es["weight"] = weights
    while lastLine != "":
        ((propertyType, propertyName, propertyFormat), currentLineIndex, lastLine) = __readXnetPropertyHeader(fp,
                                                                                                              currentLineIndex,
                                                                                                              lastLine)
        (properties, currentLineIndex, lastLine) = __readXnetProperty(fp, propertyFormat, currentLineIndex,
                                                                      lastLine)
        if propertyType == "e":
            network.es[propertyName] = properties
        elif propertyType == "v":
            network.vs[propertyName] = properties
    return network

