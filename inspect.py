import numpy as np
import csv

def getEntropy(data):
    if len(data.shape) == 2:
        # H(X, Y)
        n = float(data.shape[0])
        p = [0]*4
        for x in data:
            if x[0] == 0 and x[1] == 0: p[0] += 1
            if x[0] == 1 and x[1] == 0: p[1] += 1
            if x[0] == 0 and x[1] == 1: p[2] += 1
            if x[0] == 1 and x[1] == 1: p[3] += 1
        p = np.array(p) / n
        return np.sum([-x*np.log2(x) if x!=0 else 0 for x in p])
    elif len(data.shape) == 1:
        # H(X)
        n = data.shape[0]
        p = np.sum(data) / float(n)
        if p == 0 or p == 1: return 0
        return -p*np.log2(p)-(1-p)*np.log2(1-p)
    else:
        return 0

def getInformationGain(data):
    assert len(data.shape)==2 and data.shape[1]==2, "wrong dimension of data"
    HX = getEntropy(data[:, 0])
    HY = getEntropy(data[:, 1])
    HXY = getEntropy(data)
    return HX + HY - HXY

def getVote(data):
    if len(data.shape) == 1: 
        n = data.shape[0]
        return n - np.sum(data), np.sum(data)

def getError(data):
    n = data.shape[0]
    p = np.sum(data) / float(n)
    if (p >= 0.5): return 1-p
    else: return p

def readInput(fn, lookup=None):
    # assume fn is a .cvs file with delimiter ','
    raw_data = []
    with open(fn, 'r') as f:
        header = f.readline().strip('\n').split(',')
        reader = csv.reader(f, delimiter=',')
        for row in reader: # each row is a list
            raw_data.append(row)
        
    raw_data = np.array(raw_data)

    if lookup is None:
        # build dictionary
        lookup = []
        for i, s in enumerate(header):
            lookup.append({'name': s.strip(' '), 'attr': [], 'col': i})
            for l in raw_data[:,i]:
                if len(lookup[i]['attr']) == 1:
                    if lookup[i]['attr'][0] == l: continue
                    else: 
                        lookup[i]['attr'].append(l); break
                else: 
                    lookup[i]['attr'].append(l)

    # reorganize data
    data = []
    m = raw_data[0].shape[0]
    for l in raw_data:
        line = [0] * m
        for i, x in enumerate(l):
            if len(lookup[i]['attr'])<2:
                # only one attribution in training data
                line[i] = 1
                lookup[i]['attr'].append(x)
            if lookup[i]['attr'][1] == x: line[i] = 1
        data.append(line)

    return lookup, np.array(data)


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 3, "python inspect.py <input> <output>"

    inputf = sys.argv[1]
    outputf = sys.argv[2]

    lookup, data = readInput(inputf)
    entropy = getEntropy(data[:, -1])
    error = getError(data[:, -1])

    with open(outputf, 'w') as f:
        f.write("entropy: " + str(entropy) + '\n')
        f.write("error: " + str(error))
