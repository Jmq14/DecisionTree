import numpy as np
import csv
from inspect import getInformationGain, getVote, readInput

class Node:

    def __init__(self, depth):
        self.branch0 = None
        self.branch1 = None
        self._attr = None
        self._depth = depth
        self._classify = -1 # -1: not leaf

    def __str__(self):
        if self._depth == 0: # root
            s = '[' +  str(self._vote[0]) + ' ' + self.lookup[-1]['attr'][0] + ' / ' + \
                str(self._vote[1]) + ' ' + self.lookup[-1]['attr'][1] + ']\n' 
        else:
            s = '| ' * self._depth + self.lookup[0]['name'] + ' = ' + \
                self.lookup[0]['attr'][self._value] + ': [' +  \
                str(self._vote[0]) + ' ' + self.lookup[-1]['attr'][0] + ' / ' + \
                str(self._vote[1]) + ' ' + self.lookup[-1]['attr'][1] + ']\n'

        if self.branch0: s += str(self.branch0)
        if self.branch1: s += str(self.branch1)
        return s

    def set_attr(self, attr):
        self._attr = attr

    def set_mutual_info(self, info):
        self._mutual_info = info

    def set_vote(self, vote):
        assert len(vote) == 2, "vote should include positive and negative numbers"
        self._vote = vote

    def set_value(self, value):
        self._value = value

    def set_classify(self, iclass):
        # 1 represents positive classifier, while 0 is negative one.
        # dufault class is -1, meaning this node is not leaf
        self._classify = iclass

    def set_lookup(self, lookup):
        self.lookup = lookup.copy()

    def train(self, data, lookup, max_depth):
        attr_n = data.shape[1] - 1
        if attr_n == 0 or self._depth == max_depth:
            # Use a majority vote of the labels at each leaf to make classification decisions.
            self.set_vote(getVote(data[:, -1]))
            if self._vote[0] >= self._vote[1] : 
                self.set_classify(0)
            else:
                self.set_classify(1)

        elif self._depth < max_depth:
            # Use mutual information to split an attribution
            max_mutual_info = 0
            attr_to_split = -1
            for i in range(attr_n): # iterate over all colums
                mutual_info = getInformationGain(data[:, [i, -1]])
                if mutual_info > max_mutual_info:
                    max_mutual_info = mutual_info
                    attr_to_split = i

            if attr_to_split == -1:
                # labels are all the same
                self.set_vote(getVote(data[:, -1]))
                if self._vote[0] >= self._vote[1] : 
                    self.set_classify(0)
                else:
                    self.set_classify(1)
                return

            self.set_vote(getVote(data[:, -1]))
            self.set_mutual_info(max_mutual_info)
            self.set_attr(lookup[attr_to_split])
            
            # construct data for branches
            data1 = data[data[:,attr_to_split] == 1]
            data1 = np.delete(data1, attr_to_split, axis=1)
            self.branch1 = Node(self._depth+1)
            self.branch1.set_lookup([lookup[attr_to_split]] + [lookup[-1]])
            self.branch1.set_value(1)

            data0 = data[data[:,attr_to_split] == 0]
            data0 = np.delete(data0, attr_to_split, axis=1)
            self.branch0 = Node(self._depth+1)
            self.branch0.set_lookup([lookup[attr_to_split]] + [lookup[-1]])
            self.branch0.set_value(0)

            del lookup[attr_to_split]
            self.branch1.train(data1, lookup.copy(), max_depth)
            self.branch0.train(data0, lookup.copy(), max_depth)

    def forward(self, data):
        if self._classify != -1: 
            return self.lookup[-1]['attr'][self._classify]
        else:
            attr_id = self._attr['col']
            value = data[attr_id]
            if value == 1: 
                return self.branch1.forward(data)
            else: 
                return self.branch0.forward(data)

class DT:

    def __init__(self, data, lookup):
        self.root = None
        self.data = data
        self.lookup = lookup

    def __str__(self):
        return str(self.root)

    def train(self, max_depth):
        self.root = Node(0)
        self.root.set_lookup(self.lookup)
        self.root.train(self.data, self.lookup, max_depth)

    def predict(self, data):
        assert self.root is not None, "Model haven't been trained!"
        assert len(data.shape)==1, "Feed single row of data at a time"
        return self.root.forward(data)


def DTTrain(fn, max_depth):
    lookup, data = readInput(fn)
    dt = DT(data, lookup.copy())
    dt.train(max_depth)
    return dt, lookup

def DTPredict(dt, lookup, fn):
    _, data = readInput(fn, lookup)
    predictions = []
    for d in data:
        predictions.append( dt.predict(d) )
    return predictions

def writeOutput(predictions, fn):
    with open(fn, 'w') as f:
        if type(predictions) == list:
            f.write('\n'.join(predictions))
        elif type(predictions) == np.ndarray:
            f.write('\n'.join(predictions.reshape(1, -1)[0].tolist()))

def getPredictionError(inputf, predictions):
    ground_truth = []
    with open(inputf, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ground_truth.append(row[-1])
        ground_truth = ground_truth[1:]
    assert len(predictions) == len(ground_truth), "predictions in wrong length"
    n = len(predictions)
    error = 0.
    for x, y in zip(predictions, ground_truth):
        if x != y: error += 1
    return error / n

def writeMetrics(metrics_out, train_input, train_pred, test_input, test_pred):
    with open(metrics_out, 'w') as f:
        f.write('error(train):' + \
            str(getPredictionError(train_input, train_pred)) + '\n')
        f.write('error(test):' + \
            str(getPredictionError(test_input, test_pred)))

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 7, "python inspect.py <train input> <test input> <max depth> <train out> <test out> <metrics out>"

    # read arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    # train decision tree
    dt, lookup = DTTrain(train_input, max_depth)

    # print out the trained decision tree
    print()
    print(dt)

    # write predicted labels into files
    train_pred = DTPredict(dt, lookup.copy(), train_input)
    test_pred  = DTPredict(dt, lookup.copy(), test_input)
    writeOutput(train_pred, train_out)
    writeOutput(test_pred, test_out)

    writeMetrics(metrics_out, train_input, train_pred, test_input, test_pred)