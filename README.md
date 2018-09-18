# DecisionTree
Homework for 10-601: Introduction to Machine Learning

Split node using mutual information and classify using majority vote. 

### Usage
`inspect.py` deals with data processing and entropy/error calculation. Run 
```
python inspect.py data/small_train.csv small_inspect.txt
```
to get entropy and error (of majority vote) of the labels in `data/small_train.csv`.

`decisionTree.py` can train a decision tree and classify labels given testing data. 
```
python inspect.py <train input> <test input> <max depth> <train out> <test out> <metrics out>
```
For example,
```
python decisionTree.py data/small_train.csv data/small_test.csv 3 small_2_train.labels small_2_test.labels small_2_metrics.txt
```
