import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
import sklearn_crfsuite
from sklearn.model_selection import KFold
import numpy
plt.style.use('ggplot')

FILE_PATH = 'nkjp-morph-named.txt'

dataset = []
entry = []
new_entry = []
with open(FILE_PATH, encoding='utf-8', errors='replace') as f:
    for line in f:
        line_split = line.split()
        if not entry and line_split[0] == ".":
            continue
        tmp = (line_split[0], line_split[-2], line_split[-1])
        entry.append(tmp)
        if line_split[0] == ".":
            new_entry = entry
            dataset.append(new_entry)
            entry = []


splits = train_test_split(dataset, test_size=0.43, random_state=0)
X_train, X_test = splits


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]



train_1 = [sent2features(s) for s in X_train]
train_2 = [sent2labels(s) for s in X_train]

test_1 = [sent2features(s) for s in X_test]
test_2 = [sent2labels(s) for s in X_test]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=85,
    all_possible_transitions=True
)

crf.fit(train_1 , train_2)

labels = list(crf.classes_)
labels.remove('O')
print(labels)

y_pred = crf.predict(test_1)
metrics.flat_f1_score(test_2, y_pred,
                      average='weighted', labels=labels)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(sorted_labels)

print(metrics.flat_classification_report(
    test_2, y_pred, labels=sorted_labels, digits=3
))

A = numpy.array([[1, 2], [3, 4], [1, 2], [3, 4], [5,6],[7,8]])
b = numpy.array([1, 2, 3, 4,5,6])
kf = KFold(n_splits=3)
kf.get_n_splits(A)
print(kf)
for train_index, test_index in kf.split(A):
     print("TRAIN:", train_index, "TEST:", test_index)
     b_train, b_test = b[train_index], b[test_index]


kf = KFold(n_splits=3)
kf.get_n_splits(train_1)
y_pred_all=[]

for train_index, test_index in kf.split(train_1):
    X_tr, X_tst = numpy.array(train_1)[train_index], numpy.array(train_1)[test_index]
    y_tr, y_tst = numpy.array(train_2)[train_index], numpy.array(train_2)[test_index]
    crf.fit(X_tr, y_tr)
    y_pred = crf.predict(X_tst)
    y_pred_all.extend(y_pred)

print("all results")
print(metrics.flat_classification_report(train_2, y_pred_all, labels=sorted_labels, digits=3))