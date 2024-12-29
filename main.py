import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import random
import string
import sklearn
import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# part 1 

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

allRatings = []
userRatings = defaultdict(list)
userBooks = defaultdict(list)

for user,book,r in readCSV("train_Interactions.csv.gz"):
    userRatings[user].append(r)
    userBooks[user].append(book)
    
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
    
ratingsValid = allRatings[190000:]
ratingsTrain = allRatings[:190000]

validation = []

for user,book,rating in ratingsValid: 
    validation.append([user, book, rating, 1])
    as_list = list(userBooks[user])
    random_book = random.choice(as_list)
    validation.append([user, random_book, rating, 0])
    
userSet = set()
bookSet = set()
readSet = set()

for u,b,r in allRatings:
    userSet.add(u)
    bookSet.add(b)
    readSet.add((u,b))

lUserSet = list(userSet)
lBookSet = list(bookSet)

readValid = set()
for u,b,r in ratingsValid:
    readValid.add((u,b))
    
notRead = set()
for u,b,r in ratingsValid:
    
    b = random.choice(lBookSet)
    while ((u,b) in readSet or (u,b) in notRead):
        b = random.choice(lBookSet)
    notRead.add((u,b))

acc_list = {}
thresehold = 0 
li = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for j in li: 
    
    count = 0 
    return1 = set()
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead * j: break
            
    correct = 0
    p0, p1 = 0,0
    for (label,sample) in [(1, readValid), (0, notRead)]:
        for (u,b) in sample:
            pred = 0
            if b in return1:
                pred = 1
            if pred == label:
                correct += 1
    acc = correct / (len(readValid) + len(notRead))
    acc_list[acc] = j
 

acc2 = max(acc_list)
threshold = acc_list[acc2]



return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead * threshold: break #from hw3 num 2 
#     if count > totalRead/2: break

train = pd.read_csv("train_Interactions.csv.gz")
train = train.set_index("bookID")
new = train.drop(['rating'], axis=1)
l = []

for bookID in new.index:
    book = bookID
    if bookID not in return1: 
        l.append(0)
    else: 
        l.append(1)
new["prediction"] = l

trial = new.reset_index()

x = trial[["bookID"]]
y = trial[["prediction"]]

preproc = ColumnTransformer(transformers = [
        ('one-hot1', OneHotEncoder(handle_unknown='ignore'), 
         ["bookID"])],
        remainder = "drop")

pipe = Pipeline([('preprocessor', preproc), ('dt_classifier', 
                                             DecisionTreeClassifier())])
pipe = pipe.fit(x, y)

test = pd.read_csv("pairs_Read.csv")
X_test = test[["bookID"]]
 
prediction = pipe.predict(X_test)
test["prediction"] = prediction
test = test.set_index("bookID")


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    try:
        a = (set(test["prediction"][b]))
    except: 
        a = ({test["prediction"][b]})
        

    if a == {1}:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")

predictions.close()

#part 2

train_cateogry = readGz("train_Category.json.gz")
dataset = []
for l in train_cateogry:
    dataset.append(l)
    
punctuation = string.punctuation

ys = []

for l in dataset:
    ys.append(l['genreID'])
    
word_list = []
for l in dataset:
    words = l['review_text'].lower()
    word_list.append(words)

text_transformer = TfidfVectorizer()
transformed_tfidf = text_transformer.fit_transform(word_list)

Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(word_list,ys, test_size = .25)

model = Pipeline([("count", TfidfVectorizer()),('svc', LinearSVC())])

model.fit(Xtrain, ytrain)

x_valid = []
for l in readGz("test_Category.json.gz"):
    to_add =  l['review_text']
    x_valid.append(to_add)

pred = model.predict(x_valid)

predictions = open("predictions_Category.csv", 'w')
predictions.write("userID,reviewID,prediction\n")
i = 0
for l in readGz("test_Category.json.gz"):
    cat = pred[i]
    i  = i + 1
    predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(cat) + "\n")
predictions.close()