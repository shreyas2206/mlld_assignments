import nltk
import time
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
import numpy as np

# To remove all unwanted stuff from the documents
def process_text(soup):
    soup_temp=(BeautifulSoup(soup).get_text().lower().translate(str.maketrans('','',string.punctuation)))[:-4].split()
    soup=[[w for w in soup_temp if ((not w in stop_words) and (len(w) > 2)) ]]
    return soup

# Function to increment count(word,y') in a dictionary.
def add2dict(d,word,label,init_dict):
    if word in d:
        d[word][label]+=1
    else:
        d[word]=dict(init_dict) #Add a word which is seen for the first time to the Dictionary
        d[word][label]+=1

# To calculate the label counts (Y=y')
def inc_count(Y,label):
    if label in Y:
        Y[label]+=1
    else:
        Y[label]=1

# To output the class decision by argmax on posteriors (MAP detection)
def argmax_class(post):
    v=list(post.values())
    k=list(post.keys())
    return k[v.index(max(v))]

# Read and process the train data
start_time=time.time()
train_data = open('/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt').readlines()
a=[train_data[i].split(" ",1) for i in range(3,len(train_data))]

for l in a:
	l[0]=l[0].split(",")
	l[1]=process_text(l[1])[0]

#Get Unique train Labels
train_labels = list(set([s for ss in [l[0] for l in a] for s in ss]))
init_dict = dict(zip(train_labels,[0 for i in range(len(train_labels))]))

d=dict()
Y=dict()
cYany=0

# Get the various counts
for eachline in a:
    for eachlabel in eachline[0]:
        inc_count(Y,eachlabel)
        cYany+=1
        for eachword in eachline[1]:
            add2dict(d,eachword, eachlabel,init_dict)

# Read and process the test data
test_data = open('/scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt').readlines()
b=[test_data[i].split(" ",1) for i in range(3,len(test_data))]

for l in b:
	l[0]=l[0].split(",")
	l[1]=process_text(l[1])[0]

# Compute counts (X=ANY, Y=y')
joint_count_matrix=np.array([[d[word][c] for c in train_labels] for word in d])
class_counts=np.sum(joint_count_matrix,axis=0)
fin_dict=dict(zip(train_labels,class_counts))

qx=1/len(d)
qy=1/len(Y)

# Obtain a vector of posteriors for each test document
posts=[dict(zip(train_labels,
                [np.sum(np.log(np.array([(d[eachword][y]+qx)/(fin_dict[y]+1) for eachword in eachline[1]])))+np.log((Y[y]+qy)/cYany+1) for y in train_labels])) for eachline in b]
    

# Obtain accuracies
ground_truth=[set(l[0]) for l in b]
decisions=[argmax_class(post) for post in posts]


correctly_classified=0
for i in range(len(b)):
    if decisions[i] in ground_truth[i]:
        correctly_classified+=1

accuracy=100*correctly_classified/len(b)
end_time=time.time()

time_taken=end_time-start_time

accuracy
time_taken
