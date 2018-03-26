import random
import numpy as np
import igraph
#import gensim
#import graph_tool.all as gt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RndForestClass
from sklearn.ensemble import AdaBoostClassifier as AdaClass
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import pickle as pc
import nltk
import csv
import networkx as netx
nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()
path = "C:\\Users\\Guillaume\\Desktop\\kaggle\\"
#path = "/media/plays/Windows8_OS/Users/Guillaume/Desktop/kaggle"
with open(path + "testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

###################
# random baseline #
###################
#
#random_predictions = np.random.choice([0, 1], size=len(testing_set))
#random_predictions = zip(range(len(testing_set)),random_predictions)
#
#with open("random_predictions.csv","wb") as pred:
#    csv_out = csv.writer(pred)
#    for row in random_predictions:
#        csv_out.writerow(row)
#        
# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

#load Google word2vec module the use of word2vec was abandonned during the
# project
#model = gensim.models.word2vec.Word2Vec(corpus, min_count=1)
#model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
#model.build_vocab(corpus)  # can be a non-repeatable, 1-pass generator

#model.intersect_word2vec_format(path + 'GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.Word2Vec.load_word2vec_format(path + 'GoogleNews-vectors-negative300.bin', binary=True)

## the following is the construction of a graph with igraph

edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

## some nodes may not be connected to any other node
## hence the need to create the nodes of the graph from node_info.csv,
## not just from the edge list

nodes = IDs

## create empty directed graph
g = igraph.Graph(directed=True)
#g = gt.Graph()

## add vertices
g.add_vertices(nodes)
#g.add_vertex(n=nodes[-1])

## add edges
g.add_edges(edges)
#g.add_edge_list(edges)

## Addition of PageRankFeatures
pageR = g.pagerank(nodes)

## Additon of Katz Feature
## Not used in final version

#print '"begin Katz computation"
#katzCent = gt.katz(g, alpha=0.01, beta=None, weight=None, vprop=None, epsilon=1e-06, max_iter=None, norm=True)
#array_katzCent = [ele for ele in katzCent.get_arrray()]

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set
noDump = False
training_features = []
# randomly select 5% of training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
training_set_reduced = [training_set[i] for i in to_keep]
#training_set_reduced=training_set

# we will use three basic features:

# number of overlapping words in title
overlap_title = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []

#distance between abstract according to TF_IDF
distance_tfidf = []

#adhesion between two vertex
adhesion = []

# pageRank probability
pageRankP = []
pageRankTarget = []
counter = 0
for i in xrange(len(training_set_reduced)):
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]

    index_source = IDs.index(source)
    index_target = IDs.index(target)

    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]

    # convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
    # remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    distance_tfidf.append(cosine_similarity(features_TFIDF[index_source], features_TFIDF[index_target]))
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    pageRankP.append(pageR[index_source]*pageR[index_target])
    pageRankTarget.append(pageR[index_target])
    adhesion.append(g.adhesion(source=index_source, target=index_target)-1)
    counter += 1
    if counter % 1000 == True:
        print counter, "training examples processed"

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_title, temp_diff, comm_auth,distance_tfidf,pageRankP,adhesion,pageRankTarget]).T


# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

# initialize basic SVM
#classifier = svm.LinearSVC()
#classifier = RndForestClass()
classifier = AdaClass(learning_rate=0.5)
#classifier = GaussianNB()

# train
classifier.fit(training_features, labels_array)

# test
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
distance_tfidf_test = []
pageRankP_test = []
pageRankTarget_test = []
adhesion_test = []

counter = 0
for i in xrange(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]

    index_source = IDs.index(source)
    index_target = IDs.index(target)

#    source = training_set_reduced[i][0]
#    target = training_set_reduced[i][1]
#
#    index_source = IDs.index(source)
#    index_target = IDs.index(target)

    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]

    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    distance_tfidf_test.append(cosine_similarity(features_TFIDF[index_source], features_TFIDF[index_target]))
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    pageRankP_test.append(pageR[index_source]*pageR[index_target])
    pageRankTarget_test.append(pageR[index_target])
    adhesion_test.append(g.adhesion(source=index_source, target=index_target))
    counter += 1
    if counter % 1000 == True:
        print counter, "testing examples processed"

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([overlap_title_test,temp_diff_test,comm_auth_test,distance_tfidf_test,pageRankP_test,adhesion_test,pageRankTarget_test]).T


# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

with open("improved_predictions_Ada2.csv","wb") as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(["id","category"])
    for row in predictions_SVM:
        csv_out.writerow(row)