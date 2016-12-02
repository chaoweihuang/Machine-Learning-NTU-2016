from gensim.models import Doc2Vec, Word2Vec
from utility import *
from nltk.corpus import stopwords
from os import path
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

import numpy as np
import re
import sys


datadir = sys.argv[1]
output = sys.argv[2]
titles_path = path.join(datadir, 'title_StackOverflow.txt')
docs_path = path.join(datadir, 'docs.txt')
index_path = path.join(datadir, 'check_index.csv')

print('reading test indices...')
indices = get_index(index_path)

print('getting stopwords from nltk...')
stop_words = stopwords.words('english')

print('reading data from title_StackOverflow.txt')
docs = [re.findall('\w+', line.strip().lower()) for line in open(titles_path)]
#docs = [line.strip().lower().split() for line in open(titles_path)]
lines = [line.strip().lower() for line in open(titles_path)]

print('reading data from docs.txt')
docs_all = docs + [re.findall('\w+', line.strip().lower()) for line in open(docs_path)]
#docs_all = docs + [line.strip().lower().split() for line in open(docs_path)]
lines_all = lines + [line.strip().lower() for line in open(docs_path)]

print('loading Doc2Vec model')
model = Doc2Vec.load('d2v_dbow_bulk_mc5_re')

print('calculating average vector for each title')
vecs = []
for i in range(20000):
    title_vecs = []
    for w in docs[i]:
        try:
            if w not in stop_words:
                title_vecs.append(model[w])
        except:
            pass
    if not len(title_vecs) == 0:
        vecs.append(np.mean(np.asarray(title_vecs), axis=0))
    else:
        vecs.append(model.docvecs['TITLE_%d' % i])
        
anss = []

print('applying PCA')
pca = PCA(n_components=30)
normalizer = Normalizer(copy=False)
decomposition = make_pipeline(pca, normalizer)
X = decomposition.fit_transform(vecs)

passes = 5
print('applying KMeans for %d times' % passes)
for i in range(passes):
    print('\t KMeans no.%d' % (i+1))
    kmeans = KMeans(n_clusters=25)
    kmeans.fit(X)

    ans = []
    for index in indices:
        if kmeans.labels_[index[0]] == kmeans.labels_[index[1]]:
            ans.append(1)
        else:
            ans.append(0)

    anss.append(ans)

#print(len(anss))
print('calculating voting result and write submission file')
anss_ = (np.mean(anss, axis=0) > 0.5).astype(int)
write_submit_file(anss_, output)
