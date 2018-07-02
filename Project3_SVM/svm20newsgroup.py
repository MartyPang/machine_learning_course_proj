from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import os

derectory = "../data/20_newsgroups"

# delete incompatible files
count_vector = CountVectorizer()
files = load_files(derectory)
incompatible_files = []
for i in range(len(files.filenames)):
    try:
        count_vector.fit_transform(files.data[i:i + 1])
    except UnicodeDecodeError:
        incompatible_files.append(files.filenames[i])
    except ValueError:
        pass

if 0 < len(incompatible_files):
    for f in incompatible_files:
        os.remove(f)


news_files = load_files(derectory)
# calculate the BOW representation
word_counts = count_vector.fit_transform(news_files.data)

# TFIDF
tf_transformer = TfidfTransformer(use_idf=True).fit(word_counts)
X_tfidf = tf_transformer.transform(word_counts)

X = X_tfidf

# 5-fold cross validation
clf = LinearSVC()
scores = cross_val_score(clf, X, news_files.target, cv=5)

print("Mean accuracy: %0.3f (+/- %0.3f std) " % (scores.mean(), scores.std() / 2))



