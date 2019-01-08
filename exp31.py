from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
array=['Hi how are you','I am fine','How about you','I am also fine']
vect=CountVectorizer()
feature = (vect.fit_transform(array).todense())
for f in feature:
    print(euclidean_distances(feature[0],f))
