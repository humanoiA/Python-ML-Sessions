from sklearn.feature_extraction.text import CountVectorizer
array=['Hi how are you','I am fine','How about you','I am also fine']
vect=CountVectorizer()
print(vect.fit_transform(array).todense())
print(vect.vocabulary_)
#count vectorizer finds unique elements sorts them and assign value 1 and 0 in matrix form according to value assigned