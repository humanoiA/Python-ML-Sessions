from numpy import *
from operator import itemgetter
from sklearn.metrics.pairwise import euclidean_distances
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    label=['A','A','B','B']
    return group,label
def classify(p_data,dataset,label,k):
    dataSetSize=dataset.shape[0]
    diffmat=tile(p_data,(dataSetSize,1))-dataset
    sqDiffMat=diffmat**2
    sqDistance=sqDiffMat.sum(axis=1)
    #distance=sqDistance**.5
    distance=euclidean_distances(dataset,p_data)
    sortMetDistances=distance[:,0].argsort()
    classcout={}
    for i in range(k):
        votlabel=label[sortMetDistances[i]]
        classcout[votlabel]=classcout.get(votlabel,0)+1
        sortedClassCount=sorted(classcout.iteritems(),
                                key=itemgetter(1),reverse=True)
    print(classcout)
    return sortedClassCount[0][0]

