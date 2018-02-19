import numpy as np
from scipy.spatial import distance
###Distance Metric : L2 ###
class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k):
    """
    STUDENT CODE
    """

    #calculate the k near neighber
    dists = self.compute_distances(X)
    #Vote
    return self.predict_labels(dists, k=k)
   
 
  def compute_distances(self, X):
    """
    STUDENT CODE
    """
    data = X.toarray()
    result = []
    for vector in data:
        distance_to_train = []
        for vector_train in self.X_train.toarray():
            distance_to_train.append(np.linalg.norm(vector - vector_train)) #L2 norm
        result.append(distance_to_train)
    return np.array(result)

  def predict_labels(self, dists,k):
    """
    STUDENT CODE
    """

    top_k = []

    for line in dists:
        vote = [0, 0, 0, 0]
        if k > line.size:
            k = line.size
        top_k_index = line.argsort()[:k]
        for index in top_k_index:
            current_label = self.y_train[index]
            vote[current_label] += 1
        label = np.array(vote).argsort()[-1:][::-1]
        top_k.append(label[0])
    return top_k