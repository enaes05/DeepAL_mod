import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, net):
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, n):
        half_num = int(n/2)
        
        x = np.where(self.dataset.labeled_idxs==0)[0]
        class_0 = []
        class_1 = []
        for a in x:
          if(self.dataset.Y_train[a] == 0): class_0.append(a)
          elif(self.dataset.Y_train[a] == 1): class_1.append(a)


        np.random.shuffle(class_0)
        np.random.shuffle(class_1)
        
        return class_0[:half_num] + class_1[:half_num]
        # return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
