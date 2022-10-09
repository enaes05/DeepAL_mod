import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="random seed")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "HeadAndNeck"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool", 
                             "ActiveLearningByLearning"], help="dataset")
args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device)                   # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Training with initial labels...")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
print(f"Model testing accuracy: {dataset.cal_test_acc(preds)}")

for rd in range(1, args.n_round+1):
    print(f"-------------------")
    print(f"ROUND {rd}")

    # query unlabeled samples and save the indexes
    query_idxs = strategy.query(args.n_query)

    # get indexes from already labeled indexes
    labeled_idxs = [i for i, x in enumerate(dataset.labeled_idxs == True) if x]
    # merge all the indexes
    labels_to_train = query_idxs + labeled_idxs

    # apply DBSCAN
    clustering = DBSCAN(eps=6, min_samples=3,metric='l2').fit(dataset.X_train[labels_to_train])
    

    print('Unlabeled samples classes: ',dataset.Y_train[query_idxs])
    print('DBSCAN results of unlabeled: ',clustering.labels_[:args.n_query])
    # print('Labeled samples classes: ',dataset.Y_train[labeled_idxs])
    # print('DBSCAN results of labeled: ',clustering.labels_[args.n_query:])

    # update labels
    for x in range(len(query_idxs)):
      if(clustering.labels_[x]!=-1): strategy.update(query_idxs[x])

    # get labels of selected instances
    new_preds = strategy.predict(dataset.handler(dataset.X_train[query_idxs],dataset.Y_train[query_idxs]))
    # new_preds = strategy.predict(dataset.handler(dataset.random_data[query_idxs],dataset.Y_train[query_idxs]))
    print('Prediction of unlabeled samples: ',new_preds)
    
