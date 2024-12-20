import os
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from surprise import Dataset, SVD, Reader, KNNWithMeans, accuracy, BaselineOnly, get_dataset_dir
from surprise.model_selection import cross_validate, KFold
from surprise.similarities import pearson_baseline, pearson
from sympy import false

def dcg_at_k(relevance_scores, k):
    dcg = 0

    for i in range(len(relevance_scores)):
        value = 0 if(i> len(relevance_scores)-1) else relevance_scores[i]
        if value == 0: print("zero")
        dcg += value / np.log2(i + 2)  # +2 because log2(1) = 0
    return dcg
def idcg_at_k(relevance_scores, k):
    relevance_scores_sorted = sorted(relevance_scores, reverse=True)
    return dcg_at_k(relevance_scores_sorted, k)
def ndcg_at_k(predictions, testset, k=10):
    actual_ratings = defaultdict(list)
    for uid, iid, true_r in testset:
        actual_ratings[uid].append((iid, true_r))

    predicted_ratings = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        predicted_ratings[uid].append((iid, est))


    ndcg_scores = []
    for uid in actual_ratings:
        predicted_ratings_for_user = sorted(predicted_ratings[uid], key=lambda x: x[1], reverse=True)[:k]


        actual_relevances = dict(actual_ratings[uid])
        relevance_scores = [actual_relevances.get(iid, 0) for iid, _ in predicted_ratings_for_user]
        dcg = dcg_at_k(relevance_scores, k)
        idcg = idcg_at_k(relevance_scores, k)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcg_scores)

def get_top_n(predictions, n=10, sort=1):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        if not sort: user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

file_path = os.path.expanduser("./ml-100k/u.data")
reader = Reader(line_format="user item rating timestamp", sep="\t")

sim_options = {"name": "pearson"}
data = Dataset.load_from_file(file_path, reader=reader)

#KNN algorithm with Pearson Correlation Coefficient
algo = KNNWithMeans(50, sim_options=sim_options)

#Divides data set into 5 test & train set pairs with 5 fold cross validation
kf = KFold(n_splits=5)
folds = []
for trainset, testset in kf.split(data):
    folds.append((trainset, testset))


accuracies = []
top10 = []
foldPrecisions = []
foldRecalls = []
predictions = []
ndcgs = []
#Trains and calculates Mae, precision, recall and ndcg for each fold
for fold in folds:
    #Train fold operations, fold[0] is training set, fold[1] is test set
    algo.fit(fold[0])
    preds = algo.test(fold[1])
    predictions.append(preds)

    #MAE calculation
    accuracies.append(accuracy.mae(preds, verbose=0))
    #Precision calculation
    precisions, recalls = precision_recall_at_k(preds, k=10, threshold=4)
    foldRecalls.append(recalls)
    foldPrecisions.append(precisions)
    #Finds top 10 movies for each user
    top10Pred = get_top_n(preds, n=10)
    top10.append(top10Pred)

    #NDCG calculation
    ndcg = ndcg_at_k(preds, fold[1], 10)
    ndcgs.append(ndcg)

#Prints MAE accuracies for each fold
print()
index = 1
for MAE in accuracies:
    print(f"For fold {index} MAE is: {MAE}")
    index += 1

#Prints Precision and Recall values for each fold
print()
index = 1
sumPrec = 0
sumRec = 0
for precisions, recalls in zip(foldPrecisions, foldRecalls):
    prec = sum(prec for prec in precisions.values()) / len(precisions)
    rec = sum(rec for rec in recalls.values()) / len(recalls)
    sumPrec += prec
    sumRec += rec
    print(f"For Fold{index} Precision is {prec} , Recall is {rec}")
    index += 1

print(f"Avg Prec is: {sumPrec/5}, Avg Rec is {sumRec/5}")

print()
index = 1
for ndcg in ndcgs:
    print(f"For Fold{index} NDCG@10 is {ndcg}")
    index += 1

print(f"Averages: NDCG@10 Avg: {np.asarray(ndcgs).sum()/5}, Mae Avg: {np.asarray(accuracies).sum()/5}")

















