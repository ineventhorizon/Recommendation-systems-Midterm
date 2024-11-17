import os
from collections import defaultdict

from sklearn.model_selection import train_test_split
from surprise import Dataset, SVD, Reader, KNNWithMeans, accuracy, BaselineOnly, get_dataset_dir
from surprise.model_selection import cross_validate, KFold
from surprise.similarities import pearson_baseline, pearson
from sympy import false



def get_top_n(predictions, n=10):
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
        user_ratings.sort(key=lambda x: x[1], reverse=True)
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

#Calculates MAE accuracy for each train & test datasets
#Also calculates top 10 movies for Toy Story for every fold
accuracies = []
top10MoviesForEachUser = []
foldPrecisions = []
foldRecalls = []
for fold in folds:
    algo.fit(fold[0])
    preds = algo.test(fold[1])
    print(preds[0])
    accuracies.append(accuracy.mae(preds, verbose=0))
    precisions, recalls = precision_recall_at_k(preds, k=10, threshold=4)
    foldRecalls.append(recalls)
    foldPrecisions.append(precisions)
    #Finds top 10 movies for each user
    top10 = get_top_n(preds, n=10)
    top10MoviesForEachUser.append(top10)


#Prints MAE accuracies for each fold
print()
index = 1
for MAE in accuracies:
    print(f"For fold {index} MAE is: {MAE}")
    index += 1

#Prints Precision and Recall values for each fold
print()
index = 1
for precisions, recalls in zip(foldPrecisions, foldRecalls):
    prec = sum(prec for prec in precisions.values()) / len(precisions)
    rec = sum(rec for rec in recalls.values()) / len(recalls)
    print(f"For Fold{index} Precision is {prec} , Recall is {rec}")
    index += 1

















