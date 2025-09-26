import pandas as pd
import math


def naive_bayes(train_file="../dataset/bbcsports_train.csv",
                val_file="../dataset/bbcsports_val.csv",
                laplace=False):
    """
    Simple Multinomial Naive Bayes Classifier for BBC Sports dataset.
    Args:
        train_file (str): Path to training dataset CSV
        val_file (str): Path to validation dataset CSV
        laplace (bool): Whether to use Laplace smoothing
    """
    # Data
    traindata = pd.read_csv(train_file, header=None)
    validationdata = pd.read_csv(val_file, header=None)

    n_features = traindata.shape[1] - 1  # labels
    labels = [0, 1, 2, 3, 4]  # Athletics=0, Cricket=1, Football=2, Rugby=3, Tennis=4

    # Docs per class
    class_counts = {c: 0 for c in labels}
    # Features per class
    feature_totals = {c: 0 for c in labels}
    # Log probabilities
    feature_log_probs = {c: [0] * n_features for c in labels}

    # --- Training ---
    for row in traindata.values:
        features, label = row[:-1], int(row[-1])
        class_counts[label] += 1
        feature_totals[label] += sum(features)

    for row in traindata.values:
        features, label = row[:-1], int(row[-1])
        for j in range(n_features):
            if laplace:
                prob = (features[j] + 1) / (feature_totals[label] + n_features)
            else:
                if feature_totals[label] == 0 or features[j] == 0:
                    continue
                prob = features[j] / feature_totals[label]
            if prob > 0:
                feature_log_probs[label][j] += math.log(prob, 10)

    # Prior probabilities
    total_docs = len(traindata)
    priors = {c: math.log(class_counts[c] / total_docs, 10) for c in labels}

    # --- Validation ---
    accurate, inaccurate = 0, 0

    for row in validationdata.values:
        features, true_label = row[:-1], int(row[-1])

        scores = {}
        for c in labels:
            score = priors[c]
            for j in range(n_features):
                score += features[j] * feature_log_probs[c][j]
            scores[c] = score

        prediction = max(scores, key=scores.get)

        if prediction == true_label:
            accurate += 1
        else:
            inaccurate += 1

    accuracy = accurate / (accurate + inaccurate) * 100
    print("Laplace smoothing:", laplace)
    print("Accuracy =", round(accuracy, 2), "%")
    print("Correct:", accurate, "| Incorrect:", inaccurate)


if __name__ == "__main__":
    print("=== Without Laplace Smoothing ===")
    naive_bayes(laplace=False)
    print("\n=== With Laplace Smoothing ===")
    naive_bayes(laplace=True)
