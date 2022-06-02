import os
import numpy as np
from scipy.special import softmax








seed = 0

result_dir_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/" \
                  "jhuang24/models/osrci/seed_"

# Training output
train_feature_path = result_dir_base + str(seed) + "/train_features.npy"
train_feature_aug_path = result_dir_base + str(seed) + "/train_aug_features.npy"
train_label_path = result_dir_base + str(seed) + "/train_labels.npy"

train_feature = np.load(train_feature_path)
train_feature_aug = np.load(train_feature_aug_path)
train_label = np.load(train_label_path)

# Validation output
valid_feature_path = result_dir_base + str(seed) + "/valid_features.npy"
valid_feature_aug_path = result_dir_base + str(seed) + "/valid_aug_features.npy"
valid_label_path = result_dir_base + str(seed) + "/valid_labels.npy"

valid_feature = np.load(valid_feature_path)
valid_feature_aug = np.load(valid_feature_aug_path)
valid_label = np.load(valid_label_path)


# Test known output
test_known_feature_p0_path = result_dir_base + str(seed) + "/test_p0_features.npy"
test_known_feature_p1_path = result_dir_base + str(seed) + "/test_p1_features.npy"
test_known_feature_p2_path = result_dir_base + str(seed) + "/test_p2_features.npy"
test_known_feature_p3_path = result_dir_base + str(seed) + "/test_p3_features.npy"

test_known_feature_p0 = np.load(test_known_feature_p0_path)
test_known_feature_p1 = np.load(test_known_feature_p1_path)
test_known_feature_p2 = np.load(test_known_feature_p2_path)
test_known_feature_p3 = np.load(test_known_feature_p3_path)

test_known_feature_aug_p0_path = result_dir_base + str(seed) + "/test_p0_aug_features.npy"
test_known_feature_aug_p1_path = result_dir_base + str(seed) + "/test_p1_aug_features.npy"
test_known_feature_aug_p2_path = result_dir_base + str(seed) + "/test_p2_aug_features.npy"
test_known_feature_aug_p3_path = result_dir_base + str(seed) + "/test_p3_aug_features.npy"

test_known_feature_aug_p0 = np.load(test_known_feature_aug_p0_path)
test_known_feature_aug_p1 = np.load(test_known_feature_aug_p1_path)
test_known_feature_aug_p2 = np.load(test_known_feature_aug_p2_path)
test_known_feature_aug_p3 = np.load(test_known_feature_aug_p3_path)

test_known_label_p0_path = result_dir_base + str(seed) + "/test_p0_labels.npy"
test_known_label_p1_path = result_dir_base + str(seed) + "/test_p1_labels.npy"
test_known_label_p2_path = result_dir_base + str(seed) + "/test_p2_labels.npy"
test_known_label_p3_path = result_dir_base + str(seed) + "/test_p3_labels.npy"

test_known_label_p0 = np.load(test_known_label_p0_path)
test_known_label_p1 = np.load(test_known_label_p1_path)
test_known_label_p2 = np.load(test_known_label_p2_path)
test_known_label_p3 = np.load(test_known_label_p3_path)

test_known_feature = np.concatenate((test_known_feature_p0,
                                     test_known_feature_p1,
                                     test_known_feature_p2,
                                     test_known_feature_p3,),
                                    axis=0)

test_known_feature_aug = np.concatenate((test_known_feature_aug_p0,
                                         test_known_feature_aug_p1,
                                         test_known_feature_aug_p2,
                                         test_known_feature_aug_p3),
                                        axis=0)

test_known_labels = np.concatenate((test_known_label_p0,
                                    test_known_label_p1,
                                    test_known_label_p2,
                                    test_known_label_p3),
                                   axis=0)

# Test unknown output
test_unknown_feature_path = result_dir_base + str(seed) + "/test_unknown_features.npy"
test_unknown_feature_aug_path = result_dir_base + str(seed) + "/test_unknown_aug_features.npy"

test_unknown_feature = np.load(test_unknown_feature_path)
test_unknown_feature_aug = np.load(test_unknown_feature_aug_path)




def get_known_results(original_feature,
                      aug_feature,
                      labels):
    """

    :param original_feature:
    :param aug_feature:
    :param labels:
    :return:
    """
    correct = 0
    wrong = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    print(original_feature[0].shape)

    for i in range(len(original_feature)):
        label = labels[i]

        max_prob = np.max(softmax(original_feature[i]))
        pred = np.argmax(max_prob, axis=0)

        prob_aug = softmax(aug_feature[i], axis=0)
        prob_unknown = prob_aug[-1]

        if (max_prob-prob_unknown) >= 0.5:
            if pred == label:
                correct += 1
                true_positive += 1
            else:
                wrong += 1

        else:
            wrong += 1
            false_negative += 1

    print("True positive: ", true_positive)
    print("True negative: ", true_negative)
    print("False postive: ", false_positive)
    print("False negative: ", false_negative)

    accuracy = float(correct) / float(correct + wrong)
    print("Accuracy: ", accuracy)


    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)
    f1 = (2 * precision * recall) / (precision + recall)
    print("F-1: ", f1)




def get_unknown_results(original_feature,
                        aug_feature):
    """

    :param original_feature:
    :param aug_feature:
    :param labels:
    :return:
    """
    correct = 0
    wrong = 0


    for i in range(len(original_feature)):
        max_prob = np.max(softmax(original_feature[i], axis=0))

        prob_aug = softmax(aug_feature[i], axis=0)
        prob_unknown = prob_aug[-1]

        if (prob_unknown-max_prob) >= 0.5:
            correct += 1
        else:
            wrong += 1

    accuracy = float(correct) / float(correct + wrong)
    print("Accuracy: ", accuracy)




if __name__ == "__main__":
    print("*" * 40)
    print("Training results")
    get_known_results(original_feature=train_feature,
                      aug_feature=train_feature_aug,
                      labels= train_label)

    print("*" * 40)
    print("Validation results")
    get_known_results(original_feature=valid_feature,
                      aug_feature=valid_feature_aug,
                      labels=valid_label)

    print("*" * 40)
    print("Test known results")
    get_known_results(original_feature=test_known_feature,
                      aug_feature=test_known_feature_aug,
                      labels=test_known_labels)

    print("*" * 40)
    print("Test unknown results")
    get_unknown_results(original_feature=test_unknown_feature,
                      aug_feature=test_unknown_feature_aug)