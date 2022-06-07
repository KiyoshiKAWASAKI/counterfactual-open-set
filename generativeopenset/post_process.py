import os
import numpy as np
from scipy.special import softmax




seed = 2
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




def calculate_mcc(true_pos,
                  true_neg,
                  false_pos,
                  false_neg):
    """

    :param true_pos:
    :param true_neg:
    :param false_pos:
    :param false_negtive:
    :return:
    """

    return (true_neg*true_pos-false_pos*false_neg)/np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*
                                                           (true_neg+false_pos)*(true_neg+false_neg))




def get_train_valid_results(original_feature,
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
    false_negative = 0

    print(original_feature[0].shape)

    for i in range(len(original_feature)):
        label = labels[i]

        max_prob = np.max(softmax(original_feature[i]))
        pred = np.argmax(softmax(original_feature[i]), axis=0)

        prob_aug = softmax(aug_feature[i], axis=0)
        prob_unknown = prob_aug[-1]


        if (prob_unknown-max_prob) >= 0.5:
            wrong += 1

        else:
            if pred == label:
                correct += 1
            else:
                wrong += 1


    accuracy = float(correct) / float(correct + wrong)
    print("Multi class accuracy: ", accuracy)




def get_binary_results(original_known_feature,
                       aug_known_feature,
                       original_unknown_feature,
                       aug_unknown_feature):
    """

    :param original_feature:
    :param aug_feature:
    :param labels:
    :return:
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Process known samples
    for i in range(len(original_known_feature)):
        max_prob = np.max(softmax(original_known_feature[i], axis=0))

        prob_aug = softmax(aug_known_feature[i], axis=0)
        prob_unknown = prob_aug[-1]

        if (prob_unknown-max_prob) >= 0.5:
            false_negative += 1
        else:
            true_positive += 1

    # Process unknown
    for i in range(len(original_unknown_feature)):
        max_prob = np.max(softmax(original_unknown_feature[i], axis=0))

        prob_aug = softmax(aug_unknown_feature[i], axis=0)
        prob_unknown = prob_aug[-1]

        if (prob_unknown-max_prob) >= 0.5:
            true_negative += 1
        else:
            false_positive += 1

    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)
    f1 = (2 * precision * recall) / (precision + recall)
    mcc = calculate_mcc(true_pos=float(true_positive),
                        true_neg=float(true_negative),
                        false_pos=float(false_positive),
                        false_neg=float(false_negative))
    acc = float(true_negative)/float(true_negative+false_positive)

    print("True positive: ", true_positive)
    print("True negative: ", true_negative)
    print("False postive: ", false_positive)
    print("False negative: ", false_negative)
    print("Unknown accuracy: ", acc)
    print("F-1 score: ", f1)
    print("MCC score: ", mcc)





if __name__ == "__main__":

    print("*" * 40)
    print("Seed: ", seed)
    print("Muti-class results")
    get_train_valid_results(original_feature=test_known_feature,
                            aug_feature=test_known_feature_aug,
                            labels=test_known_labels)

    print("Binary results")
    get_binary_results(original_known_feature=test_known_feature,
                       aug_known_feature=test_known_feature_aug,
                       original_unknown_feature=test_unknown_feature,
                       aug_unknown_feature=test_unknown_feature_aug)
