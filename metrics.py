import numpy as np
from scipy import spatial


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum

def intersection_over_union(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2).sum()
    union = im_sum - intersection
    return intersection / union


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    prediction = prediction > 0.5
    groundtruth = groundtruth > 0.5

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))


    return FP, FN, TP, TN


def get_evaluation_metrics(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)

    #print(f"FP = {FP}, FN = {FN}, TP = {TP}, TN = {TN}, TOTAL = {FP + FN + TP + TN}")

    accuracy = np.divide(TP + TN, FP + FN + TP + TN)
    recall = np.divide(TP, TP + FN)
    specifity = np.divide(TN, TN + FP)
    precision = np.divide(TP, TP + FP)
    f1_score = 2 * np.divide(precision * recall, precision + recall)

    return accuracy, recall, specifity, precision, f1_score
