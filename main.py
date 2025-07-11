import numpy as np
import pandas as pd

from custom_logging import log, log_timer_start, log_timer_end
from constants import *

np.set_printoptions(precision=3)

start_time_total = log_timer_start()
with log("Preparation Stage"):
    start_time = log_timer_start()
    # file_path = "formatted_data.xlsx"
    # file_path = "formatted_data_cleaned.xlsx"
    file_path = "formatted_data_final.xlsx"
    # file_path = "synthetic_student_data_labels_reordered.xlsx"
    with log(f"Reading xlsx file [{file_path}] ..."):
        df = pd.read_excel(file_path)
        # print(df.head().to_string(columns=df.columns.tolist()))
        if DO_ADVANCED_LOGGING: log(df)
    log("Success!")

    with log("Convert from Pandas.Dataframe to Numpy.Array (matrix) ..."):
        all_data = np.array([[df[column_name][i] for column_name in df.columns if column_name not in BANNED_FIELDS] for i in range(len(df.index))])
        if DO_ADVANCED_LOGGING: log(f"{all_data = }")
    log("Success!")

    with log("Standartizing the data to between [0, 1] ..."):
        ustd_x = all_data[:, :-1]  # unstandartized_x
        ustd_x_ranges = [[max(ustd_x[:,j]), min(ustd_x[:,j])] for j in range(ustd_x.shape[1])]
        get_normalized_x = lambda i, j: (ustd_x[i, j] - ustd_x_ranges[j][1]) / (ustd_x_ranges[j][0] - ustd_x_ranges[j][1])
        total_x = np.array([[get_normalized_x(i, j) for j in range(ustd_x.shape[1])] for i in range(ustd_x.shape[0])])

        total_y = all_data[:, -1]

        total_xt = np.hstack([np.ones((total_x.shape[0], 1)), total_x])  # x tilde(~)

        if DO_ADVANCED_LOGGING: log(f"{total_xt = }")
        if DO_ADVANCED_LOGGING: log(f"{total_y = }")
    log("Success!")
    with log("Dividing the data among Learning, Evaluation, Testing procedures ..."):
        xt_learning = total_xt[:-TESTING_SIZE - EVALUATION_SIZE, :]
        y_learning  = total_y[ :-TESTING_SIZE - EVALUATION_SIZE]
        log("Learning array size:  ", xt_learning.shape[0])
        xt_evaluation = total_xt[-TESTING_SIZE - EVALUATION_SIZE:-TESTING_SIZE, :]
        y_evaluation  = total_y[-TESTING_SIZE - EVALUATION_SIZE:-TESTING_SIZE]
        log("Evaluation array size:", EVALUATION_SIZE)
        xt_testing = total_xt[-TESTING_SIZE:, :]
        y_testing  = total_y[-TESTING_SIZE:]
        log("Testing array size:   ", TESTING_SIZE)
        wt = np.zeros(total_xt.shape[1])  # omega tilde [w~]
    log("Success!")
    # log(f"{xt_testing = }")

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    log_timer_end(start_time)

with log("Stage Learning:"):
    start_time = log_timer_start()
    log("Learning rate:", LEARNING_RATE)
    for epoch in range(EPOCH_NUMBER):
        if DO_EPOCH_LOGGING: log(f"Epoch: {epoch+1:>5}/{EPOCH_NUMBER}")
        with log:
            linear_equations = xt_learning @ wt
            probabilities = sigmoid(linear_equations)
            errors = probabilities - y_learning
            gradient = xt_learning.T @ errors / len(y_learning)
            wt -= LEARNING_RATE * gradient
            if DO_ADVANCED_LOGGING: log(f"{wt=}")
            # log(f"{errors=}, {beta=} {ot.shape=}, {gradient=} {gradient.shape=}")
    log_timer_end(start_time)
with log("Stage Evaluation:"):
    start_time = log_timer_start()
    product = xt_evaluation @ wt
    res = sigmoid(product)
    prediction_array = np.round(res)
    actual_array = y_evaluation
    if DO_ADVANCED_LOGGING:
        log(product)
        log(res)
        log(prediction_array)
        log(y_evaluation)
    final = sum(map(int, prediction_array == y_evaluation))
    log(f"Overall Correct values: {final}/{EVALUATION_SIZE}")
    log(f"Accuracy:               {final / EVALUATION_SIZE * 100}%")

    # Confusion matrix components
    TP = np.sum((actual_array == 1) & (prediction_array == 1))
    TN = np.sum((actual_array == 0) & (prediction_array == 0))
    FP = np.sum((actual_array == 0) & (prediction_array == 1))
    FN = np.sum((actual_array == 1) & (prediction_array == 0))

    # Print confusion matrix
    with log("Confusion Matrix:"):
        log(f"         Actual")
        log(f"Predict [{TP:>4} {FN:>4}]")
        log(f"        [{FP:>4} {TN:>4}]")

    # Calculate metrics
    accuracy = (TP + TN) / len(actual_array)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # Print metrics
    with log("Metrics:"):
        log(f"Accuracy  : {accuracy:.3f}")
        log(f"Precision : {precision:.3f}")
        log(f"Recall    : {recall:.3f}")
        log(f"F1-score  : {f1:.3f}")

    log_timer_end(start_time)

with log("Stage Testing:"):
    start_time = log_timer_start()

    log("Coefficients (omega-theta / WT):")
    log(wt)
    actual_array = list()
    prediction_array = list()
    for test_index in range(TESTING_SIZE):
        with log(f"Test #{test_index+1}/{TESTING_SIZE}"):
            log("X theta values:  ", xt_testing[test_index, :])
            actual = int(y_testing[test_index])
            actual_array.append(actual)
            log("Actual Value:    ", int(y_testing[test_index]))
            product = xt_testing[test_index,:] @ wt
            log("XT @ WT product: ", product)
            res = sigmoid(product)
            log("Sigmoid(product):", res)
            prediction = round(res)
            prediction_array.append(actual)
            log("Predicted Value: ", prediction)
            log("Prediction CORRECT" if actual == prediction else "Prediction Incorrect")

    actual_array = np.array(actual_array)
    prediction_array = np.array(prediction_array)
    TP = np.sum((actual_array == 1) & (prediction_array == 1))
    TN = np.sum((actual_array == 0) & (prediction_array == 0))
    FP = np.sum((actual_array == 0) & (prediction_array == 1))
    FN = np.sum((actual_array == 1) & (prediction_array == 0))

    # Print confusion matrix
    with log("Confusion Matrix:"):
        log(f"         Actual")
        log(f"Predict [{TP:>2} {FN:>2}]")
        log(f"        [{FP:>2} {TN:>2}]")

    # Calculate metrics
    accuracy = (TP + TN) / len(actual_array)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # Print metrics
    with log("Metrics:"):
        log(f"Accuracy  : {accuracy*100:.1f}%")
        log(f"Precision : {precision*100:.1f}%")
        log(f"Recall    : {recall*100:.1f}%")
        log(f"F1-score  : {f1*100:.1f}%")

    log_timer_end(start_time)

log("Script executed successfully")
log_timer_end(start_time_total)
