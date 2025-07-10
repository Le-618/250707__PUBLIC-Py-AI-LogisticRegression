import numpy as np
import pandas as pd

from custom_logging import log
from constants import *

np.set_printoptions(precision=3)

file_path = "formatted_data.xlsx"
file_path = "formatted_data_cleaned.xlsx"
file_path = "good_relations.xlsx"
# file_path = "synthetic_student_data_labels_reordered.xlsx"
df = pd.read_excel(file_path)
# print(df.head().to_string(columns=df.columns.tolist()))
log(df)
all_data = np.array([[df[column_name][i] for column_name in df.columns if column_name not in BANNED_FIELDS] for i in range(len(df.index))])

log(f"{all_data = }")

ustd_x = all_data[:, :-1]  # unstandartized_x
ustd_x_ranges = [[max(ustd_x[:,j]), min(ustd_x[:,j])] for j in range(ustd_x.shape[1])]
get_normalized_x = lambda i, j: (ustd_x[i, j] - ustd_x_ranges[j][1]) / (ustd_x_ranges[j][0] - ustd_x_ranges[j][1])
total_x = np.array([[get_normalized_x(i, j) for j in range(ustd_x.shape[1])] for i in range(ustd_x.shape[0])])

total_y = all_data[:, -1]

total_xt = np.hstack([np.ones((total_x.shape[0], 1)), total_x])  # x tilde(~)

log(f"{total_xt = }")
log(f"{total_y = }")

xt_learning = total_xt[:-TEST_SIZE-VALIDATION_SIZE, :]
y_learning  = total_y[ :-TEST_SIZE-VALIDATION_SIZE]
xt_validation = total_xt[-TEST_SIZE-VALIDATION_SIZE:-TEST_SIZE, :]
y_validation  = total_y[ -TEST_SIZE-VALIDATION_SIZE:-TEST_SIZE]
xt_testing = total_xt[-TEST_SIZE:, :]
y_testing  = total_y[ -TEST_SIZE:]
# log(f"{xt_testing = }")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
beta = np.zeros(total_xt.shape[1])  # omega tilde [w~]

log("Stage Learning:")
with log:
    for epoch in range(EPOCH_NUMBER):
        linear_equations = xt_learning @ beta
        probabilities = sigmoid(linear_equations)
        errors = probabilities - y_learning
        gradient = xt_learning.T @ errors / len(y_learning)
        beta -= LEARNING_RATE * gradient
        log(f"{beta=}")
        # log(f"{errors=}, {beta=} {ot.shape=}, {gradient=} {gradient.shape=}")
log("Stage Validating:")
with log:
    product = xt_validation @ beta
    log(product)
    res = sigmoid(product)
    log(res)
    log(np.round(res))
    log(y_validation)
    final = sum(map(int, np.round(res) == y_validation))
    log(f"{final}/{VALIDATION_SIZE}")
    log(f"{final/VALIDATION_SIZE*100}%")
    

