import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

with open("sentence_level.csv") as my_file:
    file_name = csv.reader(my_file)
    #x = len(file_name)
    actual_class = []
    predicted_class = []
    for x in file_name:
        actual_class.append(int(x[1]))
        predicted_class.append(int(x[0]))
# plot the training loss and accuracy
print(classification_report(actual_class, predicted_class))