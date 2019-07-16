# plot the training loss and accuracy
import numpy as np
from matplotlib import pyplot as plt
import csv
a = []
b =[]
c = []
d =[]
with open('performance_librispeech') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        a.append(float(row[2]))
        b.append(float(row[3]))
        c.append(float(row[4]))
        d.append(float(row[5]))
       

N = 277
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), b, label="train_loss")
plt.title("Training Loss on Librispeech Dataset for 251 speakers")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("train_loss_libri_up.png")


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), c, label="val_loss")
plt.title(" Validation Loss on Librispeech Dataset for 251 speakers")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("val_loss_libri_up.png")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), a, label="train_acc")
plt.title("Training accuracy on Librispeech Dataset for 251 speakers")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("train_acc_libri_up.png")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), d, label="val_acc")
plt.title(" Validation Accuracy on Librispeech Dataset for 251 speakers")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("val_acc_libri_up.png")