import csv
import pandas as pd

train_labels = pd.read_csv("./train.csv")
print("training labels read successfully!")

from IPython.display import display
display(train_labels.head())
targets = train_labels['Label'].tolist()


# targets_one_hot = tf.one_hot(targets)
targets_one_hot = pd.get_dummies(pd.Series(targets))
targets_one_hot = targets_one_hot.values
targets_one_hot = np.array(targets_one_hot).astype("float32")

val_targets_one_hot = targets_one_hot[6500:]
train_targets_one_hot = targets_one_hot[0:6500]

print np.shape(train_targets_one_hot)
print train_targets_one_hot[0:5]

print np.shape(val_targets_one_hot)
print val_targets_one_hot[-5:]

print train_targets_one_hot.dtype
print val_targets_one_hot.dtype