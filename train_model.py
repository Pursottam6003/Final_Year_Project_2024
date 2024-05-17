# DEPENDENCIES ------>
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import numpy as np


# data_dict = pickle.load(open("./data_small.pickle", "rb"))
# data = data_dict["data"]

# # Determine the length of the longest sequence
# max_length = max(len(seq) for seq in data)

# # Pad the sequences
# data = pad_sequences(data, maxlen=max_length, padding="post")


data_dict = pickle.load(open("./processed_data.pickle", "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print("{}% classified correctly".format(score * 100))
f = open("data_small.p", "wb")
pickle.dump({"model": model}, f)
f.close()

# # Check the lengths of the sequences in data
# sequence_lengths = [len(seq) for seq in data_dict["data"]]

# # Print the unique lengths
# print("Unique sequence lengths: ", set(sequence_lengths))
