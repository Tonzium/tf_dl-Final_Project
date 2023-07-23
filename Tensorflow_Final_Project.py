import pandas as pd
import numpy as np
import tensorflow as tf

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


df = pd.read_csv("cover_data.csv")


#Split to features and labels
df_x = df.drop(columns="class")

df_y = df["class"]

# Multiclassification problem, implement one-hot encoding
def one_hot(data):
    data_encoded = tf.keras.utils.to_categorical(data - 1)
    return data_encoded

# Data summarization
def print_data():
    print(df.describe())
    #print(df["class"].value_counts())

# We have high std in left columns and low std in right columns
#print_data()

# Visualize classes data
def visualize_data_classes(data=df["class"]):
    class_counts = Counter(data)
    
    classes = []
    counts = []

    for class_, count_ in class_counts.items():
        classes.append(class_)
        counts.append(count_)


    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Frequency of each class')
    plt.show()

# The data visualization shows that there is high imbalance among classes - class 1 and 2 high count - others low count.
#visualize_data_classes()


# Split data
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)


# Use UNDER and OVER sampling together

under_strat = {1: 40000, 2: 40000}
smote_strat = {3: 40000, 4: 40000, 5: 40000, 6: 40000, 7: 40000}

under = RandomUnderSampler(sampling_strategy=under_strat)

smote = SMOTE(sampling_strategy=smote_strat)

pipeline = Pipeline(steps=[('u', under), ('o', smote)])

X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)
X_test_resampled, y_test_resampled = pipeline.fit_resample(X_test, y_test)

# Visualize data after resampling with SMOTE
#visualize_data_classes(y_train_resampled)

# encode resampled label data
one_hot_y_train = one_hot(y_train_resampled)
one_hot_y_test = one_hot(y_test_resampled)

# Standardize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_ready = scaler.fit_transform(X_train_resampled)
X_test_ready = scaler.transform(X_test_resampled)


def run_model():
    # Building the model

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(324, activation="relu", input_shape=(54,)))

    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(162, activation="relu"))

    model.add(tf.keras.layers.Dense(7, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=0.002)          
    model.compile(loss="categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"]
                )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(X_train_ready, one_hot_y_train, epochs=80, batch_size=250, validation_data=(X_test_ready, one_hot_y_test), callbacks=[early_stopping])

    return model, history



def plot_results(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    return plt.savefig("Results.png")


model, history = run_model()

plot_results(history)

# Predict labels on test data
y_pred = model.predict(X_test_ready)

# Convert predictions from probabilities to class labels

# classes start from 1 so +1
y_pred_classes = np.argmax(y_pred, axis=1) + 1
y_test_classes = np.argmax(one_hot_y_test, axis=1) + 1

print(classification_report(y_test_classes, y_pred_classes))


# Lastly investigate model predictions for the first rows

class_meanings = {
    1 : "Spruce/Fir",
    2 : "Lodgepole Pine",
    3 : "Ponderosa Pine",
    4 : "Cottonwood/Willow",
    5 : "Aspen",
    6 : "Douglas-fir",
    7 : "Krummholz"
}


# Select and scale the first 10 rows
first_10_rows = df_x.iloc[:10]
first_10_rows_scaled = scaler.transform(first_10_rows)

# Predict class probabilities
probs = model.predict(first_10_rows_scaled)

# Convert probabilities to class label
class_labels = np.argmax(probs, axis=1) + 1  # we add 1 because your classes start from 1

# Convert class label to class name
class_names = [class_meanings[label] for label in class_labels]

first_10_rows_true_classes = df_y.iloc[:10].values

for i, class_name in enumerate(class_names):
    true_class_label = first_10_rows_true_classes[i]
    true_class_name = class_meanings[true_class_label]
    Check_correct = "Correct" if class_name == true_class_name else "Incorrect"
    print(f"Model prediction for row {i+1} is {class_name}. While true class is {true_class_name}. Prediction is {Check_correct}.")

