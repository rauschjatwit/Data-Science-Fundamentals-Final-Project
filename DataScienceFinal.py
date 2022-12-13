import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
import seaborn as sns


#window = tk.Tk()
#window.title("Grade Calculator")
#window.geometry('700x700')
#instruct = tk.Label(window, text = "Math Grade Prediction\nEnter the Parameters Below:\n\n")
#instruct.pack()

# # Will have to adjust to set all parameters entered in each boc to a varible for predictions
#def getInput():
#    global inp_1
#    global inp_2
#    global inp_3
#    inp_1 = param1.get(1.0, "end-1c")
#    inp_2 = param2.get(1.0, "end-1c")
#    inp_3 = param3.get(1.0, "end-1c")
#    window.destroy()
  
## Parameter 1 TextBox Creation
#param1_lbl = tk.Label(window, text = "Parameter 1:")
#param1_lbl.pack()
#param1 = tk.Text(window,
#                   height = 1,
#                   width = 20)
#param1.pack()

## Parameter 2 TextBox Creation
#param2_lbl = tk.Label(window, text = "Parameter 2:")
#param2_lbl.pack()
#param2 = tk.Text(window,
#                   height = 1,
#                   width = 20)
#param2.pack()

## Parameter 3 TextBox Creation
#param3_lbl = tk.Label(window, text = "Parameter 3:")
#param3_lbl.pack()
#param3 = tk.Text(window,
#                   height = 1,
#                   width = 20)
#param3.pack()
  
## Button Creation
#printButton = tk.Button(window,
#                        text = "Enter", 
#                        command = getInput)
#printButton.pack()

# # Main Tkinter Loop
#window.mainloop()

#print(inp_1, end = " ")
#print(inp_2, end = " ")
#print(inp_3, end = " ")

 # Read in Raw Data
df = pd.read_csv("UpdatedData2.csv")

 # Split data in a training and testing dataset
train_dataset = df.sample(frac = 0.8, random_state = 0)
test_dataset = df.drop(train_dataset.index)

 # Visualizing Data Columns
#sns.pairplot(train_dataset[['age', 'absences', 'g3']])
#plt.show()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('G3')
test_labels = test_features.pop('G3')

train_dataset.describe().transpose()
 # Normalization Layer
normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(np.array(train_features))
#print(normalizer.mean.numpy())

#first = np.array(train_features[:1])

#with np.printoptions(precision=2, suppress=True):
#    print('First Example:', first)
#    print()
#    print('Normalized:', normalizer(first).numpy())

G2 = np.array(train_features["G2"])

G2_normalizer = layers.Normalization(input_shape=[1,], axis=None)
G2_normalizer.adapt(G2)

G2_model = tf.keras.Sequential([
    G2_normalizer,
    layers.Dense(units=1)
    ])

##G2_model.summary()

#G2_model.predict(G2[10:])

# # Could change the optimizer and see how the accuracy changes
G2_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#history = G2_model.fit(
#    train_features['G2'],
#    train_labels,
#    epochs=100,
#    verbose=0,
#    validation_split = 0.2)

#hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch

#def plot_loss(history):
#    plt.plot(history.history['loss'], label='loss')
#    plt.plot(history.history['val_loss'], label='val_loss')
#    plt.ylim([0, 10])
#    plt.xlabel('Epoch')
#    plt.ylabel('Error [G3]')
#    plt.legend()
#    plt.grid(True)
#    plt.show()

#plot_loss(history)

test_results = {}
test_results['G2_model'] = G2_model.evaluate(
    test_features['G2'],
    test_labels, verbose=0)

#x = tf.linspace(0.0, 20, 20)
#y = G2_model.predict(x)

def plot_G2(x, y):
    plt.scatter(train_features['G2'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('G2')
    plt.ylabel('G3')
    plt.legend()
    plt.show()

#plot_G2(x, y)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
    ])

linear_model.predict(train_features[:10])
linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split = 0.2)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
        ])
    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_G2_model = build_and_compile_model(G2_normalizer)
#dnn_G2_model.summary()

history = dnn_G2_model.fit(
    train_features['G2'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

x = tf.linspace(0.0, 20, 20)
y = dnn_G2_model.predict(x)

#plot_G2(x, y)

test_results['dnn_G2_model'] = dnn_G2_model.evaluate(
    test_features['G2'], test_labels,
    verbose=0)

dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
##print(pd.DataFrame(test_results, index=['Mean absolute error [G2]']).T)

test_predictions = dnn_model.predict(test_features).flatten()

#a = plt.axes(aspect='equal')
#plt.scatter(test_labels, test_predictions)
#plt.xlabel('True Values [G3]')
#plt.ylabel('Predictions [G3]')
#lims = [0, 20]
#plt.xlim(lims)
#plt.ylim(lims)
#_ = plt.plot(lims, lims)

error = test_predictions - test_labels

plt.hist(error, bins=25)
plt.xlabel('Prediction Error [G3]')
_ = plt.ylabel('Count')
plt.show()

#dnn_model.save('dnn_model')

#reloaded = tf.keras.models.load_model('dnn_model')

#test_results['reloaded'] = reloaded.evaluate(
#    test_features, test_labels, verbose=0)

#print(pd.DataFrame(test_results, index=['Mean absolute error [G3]']).T)

