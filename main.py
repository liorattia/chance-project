import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class SequencePredictor:
    def __init__(self, column_name, data_len, seq_length, prev_runing=None):
        self.column_name = column_name
        self.data_len = data_len
        self.seq_length = seq_length
        self.prev_runing = prev_runing
        self.model = None

    def import_csv(self):
        df = pd.read_csv(r"https://docs.google.com/spreadsheets/d/e/2PACX-1vStIvCXdHlPYd4EPogNlkrKJrJUl8Ig7tEu-5JDLKaRG9eCZ5Mmu2C6fSwo9404Ig/pub?gid=513376755&single=true&output=csv")
        df = df.sort_values('No')
        df['rank'] = df.groupby('date')['No'].rank(method='dense', ascending=True)
        df = df.set_index('No')

        if self.prev_runing is not None:
            data = df[[self.column_name]][-self.data_len:-self.prev_runing]
        else:
            data = df[[self.column_name]][-self.data_len:]

        return data

    def transform_numbers_to_arrays(self, numbers):
        result = []
        for number in numbers:
            if 7 <= number <= 14:
                array = [0] * 8
                array[number - 7] = 1
                result.append(array)
            else:
                raise ValueError(f"Number {number} is out of range. Expected numbers between 7 and 14.")
        return np.array(result)

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict_next_array(self, last_arrays):
        last_arrays = np.array(last_arrays).reshape(1, self.seq_length, 8)
        prediction = self.model.predict(last_arrays)
        return np.argmax(prediction), prediction

    def train_and_evaluate(self):
        data = self.import_csv()
        data = np.array(data).astype('int').tolist()
        numbers = [item for sublist in data for item in sublist]
        data = self.transform_numbers_to_arrays(numbers)

        X, y = self.create_sequences(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.model = self.build_model((self.seq_length, 8))

        stop_on_accuracy = StopOnAccuracy(target_acc=1.0)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=64,
            callbacks=[stop_on_accuracy]  # or EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        )

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        last_10_arrays = data[-self.seq_length:]
        predicted_array_index, predicted_array_probs = self.predict_next_array(last_10_arrays)
        predicted_number = predicted_array_index + 7

        return predicted_number, test_accuracy

class StopOnAccuracy(Callback):
    def __init__(self, target_acc=1.0):
        super(StopOnAccuracy, self).__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        if accuracy >= self.target_acc:
            print(f"\nReached {self.target_acc*100}% accuracy, stopping training!")
            self.model.stop_training = True

# Running the model for multiple columns
results = {}
columns = ['L', 'T', 'D', 'H']

for column in columns:
    predictor = SequencePredictor(column_name=column, data_len=4000, seq_length=20)#, prev_runing=2
    predicted_value, accuracy = predictor.train_and_evaluate()
    results[column] = (predicted_value, accuracy)

# Printing the results
for column, (predicted_value, accuracy) in results.items():
    print(f"{column} (Predicted Value: {predicted_value}), Accuracy: {accuracy}")
