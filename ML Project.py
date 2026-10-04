import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tkinter as tk
from tkinter import messagebox

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._step_function
        self.weights = None
        self.bias = None

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0) 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y]) 

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                #update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

try:
    df = pd.read_csv("citrus.csv")
except FileNotFoundError:
    messagebox.showerror("Error", "citrus.csv not found. Make sure the file is in the same directory as the script.")
    exit()

# encode the name column to numerical values
label_encoder = LabelEncoder()
df['name_encoded'] = label_encoder.fit_transform(df['name'])

class_mapping = {label: cls_name for label, cls_name in enumerate(label_encoder.classes_)}
print(f"Class Mapping: {class_mapping}") #
if class_mapping[0] == 'orange': # If 'orange' got mapped to 0
    target_map = {'orange': 0, 'grapefruit': 1}
else: 
    target_map = {'grapefruit': 0, 'orange': 1}

df['target'] = df['name'].map(target_map)
y = df['target'].values
X = df[['diameter', 'weight', 'red', 'green', 'blue']].values
print("treaning and testing perceptron model\n...please wait...")
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# training and evaluation
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train_scaled, y_train)
predictions = perceptron.predict(X_test_scaled)

def accuracy(y_true, y_pred):
    accuracy_val = np.sum(y_true == y_pred) / len(y_true)
    return accuracy_val

acc = accuracy(y_test, predictions)
print(f"Perceptron Accuracy on Test Set: {acc:.4f}")

#GUI 
class CitrusPredictorGUI:
    def __init__(self, master, model, scaler, target_map_inv):
        self.master = master
        self.model = model
        self.scaler = scaler
        self.target_map_inv = target_map_inv
        master.title("Citrus Fruit Predictor (Perceptron)")

        self.labels = ['Diameter:', 'Weight:', 'Red:', 'Green:', 'Blue:']
        self.entries = []

        for i, label_text in enumerate(self.labels):
            tk.Label(master, text=label_text).grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(master)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries.append(entry)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_fruit)
        self.predict_button.grid(row=len(self.labels), column=0, columnspan=2, pady=10)

        self.result_label = tk.Label(master, text="Prediction: ", font=('Arial', 12))
        self.result_label.grid(row=len(self.labels) + 1, column=0, columnspan=2, pady=5)

    def predict_fruit(self):
        try:
            input_values = [float(e.get()) for e in self.entries]
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all features.")
            return
        
        input_scaled = self.scaler.transform(np.array(input_values).reshape(1, -1))

        # prediction
        prediction_code = self.model.predict(input_scaled)[0]
        predicted_label = self.target_map_inv.get(prediction_code, "Unknown")

        self.result_label.config(text=f"Prediction: {predicted_label.capitalize()}")

#inverse mapping for displaying result
target_map_inv = {v: k for k, v in target_map.items()} # {0: 'orange', 1: 'grapefruit'} or vice-versa

#create and run the GUI
root = tk.Tk()
gui = CitrusPredictorGUI(root, perceptron, scaler, target_map_inv)
root.mainloop()