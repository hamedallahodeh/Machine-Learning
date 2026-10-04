# Citrus Classification with Perceptron

Student: Hamedallah Anwer Hamedallah Issa
Student ID: 320220603007

This is our machine learning project. We used a Perceptron to tell the difference between oranges and grapefruits using their size and color.

## The problem

Oranges and grapefruits look pretty similar, so sorting them by hand (or with simple rules) can be hard. We wanted to see if a simple model could do it automatically, like in a fruit sorting factory.

## Dataset

The data is in `citrus.csv`. Each row is one fruit and has:

- diameter
- weight
- RGB color values (red, green, blue)
- the label: `orange` or `grapefruit`

We made the dataset ourselves based on the average characteristics of oranges and grapefruits, with different values so the model has enough variety to learn from.

## How the Perceptron works

It is the simplest kind of neural network, just one neuron. It takes the inputs, multiplies each one by a weight, adds them up and adds a bias. Then a step function gives 1 or 0 as the output. If the prediction is wrong, the weights and bias get updated a little, and this repeats over many iterations until the model gets better.

## What we did

Preprocessing:

1. Loaded `citrus.csv` with pandas
2. Changed the labels to numbers (orange = 0, grapefruit = 1) with `LabelEncoder`
3. Split the data into train and test sets
4. Scaled the features with `StandardScaler` so big values (like weight) don't overpower the small ones

Training:

1. Made a Perceptron with learning rate 0.01 and 1000 iterations
2. Trained it on the scaled training data
3. Predicted on the test data
4. Compared the predictions with the real labels to get the accuracy

## Result

Accuracy on the test set: 93.3%

So the model learned to separate the two fruits pretty well, which shows a Perceptron is good enough for a simple two-class problem like this.

## Future work

- Try a Multi-Layer Perceptron
- Add more features to improve the accuracy
- This could be used in an automatic sorting system for fruit
