# Multilayer perceptron

This is a 42 project for learn how multilayer perceptron works.
We need to code one be ourself and train it on given data.

## Setup
### Virtual environement
Creation
```
python3 -m venv .venv
```
Use
```
source .venv/bin/activate
```
Quit
```
deactivate
```
### Dependencies
```
pip install -r requirements.txt
```

## Usage
### Split dataset
Script for split the dataset into two part. One for training, and the other for validating.
```
python3 splitData.py <data_path.csv> <percentage of data for training>
```
data_path.csv : must be a valid path, relative of absolute.
percentage of data for training : must be an interger between 0 and 100.
### Training
Script for training the neural network.
Display plots at the end of executions.
You need to have execute split script before.
```
python3 training.py [<hidden layers dimensions>]
```
hidden layers dimensions : must be intergers greater than 0, separated by comma.
### Validation
Script for validate the neural network training.
You need to have execute training script before.
```
python3 validation.py
```
### Compare
Script for compare training between neural network with different hidden layer.
Display plots at the end of executions.
```
python3 compare.py
```
