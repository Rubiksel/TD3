"""
A system that analyze each model and decide if the weight of the model should be updated or not
"""

import os
import pickle
import tensorflow as tf
import pandas as pd
import torch
from sklearn.model_selection import train_test_split



#data processing
df =pd.read_csv("C:\\Users\ejder\Documents\ESILV\Cours\A4\S8\Decentralization technologies\TD3-1\data\steam.csv", index_col=0)

# data processing
df["positive_ratio"] = df["positive_ratings"] / (df["positive_ratings"] + df["negative_ratings"])
df = df[(df['positive_ratings'] + df['negative_ratings']) >= 500]
df = pd.get_dummies(df)

def load_pkl_model(model_path):
    """
    Load a model from a .pkl file
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_h5_model(model_path):
    """
    Load a model from a .h5 file
    """
    model = tf.keras.models.load_model(model_path)
    return model

def load_pt_model(model_path):
    model = torch.jit.load(model_path)
    print(model)
    return model

def analyze_model(model, foo):
    X = df.drop(columns=["positive_ratio"], axis=1)
    y = df["positive_ratio"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        precision = torch.nn.MSELoss(torch.from_numpy(X_test).float().squeeze(1), torch.from_numpy(y_test).float())
        return precision
    except:
        precision = 1
    # if foo :
    #     """the model is a pickle model"""
    #     # Analyze the model
    #     precision = model.score(X_test, y_test)
    # if not foo:
    #     """the model is a h5 model"""
    #     # Analyze the model
    #     precision = model.evaluate(X_test, y_test)
    # Decide if the weight should be updated or not
    return precision
    

# Get the path to the folder containing the models
folder_path = "C:\\Users\ejder\Documents\ESILV\Cours\A4\S8\Decentralization technologies\TD3-1\Python_code"

# List all the files in the folder
files = os.listdir(folder_path)

# Filter the files to only include .pkl and .h5 files
model_files = [file for file in files if file.endswith((".pkl", ".h5", ".pt"))]

weight = []

# Load each model
models = []
for model_file in model_files:
    foo = False
    model_path = os.path.join(folder_path, model_file)
    # Load the model using the appropriate method
    if model_file.endswith(".pkl"):
        model = load_pkl_model(model_path)
        foo = True
    elif model_file.endswith(".h5"):
        model = load_h5_model(model_path)
        foo = False
    elif model_file.endswith(".pt"):
        model = load_pt_model(model_path)
    # Analyze the model and decide if the weight should be updated or not
    precision = analyze_model(model, foo)
    models.append({"model": model, "precision": precision})

# Sort the models by precision (higher precision means better performance)
models.sort(key=lambda x: x["precision"], reverse=True)

# Update the weights based on the model performance
weights = []
for i, model in enumerate(models):
    weight = model["precision"] * (0.9 ** i)  # Decrease the weight of models based on their position
    weights.append(weight)

# Normalize the weights to sum up to 1
total_weight = sum(weights)
weights = [weight / total_weight for weight in weights]

# Print the weights
for i, model_file in enumerate(model_files):
    print(f"Model: {model_file}, Weight: {weights[i]}")


#given an order in the way we deal with the models, the weight are the precision of each model
# closer to 1 means the model has more weight