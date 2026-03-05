import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils_activation_FO import *
from utils import model, preprocessing, plots
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
# ================= LOAD DATA =================
data = pd.read_csv("data/dataset20.csv")

N_REGRESSORS = 3
df = preprocessing.create_lagged(data, num_lags= 3)
X = df.drop(columns=['out']).to_numpy()
y = df['out'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Input scaling
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))

X_test = x_scaler.transform(X_test.reshape(X_test.shape[0], -1))


# Output scaling (regresija!)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))


# ============== SETUP =============
num_input = X_train.shape[1]   
num_output = y_train.shape[1]   
seed2 = 10
num_hidden_layers = 2
num_hidden = 10
ACT_FUN = ["relu", "tanh", "sigmoid"]
N_STEPS_AHEAD = [15, 30, 60, 120]

from itertools import product
from tqdm import tqdm

total_iters = len(ACT_FUN) * len(N_STEPS_AHEAD) * 5

pbar = tqdm(product(ACT_FUN, N_STEPS_AHEAD, range(5)),
            total=total_iters,
            desc="Training loops")

for function, n_steps, n in pbar:
    s_path = f"./results/{function}_{n_steps}_steps_ahead_{N_REGRESSORS}_Regressors_{n}/"

    # M1 : Classical Activation Functions 
    model_fixed = model.fixed(num_input, num_hidden, num_output, num_hidden_layers=num_hidden_layers, activation=function)
    model_fixed.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    # M2 : Shared alpha per-layer
    model_per_layer = model.per_layer(num_input, num_hidden, num_output, num_hidden_layers=num_hidden_layers, act=function, seed=seed2)
    model_per_layer.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )


    # M3 : Per-neuron alpha
    model_per_neuron = model.per_neuron(num_input, num_hidden, num_output, num_hidden_layers=num_hidden_layers, activation=function, seed=seed2)
    model_per_neuron.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    alpha_before = {
        'per_layer': model.get_initial_alpha(model_per_layer),
        'per_neuron': model.get_initial_alpha(model_per_neuron)
    }

    models = {
        "fixed": model_fixed,
        "per_layer": model_per_layer,
        "per_neuron": model_per_neuron
    }

    EPOCHS = 1000
    BATCH_SIZE = 32
    histories = model.train_models(X_train, y_train, models, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, show_progress=False, EarlyStopping=True)

    alpha_after = {
        'per_layer': model.get_trained_alpha(model_per_layer),
        'per_neuron': model.get_trained_alpha(model_per_neuron)
    }

    p = plots(histories, models, X_test, y_test, y_scaler, save_path=s_path, n_steps=n_steps)
    metrics = p.print_final_metrics(save=True)
    p.plot_alpha_comparison(alpha_before, alpha_after, save=True)
    p.plot_time_series(sample_size=-1, save=True) 
    p.plot_history(save=True)
    p.plot_mae(save=True)
    p.plot_predictions(save=True)
    p.save_y()
    p.export_alpha_comparison(alpha_before, alpha_after)


    pbar.set_postfix({
        "Activation function": function,
        "Steps ahead": n_steps,
        "iteration": n
    })