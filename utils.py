import tensorflow as tf
from tensorflow import keras
from keras import layers
import utils_activation_FO as uaf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============== PREPROCESSING ==============
class preprocessing:
    def create_data(data, n_past= 5, n_future=3, last_only=True):
        X, y = [], []
        
        for i in range(len(data) - n_past - n_future + 1):
            X.append(data.iloc[i:i+n_past].values)
            if last_only:
                y.append(data['out'].iloc[i+n_past+n_future-1])
            else:
                y.append(data['out'].iloc[i+n_past:i+n_past+n_future].values)
        
        return np.array(X), np.array(y)

# ============== MODEL BUILDERS ==============
class model:
    # def fixed(num_input, num_hidden, num_output, activation, seed=None):
    #     initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        
    #     model = keras.Sequential([
    #         layers.Input(shape=(num_input,)),
    #         layers.Dense(num_hidden, 
    #                     kernel_initializer=initializer, activation=activation),
    #         layers.Dense(num_hidden, kernel_initializer=initializer, activation=activation),
    #         layers.Dense(num_output, activation='linear', kernel_initializer=initializer)
    #     ])
    #     return model

    # def per_layer(num_input, num_hidden, num_output, activation, seed=None):
    #     initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        
    #     model = keras.Sequential([
    #         layers.Input(shape=(num_input,)),
    #         layers.Dense(num_hidden, 
    #                     kernel_initializer=initializer),
    #         activation,
    #         layers.Dense(num_hidden, kernel_initializer=initializer),
    #         activation,
    #         layers.Dense(num_output, activation='linear', kernel_initializer=initializer)
    #     ])
    #     return model

    # def per_neuron(num_input, num_hidden, num_output, bias=True, activation='Relu', seed=None):  
    #     inputs = keras.Input(shape=(num_input,))
        
    #     if activation == 'Relu':
    #         x = uaf.PerNeuronRelu(num_hidden, use_bias=bias, seed=seed)(inputs)
    #         x = uaf.PerNeuronRelu(num_hidden, use_bias=bias, seed=seed)(x)
    #     elif activation == 'Sigmoid':
    #         x = uaf.PerNeuronSigmoid(num_hidden, use_bias=bias, seed=seed)(inputs)
    #         x = uaf.PerNeuronSigmoid(num_hidden, use_bias=bias, seed=seed)(x)
    #     elif activation == 'Tanh':
    #         x = uaf.PerNeuronTanh(num_hidden, use_bias=bias, seed=seed)(inputs)
    #         x = uaf.PerNeuronTanh(num_hidden, use_bias=bias, seed=seed)(x)
        
    #     initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    #     outputs = layers.Dense(num_output, activation='linear', 
    #                         kernel_initializer=initializer)(x)
        
    #     return keras.Model(inputs, outputs)

    def fixed(num_input, num_hidden, num_output, activation, num_hidden_layers=2, seed=None):
        """
        Build a Sequential model with a variable number of dense layers.
        
        Parameters:
        - num_input: int, input shape
        - num_hidden: int, number of units in hidden layers
        - num_output: int, number of units in output layer
        - activation: str, activation function for hidden layers
        - num_hidden_layers: int, number of hidden layers. Default is 2.
        - seed: int, optional seed for initializer
        """
        # if num_hidden_layers < 1:
        #     raise ValueError("num_hidden_layers must be at least 1 (at least one hidden layer)")
        
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        
        model_layers = [layers.Input(shape=(num_input,))]
        
        # Add hidden layers
        for i in range(num_hidden_layers):
            model_layers.append(layers.Dense(num_hidden[i] if isinstance(num_hidden, list) else num_hidden,
                                            kernel_initializer=initializer, 
                                            activation=activation))
        
        # Add output layer
        model_layers.append(layers.Dense(num_output, 
                                        activation='linear', 
                                        kernel_initializer=initializer))
        
        model = keras.Sequential(model_layers)
        return model


    def per_layer(num_input, num_hidden, num_output, act, num_hidden_layers=2, seed=None):
        """
        Build a Sequential model with a variable number of dense layers, where activation is a custom layer.
        
        Parameters:
        - num_input: int, input shape
        - num_hidden: int, number of units in hidden layers
        - num_output: int, number of units in output layer
        - activation: layer, custom activation layer to apply after each hidden dense layer
        - num_hidden_layers: int, number of hidden layers. Default is 2.
        - seed: int, optional seed for initializer
        """
        
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        
        model_layers = [layers.Input(shape=(num_input,))]

        activation_layers = {
            'Relu': uaf.PerLayerRelu(),
            'Sigmoid': uaf.PerLayerSigmoid(),
            'Tanh': uaf.PerLayerTanh()
        }
                
        activation = activation_layers[act]
        # Add hidden layers with activation
        for i in range(num_hidden_layers):
            model_layers.append(layers.Dense(num_hidden[i] if isinstance(num_hidden, list) else num_hidden, kernel_initializer=initializer))
            model_layers.append(activation)
        
        # Add output layer
        model_layers.append(layers.Dense(num_output, activation='linear', kernel_initializer=initializer))
        
        model = keras.Sequential(model_layers)
        return model


    def per_neuron(num_input, num_hidden, num_output, bias=True, activation='Relu', num_hidden_layers=2, seed=None):
        """
        Build a Functional model with a variable number of hidden layers using per-neuron activations.
        
        Parameters:
        - num_input: int, input shape
        - num_hidden: int, number of units in hidden layers
        - num_output: int, number of units in output layer
        - bias: bool, whether to use bias in hidden layers
        - activation: str, type of activation ('Relu', 'Sigmoid', 'Tanh')
        - num_hidden_layers: int, number of hidden layers. Default is 2.
        - seed: int, optional seed for initializer and layers
        """
        inputs = keras.Input(shape=(num_input,))
        
        activation_classes = {
            'Relu': uaf.PerNeuronRelu,
            'Sigmoid': uaf.PerNeuronSigmoid,
            'Tanh': uaf.PerNeuronTanh
        }
        
        if activation not in activation_classes:
            raise ValueError(f"Unsupported activation: {activation}. Supported: {list(activation_classes.keys())}")
        
        act_class = activation_classes[activation]
        
        x = inputs
        for i in range(num_hidden_layers):
            x = act_class(num_hidden[i] if isinstance(num_hidden, list) else num_hidden,
                          use_bias=bias, seed=seed)(x)
        
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        outputs = layers.Dense(num_output, activation='linear', 
                            kernel_initializer=initializer)(x)
        
        return keras.Model(inputs, outputs)


    def get_initial_alpha(model):
        """
        Retrieves the initial alpha values from a model's layers.
        
        This function extracts alpha parameters from layers that contain them,
        triggering a model build if necessary by performing a forward pass with
        dummy input.
        
        Args:
            model: A Keras/TensorFlow model to extract alpha values from.
        
        Returns:
            list: A list of numpy arrays containing the alpha values copied from
                  layers that have the 'alpha' attribute. Returns an empty list
                  if no layers have alpha attributes.
        """
        alpha_before = []
        dummy_x = np.random.rand(1, model.input_shape[1])  # Dummy input to trigger build
        _ = model.predict(dummy_x, verbose=0)



        for layer in model.layers:
            if hasattr(layer, 'alpha'):
                alpha_before.append(layer.alpha.numpy().copy())

        return alpha_before


    def get_trained_alpha(model):
        """
        Extract alpha parameters from model layers.
        
        Args:
            model: A neural network model containing layers with potential alpha attributes.

        Returns:
            list: A list of numpy arrays containing alpha values from layers that have the alpha attribute.
                  Returns an empty list if no layers have an alpha attribute.
        
        """
        alpha_after = []
        for layer in model.layers:
            if hasattr(layer, 'alpha'):
                alpha_after.append(layer.alpha.numpy().copy())
                
        return alpha_after
    
    def plot_alpha_comparison(alpha_before, alpha_after):
        fig, axes = plt.subplots(len(alpha_after["per_layer"]), 1, figsize=(10, 8), sharex=True)
        labels = np.arange(len(alpha_before['per_neuron'][0]))

        for i, ax in enumerate(axes):
            ax.plot(
                labels,
                alpha_before['per_neuron'][i],
                label='Before Training',
                marker='o'
            )
            ax.plot(
                labels,
                alpha_after['per_neuron'][i],
                label='After Training',
                marker='x',
                color='orange'
            )

            ax.set_title(f'Layer {i} - Alpha Before and After Training')
            ax.set_ylabel('Alpha Value')
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel('Neuron Index')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_alpha_per_layer(alpha_before, alpha_after, model):
        np.set_printoptions(
            formatter={'float_kind': lambda x: f"{x:.4f}"},
            linewidth=np.inf
        )

        print("Per-layer alpha:")
        for idx, layer in enumerate(l for l in model.layers if hasattr(l, 'alpha')):
            if hasattr(layer, 'alpha'):
                print(f"Layer '{layer.name}':")
                print(f"Before: {alpha_before['per_layer'][idx]}")
                print(f"After:  {alpha_after['per_layer'][idx]}")

    @staticmethod
    def print_alpha_per_neuron(alpha_before, alpha_after, model):
        np.set_printoptions(
            formatter={'float_kind': lambda x: f"{x:.4f},"},
            linewidth=np.inf
        )

        print("Per-neuron alpha:")
        for idx, layer in enumerate(l for l in model.layers if hasattr(l, 'alpha')):
            if hasattr(layer, 'alpha'):
                print(f"Layer '{layer.name}':")
                print(f"Before: {alpha_before['per_neuron'][idx]}")
                print(f"After:  {alpha_after['per_neuron'][idx]}")

    @staticmethod
    def print_alpha(alpha_before, alpha_after, models, type='comparison'):
        match type:
            case 'comparison':
                model.print_alpha_per_layer(alpha_before, alpha_after, models['per_layer'])
                print("\n")
                model.print_alpha_per_neuron(alpha_before, alpha_after, models['per_neuron'])
            case 'per_layer':
                model.print_alpha_per_layer(alpha_before, alpha_after, models[type])
            case 'per_neuron':
                model.print_alpha_per_neuron(alpha_before, alpha_after, models[type])

    def train_models(X_train, y_train, models, EPOCHS=100, BATCH_SIZE=32, VAL_SPLIT=0.2, verbose=0, show_progress=True, EarlyStopping=True):
        iterator = (
            tqdm(models.items(), desc="Training models")
            if show_progress
            else models.items()
        )
        
        histories = {
            "fixed": None,
            "per_layer": None,
            "per_neuron": None
        }
        
        for name, model in iterator:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=1e-5
                    )]
            histories[name] = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VAL_SPLIT,
                verbose=verbose,
                callbacks = callbacks if EarlyStopping else None
            )
        
        return histories

# ============== PLOTS ==============
class plots:
    def __init__(self, histories, models, X_test, y_test, y_scaler):
         self.history_fixed = histories["fixed"]
         self.history_per_layer = histories["per_layer"]
         self.history_per_neuron = histories["per_neuron"]
         self.model_fixed = models["fixed"]
         self.model_per_layer = models["per_layer"]
         self.model_per_neuron = models["per_neuron"]
         self.X_test = X_test
         self.y_test = y_test
         self.y_scaler = y_scaler
         self.compute_predictions()

    # 1. Plot training and validation loss
    def plot_history(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.history_fixed.history['loss'], label='Training Loss')
        plt.plot(self.history_fixed.history['val_loss'], label='Validation Loss', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Model Fixed - Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.history_per_layer.history['loss'], label='Training Loss')
        plt.plot(self.history_per_layer.history['val_loss'], label='Validation Loss', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Model Per-Layer - Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.history_per_neuron.history['loss'], label='Training Loss')
        plt.plot(self.history_per_neuron.history['val_loss'], label='Validation Loss', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Model Per-Neuron - Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # 2. Plot training and validation MAE
    def plot_mae(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.history_fixed.history['mae'], label='Training MAE')
        plt.plot(self.history_fixed.history['val_mae'], label='Validation MAE', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Model Fixed - MAE')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.history_per_layer.history['mae'], label='Training MAE')
        plt.plot(self.history_per_layer.history['val_mae'], label='Validation MAE', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Model Per-Layer - MAE')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.history_per_neuron.history['mae'], label='Training MAE')
        plt.plot(self.history_per_neuron.history['val_mae'], label='Validation MAE', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Model Per-Neuron - MAE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # 3. Predictions vs Actual for test set
    def plot_predictions(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(self.y_test_orig, self.y_pred_fixed_orig, alpha=0.5, s=20)
        plt.plot([self.y_test_orig.min(), self.y_test_orig.max()], [self.y_test_orig.min(), self.y_test_orig.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Model Fixed - Test Predictions')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.scatter(self.y_test_orig, self.y_pred_per_layer_orig, alpha=0.5, s=20)
        plt.plot([self.y_test_orig.min(), self.y_test_orig.max()], [self.y_test_orig.min(), self.y_test_orig.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Model Per-Layer - Test Predictions')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.scatter(self.y_test_orig, self.y_pred_per_neuron_orig, alpha=0.5, s=20)
        plt.plot([self.y_test_orig.min(), self.y_test_orig.max()], [self.y_test_orig.min(), self.y_test_orig.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Model Per-Neuron - Test Predictions')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # 4. Plot actual vs predicted time series for first 100 samples
    def plot_time_series(self, sample_size=100):
        plt.figure(figsize=(15, 5))

        sample_size = min(sample_size, len(self.y_test))

        plt.subplot(1, 3, 1)
        plt.plot(self.y_test_orig[:sample_size], 'b-', label='Actual', alpha=0.7)
        plt.plot(self.y_pred_fixed_orig[:sample_size], 'r--', label='Predicted', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Output')
        plt.title('Model Fixed - Time Series')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.y_test_orig[:sample_size], 'b-', label='Actual', alpha=0.7)
        plt.plot(self.y_pred_per_layer_orig[:sample_size], 'r--', label='Predicted', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Output')
        plt.title('Model Per-Layer - Time Series')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.y_test_orig[:sample_size], 'b-', label='Actual', alpha=0.7)
        plt.plot(self.y_pred_per_neuron_orig[:sample_size], 'r--', label='Predicted', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Output')
        plt.title('Model Per-Neuron - Time Series')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # 5. Print final evaluation metrics
    def print_final_metrics(self):
        print("=" * 60)
        print("FINAL EVALUATION METRICS")
        print("=" * 60)

        # Dizionario dei modelli e delle relative predizioni
        models_preds = {
            "Fixed": self.y_pred_fixed_orig,
            "Per-Layer": self.y_pred_per_layer_orig,
            "Per-Neuron": self.y_pred_per_neuron_orig
        }

        # Lista per costruire il DataFrame
        metrics_list = []

        for name, y_pred in models_preds.items():
            mse = mean_squared_error(self.y_test_orig, y_pred)
            mae = mean_absolute_error(self.y_test_orig, y_pred)

            print(f"\nModel {name}:")
            print(f"  Test MSE: {mse:.6f}")
            print(f"  Test MAE: {mae:.6f}")

            metrics_list.append({
                "Model": name,
                "MSE": mse,
                "MAE": mae
            })

        print("=" * 60)

        # Creazione del DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df



    def compute_predictions(self):
        self.y_pred_fixed = self.model_fixed.predict(self.X_test, verbose=0)
        self.y_pred_per_layer = self.model_per_layer.predict(self.X_test, verbose=0)
        self.y_pred_per_neuron = self.model_per_neuron.predict(self.X_test, verbose=0)

        # Inverse transform to original scale
        self.y_test_orig = self.y_scaler.inverse_transform(self.y_test)
        self.y_pred_fixed_orig = self.y_scaler.inverse_transform(self.y_pred_fixed)
        self.y_pred_per_layer_orig = self.y_scaler.inverse_transform(self.y_pred_per_layer)
        self.y_pred_per_neuron_orig = self.y_scaler.inverse_transform(self.y_pred_per_neuron)