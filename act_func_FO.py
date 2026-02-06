import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x = np.linspace(-5, 5, 1000)

def ReLU_FO(x, a):
    """Fractional order ReLU activation function."""
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    
    a = tf.cast(a, dtype=dtype)
    a = tf.clip_by_value(a, -0.99, 0.99)
    gamma_term = tf.math.exp(tf.math.lgamma(2.0 + a))
    
    x_pos = tf.maximum(x, 0.0)
    return tf.pow(x_pos, 1.0 + a) / gamma_term

def sigmoid_FO(x, a):
    """Fractional order sigmoid activation function."""
    x = tf.convert_to_tensor(x)
    dtype = x.dtype

    a = tf.cast(a, dtype=dtype)
    a = tf.clip_by_value(a, -0.99, 0.99)
    a_shifted = a + 1.0

    a_shifted_expanded = tf.broadcast_to(a_shifted, tf.shape(x))
    
    h = tf.cast(1e-2, dtype=dtype)
    max_iter = 500
    tol = 1e-8
    
    res = tf.zeros_like(x, dtype=dtype)
    coef = tf.ones_like(x, dtype=dtype)
    i = tf.constant(0, dtype=tf.int32)

    def cond(i, res, coef):
        term = coef * tf.math.log1p(tf.exp(x - tf.cast(i, dtype) * h))
        max_term = tf.reduce_max(tf.abs(term))
        continue_loop = tf.logical_and(i < max_iter, max_term >= tol)
        return continue_loop
    
    def body(i, res, coef):
        term = coef * tf.math.log1p(tf.exp(x - tf.cast(i, dtype) * h))
        res = res + term

        denominator = tf.cast(i + 1, dtype)
        denominator = tf.maximum(denominator, 1.0)

        coef = coef * (-(a_shifted_expanded - tf.cast(i, dtype)) / denominator)
        i = i + 1
        return i, res, coef
    
    loop_vars = (i, res, coef)
    
    shape_invariants = (
        tf.TensorShape([]),  
        x.shape,
        x.shape
    )
    
    i_final, res_final, coef_final = tf.while_loop(
        cond, 
        body, 
        loop_vars,
        shape_invariants=shape_invariants,
        maximum_iterations=max_iter
    )
    
    h_pow = tf.pow(h, a_shifted_expanded)
    h_pow = tf.maximum(h_pow, tf.cast(1e-12, dtype=dtype))
    
    result = res_final / h_pow
    result = tf.where(tf.math.is_nan(result), tf.zeros_like(result), result)
    result = tf.where(tf.math.is_inf(result), tf.sign(result) * 1e6, result)
    
    return result


def tanh_FO(x, alpha):
    x = tf.convert_to_tensor(x)
    dtype = x.dtype

    alpha = tf.cast(alpha, dtype=dtype)
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)  # 0 ≤ α ≤ 1
    alpha_expanded = tf.broadcast_to(alpha, tf.shape(x))

    h = tf.cast(1e-3, dtype=dtype)
    max_iter = 1000
    tol = 1e-8
    
    res = tf.zeros_like(x, dtype=dtype)
    coef = tf.ones_like(x, dtype=dtype)
    
    i = tf.constant(0, dtype=tf.int32)
    
    def cond(i, res, coef):
        term = coef * tf.math.tanh(x + tf.cast(i, dtype) * h)
        max_term = tf.reduce_max(tf.abs(term))
        continue_loop = tf.logical_and(i < max_iter, max_term >= tol)
        return continue_loop
    
    def body(i, res, coef):
        term = coef * tf.math.tanh(x - tf.cast(i, dtype) * h)
        res = res + term

        denominator = tf.cast(i + 1, dtype)
        coef = coef * (tf.cast(i, dtype) - alpha_expanded) / denominator
        
        i = i + 1
        return i, res, coef

    loop_vars = (i, res, coef)
    
    shape_invariants = (
        tf.TensorShape([]),
        x.shape, 
        x.shape 
    )
    
    i_final, res_final, coef_final = tf.while_loop(
        cond, 
        body, 
        loop_vars,
        shape_invariants=shape_invariants,
        maximum_iterations=max_iter,
        name='tanh_fo_derivative_loop'
    )

    h_pow = tf.pow(h, alpha_expanded)
    h_pow = tf.maximum(h_pow, tf.cast(1e-12, dtype=dtype))
    
    result = res_final / h_pow
    result = tf.clip_by_value(result, -5.0, 5.0)
    result = tf.where(tf.math.is_nan(result), tf.zeros_like(result), result)
    
    return result

# Funkcije za plotovanje
def plot_relu_fo():
    plt.figure(figsize=(10, 6))
    for alpha in np.arange(-1.0, 1.01, 0.2):
        y = ReLU_FO(x, alpha).numpy()
        plt.plot(x, y, label=f'ReLU FO (α={alpha:.1f})')
    plt.xlabel('x')
    plt.ylabel('ReLU_FO(x, α)')
    plt.title('Fractional Order ReLU for different α values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sigmoid_fo():
    plt.figure(figsize=(10, 6))
    for alpha in np.arange(-1.0, 1.01, 0.2):
        y = sigmoid_FO(x, alpha).numpy()
        plt.plot(x, y, label=f'Sigmoid FO (α={alpha:.1f})')
    plt.xlabel('x')
    plt.ylabel('Sigmoid_FO(x, α)')
    plt.title('Fractional Order Sigmoid for different α values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tanh_fo():
    plt.figure(figsize=(10, 6))
    for alpha in np.arange(0.0, 1.01, 0.2):
        y = tanh_FO(x, alpha).numpy()
        plt.plot(x, y, label=f'Tanh FO (α={alpha:.1f})')
    plt.xlabel('x')
    plt.ylabel('Tanh_FO(x, α)')
    plt.title('Fractional Order Tanh for different α values')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    plot_relu_fo()
    plot_sigmoid_fo()
    plot_tanh_fo()