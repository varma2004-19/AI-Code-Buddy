import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os

# Disable GPU usage to avoid CUDA-related errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define collocation points with clustering near boundaries
def generate_collocation_points(K, beta=5.0):
    # Use a Beta distribution to cluster points near x=0 and x=1
    points = np.random.beta(beta, beta, K)
    points = np.sort(points)
    return points[:, None]

# Define the neural network architecture (Deeper network)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(50, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
        tf.keras.layers.Dense(50, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
        tf.keras.layers.Dense(50, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
        tf.keras.layers.Dense(50, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))
    ])
    return model

# Compute derivatives
def compute_derivatives(model, x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            u = model(x)
        u_x = tape2.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    return u, u_x, u_xx

# Define loss function (Adjust lambda for better boundary enforcement)
def compute_loss(model, x_colloc, epsilon, alpha=1e-3, kappa=1.0):
    x_colloc = tf.convert_to_tensor(x_colloc, dtype=tf.float32)
    u, u_x, u_xx = compute_derivatives(model, x_colloc)
    pde_residual = -epsilon * u_xx + u_x - 1.0
    pde_loss = tf.reduce_mean(tf.square(pde_residual))

    x_bnd = tf.convert_to_tensor([[0.0], [1.0]], dtype=tf.float32)
    u_bnd, u_x_bnd, _ = compute_derivatives(model, x_bnd)
    bc0 = -alpha * u_x_bnd[0] + kappa * u_bnd[0]
    bc1 = alpha * u_x_bnd[1] + kappa * u_bnd[1]
    bc_loss = tf.reduce_mean(tf.square(bc0)) + tf.reduce_mean(tf.square(bc1))

    lambda_ = 0.1  # Emphasize boundary conditions
    total_loss = lambda_ * pde_loss + (1 - lambda_) * bc_loss
    return total_loss

# Flatten and unflatten utilities for L-BFGS
def flatten_variables(variables):
    flat = [tf.reshape(v, [-1]) for v in variables]
    return tf.concat(flat, axis=0)

def unflatten_variables(flat, variable_shapes, variable_sizes):
    flat = tf.convert_to_tensor(flat, dtype=tf.float32)
    splits = tf.split(flat, variable_sizes)
    return [tf.reshape(split, shape) for split, shape in zip(splits, variable_shapes)]

# Training function (Improved optimization)
def train_model(model, x_colloc, epsilon, epochs=20000):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1, decay_steps=2000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x_colloc, epsilon)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 4000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4e}")

    # Custom L-BFGS implementation (Extended iterations)
    variables = model.trainable_variables
    variable_shapes = [v.shape for v in variables]
    variable_sizes = [tf.size(v).numpy() for v in variables]

    def loss_and_grads(flat_params):
        params = unflatten_variables(flat_params, variable_shapes, variable_sizes)
        for v, p in zip(variables, params):
            v.assign(p)
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x_colloc, epsilon)
        grads = tape.gradient(loss, variables)
        flat_grads = flatten_variables(grads)
        return loss, flat_grads

    def lbfgs_optimize(max_iterations=5000):
        initial_position = flatten_variables(variables)
        result = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=loss_and_grads,
            initial_position=initial_position,
            max_iterations=max_iterations,
            tolerance=1e-10
        )
        optimized_params = unflatten_variables(result.position, variable_shapes, variable_sizes)
        for v, p in zip(variables, optimized_params):
            v.assign(p)
        return result

    lbfgs_optimize()
    return model

# Exact solution (Validated implementation)
def exact_solution(x, epsilon, alpha=1e-3, kappa=1.0):
    F = 1.0
    f = 1.0
    max_exp = 50.0  # Cap exponential terms to prevent overflow
    A = np.array([
        [kappa, kappa - alpha * F / epsilon],
        [kappa, (kappa + alpha * F / epsilon) * np.clip(np.exp(F / epsilon), None, np.exp(max_exp))]
    ])
    b = np.array([alpha * f / F, -(kappa + alpha) * f])
    try:
        C1, C2 = np.linalg.solve(A, b)
        result = C1 + C2 * np.clip(np.exp(x / epsilon), None, np.exp(max_exp)) + (f / F) * x
        return np.where(np.isfinite(result), result, np.nan)
    except np.linalg.LinAlgError:
        return np.full_like(x, np.nan)

# Parameters
K = 2000  # Number of collocation points
x_colloc = generate_collocation_points(K, beta=5.0)  # Cluster points near boundaries
epsilons = [0.1, 0.01, 0.001, 0.0001]
results = {}

# Run for each epsilon
for eps in epsilons:
    print(f"\nTraining for ε = {eps}")
    model = create_model()
    model = train_model(model, x_colloc, eps)

    x_fine = np.linspace(0, 1, 1000)[:, None]
    u_pred = model(x_fine).numpy().flatten()
    u_exact = exact_solution(x_fine.flatten(), eps)

    # Compute L2 error, handling NaN values
    mask = np.isfinite(u_exact)
    l2_error = np.nan if not np.any(mask) else np.sqrt(np.mean((u_pred[mask] - u_exact[mask]) ** 2))
    results[eps] = {"x": x_fine.flatten(), "u_pred": u_pred, "u_exact": u_exact, "l2_error": l2_error}
    print(f"ε = {eps:.4g} → L2 Error = {l2_error:.2e}")

# Plotting
plt.figure(figsize=(10, 6))
for eps in epsilons:
    plt.plot(results[eps]["x"], results[eps]["u_pred"], label=f'PINN ε={eps}')
    plt.plot(results[eps]["x"], results[eps]["u_exact"], '--', label=f'Exact ε={eps}')
plt.title("Vanilla PINN vs Exact Solution for Various ε")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.savefig('vanilla_pinn_results.png')