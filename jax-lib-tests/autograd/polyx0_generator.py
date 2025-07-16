import jax
import jax.numpy as jnp
import csv


# Define the Newton solver
def newton_solver(coeffs, x0, max_iter=10):
    def f(x):
        return jnp.polyval(coeffs, x)
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = jax.grad(f)(x)
        x = x - fx / dfx
    return x


# Generate data pairs
def generate_data_pairs(num_samples, num_initial_guesses=100):
    key = jax.random.PRNGKey(42)
    data_pairs = []

    for _ in range(num_samples):
        key, subkey = jax.random.split(key)
        # Generate a polynomial with at least 3 coefficients > 0
        coeffs = jax.random.uniform(subkey, (4,), minval=-5.0, maxval=5.0)
        while jnp.sum(coeffs > 0) < 3:
            key, subkey = jax.random.split(key)
            coeffs = jax.random.uniform(subkey, (4,), minval=-5.0, maxval=5.0)

        # Find a starting point that converges in at most 4 iterations
        best_x0 = None
        for _ in range(num_initial_guesses):
            key, subkey = jax.random.split(key)
            x0 = jax.random.uniform(subkey, (), minval=-10.0, maxval=10.0)  # Random initial guess
            root_approx = newton_solver(coeffs, x0)
            if jnp.abs(jnp.polyval(coeffs, root_approx)) < 1e-6:  # Check if it converges to a root
                best_x0 = x0
                break

        # Append the data pair
        data_pairs.append((coeffs.tolist(), float(best_x0) if best_x0 is not None else 0.0))

    return data_pairs


# Example usage
num_samples = 100
data_pairs = generate_data_pairs(num_samples)
print(data_pairs)


# Example usage
num_samples = 100
data_pairs = generate_data_pairs(num_samples)

# Save the data pairs to a CSV file
with open('polyx0data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Coefficients', 'Starting Point'])  # Header
    for coeffs, x0 in data_pairs:
        writer.writerow([coeffs, x0])

print("Data pairs saved to 'polyx0data.csv'")
