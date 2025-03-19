import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate synthetic time series data
np.random.seed(42)
time_steps = 2000

# Normal pattern (sinusoidal signal)
x = np.arange(0, time_steps)
normal_data = np.sin(0.02 * x) + 0.1 * np.random.randn(time_steps)

# Introduce anomalies
anomaly_indices = np.random.choice(time_steps, size=20, replace=False)
anomaly_data = normal_data.copy()
anomaly_data[anomaly_indices] += np.random.uniform(2, 4, size=len(anomaly_indices))

# Plotting the data
plt.figure(figsize=(15, 5))
plt.plot(normal_data, label='Normal Data')
plt.plot(anomaly_data, label='Anomalous Data')
plt.title('Synthetic Time Series Data with Anomalies')
plt.legend()
plt.show()

# Save data to a DataFrame
data = pd.DataFrame({
    "value": anomaly_data
})
