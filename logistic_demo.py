import numpy as np
import matplotlib.pyplot as plt

def logistic(x):
    """
    Logistic function implementation.
    
    Parameters:
    x (float or array-like): Input value(s).
    
    Returns:
    float or ndarray: Output of the logistic function.
    """
    return 1 / (1 + np.exp(-x))

# Generate input values for plotting
x_values = np.linspace(-10, 10, 500)

t=[]
w=1.0
b=0.1
for x in x_values:
    t.append(w*x + b)

# Calculate logistic function values
y_values = logistic(t)

# Plot the logistic function
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Logistic Function', color='blue')
plt.axhline(y=0.5, color='red', linestyle='--', label='y = 0.5')
plt.axvline(x=0, color='green', linestyle='--', label='x = 0')
plt.title('Logistic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.show()
