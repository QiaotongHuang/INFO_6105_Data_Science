#Linear Algebra
#1
import numpy as np
A = np.random.randint(-10, 11, size=(5, 2))
x = np.random.randint(-10, 11, size=(2,1))
diagonal_elements = [1, 2, 3, 4, 5] 
B = np.diag(diagonal_elements)
np.dot(A, x)
np.dot(B, A)

#2
from numpy.linalg import matrix_rank 
matrix_rank(A)
matrix_rank(B)
matrix_rank(np.dot(B, A))

#4
import numpy as np
p = np.poly1d([-1, 1, 33, 57])
roots = p.roots
print(roots)

#5
import numpy as np
# Define matrices A, B, and C
A = np.array([[-2, 1, 8], [-1, -1, 7], [3, 0, 4]])
B = np.array([[5, 0, -7], [6, 3, -9], [-2, -2, 0]])
C = np.array([[6, 3, -1], [2, 4, 5], [-1, -1, 8]])
# Concatenate A, B, and C into matrix D
D = np.concatenate((A, B, C), axis=0)
# Define vector b
b = np.array([3, -10, 2])
# Solve for the least squares solution x using numpy.linalg.lstsq
x, residuals, _, _ = np.linalg.lstsq(D.T, b, rcond=None)
# Print the least squares solution x
print("Least Squares Solution x:")
print(x)
# Calculate the squared norm of the residual (‖𝐷𝑇𝑥 − 𝑏‖²)
squared_residual_norm = np.linalg.norm(np.dot(D.T, x) - b)**2
# Print the squared norm of the residual
print("Squared Norm of Residual (‖𝐷𝑇𝑥 − 𝑏‖²):")
print(squared_residual_norm)


#Statistics
#1
# Import math module
from math import comb
# Define the number of trials
n = 1000
# Define the number of successful outcomes (specifically, 1 success)
k = 1
# Define the probability of success for each individual trial
p = 1/1024
# Calculate the probability of exactly one success using the binomial probability formula
one_success = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
# Calculate the probability of at least one success using complementary probability
at_least_one_success = 1 - ((1 - p) ** n)
# Print the calculated probabilities
print("Probability of exactly one success:", one_success)
print("Probability of at least one success:", at_least_one_success)

#2
import scipy.stats as stats
# Define the given values
x = 36  # Mean (average) number of times a cat returns to its food bowl
u = 5   # Standard deviation
# Calculate the z-score for 32 and 38 using the z-score formula
z_32 = (32 - x) / u  # Z-score for 32
z_38 = (38 - x) / u  # Z-score for 38
# Calculate the cumulative probability (CDF) for the z-scores using a standard normal distribution
p_32 = stats.norm.cdf(z_32)  # Cumulative probability for 32
p_38 = stats.norm.cdf(z_38)  # Cumulative probability for 38
# Calculate the answer by finding the difference in probabilities
answer = p_38 - p_32  # Probability of returning between 32 and 38 times
# Print the calculated answer
print("Probability of returning between 32 and 38 times:", answer)

#3
import numpy as np
import scipy.stats as stats
# Define the list of prices for a particular model of digital camera
prices = [999, 1499, 1997, 398, 591, 498, 798, 849, 449, 348]
# Calculate the number of samples, sample mean, and sample standard deviation
n = len(prices)
sample_mean = np.mean(prices)
sample_std_dev = np.std(prices, ddof=1)
# Set the confidence level to 95%
confidence_level = 0.95
# Calculate the critical value (Z-score) for the confidence level
z_critical = stats.norm.ppf((1 + confidence_level) / 2)
# Calculate the margin of error
margin_of_error = z_critical * (sample_std_dev / np.sqrt(n))
# Calculate the confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
# Print the results
print("Sample Mean Price:", sample_mean)
print("Sample Standard Deviation:", sample_std_dev)
print(f"95% Confidence Interval for Mean Price: {confidence_interval}")

#4
import scipy.stats as stats
# Given data
sample_mean = 9.6
population_mean = 8.5
population_std_dev = 3.2
sample_size = 40
alpha = 0.05
# Calculate the test statistic (t)
t_statistic = (sample_mean - population_mean) / (population_std_dev / (sample_size**0.5))
# Calculate the degrees of freedom
degrees_of_freedom = sample_size - 1
# Calculate the critical values for a two-tailed test
critical_value_left = stats.t.ppf(alpha / 2, degrees_of_freedom)
critical_value_right = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
# Calculate the p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), degrees_of_freedom))
# Make a decision
if p_value < alpha:
    decision = "Reject the null hypothesis"
else:
    decision = "Fail to reject the null hypothesis"
# Draw a conclusion
if p_value < alpha:
    conclusion = "There is a significant difference from the national average."
else:
    conclusion = "There is not enough evidence to conclude a significant difference."
# Print results
print("Test Statistic (t):", t_statistic)
print("Critical Value (left):", critical_value_left)
print("Critical Value (right):", critical_value_right)
print("P-value:", p_value)
print("Decision:", decision)
print("Conclusion:", conclusion)

#5
import numpy as np
import scipy.stats as stats
# Given data (sample grades)
grades = np.array([92.3, 89.4, 76.9, 65.2, 49.1, 96.7, 69.5, 72.8, 67.5, 52.8, 88.5, 79.2, 72.9, 68.7, 75.8])
# Sample size
n = len(grades)
# Sample variance
sample_variance = np.var(grades, ddof=1)  # Use ddof=1 for sample variance
# Hypothesized population variance
hypothesized_variance = 100
# Calculate the test statistic (chi-squared)
test_statistic = (n - 1) * sample_variance / hypothesized_variance
# Degrees of freedom
degrees_of_freedom = n - 1
# Calculate the critical value from the chi-squared distribution
alpha = 0.05
critical_value = stats.chi2.ppf(1 - alpha, df=degrees_of_freedom)
# Make a decision
if test_statistic > critical_value:
    decision = "Reject the null hypothesis"
else:
    decision = "Fail to reject the null hypothesis"
# Draw a conclusion
if test_statistic > critical_value:
    conclusion = "The variance in grades exceeds 100."
else:
    conclusion = "There is not enough evidence to conclude that the variance exceeds 100."
# Print results
print("Sample Variance:", sample_variance)
print("Test Statistic (chi-squared):", test_statistic)
print("Critical Value (chi-squared):", critical_value)
print("Decision:", decision)
print("Conclusion:", conclusion)


