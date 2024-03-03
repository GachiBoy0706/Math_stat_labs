import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import cauchy
np.random.seed(42)

normal = np.random.normal(0, 1, 1000)
cauchy = np.random.standard_cauchy(1000)
student = np.random.standard_t(df=3, size=1000)
puasson = np.random.poisson(10, size=1000)
uniform = np.random.uniform(-(math.sqrt(3)), math.sqrt(3), 1000)

array_of_distributions = [normal, cauchy, student, puasson, uniform]
names = ["normal", "cauchy", "student", "puasson", "uniform"]
array_of_powers = [10, 50, 1000]
array_of_intervals = [7, 10, 20]

#normal
for power, interval in zip(array_of_powers, [7, 10, 20]):
    plt.figure(figsize=(8, 6))
    sns.histplot(normal[:power], bins=interval, kde=True, stat="density")
    plt.title(f'{'normal'}, size {power}', fontsize=18)
    plt.xlabel('Values', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.savefig(f'{'normal'}, size {power}')

#cauchy
for power, interval in zip(array_of_powers, [5, 7, 300]):
    filtered_data = cauchy[(cauchy >= -50) & (cauchy <= 50)]
    plt.figure(figsize=(8, 6))
    sns.histplot(filtered_data[:power], bins=interval, kde=True, stat="density")
    plt.title(f'{'cauchy'}, size {power}', fontsize=18)
    plt.xlabel('Values', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.savefig(f'{'cauchy'}, size {power}')

#puasson
for power, interval in zip(array_of_powers, [5, 10, 19]):
    plt.figure(figsize=(8, 6))
    sns.histplot(puasson[:power], bins=interval, kde=True, stat="density")
    plt.title(f'{'puasson'}, size {power}', fontsize=18)
    plt.xlabel('Values', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.savefig(f'{'puasson'}, size {power}')


#student
for power, interval in zip(array_of_powers, [5, 20, 30]):
    plt.figure(figsize=(8, 6))
    sns.histplot(student[:power], bins=interval, kde=True, stat="density")
    plt.title(f'{'student'}, size {power}', fontsize=18)
    plt.xlabel('Values', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.savefig(f'{'student'}, size {power}')


#uniform
for power, interval in zip(array_of_powers, [5, 20, 30]):
    plt.figure(figsize=(8, 6))
    sns.histplot(uniform[:power], bins=interval, kde=True, stat="density")
    plt.title(f'{'uniform'}, size {power}', fontsize=18)
    plt.xlabel('Values', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.savefig(f'{'uniform'}, size {power}')
