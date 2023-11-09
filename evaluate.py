import csv
import os
import numpy as np
from CustomMetrics import CustomMetrics

# Paths to the labels and predictions directories
labels_dir = 'dataset/labels'
predictions_dir = 'dataset/predictions'

# Initialize the CustomMetrics class with the labels and predictions directories
custom_metrics = CustomMetrics(labels_dir, predictions_dir)

# Calculate the custom metrics
results = custom_metrics.calculate_metrics()

# Prepare data to be written to the CSV file
csv_data = [
    ['Image File', 'Pixel Precision', 'Pixel Recall', 'Average Redundancy', 'Average Quality']
] + list(zip(
    results['label_files'],
    results['pixel_precisions'],
    results['pixel_recalls'],
    results['average_redundancies'],
    results['average_quality_metrics']
))


# Specify the name of the CSV file
csv_file = 'evaluation_results.csv'

# Write the results to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Results have been saved to {csv_file}")

