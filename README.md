This is a repository containing custom evaluation metrics for computer vision models.

There is a Python script called CustomMetrics.py which provides custom evaluation metrics for assessing the performance of a model trained to detect garbage items. 

## Prerequisites

The script is written in Python and to aid reproducibility only requires NumPy beyond the Python Standard Library:

```
pip install numpy
``` 
## Usage

The `CustomMetrics` class is initialized with the directories containing the ground truth labels and the model's predictions. These directories should contain text files, each corresponding to an image, with garbage items labeled in YOLOv5 format.

Here is a basic example of how to use the script:

```
from CustomMetrics import CustomMetrics

# Initialize with directories containing ground truth labels and predictions
metrics = CustomMetrics("path_to_labels", "path_to_predictions")

# Calculate custom metrics
results = metrics.calculate_metrics()

# Print the results
for key, value in results.items():
    print(f"{key}: {value}")
```

## Custom Metrics

1. **Pixel-Level Precision and Recall:**
   These metrics provide insights into the model’s precision and recall ability at the pixel level.

2. **Redundancy Metric:**
   This metric calculates the average redundancy of predictions for each ground truth item. It helps in understanding how often the model is over-predicting the same item as a function of IoU.

3. **Bounding Box Quality Metric:**
   This metric evaluates the quality of the predicted bounding boxes based on their aspect ratios compared to the ground truth. It helps in assessing whether the model is predicting the shapes of garbage items effectively. It is calculated as follows for predicted bounding boxes that reach the IoU threshold>=0.5:

$$ \text{Quality Metric} = 1 - \frac{\left| \frac{w_1}{h_1} - \frac{w_2}{h_2} \right|}{\max \left( \frac{w_1}{h_1}, \frac{w_2}{h_2} \right)} $$

Where $w_1, h_1$ and $w_2$, $h_2$ are the width and height of the ground truth and predicted bounding box respectively. 

The metric is bounded between 0 (very bad shape) and 1 (perfect shape).

## Functions

The main functions included in the `CustomMetrics` class are:

- `calculate_metrics()`: Computes the custom metrics for all files in the provided directories.
- `pixel_level_precision_recall()`: Calculates pixel-level precision and recall.
- `redundancy_metric()`: Computes the redundancy in overlapping predictions for each ground truth bounding box.
- `bbox_quality()`: Assesses the quality of the bounding box shapes based on their aspect ratios.

## Testing and results.
By running the CustomMetrics.py file itself, the mean and standard deviation over the entire dataset will be printed for each metric. In a production setting, this is better done in a separate file so there is also an evaluation.py which saves the calculated metrics to a .csv file.

On my local machine, I get the following results when running CustomMetrics.py using Python 3.8.10:

Mean Pixel-Level Precision: 0.57
Standard Deviation of Pixel-Level Precision: 0.26
Mean Pixel-Level Recall: 0.83
Standard Deviation of Pixel-Level Recall: 0.19
Mean Average Redundancy: 0.23
Standard Deviation of Average Redundancy: 0.35
Mean Average Quality Metric: 0.85
Standard Deviation of Average Quality Metric: 0.07

The pixel-level precision of our model is quite modest and there is a relatively large variability in the precision between the images. The pixel-level recall is quite high which translates to relatively few false negatives. However, a high recall can come at the cost of precision, leading to more false positives, which could be the case for this model. 
For an IoU threshold of 0.5, there don't appear to be many redundant detections left. This does vary substantially with the IoU threshold though. The overall aspect ratio of the bounding boxes seems to match well with our ground truth bounding boxes.
