import os
import numpy as np
from typing import List, Tuple, Optional, Dict
from statistics import mean, stdev


class CustomMetrics:
    def __init__(self, labels_dir: str, predictions_dir: str):
        """
        Initialize the CustomMetrics class with the directories of ground truth labels and predictions.

        Parameters:
        - labels_dir: A string representing the directory path where the ground truth labels are stored.
                      Each file in this directory contains ground truth labels for a specific image.

        - predictions_dir: A string representing the directory path where the predicted labels are stored.
                          Each file in this directory contains predicted labels for a specific image

        Attributes:
        - labels_dir: Stores the directory path of the ground truth labels.

        - predictions_dir: Stores the directory path of the predicted labels.

        - image_width: The width of the images for which the bounding boxes are predicted and labeled.
                       It is used to convert normalized bounding box coordinates to pixel coordinates.

        - image_height: The height of the images for which the bounding boxes are predicted and labeled.
                        It is used to convert normalized bounding box coordinates to pixel coordinates.
        """
        self.labels_dir = labels_dir
        self.predictions_dir = predictions_dir
        self.image_width = 1024
        self.image_height = 768

    def _read_boxes(
        self, file_path: str
    ) -> List[Tuple[int, float, float, float, float]]:
        """
        Read bounding boxes from a given file.

        Parameters:
        - file_path: The path of the file containing the bounding boxes data.

        Returns:
        - A list of tuples representing the bounding boxes as (class, x_center, y_center, width, height).
        """
        with open(file_path, "r") as file:
            lines = file.readlines()
            boxes = [
                (int(line.strip().split()[0]), *map(float, line.strip().split()[1:]))
                for line in lines
            ]
        return boxes

    def calculate_metrics(self) -> Dict:
        """
        Calculate custom metrics for all the files in the labels and predictions directories.

        Returns:
        - A dictionary containing lists of each custom metric for all files.
        """
        pixel_precisions = []
        pixel_recalls = []
        average_redundancies = []
        average_quality_metrics = []
        label_files = os.listdir(self.labels_dir)

        for file in label_files:
            ground_truths = self._read_boxes(os.path.join(self.labels_dir, file))
            predictions = self._read_boxes(os.path.join(self.predictions_dir, file))

            pixel_precision, pixel_recall = self.pixel_level_precision_recall(
                ground_truths, predictions
            )
            pixel_precisions.append(pixel_precision)
            pixel_recalls.append(pixel_recall)

            average_redundancy_index = self.redundancy_metric(
                ground_truths, predictions
            )
            average_redundancies.append(average_redundancy_index)

            average_quality_metric = self.bbox_quality(ground_truths, predictions)
            average_quality_metrics.append(average_quality_metric)

        metrics = {
            "label_files": label_files,
            "pixel_precisions": pixel_precisions,
            "pixel_recalls": pixel_recalls,
            "average_redundancies": average_redundancies,
            "average_quality_metrics": average_quality_metrics,
        }
        return metrics

    def pixel_level_precision_recall(
        self,
        ground_truths: List[Tuple[int, float, float, float, float]],
        predictions: List[Tuple[int, float, float, float, float]],
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate the pixel-level precision and recall by comparing binary masks of ground truth
        and predicted bounding boxes.

        Parameters:
        - ground_truths: A list of ground truth bounding boxes.
        - predictions: A list of predicted bounding boxes.

        Returns:
        - A tuple containing pixel-level precision and recall as floats.
        - If there are no ground truths, i.e. the image has no object of interest in the first place,
        it returns None.
        """

        # First index of numpy array corresponds to rows i.e. height
        gt_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        pred_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        for gt in ground_truths:
            x, y, w, h = self._convert_to_pixel_coordinates(gt[1:])
            gt_mask[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2] = 1

        for pred in predictions:
            x, y, w, h = self._convert_to_pixel_coordinates(pred[1:])
            pred_mask[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2] = 1

        true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
        false_positive = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
        false_negative = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))

        # Conditional statement ensures no divide by zero.
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0
        )

        return precision, recall if true_positive + false_negative > 0 else None

    def redundancy_metric(
        self,
        ground_truths: List[Tuple[int, float, float, float, float]],
        predictions: List[Tuple[int, float, float, float, float]],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Calculate the redundancy in overlapping predictions for each ground truth bounding box.

        Parameters:
        - ground_truths: A list of ground truth bounding boxes.
        - predictions: A list of predicted bounding boxes.
        - iou_threshold: The IoU threshold to consider a predicted box as a significant overlap.

        Returns:
        - The average redundancy for the ground truth bounding boxes with at least one overlapping prediction.
        """
        redundancies = []

        for gt in ground_truths:
            count = 0
            for pred in predictions:
                iou = self._compute_iou(
                    self._convert_to_pixel_coordinates(gt[1:]),
                    self._convert_to_pixel_coordinates(pred[1:]),
                )
                if iou > iou_threshold:
                    count += 1

            if count > 0:
                redundancy = (
                    count - 1
                )  # subtracting 1 to ignore the first (desired) overlap
                redundancies.append(redundancy)
        average_redundancy = np.mean(redundancies) if redundancies else 0
        return average_redundancy

    def bbox_quality(
        self,
        ground_truths: List[Tuple[int, float, float, float, float]],
        predictions: List[Tuple[int, float, float, float, float]],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Calculate the quality of the bounding box shapes based on their aspect ratio w.r.t the ground truth bounding box.

        Parameters:
        - ground_truths: A list of ground truth bounding boxes.
        - predictions: A list of predicted bounding boxes.
        - iou_threshold: The IoU threshold to match ground truths and predictions.

        Returns:
        - The average quality of the predicted bounding boxes (which reach the IoU threshold) as float.
        """
        quality_metrics = []

        for gt in ground_truths:
            for pred in predictions:
                iou = self._compute_iou(
                    self._convert_to_pixel_coordinates(gt[1:]),
                    self._convert_to_pixel_coordinates(pred[1:]),
                )

                if iou > iou_threshold:
                    # Compute the aspect ratio difference
                    aspect_ratio_gt = gt[3] / gt[4] if gt[4] != 0 else 0
                    aspect_ratio_pred = pred[3] / pred[4] if pred[4] != 0 else 0
                    norm_aspect_ratio_diff = abs(
                        aspect_ratio_gt - aspect_ratio_pred
                    ) / max(
                        aspect_ratio_gt, aspect_ratio_pred
                    )  # min = 0 (good), max = 1 (bad)

                    quality_metric = 1 - norm_aspect_ratio_diff
                    quality_metrics.append(quality_metric)

        average_quality_metric = np.mean(quality_metrics) if quality_metrics else 0
        return average_quality_metric

    def _compute_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> Optional[float]:
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        - bbox1: a tuple of pixel bounding box coordinates
        - bbox2: a tuple of pixel bounding box coordinates

        Returns:
        - The IoU as a float
        - If there is no area of intersection between the bounding boxes, it returns zero.
        """
        intersection = self._compute_intersection(bbox1, bbox2)
        if intersection:
            # add 1 to width and height to counter zero-division errors
            intersection_area = (1 + intersection[2]) * (1 + intersection[3])
            bbox1_area = (1 + bbox1[2]) * (1 + bbox1[3])
            bbox2_area = (1 + bbox2[2]) * (1 + bbox2[3])
            union_area = bbox1_area + bbox2_area - intersection_area
            return intersection_area / union_area if union_area > 0 else 0
        else:
            return 0

    def _convert_to_pixel_coordinates(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        """
        Convert normalized bounding box coordinates to pixel coordinates based on the given image resolution.

        Parameters:
        - bbox: a tuple of normalized bounding box coordinates

        Returns:
        - a tuple of the bounding box pixel coordinates.
        """
        x_center = int(bbox[0] * self.image_width)
        y_center = int(bbox[1] * self.image_height)
        width = int(bbox[2] * self.image_width)
        height = int(bbox[3] * self.image_height)
        return x_center, y_center, width, height

    def _compute_intersection(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute the intersection of two bounding boxes.

        Parameters:
        - bbox1: a tuple of pixel bounding box coordinates
        - bbox2: a tuple of pixel bounding box coordinates

        Returns:
        - The top-left coordinates and dimensions of the intersecting region between two given bounding boxes.
        - If there is no intersection between the bounding boxes, it returns None.
        """
        x1 = max(bbox1[0] - bbox1[2] // 2, bbox2[0] - bbox2[2] // 2)
        y1 = max(bbox1[1] - bbox1[3] // 2, bbox2[1] - bbox2[3] // 2)
        x2 = min(bbox1[0] + bbox1[2] // 2, bbox2[0] + bbox2[2] // 2)
        y2 = min(bbox1[1] + bbox1[3] // 2, bbox2[1] + bbox2[3] // 2)
        if x1 < x2 and y1 < y2:
            return x1, y1, x2 - x1, y2 - y1
        else:
            return None


if __name__ == "__main__":
    # Example usage
    labels_dir = "dataset/labels"
    predictions_dir = "dataset/predictions"
    custom_metrics = CustomMetrics(labels_dir, predictions_dir)

    metrics = custom_metrics.calculate_metrics()
    print(metrics)

    pixel_precisions = metrics["pixel_precisions"]
    pixel_recalls = metrics["pixel_recalls"]
    average_redundancies = metrics["average_redundancies"]
    average_quality_metrics = metrics["average_quality_metrics"]

    # Displaying summary statistics
    print(f"Mean Pixel-Level Precision: {mean(pixel_precisions):.2f}")
    print(f"Standard Deviation of Pixel-Level Precision: {stdev(pixel_precisions):.2f}")
    print(f"Mean Pixel-Level Recall: {mean(pixel_recalls):.2f}")
    print(f"Standard Deviation of Pixel-Level Recall: {stdev(pixel_recalls):.2f}")
    print(f"Mean Average Redundancy: {mean(average_redundancies):.2f}")
    print(
        f"Standard Deviation of Average Redundancy: {stdev(average_redundancies):.2f}"
    )
    print(f"Mean Average Quality Metric: {mean(average_quality_metrics):.2f}")
    print(
        f"Standard Deviation of Average Quality Metric: {stdev(average_quality_metrics):.2f}"
    )
