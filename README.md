# post-ocr-text-correction
Repository for post-OCR text correction for Bulgarian historical documents

## Overview

This repository contains a comprehensive framework for correcting text errors that occur during Optical Character Recognition (OCR) of Bulgarian historical documents. The project includes various components for handling datasets, training and evaluating error correction models, and running different error correction algorithms.

## Directory Structure

- `config.py`: Manages configuration settings for the project.
- `config.ini`: Configuration file with settings.
- `data_utils.py`: Contains utility functions for reading and processing datasets.
- `dataset_parser.py`: Parses datasets for training and testing, including synthetic data generation.
- `correction_utils.py`: Contains utility functions and classes for handling text correction tasks.
- `error_corrector_seq2seq.py`: Implements a sequence-to-sequence (Seq2Seq) model for error correction.
- `error_corrector_knn.py`: Implements a k-nearest neighbors (KNN) model for error correction.
- `error_corrector_runner.py`: Runs the error correction models.
- `error_corrector_evaluator.py`: Evaluates the performance of error correction models.
- `error_detection_models.py`: Implements models for error detection.
- `error_detection_model_runner.py`: Runs the error detection models.
- `error_detection_baselines.py`: Contains baseline methods for error detection.
- `generate_synthetic_data.py`: Generates synthetic data for training and testing.
- `converter/`: Contains modules for converting datasets, including Drinovski and Ivanchevski converters.
- `data/`: Directory for storing datasets.

## Installation

To install the required dependencies, run:

```bash
pip install -r
