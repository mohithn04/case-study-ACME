# Acme Optimization Platform Case Study
Copyright © 2025 Mohith Nagendra. All rights reserved.

This code/program is the sole and exclusive intellectual property of Mohith Nagendra.
Unauthorized copying, modification, distribution, or use of this code, in whole or in part,
is strictly prohibited without the express written consent of Mohith Nagendra.

Author: Mohith Nagendra  
Date: 2025-02-24

# Acme Sales & Margin Optimizer

A sophisticated Python tool designed to optimize sales and profit margins across different business segments using linear programming techniques. It handles hierarchical business structures with multiple levels including Portfolio, Geography, Category, Brand, and Segment.

## Table of Contents

[Overview](#overview)\
[Features](#features)\
[Requirements](#requirements)\
[Environment Setup](#environment-setup)\
[Usage](#usage)\
[Output Files](#output-files)\
[Data Structure](#data-structure)\
[Limitations](#limitations)\
[Class Structure](#class-structure)\
[Example Output](#example-output)\
[Error Handling](#error-handling)


## Overview

This tool provides comprehensive optimization capabilities for complex business hierarchies:

- **Portfolio Level Analysis**
- **Geographic Optimization**
- **Category Performance**
- **Brand Analysis**
- **Segment-specific Optimization**

> Please scroll to bottom of this README for the output and generated files details.

## Features

### Multiple Optimization Strategies
- Sales maximization
- Margin maximization
- Sales target achievement with margin optimization
- Margin target achievement with sales optimization
- Five-year projections

### Hierarchical Analysis
- Portfolio level insights
- Geographical breakdown
- Category performance
- Brand analysis
- Segment-specific optimization

### Visualization & Reporting
- Comparative visualizations
- Detailed JSON reports
- Five-year trend analysis
- Performance metrics by segment

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- pulp
- matplotlib

## Environment Setup

If you want to do this in a virtual environment for easy clean up start at step 1, if not skip to step 3.
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
>**NOTE:** You may also have to install the CBC solver if you don't already have it.
- Using Homebrew:
    ```
    brew install cbc
    ```
- Ubuntu/Debian:
    ```
    sudo apt-get install coinor-cbc
    ```
- Windows:
  - Download the CBC executable from the COIN-OR website.
  - Extract the files and place the `cbc.exe` executable in a directory (e.g., `C:\cbc`).
  - Add this directory to your system’s PATH environment variable:
    1. Search for "Environment Variables" in the Windows search bar.
    2. Edit the "Path" variable under "System variables" or "User variables."
    3. Add the full path to the directory containing `cbc.exe`.

## Usage

### Basic Usage

```python
from casestudy import AcmeOptimizer

# Create optimizer instance
optimizer = AcmeOptimizer()

# Run optimization
optimizer.optimize_sales(constraints)
```

### Running the Main Program

```bash
python casestudy.py
```

## Output Files

The program generates several output files:

### Reports
- `sales_optimization_report.json`: Detailed sales optimization results
- `margin_optimization_report.json`: Margin optimization analysis
- `sales_target_report.json`: Results for sales target scenarios
- `margin_target_report.json`: Results for margin target scenarios

### Visualizations
- `sales_optimization.png`: Sales visualization
- `margin_optimization.png`: Margin visualization
- `five_year_projection.png`: 5-year trend visualization

## Data Structure

The system expects data in a hierarchical format:

```
Portfolio
├── Geography
    ├── Category
        ├── Brand
            └── Segment
                ├── Sales figures
                ├── Margin percentages
                ├── Growth trends
                └── Contribution metrics
```

### Constraints Configuration
```python
constraints = {
    "Brand:BrandName": {
        "min_trend": -0.01,
        "max_trend": 0.03,
        "max_contribution": 0.14
    },
    "Geography:Region": {
        "max_trend": 0.13,
        "Category:CategoryName": {
            "max_contribution": 0.05
        }
    }
}
```

### Visualization Settings
```python
optimizer.visualize_results(
    results,
    metric='Sales',
    level='Geography'
)
```

## Limitations

- Assumes linear relationships in optimization models
- Requires well-structured hierarchical data
- Memory usage scales with data size
- Single-threaded implementation

## Class Structure

### AcmeOptimizer
- `optimize_sales()`: Sales optimization
- `optimize_margin()`: Margin optimization
- `hit_sales_target_maximize_margin()`: Target-based optimization
- `visualize_results()`: Visualization methods
- `generate_reports()`: Report generation

## Example Output

The optimizer generates comprehensive results including:

### Optimization Results
- **Sales Maximization:** 20.00% increase ($83M → $99.6M)
- **Margin Maximization:** 20.00% increase ($25.7M → $30.8M)
- **Target Achievement:** Successfully hit sales target of $91.3M while maximizing margin
- **Average Margin:** Improved to 30.96%

### Five-Year Projections
| Year | Sales (M) | Profit (M) | Growth |
|------|-----------|------------|--------|
| 1    | $99.6     | $30.8      | 20.00% |
| 2    | $119.5    | $37.0      | 20.00% |
| 3    | $143.4    | $44.4      | 20.00% |
| 4    | $172.1    | $53.3      | 20.00% |
| 5    | $206.5    | $63.9      | 20.00% |

### Generated Files Which are Included
- **Reports:**
  - `sales_optimization_report.json`
  - `margin_optimization_report.json`
  - `sales_target_report.json`
  - `margin_target_report.json`
- **Visualizations:**
  - `sales_optimization.png`
  - `margin_optimization.png`
  - `five_year_projection.png`

## Error Handling

### Common Issues and Solutions
- **CBC Solver Not Found**: Ensure CBC solver is properly installed and accessible in PATH
- **Data Format Errors**: Verify input data matches the expected hierarchical structure
- **Memory Issues**: For large datasets, consider using the batch processing option

### Logging
The system logs errors and operations to `optimizer.log`. Log levels can be configured in the settings.