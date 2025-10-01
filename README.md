# Predicting Subscription Churn Using Feature Importance Ranking and Customer Segmentation

## Overview....

This project analyzes subscription churn data to identify key drivers of churn and develop targeted retention strategies.  The analysis utilizes feature importance ranking techniques to pinpoint the most influential factors contributing to churn.  Furthermore, customer segmentation is employed to identify distinct groups of subscribers with varying churn probabilities, enabling the creation of tailored retention campaigns.  The project delivers insights into customer behavior and provides a framework for proactive churn management.

## Technologies Used....

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run....

1. **Install Dependencies:**  Ensure you have Python 3 installed.  Then, install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis:** Execute the main script:

   ```bash
   python main.py
   ```

## Example Output

The script will print key findings to the console, including:

* Feature importance scores, ranked in order of influence on churn.
* Summary statistics for each identified customer segment.
*  Details on the chosen segmentation model and its performance metrics.

Additionally, the analysis generates several visualization files (e.g., plots showing feature importance, churn rates by segment, etc.) in the `output` directory.  These files provide a visual representation of the analysis results.  The exact file names and contents may vary depending on the specific analysis performed.
