# COVID-19 Data Analysis and Prediction

## 1. Project Objectives

The project aims to:

- **Analyze COVID-19 Data:** Gather, process, and visualize COVID-19 data to understand both global and local trends.
- **Predict Cases and Deaths:** Develop predictive models to forecast future COVID-19 cases and deaths, with a focus on local analysis. In this project, the local analysis is performed for Lithuania due to its minimal missing data.
- **Compare Machine Learning Approaches:** Evaluate and compare various regression models to determine the best approach for predicting COVID-19-related metrics.
- **Support Decision-Making:** Provide analytical tools that can assist policymakers and researchers in assessing the effectiveness of health interventions.

## 2. Project Execution

### 2.1. Data Acquisition and Exploration
- **Data Source:**  
  The project utilizes COVID-19 data obtained via the `covid19dh` library. 
- **Initial Analysis:**  
  - The data is loaded in a tabular format (CSV).  
  - The dataset structure is examined using methods such as `head()`, `shape`, `info()`, and `describe()`.
  - Lithuania is chosen as a country to analyse.
  - The data, originally provided in cumulative form, is converted to daily figures in order to capture the day-to-day trends effectively.
  - A correlation matrix is visualized using a heatmap (with Seaborn) to identify patterns and relationships among key variables.

### 2.2. Data Preparation and Cleaning
- **Data Cleaning:**  
  - Unnecessary columns (e.g., "WHO Region") are removed to streamline the dataset.
- **NA Replacement Strategy:**  
  - For features with missing values (NaN), the project employs a strategy of imputing missing values with the latest available observation. This approach assumes that variables, particularly those related to government intervention policies, change infrequently over the analyzed period.
- **Feature Engineering:**  
  - The conversion of cumulative metrics to daily figures provides a clearer view of trends.
  - Data types are optimized (e.g., converting to int32) to reduce memory usage, which is especially important for large global datasets.

### 2.3. Exploratory Analysis and Visualization
- **Descriptive Analysis:**  
  - Statistical summaries are generated using the `describe` method from the Pandas library, providing basic descriptive statistics.
- **Correlation Analysis:**  
  - A heatmap is generated to display the correlations between variables, helping to identify potential relationships.
- **Trend Visualization for Lithuania:**  
  - Specific visualizations (such as line plots) are created to illustrate the progression of deaths and confirmed cases over time for Lithuania, both for cumulative and daily data. These plots serve to assess the overall dynamics of the pandemic in the country.

### 2.4. Building and Validating Predictive Models
- **Regression Models:**  
  The project implements multiple regression techniques to predict COVID-19 outcomes. The models employed include:
  - **Linear Regression**
  - **Support Vector Regression (SVR)**
  - **Decision Tree Regression**
  - **Random Forest Regression**

- **Model Variables and Evaluation Metrics:**  
  For each target variable (*deaths* and *confirmed cases*), several models are evaluated. Below is a summary of the performance metrics (Mean Squared Error (MSE) and R²) for the various models:

  | Model                         | Predicted Variable | Train MSE   | Test MSE    | Train R² | Test R² |
  | ----------------------------- | ------------------ | ----------- | ----------- | -------- | ------- |
  | LinearRegression              | deaths             | 16.50       | 12.84       | 0.88     | 0.91    |
  | LinearRegression              | confirmed          | 559008.13   | 959256.17   | 0.85     | 0.81    |
  | SVR                           | deaths             | 34.88       | 34.11       | 0.75     | 0.77    |
  | SVR                           | confirmed          | 1241668.35  | 168147.61   | 0.67     | 0.66    |
  | DecisionTreeRegressor         | deaths             | 16.71       | 14.80       | 0.88     | 0.90    |
  | DecisionTreeRegressor         | confirmed          | 609452.41   | 1694537.60  | 0.84     | 0.66    |
  | RandomForestRegressor         | deaths             | 14.86       | 13.68       | 0.90     | 0.91    |
  | RandomForestRegressor         | confirmed          | 561654.08   | 1171424.96  | 0.85     | 0.76    |

- **Variable Selection and Multicollinearity:**  
  - Variables for the regression models are selected based on Pearson correlation analysis and the Variance Inflation Factor (VIF) to ensure low multicollinearity.
- **Model Assumptions Validation:**  
  - **Statistical Tests:**  
    - The Shapiro-Wilk test is used to assess the normality of the residuals.
    - The Breusch-Pagan test is performed to test for homoscedasticity.
  - **Visualization of Residuals:**  
    - Histograms and scatter plots of residuals are created to visually evaluate the normality and constant variance assumptions of the models.

## 3. Technical and Methodological Details

The project is built on rigorous statistical analysis and a structured methodological approach. Key aspects include:

- Advanced data processing techniques using Pandas and NumPy.
- Transformation of cumulative data into daily metrics for enhanced trend analysis.
- Focused local analysis for Lithuania, selected due to its high-quality dataset with minimal missing values.
- A comprehensive evaluation and comparison of multiple regression models, aiming to identify the best predictive approach.
- Validation of model assumptions to ensure the reliability of the predictions.

## 4. Running the Project

### Requirements
- Python 3.7 or higher.
- Libraries: pandas, numpy, matplotlib, seaborn, covid19dh, scikit-learn.

### Installation Instructions
1. Clone the repository to your local machine.
2. Install the dependencies by executing the following command:

```bash
pip install -r requirements.txt
```

3. Launch the Jupyter Notebook by running:
```bash
 jupyter notebook covid.ipynb
```

## 5. References

- Guidotti, E., & Ardia, D. (2020). COVID-19 Data Hub. *Journal of Open Source Software*, 5(51):2376. doi:10.21105/joss.02376
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95.
- Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362. doi:10.1038/s41586-020-2649-2
- Waskom, M. L. (2021). Seaborn: Statistical Data Visualization. *Journal of Open Source Software*, 6(60), 3021.

## 6. License

This project is licensed under the MIT License. See the LICENSE file for details.

## 7. Contact

For questions or suggestions, please contact: [mar.kostrz@onet.pl].