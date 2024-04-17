# Vessel-Estimated-Arrival-Time-Prediction-using-a-Tree-based-Stacking-Approach-CTG-port.

This project predicts vessel ETAs using historical AIS data and a tree-based stacking ensemble model, outperforming individual models. It highlights the value of ensemble learning for improved accuracy in maritime logistics.


## Overview

Utilizing AIS data, our project predicts vessel arrival times. Employing a tree-based stacking approach and hyperparameter tuning, we outperform individual models, impacting maritime logistics. Future work involves incorporating external data for enhanced model robustness.

## Source

The raw AIS database used for this research can be found here - https://ieee-dataport.org/open-access/vessel-tracking-ais-vessel-metadata-and-dirway-datasets

In addition, the codes for the employed algorithms can be found in the following Google Colab notebook links:

1)	LR - https://colab.research.google.com/drive/1S6gbeeGD3stDupFOZ9kWkYDHY-tTjYI5?usp=sharing

2)	RFR - https://colab.research.google.com/drive/1roIH8xhUlMsK969WCurvJHCDaQiPQ4R9?usp=sharing

3)	SVR - https://colab.research.google.com/drive/1RpLXNZh8JHo5MVdGnoFgSWriWQNnr9u-?usp=sharing

4)	ANN - https://colab.research.google.com/drive/1HNkWXIZcS-YwzSnv44Ssb93Saga0jxNS?usp=sharing

5)	KNN - https://colab.research.google.com/drive/1nftgOcoJlpbkhakI5U9wpwY6FF_Sd9aq?usp=sharing

6)	LightGBM - https://colab.research.google.com/drive/1Fhtg_VVGzjTDn0sQeNQl3hV20kQ5iT2Z?usp=sharing

7)	XGBoost - https://colab.research.google.com/drive/11h6xOuHLTk-zybDylSQOW6VdHHrEihkv?usp=sharing

8)	Stacking Ensemble Model: https://colab.research.google.com/drive/1BEIkqTgcbo47hkNEknQtnRfEF51ex68a?usp=sharing

Moreover, the codes for the distance formulas utilized for feature engineering can be found on the following links -

1. Vincenty Formula - https://colab.research.google.com/drive/1ofT1dpx4k6b3z_oa4_ga8Fq1_IgHMWeC?usp=sharing

2. Haversine Formula - https://colab.research.google.com/drive/1P_lgnTbcOGMJs4P4zxy7iEVsTei7olYt?usp=sharing

## Research Objective

The research objective is to enhance maritime logistics by accurately predicting vessel arrival times at Chattogram Port using AIS data and machine learning algorithms. It aims to develop a novel hybrid regression stacking model and identify features, output variables, and algorithms for precise ETA prediction. This study addresses gaps in existing research by introducing a unique approach and contributing to the advancement of port management efficiency.


## Keyword Understanding

1. MMSI (Maritime Mobile Service Identity): A unique 9-digit code assigned to vessels for identification in maritime communication systems.

2. Departure Time: The time when a vessel leaves the departure port to begin its journey.

3. LATd (Latitude of Departure Port): The geographic coordinate specifying the north-south position of the departure port on the Earth's surface.

4. LONd (Longitude of Departure Port): The geographic coordinate specifying the east-west position of the departure port on the Earth's surface.

5. Arrival Time: The time when a vessel reaches the arrival port to conclude its journey.

6. LATa (Latitude of Arrival Port): The geographic coordinate specifying the north-south position of the arrival port on the Earth's surface.

7. LONa (Longitude of Arrival Port): The geographic coordinate specifying the east-west position of the arrival port on the Earth's surface.

8. AVGSPDkmph (Average Speed in Kilometer per Hour): The mean rate of travel of the vessel over the duration of its journey, measured in kilometers per hour.

9. DistanceKm (Distance in Kilometer): The total length of the route traveled by the vessel between the departure and arrival ports, measured in kilometers.

10. Vessel Type: The classification of the vessel based on its design, purpose, and characteristics, such as cargo ship, tanker, or passenger vessel.

11. Length: The longitudinal measurement of the vessel from bow to stern, typically measured in meters.

12. Width: The lateral measurement of the vessel from port to starboard, typically measured in meters.

13. Draft: The vertical distance between the waterline and the bottom of the hull of the vessel, indicating the depth of water needed to float the vessel fully loaded, typically measured in meters.

## Key Technical Frameworks

1. Linear Regression: A statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

2. Random Forest Regression: A machine learning algorithm that builds multiple decision trees during training and outputs the mean prediction of the individual trees for regression tasks.

3. Support Vector Regression (SVR): A machine learning algorithm that finds the hyperplane in a high-dimensional space that best fits the training data while minimizing the error, suitable for regression tasks.

4. XGBoost: An optimized gradient boosting machine learning library that implements the gradient boosting decision tree algorithm, known for its speed and performance.

5. LightGBM: A gradient boosting framework that uses tree-based learning algorithms, designed for distributed and efficient training of large-scale datasets.

6. KNN (K-Nearest Neighbors): A non-parametric algorithm used for classification and regression tasks, where the output is a class membership or the value of a continuous variable based on the majority vote or average of the k-nearest data points in the feature space.

7. ANN (Artificial Neural Network): A computational model inspired by the structure and function of the human brain, consisting of interconnected nodes (neurons) arranged in layers, used for various machine learning tasks including regression.

8. Ensemble Model: A machine learning model that combines the predictions of multiple individual models to improve performance and robustness, often achieving better results than any single model alone.

9. Cross Validation: A technique used to evaluate the performance of a machine learning model by splitting the dataset into multiple subsets, training the model on some subsets, and validating it on the remaining subsets to assess its generalization ability.

10. MAE (Mean Absolute Error): A metric used to measure the average absolute differences between predicted values and actual values, providing a measure of the model's accuracy.

11. MSE (Mean Squared Error): A metric used to measure the average squared differences between predicted values and actual values, providing a measure of the variance of the errors.

12. R^2 (Coefficient of Determination): A statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables in a regression model, indicating the goodness of fit of the model.

## Distance Calculation Algorithms

1. Haversine Formula: A formula used to calculate the shortest distance between two points on a sphere given their latitude and longitude coordinates. It considers the spherical shape of the Earth and calculates the great-circle distance, which is the shortest path between the two points along the surface of the sphere.

2. Vincenty Formula: A more accurate method for calculating the distance between two points on the Earth's surface compared to the Haversine formula. It takes into account the flattening of the Earth's shape at the poles and the equator, resulting in more precise distance calculations over long distances. The Vincenty formula considers the ellipsoidal shape of the Earth and accounts for variations in the Earth's radius at different latitudes.

## Libraries

1. Pandas: A Python library used for data manipulation and analysis. It provides data structures like DataFrame and Series, which are powerful tools for working with structured data, making tasks like data cleaning, transformation, and analysis more efficient.

2. NumPy: Short for Numerical Python, it is a fundamental package for scientific computing in Python. NumPy provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

3. Scikit-learn (sklearn): A popular machine learning library in Python that provides simple and efficient tools for data mining and data analysis. It features various algorithms for classification, regression, clustering, dimensionality reduction, and more, along with utilities for model selection and evaluation.

4. Matplotlib: A plotting library for Python that provides a MATLAB-like interface for creating static, interactive, and animated visualizations. It enables users to generate plots, histograms, bar charts, scatter plots, and more, allowing for the visualization of data and analysis results.

## Training Methodology

1. Utilize 80% of historical AIS data.

2. Preprocess data for quality and consistency.

3. Select and implement machine learning algorithms: Linear Regression, Random Forest Regression, Support Vector Regression, XGBoost, LightGBM, K-Nearest Neighbors, and Artificial Neural Network.

4. Train models with the selected algorithms, with an 80/20 train-test split.

5. Evaluate model performance using metrics like MAE, MSE, and R-squared.

6. Create ensemble models by combining predictions from individual models(LightGBM, XGBoost, KNN with Linear Regression as Meta-model).

7. Utilize cross-validation techniques to validate model robustness.

8. Iteratively refine models based on performance feedback.

## Methodology and Experimentation

1. Utilized Baltic Sea ports dataset due to unavailability of AIS data for Chattogram port.

2. Dataset comprised 1,048,576 data points from 144 vessels, spanning November 2017 to October 2018.

3. Applied Feature Engineering to extract 1250 vessel voyage data.

4. Divided dataset into 80% training and 20% validation sets.

5. Stacking Ensemble Model combining LightGBM, XGBoost, and KNN with Linear Regression as meta-model achieved highest R-squared (0.9949) and lowest MAE (58.9 minutes).

6. XGBoost demonstrated strong individual performance with R-squared of 0.9958 and MAE of 55 minutes.

7. LightGBM exhibited R-squared of 0.9933 and MAE of 4 hours 19 minutes.

8. Cross-validation confirmed Stacking Ensemble's superior generalization (Cross-Val MAE 2 hours 57 minutes, R2 0.9956).

9. Ensemble model visually outperformed individual models.

## Results

1. The Stacking Ensemble Model combining LightGBM, XGBoost, and KNN with Linear Regression as meta-model achieved highest R-squared (0.9949) and lowest mean absolute error (MAE) of 58.9 minutes, outperforming individual models.

2. Cross-validation confirmed its superior generalization (Cross-Val MAE 2 hours 57 minutes, R2 0.9956).

3. XGBoost was next best individual model (R2 0.9958, MAE 55 minutes, Cross-Val MAE 40 minutes, R2 0.9997), followed by LightGBM (R2 0.9933, MAE 4 hours 19 minutes, Cross-Val MAE 3 hours 56 minutes, R2 0.9907).

4. Feature importance analysis revealed LightGBM prioritized average speed, distance, location; XGBoost emphasized distance, average speed, vessel length, draft; both consistently identified average speed and distance as crucial.

5. Stacking Ensemble leveraged individual strengths for superior accuracy, followed by XGBoost and LightGBM tree-based models.

## Limitations

1. The study is limited to the Baltic Sea region, which may restrict the generalizability of the findings to other maritime areas.

2. The AIS data used might not encompass all relevant factors influencing ETA prediction, such as weather conditions, traffic patterns, wind speed, current speed, and tidal directions.

3. Reliance on foundational models like LightGBM, XGBoost, and KNN without introducing novel adaptations may limit the exploration of alternative modeling approaches tailored to AIS data characteristics.

4. The absence of real-time data integration and dynamic external factors like changing environmental conditions and geopolitical influences may affect the predictive accuracy and responsiveness of the models.

5. The research does not explore the computational and resource requirements associated with implementing the proposed models in real-world maritime logistics operations.

## Conclusion

1. The study showcases the effectiveness of the Stacking Ensemble Model, combining LightGBM, XGBoost, and KNN with Linear Regression, for ETA prediction in the Baltic Sea.

2. Achieving an impressive R-squared value of 0.9949 and a minimal mean absolute error (MAE) of 58.9 minutes, the ensemble model outperforms individual models.

3. Cross-validation confirms the model's robust generalization with a Cross-Validation MAE of 2 hours 57 minutes and a Cross-Validation R-squared of 0.9956.

4. Feature importance analysis underscores the importance of average speed and distance, providing valuable insights for model optimization.

5. Despite limitations, including regional focus and data constraints, the study highlights the potential of advanced modeling techniques for enhancing maritime logistics.
