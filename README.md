# Clustering Project: Wine Dataset
![winetaste](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRky2ygrKbF3Mz9utL7qEsBBOhL44RiUlzWcw&usqp=CAU)

## <u>Project Description</u>
Unraveling the complexities of wine quality prediction requires considering both objective and subjective qualities. Aggregating and analyzing these attributes can assist vintners in crafting superior products. This project aims to uncover key predictors that influence wine quality, leveraging data from data.world on chemical properties of red and white wine. We will explore their relationship to overall wine quality, utilizing cluster analysis to identify contributing factors and regression models for quality rating predictions. Enhance wine production with data-driven insights and elevate the quality of wines..

## <u>Project Goal</u>

* Identify key variables (drivers) influencing wine quality ratings.
* Apply Kmeans clustering to explore relationships among wine chemical properties.
* Evaluate the effectiveness of the resulting clusters in linear and polynomial regression models for predicting wine quality ratings and implement the final model for future predictions.

## <u>Initial Questions</u>
1. Does category of wine "red or white" have a relationship to wine quality?
2. Does density have a relationship with wine quality?
3. Does chlorides have a relationship with wine quality?
4. Does volitale acidity have a relationship with wine quality?

##<u>Data Dictionary</u>

There were 13 columns in the initial data and 13 columns after preparation; 6497 rows in the intial data and 5320 after preparation. The target variable is quality: 

|     Target         |  Datatype  |       Definition                                |
|:-------------------|:-----------|:------------------------------------------------|
|  quality           |  int       |  wine quality median rating from 3 wine experts |


|  Column Name       |  Datatype  |     Definition                                  |
|:-------------------|:-----------|:------------------------------------------------|
|  fixed acidity     |  float     |  acidic compunds contributing to tartness       |
|  volatile acidity  |  float     |  acidic compounds contributing vinegar flavor   |
|  citric acid       |  float     |  specific acid contributing to tartness         |
|  residual sugar    |  float     |  sugar left after fermenting                    |
|  chlorides         |  float     |  chloride-based salts (saltiness                |
|free sulfur dioxide |  float     |  sulfur in wine that has not yet reacted        |
|total sulfur dioxide|  float     |  total sulfur, reacted and not reacted          |
|  density           |  float     |  hydrometer reading of alcohol content          | |  pH                |  float     |  acidity vs alkilinity                          | |  sulphates         |  float     |  type of sulfur-based salt                      | |  alcohol           |  float     |  alcohol as a percentage of wine                | |  Yes_white         |  int       |  white wine = 1, red wine = 0                   |     

## Initial Hypotheses
Hypothesis 1 - Pearson R

alpha = .05
H0 = Chlorides has no relationship with wine quality
Ha = Chlorides has a relationship with wine quality
Outcome: We reject the Null Hypothesis.

Hypothesis 2 - Pearson R

alpha = .05
H0 = Volatile acidity has no relationship with wine quality
Ha = Volatile acidity has a relationship  wine quality
Outcome: We reject the Null Hypothesis.

## <u>Planning Process</u>

##### Planning
Clearly define the problem to be investigated, such as the impact square feet on property assessed tax value.
Obtain the required data from the Data.world Wine Quality dataset at https://data.world/food/wine-quality"Zillow.csv" database.
Create a comprehensive README.md file documenting all necessary information.
* Acquire dataset from data.world Wine Quality dataset

##### Acquisition and Preparation
Develop the acquire.py and prepare.py scripts, which encompass functions for data acquisition, preparation, and data splitting.
Implement a .gitignore file to safeguard sensitive information and include files (e.g., env file) that require discretion.
##### Exploratory Analysis
Create preliminary notebooks to conduct exploratory data analysis, generate informative visualizations, and perform relevant statistical tests (e.g., Pearsonr, t-test) utilizing Random Seed value 210 and alpha = .05.
##### Modeling
Train and evaluate various models, such as Ordinary Least Squares (OLS) Linear Regression, Least Absolute Shrinkage and Selection Operator (LASSO)+ Least Angle Regression (LARS), Tweedie Regression and Polynomial Regressionutilizing a Random Seed value of 123 and alpha= 1.0.
Train the models using the available data.
Validate the models to assess their performance.
Select the most effective model (e.g., OLS) for further testing.
##### Product Delivery
Prepare a final notebook that integrates the best visuals, models, and pertinent data to present comprehensive insights.

## <u>Instructions  to Reproduce the Final Project Notebook</u>
To successfully run/reproduce the final project notebook, please follow these steps:
1. Read this README.md document to familiarize yourself with the project details and key findings.
2. Import separate white and red .csv files from data.world Wine Quality dataset at https://data.world/food/wine-quality
3. Open the final_report.ipynb notebook in your preferred Jupyter Notebook environment or any compatible Python environment.
4. Ensure that all necessary libraries or dependent programs are installed. You may need to install additional packages if they are not already present in your environment.
5. Run the final_report.ipynb notebook to execute the project code and generate the results.

By following these instructions, you will be able to reproduce the analysis and review the project's final report. Feel free to explore the code, visualizations, and conclusions presented in the notebook.


## <u>Key Findings</u>
- White wine outperformed red wine in quality by three fold.
- Consider prolonging fermentation to reduce density, leading to an increase in alcohol content and potentially improving wine quality.
- Prioritize efforts on minimizing chlorides (associated with saltiness) and volatile acidity (linked to vinegar flavor) to enhance wine quality.
- Inconclusive cluster analysis due to high-density data in certain regions making it challenging to differentiate.



## <u>Conclusion</u>
This project utilized ML regression models and KMeans cluster analysis to identify unique value clusters in wine data. However, cluster analysis was not effective in predicting future wine quality ratings. White wine generally received higher ratings than red wine. Key drivers for quality ratings included volatile acidity, chlorides, and density. The LASSO + LARS regression model was the top performer, consistently outperforming the baseline by 17%. This model proves valuable for predicting wine quality.


## <u>Next Steps</u>
Based on the findings, the following recommendations and next steps are proposed:
1. Considering conducting DBSCAN cluster analysis can help in identifying and eliminating outliers; may produce more distinct clusters 
2. To gain more nuanced insights, it is recommended to conduct separate evaluations for white and red wines. Particular emphasis should be placed on minimizing chlorides and volatile acidity, as these factors significantly influence poor wine quality.
   
## <u>Recommendations</u>
- Vintners are encouraged to explore collecting data on various variables, such as temperature and duration of fermentation, with a focus on identifying factors that predict both poor and superior wine quality.
- To enhance data collection, it is advisable to include more wine experts (e.g., 4-6) to gather additional information for comprehensive analysis. The expertise of wine experts can provide valuable insights for refining the clustering approach.
