import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os

# Exploring
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr

# Visualizing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

#Clustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


#----------AQUIRE AND PREPARE-------

def prepare_wine_data(): 
    "prepares data from wine dataset by taking separate csvs and combining them, concatenating this into a new dataframe, dropping duplicates, and renaming columns"

    # this reads in red wine dataframe from csv
    df_red = pd.read_csv('https://query.data.world/s/cjdbzy2v64s7prdtjzqbpycvvmar5c?dws=00000')
    # this reads in white wine dataframe from csv
    df_white = pd.read_csv('https://query.data.world/s/mymzmddiphbb65cotvuinb2pgyzamt?dws=00000')   
    #this creates a column labeling all items in red wine dataframe as red
    df_red['type'] = 'red'
    #this creates a column labeling all items in white wine dataframe as whie
    df_white['type'] = 'white'
    #this creates a variables containing both dataframes created above
    frames = [df_white, df_red]
    #this creates a new dataframe using the above variable and combining both dataframes
    df_wine = pd.concat(frames)   
    # Remove nulls if any
    df_wine_nonulls = df_wine.dropna()
    #this drops duplicate rows
    df_wine = df_wine.drop_duplicates(keep='last')
    return df_wine

def get_wine_data(df_wine):
    '''This function creates a csv for concat wine csv'''
    # Assuming you have a function 'get_wine()' that retrieves the wine data and returns a DataFrame
    df_wine2 = df_wine

    # Save the DataFrame to a CSV file
    df_wine.to_csv("wine.csv", index=False)  # Specify 'index=False' to exclude the index column in the CSV

    filename = 'wine.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)

def hot_wine(df_wine):
    '''This function takes a the previously concated red.csv and white.csv and creates a new dataframe wine.csv.'''
    # Assuming you have a previously concated red_wine.csv and white_wine.csv into 'df'DataFrame
    # Create one-hot encoding for the "type" column
    wine_type_df = pd.get_dummies(df_wine['type'],  prefix='Yes', drop_first=True)
    
    # Concatenate the DataFrames df and wine_type_df
    wine_df = pd.concat([df_wine, wine_type_df], axis=1)
    # Drop the original 'type' column
    wine_df.drop(columns=['type'], inplace=True)
    
    #Make Yes_wine data type int
    wine_df['Yes_white'] = wine_df['Yes_white'].astype(int)
    return wine_df

def split_wine(wine_df):
    '''Before split must run: wine_df = hot_wine(). to rename the dataframe.
    This function splits wine_df into 60%, 20% 20% ad prints the shape .'''
    df= wine_df
    train, wine_test = train_test_split(wine_df, test_size=0.2, random_state=210)
    wine_train, wine_validate = train_test_split(train, test_size=0.25, random_state=210) 
    print(f'Training set shape: {wine_train.shape}')
    print(f'Validation set shape: {wine_validate.shape}')
    print(f'Test set shape: {wine_test.shape}')
    return wine_train, wine_validate, wine_test

#-----------VISUALS FOR FINAL REPORT---
def red_wine():
    '''Red wine graph'''
    df_red = pd.read_csv('https://query.data.world/s/cjdbzy2v64s7prdtjzqbpycvvmar5c?dws=00000')
    rw = sns.histplot(data=df_red, x='quality', color='#800020', edgecolor='black', bins=range(11), kde=False)
    
    # Set the x-axis label and y-axis label
    plt.xlabel('Quality of Red Wine')
    plt.ylabel('Number of Observations')
    plt.title('Red Wine by Qualtiy')
    
    # Add count numbers on bars
    for p in rw.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        offset = width * 0.02  # Adjust the offset percentage as needed
        rw.annotate(format(height, '.0f'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    plt.show()

def white_wine():
    '''white wine graph''' 
    df_white = pd.read_csv('https://query.data.world/s/mymzmddiphbb65cotvuinb2pgyzamt?dws=00000')
    ww = sns.histplot(data=df_white, x='quality', color='#FF8C00', edgecolor='black', bins=range(11), kde=False)
    
    # Set the x-axis label and y-axis label
    plt.xlabel('Quality of White Wine')
    plt.ylabel('Number of Observations')
    plt.title('White Wine by Qualtiy')
    for p in ww.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        offset = width * 0.02  # Adjust the offset percentage as needed
        ww.annotate(format(height, '.0f'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.show()

def wine_KDE(wine_train):
    '''KDE graph''' 
    custom_palette = ['#FF8C00', '#800020']
    sns.kdeplot(data=wine_train, x='density', y='volatile acidity', hue='Yes_white',palette=custom_palette)
    plt.xlabel('Density Scale')
    plt.ylabel('Volatile Acidity')
    plt.title('Does Volatile Acidity Relate to Density?')
    plt.show() 

def chloride_lmplot(wine_train):
    '''lmplot 1''' 
    #changes color palette to match slides red yellow"
    custom_palette = ['#800020', '#FF8C00']
    new_labels = {'0': 'Red', '1': 'White'}
    cl = sns.lmplot(x='chlorides', y='quality', data=wine_train, hue='Yes_white', palette=custom_palette)
    plt.xlabel('Scale of Chlorides')
    plt.ylabel('Wine Quality')
    plt.title('Do Chlorides Relate to Wine Quality?')
    # Rename the hue legend
    leg = plt.legend(title='Type of Wine')
    for t, label in zip(leg.texts, new_labels.values()):
        t.set_text(label)
    plt.show()

def density_lmplot(wine_train):
    '''lmplot 2''' 
    custom_palette = ['#800020', '#FF8C00']
    new_labels = {'0': 'Red', '1': 'White'}
    cl = sns.lmplot(x='density', y='quality', data=wine_train, hue='Yes_white', palette=custom_palette)
    plt.xlabel('Density Scale')
    plt.ylabel('Wine Quality')
    plt.title('Does Density Relate to Wine Quality?')
    leg = plt.legend(title='Type of Wine')
    for t, label in zip(leg.texts, new_labels.values()):
        t.set_text(label)
    plt.show()
def acid_lmplot(wine_train):
    custom_palette = ['#800020', '#FF8C00']
    new_labels = {'0': 'Red', '1': 'White'}
    cl = sns.lmplot(x='volatile acidity', y='quality', data=wine_train, hue='Yes_white', palette=custom_palette)
    plt.xlabel('Volatile Acidity Scale')
    plt.ylabel('Wine Quality')
    plt.title('Does Volatile Acidity Relate to Wine Quality?')

    # Rename the hue legend
    leg = plt.legend(title='Type of Wine')
    for t, label in zip(leg.texts, new_labels.values()):
        t.set_text(label)
    plt.show()

def vc_jplot(wine_train):
    '''jointplot'''
    custom_palette = ['#FF8C00', '#800020']
    vc = sns.jointplot(data=wine_train, x="volatile acidity", y="chlorides", hue="Yes_white", kind="hist", palette=custom_palette)
    plt.xlabel('Scale of Volatile Acidity')
    plt.ylabel('Scale of Chlorides')
    plt.title('Does Volatile Acidity and Chloride have Relation?')
    plt.tight_layout()
    plt.show()
    
#-------STATISTICAL TESTING

def chlorides_stat(wine_train, wine_validate, wine_test):
    '''Pearson R stat for chlorides'''
    alpha = 0.05
    train_r, train_p = pearsonr(wine_train.chlorides, wine_train.quality)
    validate_r, validate_p = pearsonr(wine_validate.chlorides, wine_validate.quality)
#     test_r, test_p = pearsonr(test.chlorides, test.quality)
    print('train_r:', train_r)
    print('train_p:',train_p)
    print('validate_r:', validate_r)
    print('validate_p:', validate_p)
    print(f'The p-value is less than the alpha: {validate_p < alpha}')
    if validate_p < alpha:
        print('Outcome: We reject the null')
    else:
        print("Outcome: We fail to reject the null")
        
        
def density_stat(wine_train, wine_validate, wine_test):
    '''Pearson R stat for density'''
    alpha = 0.05
    train_r, train_p = pearsonr(wine_train.density, wine_train.quality)
    validate_r, validate_p = pearsonr(wine_validate.density, wine_validate.quality)
#     test_r, test_p = pearsonr(test.alcohol, test.quality)
    print('train_r:', train_r)
    print('train_p:',train_p)
    print('validate_r:', validate_r)
    print('validate_p:', validate_p)
    print(f'The p-value is less than the alpha: {validate_p < alpha}')
    if validate_p < alpha:
        print('Outcome: We reject the null')
    else:
        print("Outcome: We fail to reject the null")
        

# ----CLUSTERING FUNCTIONS----

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def X_scaled():
#Create X using volatiel_acid and density
    X = wine_train[['volatile acidity', 'density','chlorides']]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])
    

# 1. Create X using volatile acidity, density and chloride
# X = wine_train[['volatile acidity', 'density','chlorides']]
# X.head()

# 2. # Scarler using X.columns
# scaler = StandardScaler().fit(X)
# X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])
# X_scaled.head()

# 3. Elbow approach for to determine k. we went with k = 3 "Codeup instructor "
def k_elbow(X_scaled):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X_scaled).inertia_ for k in range(2, 12)}).plot(marker='x', color='#800020')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')

#  4. Create and graph cluster Models 
def create_cluster_models(wine_train, X, k):
    """Takes in df, X (dataframe with variables you want to cluster on), and k
    It scales the X, calculates the clusters, and returns the DataFrame (with clusters), 
    the scaled DataFrame, the scaler, kmeans object, and unscaled centroids as a DataFrame"""

    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values)
    
    #Model 1
    kmeans = KMeans(n_clusters=k, random_state=210)
    kmeans.fit(X_scaled[['volatile acidity', 'density']])
    centroids1 = pd.DataFrame((kmeans.cluster_centers_), columns=['volatile acidity', 'density'])
    # Add cluster columns to the X_scaled DataFrame
    X_scaled['cluster_M1'] = 'cluster_' + pd.Series(kmeans.predict(X_scaled[['volatile acidity', 'density']]).astype(str))
    
    #Model 2
    kmeans2 = KMeans(n_clusters=(k), random_state=210)
    kmeans2.fit(X_scaled[['volatile acidity', 'chlorides']])
    
    centroids2 = pd.DataFrame((kmeans2.cluster_centers_), columns=['volatile acidity', 'chlorides'])
    X_scaled['cluster_M2'] = 'cluster_' + pd.Series(kmeans2.predict(X_scaled[['volatile acidity', 'chlorides']]).astype(str))
    
    #Model 3 
    kmeans3 = KMeans(n_clusters=(k), random_state=210)
    kmeans3.fit(X_scaled[['chlorides', 'density']])
    centroids3 = pd.DataFrame((kmeans3.cluster_centers_), columns=['chlorides', 'density'])
    X_scaled['cluster_M3'] = 'cluster_' + pd.Series(kmeans3.predict(X_scaled[['chlorides', 'density']]).astype(str))
    

    
    # Create cluster Model graphs using features volatile acidity and density
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=X_scaled, x='volatile acidity', y='density', hue='cluster_M1', palette='YlOrRd')
    centroids1.plot.scatter(x='volatile acidity', y='density', ax=plt.gca(), color='k', alpha=0.9, s=200, marker=(4, 1, 0), label='centroids')
    plt.title('Cluster Model 1')
    
    # Create cluster Model graphs using features volatile acidity and chlorides
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=X_scaled, x='volatile acidity', y='chlorides', hue='cluster_M2', palette='YlOrRd')
    centroids2.plot.scatter(x='volatile acidity', y='chlorides', ax=plt.gca(), color='k', alpha=0.9, s=200, marker=(4, 1, 0), label='centroids')
    plt.title('Cluster Model 2')
    
    # Create cluster Model graphs using features chlorides and density
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=X_scaled, x='chlorides', y='density', hue='cluster_M3', palette='YlOrRd')
    centroids3.plot.scatter(x='chlorides', y='density', ax=plt.gca(), color='k', alpha=0.9, s=200, marker=(4, 1, 0), label='centroids')
    plt.title('Cluster Model 3')
    plt.tight_layout()
    plt.show()
    
    return wine_train, X_scaled, scaler, kmeans, centroids1, centroids2, centroids3 
# plot out volatile acidity vs density with regard to the cluster and age
def clusters_1():
    sns.relplot(data=X_scaled, x="volatile acidity", y="density", col="cluster_M1", hue="cluster_M1", col_wrap=2,palette='YlOrRd')
    X_scaled.cluster = X_scaled.cluster_M1.map({
        0: "density",
        1: "volatile acidity",
        2: "density_vacid"
    })
    plt.show()
## ----------------------------------REGRESSION MODELING -------------------    
def x_y_split(wine_train, wine_validate, wine_test):
    X_train, y_train = wine_train.drop(columns=['quality']), wine_train.quality
    X_validate, y_validate = wine_validate.drop(columns=['quality']), wine_validate.quality
    X_test, y_test = wine_test.drop(columns=['quality']), wine_test.quality
    return X_train, y_train, X_validate, y_validate, X_test, y_test 

def wine_distplot(y_train):
    plt.hist(y_train.quality, color= 'brown')
    plt.title("Distribution of Target (Wine Quality)")
    plt.xlabel("Final algorithm distribution (Wine")
    plt.ylabel("Number of Features")
    plt.show()
def get_baseline(y_train, y_validate):
    '''This function gets the baseline for modeling and creates a metric df '''

    #  y_train and y_validate to be dataframes to append the new metric columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # Baseline for mean 
    # 1. Predict quality_pred_mean  make columns for train and validate
    quality_pred_mean = y_train.quality.mean()
    y_train['quality_pred_mean'] = quality_pred_mean
    y_validate['quality_pred_mean'] = quality_pred_mean 

  # 3. RMSE of quality_pred_mean
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_mean) ** (.5)
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_mean) ** (.5)

    # create a df to easily view results of models
    metric_df = pd.DataFrame(data = [
        {
            'model': "mean_baseline",
            'RMSE_train': rmse_train,
            'RMSE_validate': rmse_validate,
            "R2_validate": explained_variance_score(y_validate.quality, y_validate.quality_pred_mean)
        }
    ])

    return y_train, y_validate, metric_df
def wine_distplot(y_train):
    plt.hist(y_train.quality, color= 'brown')
    plt.title("Distribution of Target (Wine Quality)")
    plt.xlabel("Final algorithm distribution (Wine")
    plt.ylabel("Number of Features")
    plt.show()

def act_vs_pred(y_train):
    ''' This function graphs actual vs prediction for Wine Quality '''
    plt.hist(y_train.quality, color='brown', alpha=.5, label="Actual Wine Quality")
    plt.hist(y_train.quality_pred_mean, bins=1, color='red', alpha=.5,  label="Predicted Wine Quality - Mean")
    #plt.hist(y_train.quality_pred_median, bins=1, color='orange', alpha=.5, label="Predicted Wine Quality - Median")
    plt.xlabel("Final Wine Quality Score (quality)")
    plt.ylabel("Number of Observations")
    plt.legend()
    plt.show()
    
    
def ols_lasso_tweedie_poly(X_train, X_validate, y_train, y_validate, X_test, metric_df):
    ''' This function runs 4 models at once: OLS, Lasso, Tweedie Regression and Ploynomial Regression '''
    #---OLS------
    # make and fit OLS model
    lm = LinearRegression()

    OLSmodel = lm.fit(X_train, y_train.quality)

    # make a prediction and save it to the y_train
    y_train['quality_pred_ols'] = lm.predict(X_train)

    #evaluate RMSE
    rmse_train_ols = mean_squared_error(y_train.quality, y_train.quality_pred_ols) ** .5

    # predict validate
    y_validate['quality_pred_ols'] = lm.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_ols = mean_squared_error(y_validate.quality, y_validate.quality_pred_ols) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'ols',
        'RMSE_train': rmse_train_ols,
        'RMSE_validate': rmse_validate_ols,
        'R2_validate': explained_variance_score(y_validate.quality, y_validate.quality_pred_ols)    
    }, ignore_index=True)

    print(f"""RMSE for OLS using LinearRegression
        Training/In-Sample:  {rmse_train_ols:.2f} 
        Validation/Out-of-Sample: {rmse_validate_ols:.2f}\n""")


    #------LassoLars----------
    # make and fit Lasso+Lars model
    lars = LassoLars(alpha=0.01)

    Larsmodel = lars.fit(X_train, y_train.quality)

    # make a prediction and save it to the y_train
    y_train['quality_pred_lars'] = lars.predict(X_train)

    #evaluate RMSE
    rmse_train_lars = mean_squared_error(y_train.quality, y_train.quality_pred_lars) ** .5

    # predict validate
    y_validate['quality_pred_lars'] = lars.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_lars = mean_squared_error(y_validate.quality, y_validate.quality_pred_lars) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'lasso_alpha0.01',
        'RMSE_train': rmse_train_lars,
        'RMSE_validate': rmse_validate_lars,
        'R2_validate': explained_variance_score(y_validate.quality, y_validate.quality_pred_lars)    
    }, ignore_index=True)

    print(f"""RMSE for LassoLars
        Training/In-Sample:  {rmse_train_lars:.2f} 
        Validation/Out-of-Sample: {rmse_validate_lars:.2f}\n""")

    #-----------Tweedie Model--------
    # make and fit Tweedie model
    tr = TweedieRegressor(power=0, alpha=1.0)

    Tweediemodel = tr.fit(X_train, y_train.quality)

    # make a prediction and save it to the y_train
    y_train['quality_pred_tweedie'] = tr.predict(X_train)

    #evaluate RMSE
    rmse_train_tweedie = mean_squared_error(y_train.quality, y_train.quality_pred_tweedie) ** .5

    # predict validate
    y_validate['quality_pred_tweedie'] = tr.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_tweedie = mean_squared_error(y_validate.quality, y_validate.quality_pred_tweedie) ** .5

    # append metric
    metric_df = metric_df.append({
        'model': 'tweedie_power0_alpha1.0',
        'RMSE_train': rmse_train_tweedie,
        'RMSE_validate': rmse_validate_tweedie,
        'R2_validate': explained_variance_score(y_validate.quality, y_validate.quality_pred_tweedie)    
    }, ignore_index=True)

    print(f"""RMSE for TweedieRegressor
        Training/In-Sample:  {rmse_train_tweedie:.2f} 
        Validation/Out-of-Sample: {rmse_validate_tweedie:.2f}\n""")
    
    #----------Polynomial--------------
    #1. Create the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2) #Quadratic aka x-squared
    
    #2. Fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    
    #3. Transform X_validate_scaled & X_test_scaled 
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)
    
    
    #2.1 MAKE THE THING: create the model object
    poly = LinearRegression()
    
    #2.2 FIT THE THING: fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    poly.fit(X_train_degree2, y_train.quality)
    
    #3. USE THE THING: predict train
    y_train['quality_pred_poly'] = poly.predict(X_train_degree2)
    
    #4. Evaluate: rmse
    poly_rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_poly) ** .5
    
    #5. REPEAT STEPS 3-4
    
    # predict validate
    y_validate['quality_pred_poly'] = poly.predict(X_validate_degree2)
    
    # evaluate: rmse
    poly_rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_poly) ** .5
    
    print(f"""RMSE for Polynomial Model, degrees=2
    Training/In-Sample:  {poly_rmse_train:.2f}
    Validation/Out-of-Sample:  {poly_rmse_validate:.2f}\n""")
    #Append
    metric_df = metric_df.append({
        "model":"poly_alpha.2",
        "RMSE_train": poly_rmse_train,
        "RMSE_validate": poly_rmse_validate,
        "R2_validate": explained_variance_score(y_validate.quality, y_validate.quality_pred_poly)
    }, ignore_index=True)
    return y_train, y_validate, metric_df

def lasso_test_model(X_train, y_train, X_test, y_test):

    ''' This function graphs test model '''
    # Convert y_test Series to a df
    y_test = pd.DataFrame(y_test)
    
    lars = LassoLars(alpha=0.01)
    
    LarsTest = lars.fit(X_test, y_test.quality)
    # USE THE THING: predict on test
    y_test['quality_pred_lars'] = lars.predict(X_test)
    
    # Evaluate: rmse
    rmse_test = mean_squared_error(y_test.quality, y_test.quality_pred_lars) ** (.5)
    
    print(f"""RMSE for LassoLars alpha=0.01
    Test Performance: {rmse_test:.2f}
    Baseline: {y_train.quality.mean():.2f}\n""")

def plt_regmods(y_validate):
    ''' This function plots Wine Quality regression models'''
    plt.figure(figsize=(16,8))
    #actual vs mean
    plt.plot(y_validate.quality, y_validate.quality_pred_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    
    #actual vs. actual
    plt.scatter(y_validate.quality, y_validate.quality, alpha=0.5, cmap="autumn", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)
    
#     #actual vs. LinearReg model
    plt.scatter(y_validate.quality, y_validate.quality_pred_ols, 
               alpha=.5, color="brown", s=100, label="Model: LinearRegression")
    # #actual vs. LassoLars model
    plt.scatter(y_validate.quality, y_validate.quality_pred_lars, 
                alpha=.5, color="orange", s=100, label="Model: Lasso Lars")
#     #actual vs. Tweedie/GenLinModel
#     plt.scatter(y_validate.quality, y_validate.quality_pred_tweedie, 
#                alpha=.5, color="red", s=100, label="Model: TweedieRegressor")
    # #actual vs. PolynomReg/Quadratic
    # plt.scatter(y_validate.quality, y_validate.quality_pred_poly, 
    #             alpha=.5, color="green", s=100, label="Model 2nd degree Polynomial")
    plt.legend()
    plt.xlabel("Actual Wine Quality")
    plt.ylabel("Predicted Wine Quality")
    plt.title("Where are predictions more extreme? More modest?")
    plt.show()
    
def hist_mods(y_validate):
    ''' This function graphs Wine Quality regression models'''
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8)) 
    plt.hist(y_validate.quality, color='red', alpha=.5, label="Actual Wine Quality ")
    plt.hist(y_validate.quality_pred_ols, color='brown', alpha=.5, label="Model: LinearRegression")
    plt.hist(y_validate.quality_pred_lars, color='orange', alpha=.5, label="Model: Lasso Lars")
#     plt.hist(y_validate.quality_pred_tweedie, color='yellow', alpha=.5, label="Model: TweedieRegressor")
    # plt.hist(y_validate.quality_pred_poly, color='green', alpha=.5, label="Model 2nd degree Polynomial") 
    plt.xlabel("Actual Wine Quality")
    plt.ylabel("Number of Observations")
    plt.title("Comparing the Distribution of Actual Wine Quality to Distributions of Predicted Wine Quality for the Top Models")
    plt.legend()
    plt.show()


# to call function use: wine_train, X_scaled, scaler, kmeans, centroids1, centroids2, centroids3 = create_cluster_models(wine_train, X_scaled, 3)
#-------------SCALING FUNCTIONS-------
#CodeUp  visualize scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,QuantileTransformer
def visualize_scaler(scaler, df, features_to_scale, bins=50):
    #create subplot structure
    fig, axs = plt.subplots(len(features_to_scale), 2, figsize=(24, 24,))
    #copy the df for scaling
    df_scaled = df.copy()
    
    #fit and transform the df
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    #plot the pre-scaled data next to the post-scaled data in one row of a subplot
    for (ax1, ax2), feature in zip(axs, features_to_scale):
        ax1.hist(df[feature], bins=bins, color='#E97451')
        ax1.set(title=f'{feature} before scaling', xlabel=feature, ylabel='count',)
    
        ax2.hist(df[feature], bins=bins, color='#E97451')
        ax2.set(title=f'{feature} after scaling with {scaler.__class__.__name__}', xlabel=feature, ylabel='count')
    plt.tight_layout()

#call function visualize_scaler(scaler=standard_scaler, df=wine_train, features_to_scale=to_scale, bins=50)

