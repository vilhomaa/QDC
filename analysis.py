import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd
from datetime import datetime,timedelta
import statsmodels.api as sm
from pathlib import Path
import time
import miceforest as mf
starttime = time.time()

# Import data
data = pd.read_csv('final_data.csv')

# Day of week dummies
data['time'] = pd.to_datetime(data['time'])

data = pd.concat((data, pd.get_dummies(data['time'].dt.day_name())), axis=1)

# date as ordinal, counts the days since jesus birth
data['time_ord']=data['time'].map(datetime.toordinal)
# normalize it to fit our data better - to count the days since first observation
data['time_ord']=data['time_ord'] - min(data['time_ord'])

# Country dummies
data = pd.concat((data, pd.get_dummies(data['country_region_code'])), axis=1)


######### VARIABLE DECLARATIONS AND FEATURE ENGINEERING ########

## Feature engineering

# In the main model, we wanted to use the incidence rate to measure
# both the individuals' own cautiousness related due to the amount of corona 
# present - > we will count the measure for the effect of the governments'
# measures as:
#   total_cases - neurohm_idx*total_cases 
# => (1- neurohm_idx)*total_cases

data['gov_measures_effect'] = (1 + data['neurohm_idx']) * data['cases_per_100k']
data['log_gov_measures_effect'] = np.log(data['gov_measures_effect']+0.0001)

# Lockdown effects -> France and Germany have had nationwide lockdowns

# France -> 3 lockdowns:
# 1.) early april -> 11.may (no data from that peroiod)
# 2.) 30 October 2020 that would last until at least 1 December 2020
# 3.) 3.4 - 3.5

# Germany: 
# 1.) 16.3: Merkel encourages to stay home. official recommendations
#   announced 22.3, some states lock down. Easing started 20.4
# https://www.frontiersin.org/articles/10.3389/fpubh.2020.568287/full
# 2.) 16.12 - 10.1 https://www.bbc.com/news/world-europe-55292614
# 3.) 25.4 - now - notbroms

data['lockdown_2_FR'] = (
    ((data['time']>=datetime(2020,10,30,0,0)) & (data['time']<=datetime(2020,12,1,0,0))) 
    & (data['country_region_code']=='FR')
).astype(int)
data['lockdown_3_FR'] = (
    ((data['time']>=datetime(2021,4,3,0,0)) & (data['time']<=datetime(2021,5,3,0,0))) 
    & (data['country_region_code']=='FR')
).astype(int)

data['lockdown_1_DE'] = (
    ((data['time']>=datetime(2020,3,16,0,0)) & (data['time']<=datetime(2020,4,20,0,0))) 
    & (data['country_region_code']=='DE')
).astype(int)
data['lockdown_2_DE'] = (
    ((data['time']>=datetime(2020,12,16,0,0)) & (data['time']<=datetime(2021,1,10,0,0))) 
    & (data['country_region_code']=='DE')
).astype(int)
data['lockdown_3_DE'] = (
    ((data['time']>=datetime(2021,4,25,0,0)) & (data['time']<=datetime.now())) 
    & (data['country_region_code']=='DE')
).astype(int)



# Dependent variables

mobility_locations = [ 
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline'
]


## Explanatory Variable declarations
explanatory_variables = [
    'time_ord',
    'log_gov_measures_effect',
    'tavg',
    'prcp',
    'wspd',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
    'SE',
    'FR',
]
lockdowns = [
    'lockdown_2_FR',
    'lockdown_3_FR',
    'lockdown_1_DE',
    'lockdown_2_DE',
    'lockdown_3_DE'
]



# Data imputations
max_obs = len(data)
for column, count in zip(data.columns,data.count()):
    missing_values = ((max_obs-count)/max_obs)*100
    print('Column: ' + column + ' has  '+ str(round(missing_values,1))+'% missing values')

data_amp = mf.ampute_data(data[explanatory_variables +['DE'] + lockdowns+mobility_locations],perc=0.25,random_state=2021)

kds = mf.KernelDataSet(
  data_amp,
  save_all_iterations=True,
  random_state=2021
)

# Run the MICE algorithm for 3 iterations
kds.mice(3)

# IMPORTANT DISTINCTION: 
# The imputed dataset, called 'data_completed' will be used for the regression and predictive
# analytics. 
# The non-inputed data vill be used for visualizations
data_completed = kds.complete_data()


######################## VISUALISATIONS ###########################

## Visualizing mobility
# Plots for all locations for all countries


def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i+window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average



## Group the dataframe based on countries

grouped_data_aggregations = {loc: np.mean for loc in mobility_locations}
grouped_data_aggregations['cases'] = np.sum
grouped_data_aggregations['population'] = np.sum
grouped_data_aggregations['cases_per_100k'] = np.mean
# add here if more variables needed for aggregation

grouped_data = data.groupby(['country_region_code','time'],as_index=False).agg(grouped_data_aggregations)
grouped_data['mobility_avg'] = grouped_data[mobility_locations].mean(axis=1)

## Graph for the total mobility of in countries with bins

def date_is_weekend(date):
    if date.weekday() > 4:
        return True
    else:
        return False



def total_mobility_graph(grouped_df,country):

    grouped_df = grouped_df[grouped_df['country_region_code'] == country]

    weekday_mobility_dates = []
    weekend_mobility_dates = []

    weekday_mobility = []
    weekend_mobility = []

    for date, mobility in zip(grouped_df['time'],grouped_df['mobility_avg']):
        if date_is_weekend(date):
            weekend_mobility_dates.append(date)
            weekend_mobility.append(mobility)
        else:
            weekday_mobility_dates.append(date)
            weekday_mobility.append(mobility)



    weekday_mobility_average = moving_average(weekday_mobility, 7)
    weekend_mobility_average = moving_average(weekend_mobility, 7)
    
    plt.figure(figsize=(12,7))
    plt.bar(weekday_mobility_dates, weekday_mobility, color='cornflowerblue')
    plt.plot(weekday_mobility_dates, weekday_mobility_average, color='green')
    
    plt.bar(weekend_mobility_dates, weekend_mobility, color='salmon')
    plt.plot(weekend_mobility_dates, weekend_mobility_average, color='black')
    
    plt.legend(['Moving average (7 days) weekday mobility', 'Moving Average (7 days) weekend mobility', 'Weekday mobility', 'Weekend mobility'], prop={'size': 15})
    plt.title('{} Total Mobility Data'.format(country), size=25)
    date_since = min(grouped_df['time']).date().strftime('%d.%m.%Y')
    plt.xlabel('Date', size=15)
    plt.ylabel('Mobility % change from average', size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.savefig('graphs_tables/{}_mobility_total.png'.format(country))



total_mobility_graph(grouped_data,'DE')
total_mobility_graph(grouped_data,'FR')
total_mobility_graph(grouped_data,'SE')


## graphs of daily mobility comparing countries (with MA)

countries = ['DE','FR','SE']

plt.figure(figsize=(12,7))
for country in countries:
    plt.plot(
        grouped_data[grouped_data['country_region_code'] == country]['time'],
        moving_average(grouped_data[grouped_data['country_region_code'] == country]['mobility_avg'],7)
        )
plt.axhline(y=0, color='r', linestyle='--')
plt.legend(countries, prop={'size': 10})
plt.xlabel('date', size=15)
plt.ylabel('Mobility as % from normal', size=15)
plt.title('Comparison of daily average mobility', size=20)
plt.xticks(size=10)
plt.yticks(size=10)
plt.savefig('graphs_tables/countries_comparison_avg_mobility.png')




## Graph of comparison between locations 

def location_comparison_graph(grouped_data, country,locations):
    plt.figure(figsize=(12,7))
    for location in locations:
        plt.plot(
            grouped_data[grouped_data['country_region_code'] == country]['time'],
            moving_average(grouped_data[grouped_data['country_region_code'] == country][location],7)
            )
    plt.axhline(y=0, color='r', linestyle='--')
    locations_title = [location.replace('_percent_change_from_baseline','').replace('_',' ').capitalize() for location in locations]
    plt.legend(locations_title, prop={'size': 10})
    plt.xlabel('Date', size=15)
    plt.ylabel('Mobility as % from normal', size=15)
    plt.title('Comparison of daily average mobility between locations in {}'.format(country), size=20)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.savefig('graphs_tables/{}_mobility_comparison_locations.png'.format(country))

location_comparison_graph(grouped_data,'DE',mobility_locations)
location_comparison_graph(grouped_data,'FR',mobility_locations)
location_comparison_graph(grouped_data,'SE',mobility_locations)



# Plotting relationships
plt.figure(figsize=(12,7))
plt.scatter(x = data['cases_per_100k'],y = data['retail_and_recreation_percent_change_from_baseline'],s = 0.5)
plt.ylabel('Mobility as % from normal', size=15)
plt.xlabel('Cases during last 7 days per 100k inhabitants', size=15)
plt.savefig('graphs_tables/mobility_x_cases.png')


graph_data = data[['tavg','parks_percent_change_from_baseline']].dropna()
x_vals = graph_data['tavg']
y_vals = graph_data['parks_percent_change_from_baseline']
plt.figure(figsize=(12,7))
plt.scatter(x = x_vals,y = y_vals,s = 0.5)
m, b = np.polyfit(x_vals, y_vals, 1)
plt.plot(x_vals, m*x_vals + b,label = 'OLS best fitted line',color = 'orange')
plt.ylabel('Mobility in parks as % from normal', size=15)
plt.xlabel('Average temperature of the day', size=15)
plt.legend(loc="best")
plt.savefig('graphs_tables/park_mobility_x_tavg.png')



## Descriptive statistics

data.describe()
with open('descriptive_statistics.txt', 'w') as fn:
            fn.write(data[
                mobility_locations 
                + explanatory_variables[:5]
                + ['log_gov_measures_effect']]
                .describe()
                .round(decimals = 2)
                .T
                .to_string())


## regression




def make_regression_summaries(data,locations,x_variables,savefilename = '_reg.txt'):
    for location in locations:
            
        lags_for_location = []
        for lag in range(1,8):
            first_part_of_location_name = location.split('_')[0]
            lag_name = '{y}_lag{l}'.format(y = first_part_of_location_name,l = lag)
            data[lag_name] = data[location].shift(-lag).fillna(0)
            lags_for_location.append(lag_name)


        data = data.dropna(subset = [location])

        Y = data[location]
        X = data[x_variables+lags_for_location]

        result = sm.OLS(endog=Y,exog=X,missing = 'drop').fit()

        with open('{}{}'.format(location,savefilename), 'w') as fh:
            fh.write(result.summary(xname=x_variables+lags_for_location,yname=location).as_text())

make_regression_summaries(data_completed,mobility_locations,explanatory_variables+lockdowns,'_reg_log_w_lockdowns.txt')



# =====================================================
# ============== PREDICTIVE ANALYTICS =================
# =====================================================


def ML_analytics_summarizer(data, pred_y_value,explanatory_variables,lockdowns,days_to_predict):

    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor

    print('Starting analysis for target variable: ' + pred_y_value )


    pred_x_variables = explanatory_variables + lockdowns 
    data_for_pred = data[pred_x_variables + [pred_y_value]].dropna()

    data_cutoff_day = max(data_for_pred['time_ord'])-days_to_predict

    X_train = data_for_pred[data_for_pred['time_ord'] <= data_cutoff_day][pred_x_variables]
    X_test = data_for_pred[data_for_pred['time_ord'] > data_cutoff_day][pred_x_variables]
    y_train = data_for_pred[data_for_pred['time_ord'] <= data_cutoff_day][pred_y_value]
    y_test = data_for_pred[data_for_pred['time_ord'] > data_cutoff_day][pred_y_value]
    ## Regression plot 
    print('Conduction regression')
    ols_model = sm.OLS(endog=y_train,exog=X_train,missing = 'drop').fit()
    y_pred = ols_model.predict(X_test)

    if days_to_predict % 7 == 0:
        Path("graphs_days_{}".format(days_to_predict)).mkdir(parents=True, exist_ok=True)
        x1 = X_test['time_ord']
        plt.figure(figsize=(12,7))
        plt.plot(x1, y_test, '.', label="Data")
        plt.plot(x1, y_pred, 'r.', label="Predicted")
        plt.xlabel('Days from 15.2.2020', size=15)
        #ax.plot(x1,y_pred, 'r', label="OLS prediction")
        plt.legend(loc="best")
        plt.savefig('graphs_days_{days}/reg_pred_{y}.png'.format(y = pred_y_value,days = days_to_predict))
    

    assert y_pred.count() == y_test.count(), "NA:s left in the data"                                          

    ## Regression evaluation RMSE:
    error_test_reg = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RSME for regression: ' + str(error_test_reg))

    ## decision tree

    print('Starting decision trees')

    def find_reg_tree_RMSE_for_depth(data,iterations,seed,X_train, X_test, y_train, y_test):

        

        iteration = []
        RSME = []

        for i in range(1,iterations):


            regr = DecisionTreeRegressor(max_depth=i)
            regr.fit(X_train,y_train)
            y = regr.predict(X_test)
            iteration.append(i)
            RSME.append(np.sqrt(mean_squared_error(y_test, y)))
        
        return {
                'iteration' : iteration,
                'RSME' : RSME
                }
        
            

    dec_tree_dephts = find_reg_tree_RMSE_for_depth(data_for_pred,25,2021,X_train, X_test, y_train, y_test)

    error_test_dec_tree = min(dec_tree_dephts['RSME'])
    dec_tree_optimal_depth = dec_tree_dephts['iteration'][dec_tree_dephts['RSME'].index(error_test_dec_tree)]
    print('Decision tree RSME:' + str(error_test_dec_tree))

    if days_to_predict % 7 == 0:
        plt.figure(figsize=(12,7))
        plt.plot(dec_tree_dephts['iteration'],dec_tree_dephts['RSME'])
        plt.legend(['RSME'], prop={'size': 15})
        plt.title('RSME for different dephts of the regression tree', size=25)
        plt.xlabel('Depth', size=15)
        plt.ylabel('RSME', size=20)
        plt.savefig('graphs_days_{days}/{a}_Reg_tree_dephts_RSME.png'.format(a = pred_y_value,days = days_to_predict))



    ## random forest
    print('Starting RF')
    # Random forest model (complete the following 2 lines)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state = 2021)

    # Train the random forest model
    rf.fit(X_train,y_train)

    # Compute the prediction over the training and testing sets
    y_pred_test = rf.predict(X_test)

    # Compute the MSE for the training and testing sets
    # (Complete the lines beginning with 'error_train =' and 'error_test ='.)

    error_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("RMSE on testing set:", error_test_rf)

    # Get the name and the importance measure of the variable
    # with the highest importance measure

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    name = data_for_pred.columns[indices[0]]
    importance = importances[indices[0]]

    print("Name of the variable with the highest importance measure:", name)
    print("Corresponding importance measure:", importance)


    ## gradient boosting

    print('Starting GBM')

    # Gradient boosting model
    from sklearn.ensemble import GradientBoostingRegressor
    gbm = GradientBoostingRegressor(loss = 'ls', criterion = "friedman_mse", learning_rate = 0.1,
                                    n_estimators = 1000, min_samples_leaf = 5, max_depth = 12,
                                    random_state = 2021, verbose = 0)

    # Train the gradient boosting model
    gbm.fit(X_train, y_train)

    # Compute the prediction over the training and testing sets
    y_pred_test = gbm.predict(X_test)

    # Compute the MSE for the training and testing sets
    error_test_gbm = np.sqrt(mean_squared_error(y_test, y_pred_test))


    print("RMSE on testing set:", error_test_gbm)


    # Graph of GBM Feature importances
    feature_importance = gbm.feature_importances_
    std = np.std([tree[0].feature_importances_ for tree in gbm.estimators_],
                axis=0)
    indices = np.argsort(feature_importance)

    names = data_for_pred.columns[indices]
    importances = feature_importance[indices]

    if days_to_predict % 7 == 0:
        pos = np.arange(indices.shape[0]) + 2
        plt.figure(figsize=(11,7))
        plt.barh(pos, importances, align='center')
        plt.yticks(pos, np.array(names))
        plt.title('GBM Feature Importance when predicting: {}'.format(pred_y_value))
        plt.tight_layout()
        plt.savefig('graphs_days_{days}/GBM_ft_importance_{y}'.format(y = pred_y_value,days = days_to_predict))


    dict_for_analysis_summary = {
        'Target_variable' : pred_y_value,
        'Target variable std' : data[pred_y_value].std(),
        'Regression RSME' : error_test_reg,
        'Decision tree RSME' : error_test_dec_tree,
        'Decision tree optimal depth' : dec_tree_optimal_depth,
        'Random Forests RSME' : error_test_rf,
        'Random forests highest importance measure': name,
        'GBM RSME' : error_test_gbm
    }

    return dict_for_analysis_summary


def ML_summarizer_loop_and_save(data,mobility_locations,explanatory_variables,lockdowns,days_to_predict):

    analysis_summary = []
    for ndays in days_to_predict:

        
        for location in mobility_locations:
            buffer_dict = ML_analytics_summarizer(data, location,explanatory_variables,lockdowns,ndays)
            buffer_dict['days_predicted'] = ndays
            analysis_summary.append(buffer_dict)


    analysis_summary_df = pd.DataFrame.from_dict(analysis_summary)
    analysis_summary_df['GBM error as % of target variable std'] = analysis_summary_df['GBM RSME'] / analysis_summary_df['Target variable std']
    return analysis_summary_df


day_amounts = [i+1 for i in range(30)]

analysis_summary_df = ML_summarizer_loop_and_save(data_completed,mobility_locations,explanatory_variables,lockdowns,day_amounts)

analysis_summary_df.to_csv('Analysis_summary_df_{mini}_to_{maxi}_days.csv'.format(mini = min(day_amounts), maxi = max(day_amounts)))




def make_RMSE_comparison_graphs(analysis_summary_df,mobility_locations):

    RSME_columns = [
        'Regression RSME', 
        'Decision tree RSME',
        'Random Forests RSME',
        'GBM RSME',    
    ]
    for location in mobility_locations:
        print('Creating a RSME Comparison table for {}'.format(location))
        single_location_results = analysis_summary_df[analysis_summary_df['Target_variable'] == location]

        # create figure and axis objects with subplots()
        fig,ax = plt.subplots(figsize = (11,6))
        # make a plot
        for ML_tool in RSME_columns:
            ax.plot(
                single_location_results['days_predicted'],
                single_location_results[ML_tool]
            )# set x-axis label
        ax.set_xlabel("Days Forecasted",fontsize=12)
        # set y-axis label
        ax.set_ylabel("RSME",fontsize=12)
        # twin object for two different y-axis on the sample plot
        ax2=ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(single_location_results['days_predicted'],single_location_results['GBM error as % of target variable std'],color = 'black')
        ax2.set_ylabel('GBM error as % of target variable std',fontsize=12)
        fig.legend(RSME_columns + ['GBM RSME as % of target variable std'], prop={'size': 10},loc = 'upper center')
        plt.savefig('graphs_tables/RSME_comparison_{}'.format(location))


make_RMSE_comparison_graphs(analysis_summary_df,mobility_locations)

print("--- %s seconds ---" % (time.time() - starttime))


