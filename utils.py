import os
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

import plotly.io as pio
import plotly.express as px
from ipywidgets import interact
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numerize import numerize

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.regression.linear_model as sm
from statsmodels.tools.tools import add_constant
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.linear_model import Lasso, LassoCV, Ridge, BayesianRidge, ElasticNet, SGDRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# MAPE metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# XGB regression
from xgboost import XGBRegressor

# LGBM regression
# from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')

pio.renderers.default = "iframe"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)




# Removing redundant features based on their statistical significance
def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant.astype(float)).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features


def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true)))*100


def WAPE(y_true, y_pred):
    return (y_true - y_pred).abs().sum()*100 / y_true.abs().sum()


def feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp):
    global regression_features
    X[selected_features] = np.multiply(X[selected_features], weights)
    X['sum'] = X[selected_features].abs().sum(axis = 1)
    for feature in selected_features:
        X[feature] = (X[feature])/X['sum']
    
    sel_oh_cols = list(set(selected_features).difference(set(regression_features)))
    sel_reg_features = list(set(regression_features).intersection(set(selected_features)))
    X['Month'] = X[sel_oh_cols].sum(axis = 1)
    X = X.drop(columns= sel_oh_cols)
    X_array = np.array(X[sel_reg_features + ['Month']])
    feature_names = sel_reg_features + ['Month']
    rf_resultX = pd.DataFrame(X_array, columns = feature_names)

    vals = rf_resultX.values.mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                      columns=['Feature','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                   ascending=False, inplace=True)
    shap_importance = shap_importance[(shap_importance['feature_importance_vals']>0.001)|
                                  (shap_importance['feature_importance_vals']<-0.001)]
    shap_importance = shap_importance.sort_values(by = 'feature_importance_vals',  ascending= True)

    shap_importance["Color"] = np.where(shap_importance["feature_importance_vals"]<0, '#C9002B', '#005CB4')
    shap_importance[["Campaign_Type", "Brand", "Sub_Brand"]] = campaign, brnd, sbrnd 
    df_feat_imp = pd.concat([df_feat_imp, shap_importance])
    
    return df_feat_imp


def kpi_prediction_best_model(data, kpi, predict_future_months, test_months, combination, date_col):
    data = data[[date_col, kpi]+combination]
    data[date_col] = pd.to_datetime(data[date_col])
    data[kpi] = data[kpi].astype(float)

    df_pred = pd.DataFrame()
    for keys, df_grp in data.groupby(combination):
        brand, sbrand, campaign_type = keys
        print(keys)
        df_grp = df_grp.sort_values(by=date_col)
        mape_dict = {}
        wape_dict = {}

        ########################################################
        #exp smoothing
        train_data =  df_grp[~df_grp[date_col].isin(sorted(df_grp[date_col].unique())[-1*test_months:])]
        test_data  =  df_grp[df_grp[date_col].isin(sorted(df_grp[date_col].unique())[-1*test_months:])]

        train_data = train_data[[date_col, kpi]].set_index(date_col, drop=True)
        train_data[kpi] = train_data[kpi].astype('float64')                         
        train_data = train_data.to_period(freq="M")

        # modeling - exp smoothing
        # model_1 = ExponentialSmoothing(trend='mul', seasonal='multiplicative', sp=12)
        model_1 = ExponentialSmoothing()
        model_1.fit(train_data)
        future_pred = pd.DataFrame(model_1.predict(fh=np.arange(1, test_months+1)))
        future_pred.columns = [f'{kpi}_hat_expsm']
        future_pred = future_pred.to_timestamp().reset_index().rename(columns= {'index': date_col})

        test_data = test_data.merge(future_pred, on= [date_col], how= 'outer')
        test_mape_exp = MAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_expsm'])
        test_wape_exp = WAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_expsm'])

        #########################################################
        #Prophet
        df1 = df_grp[[date_col, kpi]].rename(columns={kpi:'y', date_col:'ds'})
        df1 = df1.sort_values(by='ds').reset_index(drop=True)
        train, test = df1.iloc[:-1*test_months], df1.iloc[-1*test_months:]

        # modeling
        model_2 = Prophet(daily_seasonality= False,
                        weekly_seasonality= False,
                        yearly_seasonality=False
                       ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                        .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)
        # model_2 = Prophet()
        model_2.fit(train)

        forecast_test = model_2.predict(test[['ds']]).merge(test[['ds', 'y']], on= ['ds'])

        forecast_test = forecast_test.reset_index().rename(columns = {'ds' : date_col, 'y' : f'{kpi}', 'yhat' : f'{kpi}_hat_prophet'})

        test_data = test_data.merge(forecast_test[[date_col, f'{kpi}_hat_prophet']], on = [date_col], how =  'outer')

        test_mape_pro = MAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_prophet'])
        test_wape_pro = WAPE(test_data[f'{kpi}'], test_data[f'{kpi}_hat_prophet'])

        ########################################################

        # mape_dict['expsm_mape'] = test_mape_exp
        # mape_dict['prophet_mape'] = test_wape_pro
        
        wape_dict['expsm_wape'] = test_wape_exp
        wape_dict['prophet_wape'] = test_wape_pro

        model_select = min(wape_dict, key = wape_dict.get)

        if model_select == 'expsm_wape':
            model = model_1
            future_pred = pd.DataFrame(model.predict(fh=np.arange(1, predict_future_months+1)))
            future_pred.columns = [f'{kpi}_hat']
            future_pred = future_pred.to_timestamp().reset_index().rename(columns= {'index': date_col})
            future_pred[['Campaign_Type', 'Brand','Sub_Brand', 'model']] = campaign_type, brand, sbrand, 'exponential_smoothing'
            df_grp = df_grp.merge(future_pred, on= [date_col,'Campaign_Type', 'Brand','Sub_Brand'], how= 'outer')

        elif model_select == 'prophet_wape':
            model = Prophet(daily_seasonality= False,
                        weekly_seasonality= False,
                        yearly_seasonality=False
                       ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                        .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)
            # model = Prophet()
            train_data_new = df_grp[[date_col, kpi]].rename(columns={kpi:'y', date_col:'ds'})
            train_data_new = train_data_new.sort_values(by='ds').reset_index(drop=True)
            model.fit(train_data_new)
            
            future_data = model.make_future_dataframe(periods = predict_future_months,
                                                      freq='m', 
                                                      include_history=False)
            future_data['ds'] = pd.DatetimeIndex(future_data['ds']) + pd.DateOffset(1)

            forecast = model.predict(future_data[['ds']])
            forecast = forecast[['ds', 'yhat']].rename(columns = {'ds': date_col, 'yhat':f'{kpi}_hat'})
            forecast[['Campaign_Type', 'Brand','Sub_Brand', 'model']] = campaign_type, brand, sbrand, 'prophet'
            df_grp = df_grp.merge(forecast, on = [date_col,'Campaign_Type', 'Brand','Sub_Brand' ], how = 'outer')
        
        df_pred = pd.concat([df_pred, df_grp])
        
    return df_pred


def sales_prediction_best_model(data, combination, future_months, test_months, df_feat_imp, regression_features):

    wape_dict = {}
    model_selected_feat = pd.DataFrame(columns= ['Campaign', "Brand", "Sub_Brand", 'model', 'features'])
    df_sales_pred = pd.DataFrame()
    
    for keys, df_slice in data.groupby(combination):
        brnd, sbrnd, campaign = keys
        df_slice = df_slice.sort_values(by='Date')
        
        oh_cols = list(pd.get_dummies(df_slice['Month']).columns.values)[1:]
        encoded_features = pd.get_dummies(df_slice['Month'])
        df_slice2 = pd.concat([df_slice, encoded_features],axis=1)
        df_slice3 = df_slice2.sort_values(by='Date').iloc[:-1*future_months]
        X, y = df_slice3[regression_features + oh_cols], df_slice3[['Sales']]
        X[regression_features] = np.where(X[regression_features] != 0, X[regression_features].apply(np.log10), 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_months, shuffle = False)

        # Feature selection for regression based models
        selected_features = backward_elimination(X_train, y_train)

        X_train, X_test = X_train[selected_features], X_test[selected_features]
        
        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        linear_test_wape = WAPE(y_test['Sales'], y_test_pred.reshape(-1).tolist())
        
        # Ridge Regression
        grid = dict()
        grid['alpha'] = np.arange(0, 1, 0.01)
        model = Ridge()
        search_ridge = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
        results_ridge = search_ridge.fit(X_train, y_train)
        model = Ridge(**results_ridge.best_params_)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        ridge_test_wape = WAPE(y_test['Sales'], y_test_pred.reshape(-1).tolist())

        # Lasso Regression Tuned
        model = Lasso()
        search_lasso = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
        results_lasso = search_lasso.fit(X_train, y_train)
        model = Lasso(**results_lasso.best_params_)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        lasso_test_wape = WAPE(y_test['Sales'], y_test_pred.reshape(-1).tolist())

        # Bayesian Regression
        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        bayesian_test_wape = WAPE(y_test['Sales'], y_test_pred.reshape(-1).tolist())
        
        # ElasticNet Regression Tuned
        grid = {}
        grid['l1_ratio'] = np.arange(0, 1, 0.01)
        grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        model = ElasticNet()
        search_elastic = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
        results_elastic = search_elastic.fit(X_train, y_train)
        model = ElasticNet(**results_elastic.best_params_)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        elastic_test_wape = WAPE(y_test['Sales'], y_test_pred.reshape(-1).tolist())

        # SGD Regressor
        model = SGDRegressor()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        sgd_test_wape = WAPE(y_test['Sales'], y_test_pred.reshape(-1).tolist())

        # Prophet
        df1 = df_slice[['Date', 'Sales']+regression_features].rename(columns={'Sales':'y', 'Date':'ds'})
        df1 = df1.sort_values(by='ds').reset_index(drop=True).iloc[:-1*future_months]
        train, test = df1.iloc[:-1*test_months], df1.iloc[-1*test_months:]

        X_prophet, y_prophet = train[regression_features], train[['y']]
        X_prophet[regression_features] = np.where(X_prophet[regression_features]>0, 
                                                  X_prophet[regression_features].apply(np.log10), 0)
        train[regression_features] = np.where(train[regression_features]>0, 
                                              train[regression_features].apply(np.log10), 0)
        test[regression_features] = np.where(test[regression_features]>0, 
                                             test[regression_features].apply(np.log10), 0)
        selected_features = backward_elimination(X_prophet, y_prophet)

        model = Prophet(daily_seasonality= False,
                    weekly_seasonality= False,
                    yearly_seasonality=False
                       ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                             .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)

        for x in selected_features:
            model.add_regressor(x)

        model.fit(train)
        forecast_train = model.predict(train[['ds']+selected_features]).merge(train[['ds', 'y']], on= ['ds'])
        forecast_test = model.predict(test[['ds']+selected_features]).merge(test[['ds', 'y']], on= ['ds'])

        prophet_test_wape =  WAPE(forecast_test['y'], forecast_test['yhat'])

        # Prophet with Covid as Holiday
        # holiday --> Feb 2020 to Mar 2021
        # holiday = pd.DataFrame(columns= ['ds', 'holiday'])
        # dates = pd.date_range(start='2020-02-01', end= '2021-03-01', freq='MS')
        # holiday['ds'] = dates
        # holiday['holiday'] = 'covid'

        # model = Prophet(daily_seasonality= False,
        #             weekly_seasonality= False,
        #             yearly_seasonality=False, holidays = holiday,
        #                ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
        #                      .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)

        # for x in selected_features:
        #     model.add_regressor(x)

        # model.fit(train)
        # forecast_train = model.predict(train[['ds']+selected_features]).merge(train[['ds', 'y']], on= ['ds'])
        # forecast_test = model.predict(test[['ds']+selected_features]).merge(test[['ds', 'y']], on= ['ds'])

        # prophet_holiday_test_wape =  WAPE(forecast_test['y'], forecast_test['yhat'])
        
        # Selecting model having minimum test wape
        wape_dict['wape_linear'] = linear_test_wape
        wape_dict['wape_ridge'] = ridge_test_wape
        wape_dict['wape_lasso'] = lasso_test_wape
        wape_dict['wape_bayesian'] = bayesian_test_wape
        wape_dict['wape_elastic'] = elastic_test_wape
        wape_dict['wape_sgd'] = sgd_test_wape
        wape_dict['wape_prophet'] = prophet_test_wape
        # wape_dict['wape_prophet_holiday'] = prophet_holiday_test_wape

        model_select = min(wape_dict, key = wape_dict.get)
        
        if model_select == 'wape_linear':
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)] =  [campaign, brnd, sbrnd, "Linear", selected_features]
            X = X[selected_features]

            model = LinearRegression()
            model.fit(X, y)
            
            filename = f'models/Linear_{campaign}_{brnd}_{sbrnd}.sav'
#             pickle.dump(model, open(filename, 'wb'))
            
            # Feature importance
            weights = model.coef_[0]
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0, 
                                                      df_slice2[regression_features].apply(np.log10), 0)
            df_slice['sales_pred'] = model.predict(df_slice2[selected_features])
            df_sales_pred = pd.concat([df_sales_pred, df_slice])
        
        elif model_select == 'wape_ridge':
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)] = [campaign, brnd, sbrnd, "Ridge", selected_features]
            X = X[selected_features]

            model = Ridge(**results_ridge.best_params_)
            model.fit(X, y)
            
            filename = f'models/Ridge_{campaign}_{brnd}_{sbrnd}.sav'
#             pickle.dump(model, open(filename, 'wb'))
            
            # Feature importance
            weights = model.coef_[0]
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0, 
                                                      df_slice2[regression_features].apply(np.log10), 0)
            df_slice['sales_pred'] = model.predict(df_slice2[selected_features])
            df_sales_pred = pd.concat([df_sales_pred, df_slice])

        elif model_select == 'wape_lasso':
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)] = [campaign, brnd, sbrnd, "Lasso", selected_features]
            X = X[selected_features]

            model = Lasso(**results_lasso.best_params_)
            model.fit(X, y)
            
            filename = f'models/Lasso_{campaign}_{brnd}_{sbrnd}.sav'
#             pickle.dump(model, open(filename, 'wb'))
            
            # Feature importance
            weights = model.coef_[0]
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0, 
                                                      df_slice2[regression_features].apply(np.log10), 0)
            df_slice['sales_pred'] = model.predict(df_slice2[selected_features])
            df_sales_pred = pd.concat([df_sales_pred, df_slice])


        elif model_select == 'wape_bayesian':
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)] = [campaign, brnd, sbrnd, "Bayesian", selected_features]
            X = X[selected_features]

            model = BayesianRidge()
            model.fit(X, y)
            filename = f'models/Bayesian_{campaign}_{brnd}_{sbrnd}.sav'
#             pickle.dump(model, open(filename, 'wb'))
            
            # Feature importance
            weights = model.coef_[0]
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0, 
                                                      df_slice2[regression_features].apply(np.log10), 0)
            df_slice['sales_pred'] = model.predict(df_slice2[selected_features])
            df_sales_pred = pd.concat([df_sales_pred, df_slice])


        elif model_select == 'wape_elastic':
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)]=[campaign, brnd, sbrnd, "ElasticNet", selected_features]
            X = X[selected_features]

            model = ElasticNet(**results_elastic.best_params_)
            model.fit(X, y)
            filename = f'models/ElasticNet_{campaign}_{brnd}_{sbrnd}.sav'
#             pickle.dump(model, open(filename, 'wb'))
            
            # Feature importance
            weights = model.coef_[0]
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0, 
                                                      df_slice2[regression_features].apply(np.log10), 0)
            df_slice['sales_pred'] = model.predict(df_slice2[selected_features])
            df_sales_pred = pd.concat([df_sales_pred, df_slice])
        
        elif model_select == 'wape_sgd':
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)] = [campaign, brnd, sbrnd, "SGD", selected_features]
            X = X[selected_features]

            model = BayesianRidge()
            model.fit(X, y)
            filename = f'models/SGD_{campaign}_{brnd}_{sbrnd}.sav'
#             pickle.dump(model, open(filename, 'wb'))
            
            # Feature importance
            weights = model.coef_[0]
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df_slice2[regression_features] = np.where(df_slice2[regression_features] != 0, 
                                                      df_slice2[regression_features].apply(np.log10), 0)
            df_slice['sales_pred'] = model.predict(df_slice2[selected_features])
            df_sales_pred = pd.concat([df_sales_pred, df_slice])

        elif model_select == 'wape_prophet':
            df1 = df_slice[['Date', 'Sales']+regression_features].rename(columns={'Sales':'y', 'Date':'ds'})
            train = pd.concat([train, test])
            
            X, y = train[regression_features], train[['y']]
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)] = [campaign, brnd, sbrnd, "Prophet", selected_features]
            
            model = Prophet(daily_seasonality= False,
                    weekly_seasonality= False,
                    yearly_seasonality=False, 
                       ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                             .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)
            for x in selected_features:
                model.add_regressor(x)

            model.fit(train)
            filename = f'models/Prophet_{campaign}_{brnd}_{sbrnd}.sav'
#             joblib.save(model, open(filename, 'wb'))
            
            # Feature importance
            regressor_coef = regressor_coefficients(model)
            weights = regressor_coef['coef'].values
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df1[regression_features] = np.where(df1[regression_features] != 0, 
                                                      df1[regression_features].apply(np.log10), 0)
            df_slice = df_slice.sort_values(by= 'Date')
            df_slice['sales_pred'] = model.predict(df1[selected_features + ['ds']])['yhat'].values
            df_sales_pred = pd.concat([df_sales_pred, df_slice])

        elif model_select == 'wape_prophet_holiday':
            df1 = df_slice[['Date', 'Sales']+regression_features].rename(columns={'Sales':'y', 'Date':'ds'})
            train = pd.concat([train, test])
            
            X, y = train[regression_features], train[['y']]
            selected_features = backward_elimination(X, y)
            model_selected_feat.loc[len(model_selected_feat)]=[campaign, brnd, sbrnd, "ProphetHoliday", selected_features]
            
            model = Prophet(daily_seasonality= False,
                    weekly_seasonality= False,
                    yearly_seasonality=False, holidays = holiday,
                       ).add_seasonality(name='yearly', period= 365, fourier_order= 5)\
                             .add_seasonality(name='quarterly', period= 365/4, fourier_order= 5)
            for x in selected_features:
                model.add_regressor(x)

            model.fit(train)
            filename = f'models/ProphetHoliday_{campaign}_{brnd}_{sbrnd}.sav'
#             pickle.dump(model, open(filename, 'wb'))
            
            # Feature importance
            regressor_coef = regressor_coefficients(model)
            weights = regressor_coef['coef'].values
            df_feat_imp = feature_importance(brnd, sbrnd, campaign, selected_features, X, weights, df_feat_imp)
            
            df1[regression_features] = np.where(df1[regression_features] != 0, 
                                                      df1[regression_features].apply(np.log10), 0)
            df_slice = df_slice.sort_values(by= 'Date')
            df_slice['sales_pred'] = model.predict(df1[selected_features + ['ds']])['yhat'].values
            df_sales_pred = pd.concat([df_sales_pred, df_slice])
        
        # if os.path.exists('models'):
        #     joblib.dump(model, filename)
        # else:
        #     os.makedirs('models')
        #     joblib.dump(model, filename)
        

    df_sales_pred['sales_pred'] = np.where(df_sales_pred['sales_pred']<0, 0, df_sales_pred['sales_pred'])
    
    return df_sales_pred, model_selected_feat, df_feat_imp
