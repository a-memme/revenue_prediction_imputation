# Predicting Revenue - Timeseries Imputation + Random Forest Regression
*Utilizing imputation (via ARIMA timeseries modelling) in combination with random forest regression to predict daily revenue*

## Purpose 
In social media advertising, revenue gained via ad servicing can often lag in its calibration, and thus, leave businesses reliant on said revenue streams potentially numerous days behind in tracking/understanding their own performance. In the extremely agile and fickle environment of online media, a couple of days could become detrimental when considering strategy and overall flexibility needed to stay relevant in the industry, working against a business' ability to gain or even maintain a competitive advantage in the market. 

In the specific case of this analysis, revenue data (and all associated revenue metrics such as eCPM and ad impressions) are lagged by 3 days. The current analysis looks to build a flexible and accurate predictive model leveraging imputed revenue metrics (tvv via timeseries forecasting) and other strongly correlated metrics (such as demographic data, viewership data, etc) to predict revenue accurately for the lagged days, thus providing consistent, up-to-date reporting.

## Method 
- Utilize timeseries forecasting (Auto-ARIMA) to impute values of a revenue metric that is strongly correlated to revenue (i.e TVV - total view value)
    - manually train and test via BigQueryML
- Split data that will be used to build the random forest model into train and test sets using random sampling
- Leverage hyperparameter tuning to evaluate some of the best performing random forest regression models to predict revenue
    - utilizing the imputed values from step 1 + a number of other relevant performance metrics (demographics, viewership metrics, etc.)
    - BigQueryML has built in functionality for hyperparameter tuning
-  Test competing models on the test set
-  Create a view utilizing the organic data + model results to integrate into reporting and provide up-to-date revenue.

## Results & Analysis 
*See ... for full SQL code* 

### ARIMA Imputation - TVV 
- In this analysis, a created KPI is TVV (total view value) that is simply revenue / views *10k.
    - As an aggregated metric containing revenue in its calculation, this metric is also lagged by 3 days, however contains very important information on rates and ad spend otherwise not included in available real-time metrics.
    - Based on its profile (of being an aggregated metric) and its behaviour over time (analyzed in previous instances not included in this analysis), it sees generally steady trends and is a good contender for timeseries prediction.
- Using the code below, an initial auto-ARIMA model is trained and then test on unseen data.
    - Because the past 3 days contain no revenue data information, the initial model omits the past 6 days, where the predicted values of days 3-6 are measured against the actual values of days 3-6. See code below:

```
-- TEST ARIMA IMPUTATION (6 DAYS LAG for testing purposes)

CREATE OR REPLACE MODEL `project.revenue_estimation.arima_impute_tvv`
OPTIONS(MODEL_TYPE='ARIMA_PLUS',
       time_series_timestamp_col='date',
       time_series_data_col='tvv',
       TIME_SERIES_ID_COL = 'publisher_name',
       AUTO_ARIMA = TRUE,
       DATA_FREQUENCY = 'DAILY',
       CLEAN_SPIKES_AND_DIPS = TRUE,
       ADJUST_STEP_CHANGES = TRUE,
       TREND_SMOOTHING_WINDOW_SIZE =3,
       MAX_TIME_SERIES_LENGTH = 90,
       SEASONALITIES = ['NO_SEASONALITY'])
       AS
SELECT *
FROM EXTERNAL_QUERY(
                                                 "[sample_external_connection]",
                                                 """
                                                 SELECT date,
                                                        publisher_name,
                                                        CASE WHEN topsnap_views <= 0
                                                                      OR revenue_cad <= 0
                                                                      THEN NULL
                                                               ELSE revenue_cad*0.5/topsnap_views*10000
                                                          END AS tvv
                                                 FROM snap_studio_metrics_daily
                                                 WHERE date >= current_date - 126
                                                 AND date < current_date - 6
                                                 ORDER BY publisher_name, date ASC;
                                                 """
);

WITH cte AS
(
SELECT date(arima.forecast_timestamp) AS date,
       arima.publisher_name,
       arima.forecast_value AS forecast_tvv,
       actual.tvv
FROM ML.FORECAST(MODEL`project.revenue_estimation.arima_impute_tvv`,
                     STRUCT(3 AS horizon, 0.90 as confidence_level)) arima
LEFT JOIN EXTERNAL_QUERY(
                                                 "[sample_external_connection]",
                                                 """
                                                 SELECT date,
                                                        publisher_name,
                                                        topsnap_views,
                                                        sold_impressions,
                                                        revenue_cad*0.5/topsnap_views*10000 AS tvv
                                                 FROM snap_studio_metrics_daily
                                                 WHERE TRUE
                                                 AND date < current_date - 3
                                                 AND date >= current_date - 30
                                                 ORDER BY publisher_name, date ASC;
                                                 """
                     ) actual
ON (date(arima.forecast_timestamp) = actual.date) AND arima.publisher_name = actual.publisher_name
ORDER BY publisher_name, date ASC
)
SELECT publisher_name,
       AVG(ABS(cte.forecast_tvv - tvv)) AS mae,
       AVG(POW(cte.forecast_tvv - tvv, 2)) as mse,
       SQRT(AVG(POW(forecast_tvv - tvv, 2))) rmse
FROM cte
GROUP BY publisher_name
ORDER BY mae ASC;
```
- Through trial and error, the above model performed best, eliciting the following peformance metrics:
    - Avg MAE of 0.97 and median of 0.75
    - Avg RMSE of 1.11 and median of 0.90

![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/ba3e1ae2-d8ec-4a5a-b3a6-7384b8e9a8eb)

### Random Forest Regression - Train, Evaluate, Test

#### Train & Evaluate
- Utilizing the training data (See lines 103-141 in ...), hyperparameter tuning is applied and results are evaluated for best performing models based on MAE and MSE:

```
--RANDOM FOREST REGRESSOR
-- HYPERPARAMATER TUNING EVALUATION

CREATE OR REPLACE MODEL `project.revenue_estimation.rf_tuning`
OPTIONS(model_type='RANDOM_FOREST_REGRESSOR',
       num_trials=50,
       max_parallel_trials=5,
       NUM_PARALLEL_TREE = hparam_candidates([50, 100, 500, 1000]),
       l1_reg=hparam_candidates([0, 0.1, 0.5, 1, 2]),
       l2_reg=hparam_candidates([0, 0.1, 0.5, 1, 2]),
       max_tree_depth=hparam_candidates([5, 10, 15, 20]),
       subsample=hparam_candidates([0.7, 0.8, 0.9]),
       MIN_SPLIT_LOSS = hparam_candidates([0, 1, 2, 5]),
       TREE_METHOD = 'HIST',
       DATA_SPLIT_METHOD = 'RANDOM',
       HPARAM_TUNING_OBJECTIVES = (['mean_absolute_error']),
       INPUT_LABEL_COLS = ['post_snap_revenue_cad']) AS
SELECT date,
       publisher_name,
       topsnap_views,
       demo_age_25_to_34,
       post_snap_revenue_cad,
       total_time_viewed,
       mature_audience,
       tvv
FROM `project.revenue_estimation.revenue_training_data`;


-- EVALUATE MODEL -- first sorting for MAE, then considering MSE/R-squared measures
SELECT hype.hyperparameters.l1_reg,
       hype.hyperparameters.l2_reg,
       hype.hyperparameters.max_tree_depth,
       hype.hyperparameters.num_parallel_tree,
       hype.hyperparameters.subsample,
       hype.hyperparameters.min_split_loss,
       eval.*,
       SQRT(eval.mean_squared_error) rmse
FROM
  ML.EVALUATE(MODEL `project.revenue_estimation.rf_tuning`) AS eval
LEFT JOIN ML.TRIAL_INFO(MODEL `project.revenue_estimation.rf_tuning`) AS hype
ON eval.trial_id = hype.trial_id
ORDER BY mean_absolute_error ASC;
```

- Results are as follows where trial number 44 elicits the best evaluation results (MAE):

![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/70a23c04-5338-4586-a9f0-69c3f834c70a)
![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/bdd024cf-99cb-4135-bdfc-a54b3f01ad87)

#### Test 
- Given the information above, 2 top performing models based on MAE at evaluation (trial 44 shown above) and MSE at evaluation (trial 34 not shown above) are tested on the unseen data (test data) eliciting the following results:
    - Trial 44 (best performing***):
       ![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/3864c92e-e1a6-4503-a6a9-3c44c4ec41dd)
    - Trial 34
      ![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/3aa99b74-7184-45a8-b512-d70879b27067)

 

## Discussion 
