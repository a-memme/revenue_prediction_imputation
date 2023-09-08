# Revenue Forecasting - Timeseries Imputation + Random Forest Regression
*Utilizing imputation (via ARIMA timeseries modelling) in combination with random forest regression to forecast daily revenue via BigQuery ML*

## Purpose 
In social media advertising, revenue gained via ad servicing can often lag in its calibration thus leaving businesses that are reliant on said revenue streams potentially numerous days behind in tracking/understanding their own performance. In the extremely agile and fickle environment of online media, a couple of days could become detrimental when considering strategy and overall flexibility needed to stay relevant in the industry, working against a business' ability to gain or even maintain a competitive advantage in the market. 

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
*See [revenue_prediction_imputation.sql](https://github.com/a-memme/revenue_prediction_imputation/blob/main/revenue_prediction_imputation.sql) for code details* 

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
- Utilizing the training data (See lines 103-141 in [revenue_prediction_imputation.sql](https://github.com/a-memme/revenue_prediction_imputation/blob/main/revenue_prediction_imputation.sql)), hyperparameter tuning is applied and results are evaluated for best performing models based on MAE and MSE:

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
- Given the information above, 2 top performing models based on MAE at evaluation (trial 44; shown above) and MSE at evaluation (trial 34; not shown above) are tested on the unseen data (test data) eliciting the following results:
    - Trial 44 (best performing***):
       ![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/3864c92e-e1a6-4503-a6a9-3c44c4ec41dd)
    - Trial 34
      ![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/3aa99b74-7184-45a8-b512-d70879b27067)
- Feature importance is assessed in the best performing model, identifying factors in the model that account for the most variance in the response:
![image](https://github.com/a-memme/revenue_prediction_imputation/assets/79600550/aba0cb4b-e802-4718-a970-4d8566448f44)

### Integrate into Reporting 
- Using the best performing regression model, organic performance data is combined with model results to create a view including:
    - an "estimated_revenue_cad" column representing original + predicted revenue (revenue is updated as actuals are realized)
    - a "predictived_tvv" column representing results of the ARIMA model for imputation
    - all other relevant metrics to revenue (date, actual revenue, channel names, etc)
- See code below:
```
--INTEGRATE INTO REPORTING  -- CREATE VIEW
CREATE OR REPLACE VIEW `Views.snap_revenue_est` AS
-- REVENUE REPORTING VIEW 
WITH cte AS (
              SELECT
              *
              FROM
              ML.PREDICT(MODEL `project.revenue_estimation.model_1`,
                            (SELECT       actual.general_show_split, 
                                          actual.net_proceeds_cad,
                                          arima.date,
                                          arima.publisher_name, 
                                          actual.topsnap_views,
                                          actual.demo_age_25_to_34,
                                          actual.post_snap_revenue_cad, 
                                          actual.total_time_viewed, 
                                          actual.mature_audience,
                                          arima.tvv, 
                            FROM (SELECT  date(forecast_timestamp) AS date,
                                          publisher_name, 
                                          forecast_value AS tvv
                                   FROM ML.FORECAST(MODEL`project.revenue_estimation.arima_impute_tvv`,
                                                        STRUCT(3 AS horizon, 0.90 as confidence_level))
                                   UNION ALL 
                                   SELECT date, 
                                          publisher_name, 
                                          tvv
                                   FROM  EXTERNAL_QUERY(
                                                                                    "[sample_external_connection]",
                                                                                    """
                                                                                    SELECT date,
                                                                                           publisher_name, 
                                                                                           revenue_cad*0.5/topsnap_views*10000 AS tvv
                                                                                    FROM snap_studio_metrics_daily
                                                                                    WHERE TRUE 
                                                                                    AND date < current_date - 3
                                                                                    AND topsnap_views > 0
                                                                                    ORDER BY date DESC;
                                                                                    """
                                                        )
                                   ORDER BY publisher_name, date DESC
                                   ) arima
                            LEFT JOIN EXTERNAL_QUERY(
                                                                                    "[sample_external_connection]",
                                                                                    """
                                                                                    SELECT s.date,
                                                                                           s.publisher_name, 
                                                                                           s.topsnap_views,
                                                                                           demo_age_25_to_34,
                                                                                           s.revenue_cad*0.5 AS post_snap_revenue_cad, 
                                                                                           total_time_viewed,
                                                                                           ROUND((demo_age_25_to_34 + demo_age_35_plus) / (demo_age_18_to_17 + demo_age_18_to_24 + demo_age_25_to_34 + demo_age_35_plus + demo_age_unknown)::NUMERIC, 4) AS mature_audience,    
                                                                                           profile.general_show_split, 
                                                                                           share.net_proceeds, 
                                                                                           share.net_proceeds*ex.cad AS net_proceeds_cad                
                                                                                    FROM snap_studio_metrics_daily s
                                                                                    LEFT JOIN snap_publisher_profile AS profile 
                                                                                    ON s.publisher_name = profile.publisher_name
                                                                                    LEFT JOIN snapchat_daily_revenue_share_new AS share
                                                                                    ON (s.date = share.date) AND (s.publisher_name = share.page_name)
                                                                                    LEFT JOIN exchange_rates AS ex
                                                                                    ON s.date = ex.date
                                                                                    WHERE TRUE 
                                                                                    AND s.topsnap_views > 0
                                                                                    AND demo_age_25_to_34 > 0
                                                                                    ORDER BY date DESC;
                                                                                    """
                                                        ) AS actual 
                            ON (arima.date = actual.date) AND (arima.publisher_name = actual.publisher_name)
                            ORDER BY date DESC
                            )
                        )
              ORDER BY date DESC
              ) 
SELECT publisher_name, 
       date, 
       post_snap_revenue_cad AS actual_ps_revenue_cad,
       predicted_post_snap_revenue_cad, 
       CASE WHEN date >= current_date - 3
              OR date >= current_date - 7 AND post_snap_revenue_cad <=0
              OR post_snap_revenue_cad <=0 AND publisher_name NOT IN ([list_of_publishers])
              THEN predicted_post_snap_revenue_cad
            ELSE post_snap_revenue_cad
         END AS estimated_revenue_cad,
       net_proceeds_cad AS actual_np_cad, 
       (predicted_post_snap_revenue_cad * general_show_split) AS predicted_np_cad, 
       CASE WHEN date >= current_date - 3
              OR date >= current_date - 7 AND post_snap_revenue_cad <=0
              OR post_snap_revenue_cad <=0 AND publisher_name NOT IN ([list_of_publishers])
              THEN (predicted_post_snap_revenue_cad * general_show_split)
            ELSE net_proceeds_cad
         END AS estimated_net_proceeds_cad,
       topsnap_views, 
       tvv AS predicted_tvv,
       post_snap_revenue_cad/topsnap_views *10000 AS actual_tvv
FROM cte
WHERE TRUE 
ORDER BY date DESC;
```

## Discussion 
### Why use Imputation AND Random Forest Regression?
- Given that TVV is an aggregated metric, we could simply just predict TVV and then refactor its equation and substitute the predicted values into the formula:
  - i.e Revenue = (predicted tvv value) / 10k * views
- Problem here is a strong reliance on the ARIMA model to be VERY accurate - small swings in errors from the tvv timeseries prediction are magnified once aggregated in the formula.
    - This option actually performs significantly WORSE than the imputation + rf option (we've tried it).
    - The random forest model adds another layer that considers other strongly correlated metrics other than tvv - see point directly below.
- Compounding error exposure
    - typically a risk in imputation is the exposure of compounding errors from the imputation model (ARIMA model) + the final model used to predict the response variable (rf model).
    - HOWEVER, in our case, through extensive evaluation and testing, we found that the rf model capturing relationships in other highly correlated metrics (such as date or demographics - see feature importance in the Test section above) can actually make up for errors in imputation and still elicit good results in evaluation and testing on unseen data with the help of standardization (i.e l1 or lasso regression).
- Amount of data to impute
    - As there are only 3 data points that need to be imputed, we follow the 5% rule (per dimension) by providing sufficient training and test data that also helps elicit the best test results (data recency is also important due to consistent algorithm changes on social media platforms - date representing the most important dimension via feature importance analysis reflects this as well).

### Why choose Random Forest over other model options?
- Overall best performing over other model options 
- In its essence, random forest models contain many benefits specific to this analysis such as:
    - naturally robust to overfitting due to its natural structure (combination of many weak predictors = one strong predictor)
    - ability to capture many complex relationships in variables - especially appropriate when considering the imputation discussion above.
 
## Conclusion 
The analysis above shows a compounding solution to providing up-to-date revenue analytics, prepared to be loaded into a BI tool for visualization. Although the problem may seem fairly simple at first, the relationships of available data, resources available, and overall domain knowledge ultimately determine what approach to take. Here, we see that a compounding solution utilizing missing data imputation and random forest regression elicits more than satisfactory results, solving a potentially adverse business analytics issue.
