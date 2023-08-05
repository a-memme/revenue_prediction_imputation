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
#### ARIMA Imputation - TVV 
- In this analysis, a created KPI is TVV (total view value) that is simply revenue / views *10k.
    - As an aggregated metric containing revenue in its calculation, this metric is also lagged by 3 days, however contains very important information on rates and ad spend otherwise not included in available real-time metrics.
    - Based on its profile (of being an aggregated metric) and its behaviour over time (analyzed in previous instances not included in this analysis), it sees generally steady trends and is a good contender for timeseries prediction.
- Using the code below, an initial auto-ARIMA model is trained and then test on unseen data.
    - Because the past 3 days contain no revenue data information, the initial model omits the past 6 days, where the predicted values of days 3-6 are measured against the actual values of days 3-6. See below:

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

