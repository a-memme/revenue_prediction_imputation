
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

-- IMPUTE TVV USING ARIMA (3 DAYS LAG)
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
       --SEASONALITIES = ['QUARTERLY'])
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
                                                 WHERE date >= current_date - 123
                                                 AND date < current_date - 3
                                                 ORDER BY publisher_name, date ASC;
                                                 """
);


-- TRAIN TEST SPLIT
CREATE TEMPORARY TABLE randomized_data AS
SELECT *,
       RAND() AS rand
FROM EXTERNAL_QUERY(
                                                 "[sample_external_connection]",
                                                 """
                                                 SELECT date,
                                                        publisher_name, 
                                                        topsnap_views,
                                                        demo_age_25_to_34,
                                                        revenue_cad*0.5 AS post_snap_revenue_cad, 
                                                        total_time_viewed,
                                                        ROUND((demo_age_25_to_34 + demo_age_35_plus) / (demo_age_18_to_17 + demo_age_18_to_24 + demo_age_25_to_34 + demo_age_35_plus + demo_age_unknown):: NUMERIC, 4) AS mature_audience, 
                                                        revenue_cad*0.5/topsnap_views*10000 AS tvv                     
                                                 FROM snap_studio_metrics_daily
                                                 WHERE date >= current_date - 63
                                                 AND date < current_date - 3
                                                 AND topsnap_views > 0
                                                 AND demo_age_25_to_34 > 0
                                                 AND revenue_cad > 0
                                                 ORDER BY date DESC;
                                                 """
)
WHERE TRUE 
AND tvv IS NOT NULL;

CREATE OR REPLACE TABLE `project.revenue_estimation.revenue_training_data` AS
SELECT *
FROM randomized_data
WHERE TRUE
AND rand <= 0.8;


CREATE OR REPLACE TABLE `project.revenue_estimation.revenue_testing_data` AS
SELECT *
FROM randomized_data
WHERE TRUE 
AND rand > 0.8;



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


--MODEL 1 
-- BUILD RF MODEL
CREATE OR REPLACE MODEL `distribution-engine.revenue_estimation.model_1`
OPTIONS(MODEL_TYPE='RANDOM_FOREST_REGRESSOR',
        NUM_PARALLEL_TREE = 50,
        TREE_METHOD = 'HIST',
        MAX_TREE_DEPTH = 10,
        L1_REG = 1.0,
        L2_REG = 0,
        EARLY_STOP = TRUE,
        SUBSAMPLE = 0.9,
        MIN_SPLIT_LOSS = 0,
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

--ASSESS FEATURE IMPORTANCE
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL `project.revenue_estimation.model_1`)
ORDER BY importance_weight DESC;

--TEST MODEL
SELECT *,
       SQRT(mean_squared_error) rmse
FROM ML.EVALUATE (MODEL `project.revenue_estimation.model_1`,
                    (SELECT date,
                            publisher_name,
                            topsnap_views,
                            demo_age_25_to_34,
                            post_snap_revenue_cad,
                            total_time_viewed,
                            mature_audience,
                            tvv
                     FROM `distribution-engine.project.revenue_testing_data`)
);

--MODEL 2 
-- BUILD RF MODEL
CREATE OR REPLACE MODEL `distribution-engine.revenue_estimation.model_2`
OPTIONS(MODEL_TYPE='RANDOM_FOREST_REGRESSOR',
        NUM_PARALLEL_TREE = 50,
        TREE_METHOD = 'HIST',
        MAX_TREE_DEPTH = 15,
        L1_REG = 0,
        L2_REG = 0,
        EARLY_STOP = TRUE,
        SUBSAMPLE = 0.9,
        MIN_SPLIT_LOSS = 0,
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

--ASSESS FEATURE IMPORTANCE
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL `project.revenue_estimation.model_2`)
ORDER BY importance_weight DESC;

--TEST MODEL
SELECT *,
       SQRT(mean_squared_error) rmse
FROM ML.EVALUATE (MODEL `project.revenue_estimation.model_2`,
                    (SELECT date,
                            publisher_name,
                            topsnap_views,
                            demo_age_25_to_34,
                            post_snap_revenue_cad,
                            total_time_viewed,
                            mature_audience,
                            tvv
                     FROM `distribution-engine.project.revenue_testing_data`)
);



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