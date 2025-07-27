import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from datetime import timedelta
from sklearn.pipeline import Pipeline

def one_hot_encode_columns(df, categorical_columns):
    df_encoded = df.copy()
    for col in categorical_columns:
        if col in df_encoded.columns:
            one_hot = pd.get_dummies(df_encoded[col], dtype=float)
            df_encoded = pd.concat([df_encoded, one_hot], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
    return df_encoded

def train_rf_with_gridsearch(X_train, y_train, X_val, y_val, param_grid, cv=2, verbose=2, n_jobs=-1):
    """
    Train a RandomForestClassifier with GridSearchCV and evaluate on validation set.
    Returns the best model, feature importances, and validation report.
    """
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        verbose=verbose,
        n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    val_predictions = best_rf.predict(X_val)
    val_report = classification_report(y_val, val_predictions, output_dict=True)
    feature_importance = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return best_rf, feature_importance, val_report, grid_search.best_params_


def calculate_features_with_window(
   anes_df, 
    news_sentiment, 
    time_window, 
    topics,
    candidate_scores=('trump_sentiment_score', 'biden_sentiment_score'),
    sources=None
):
    """
    Calculate features for each respondent based on news sentiment in a time window.
    topics: list of topic column names to use.
    candidate_scores: tuple of (trump_score_col, biden_score_col) column names.
    """
    features = []
    trump_score_col, biden_score_col = candidate_scores

    for _, respondent in anes_df.iterrows():
        respondent_date = respondent['date']
        read_sources = [col for col in anes_df.columns if col in news_sentiment['source'].unique() and respondent.get(col, 0) == 1]

        # Filter news by sources and time window
        filtered_news = news_sentiment[
            (news_sentiment['source'].isin(read_sources)) &
            (news_sentiment['date'] < respondent_date) &
            (news_sentiment['date'] >= respondent_date - timedelta(days=time_window))
        ]

        feature_dict = {}

        if not filtered_news.empty:
            filtered_news = filtered_news.dropna(subset=[trump_score_col, biden_score_col])

            if not filtered_news.empty:
                days_since_start = np.array((filtered_news['date'] - filtered_news['date'].min()).dt.days).reshape(-1, 1)

                # Calculate slopes for Trump and Biden sentiment scores
                reg_trump = LinearRegression().fit(days_since_start, filtered_news[trump_score_col].values)
                reg_biden = LinearRegression().fit(days_since_start, filtered_news[biden_score_col].values)

                feature_dict['trump_slope'] = reg_trump.coef_[0]
                feature_dict['biden_slope'] = reg_biden.coef_[0]
            else:
                feature_dict['trump_slope'] = 0
                feature_dict['biden_slope'] = 0

            feature_dict['trump_avg_sentiment'] = filtered_news[trump_score_col].mean()
            feature_dict['biden_avg_sentiment'] = filtered_news[biden_score_col].mean()
            feature_dict['news_count'] = len(filtered_news)

            # Topic-specific calculations
            for topic in topics:
                topic_news = filtered_news[filtered_news[topic] == 1]
                if not topic_news.empty:
                    days_since_start_topic = np.array((topic_news['date'] - topic_news['date'].min()).dt.days).reshape(-1, 1)

                    reg_trump_topic = LinearRegression().fit(days_since_start_topic, topic_news[trump_score_col].values)
                    reg_biden_topic = LinearRegression().fit(days_since_start_topic, topic_news[biden_score_col].values)

                    feature_dict[f'trump_slope_{topic}'] = reg_trump_topic.coef_[0]
                    feature_dict[f'biden_slope_{topic}'] = reg_biden_topic.coef_[0]

                    feature_dict[f'trump_avg_sentiment_{topic}'] = topic_news[trump_score_col].mean()
                    feature_dict[f'biden_avg_sentiment_{topic}'] = topic_news[biden_score_col].mean()
                else:
                    feature_dict[f'trump_slope_{topic}'] = 0
                    feature_dict[f'biden_slope_{topic}'] = 0
                    feature_dict[f'trump_avg_sentiment_{topic}'] = 0
                    feature_dict[f'biden_avg_sentiment_{topic}'] = 0
        else:
            feature_dict = {
                'trump_avg_sentiment': 0, 'biden_avg_sentiment': 0, 'news_count': 0,
                'trump_slope': 0, 'biden_slope': 0
            }
            for topic in topics:
                feature_dict.update({
                    f'trump_slope_{topic}': 0, f'biden_slope_{topic}': 0,
                    f'trump_avg_sentiment_{topic}': 0, f'biden_avg_sentiment_{topic}': 0
                })

        features.append(feature_dict)

    # Convert list of dictionaries to a DataFrame
    features_df = pd.DataFrame(features)

    # Drop the 'date' column and merge new features
    X_cleaned = anes_df.drop(columns=['date'], errors='ignore').reset_index(drop=True)
    return pd.concat([X_cleaned, features_df], axis=1)

def tune_time_window(
    X_train, y_train, X_val, y_val, news_df,
    time_windows,
    topics,
    candidate_scores,
    sources,
    param_grid,
    estimator_cls,
    estimator_kwargs=None,
    cv=2,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
):
    if estimator_kwargs is None:
        estimator_kwargs = {}
    results = []
    for window in time_windows:
        X_train_features = calculate_features_with_window(
            X_train, news_df, time_window=window,
            topics=topics, candidate_scores=candidate_scores, sources=sources
        )
        X_val_features = calculate_features_with_window(
            X_val, news_df, time_window=window,
            topics=topics, candidate_scores=candidate_scores, sources=sources
        )
        X_val_features = X_val_features[X_train_features.columns]
        estimator = estimator_cls(**estimator_kwargs)
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
        grid_search.fit(X_train_features, y_train)
        best_model = grid_search.best_estimator_
        val_score = best_model.score(X_val_features, y_val)
        results.append({
            'time_window': window,
            'best_params': grid_search.best_params_,
            'train_score': grid_search.best_score_,
            'val_score': val_score,
            'best_model': best_model,
            'X_train_features': X_train_features,
            'X_val_features': X_val_features
        })
    best_result = max(results, key=lambda x: x['val_score'])
    return best_result, results
