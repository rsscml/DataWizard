"""
Enhanced Prompt templates for the DataAnalysisAgent with comprehensive business analytics, 
advanced visualizations, and predictive modeling capabilities
"""


def get_advanced_ml_guide():
    """Return comprehensive guide for advanced ML algorithms"""
    return """
# ADVANCED MACHINE LEARNING ALGORITHMS GUIDE

## 1. XGBOOST

### When to Use XGBoost
- **Best for**: Structured/tabular data, competitions, high accuracy requirements
- **Strength**: Handles missing values, feature importance, regularization
- **Use cases**: Sales forecasting, customer churn, risk assessment, demand planning

### XGBoost Implementation Patterns
```python
# Regression Example
def xgboost_regression(X, y, optimize_params=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if optimize_params:
        # Use automated hyperparameter optimization
        optimization_result = advanced_ml_optimize(X_train, y_train, 'xgboost', 'regression')
        model = optimization_result['best_model']
    else:
        # Default parameters
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'rmse': rmse,
        'r2_score': r2,
        'feature_importance': feature_importance,
        'predictions': y_pred
    }

# Classification Example
def xgboost_classification(X, y, optimize_params=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if optimize_params:
        optimization_result = advanced_ml_optimize(X_train, y_train, 'xgboost', 'classification')
        model = optimization_result['best_model']
    else:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'predictions': y_pred,
        'probabilities': y_proba
    }
```

### XGBoost Business Applications
```python
# Sales Forecasting with XGBoost
def sales_forecasting_xgboost(data, date_col, sales_col, external_features=[]):
    # Create time-based features
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)

    # Lag features
    data['sales_lag_1'] = data[sales_col].shift(1)
    data['sales_lag_7'] = data[sales_col].shift(7)
    data['sales_lag_30'] = data[sales_col].shift(30)

    # Rolling features
    data['sales_rolling_7'] = data[sales_col].rolling(7).mean()
    data['sales_rolling_30'] = data[sales_col].rolling(30).mean()

    # Date features
    data['month'] = data[date_col].dt.month
    data['quarter'] = data[date_col].dt.quarter
    data['day_of_week'] = data[date_col].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Prepare features
    feature_cols = ['sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_rolling_7', 
                    'sales_rolling_30', 'month', 'quarter', 'day_of_week', 'is_weekend'] + external_features

    clean_data = data.dropna()
    X = clean_data[feature_cols]
    y = clean_data[sales_col]

    return xgboost_regression(X, y, optimize_params=True)
```

## 2. LIGHTGBM (LIGHT GRADIENT BOOSTING MACHINE)

### When to Use LightGBM
- **Best for**: Large datasets, fast training, memory efficiency
- **Strength**: Faster than XGBoost, handles categorical features natively
- **Use cases**: Real-time predictions, large-scale analytics, categorical-heavy data

### LightGBM Implementation
```python
def lightgbm_analysis(X, y, problem_type='regression', categorical_features=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if problem_type == 'regression':
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train, categorical_feature=categorical_features)

        y_pred = model.predict(X_test)
        performance = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred)
        }
    else:  # classification
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train, categorical_feature=categorical_features)

        y_pred = model.predict(X_test)
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }

    return {
        'model': model,
        'performance': performance,
        'feature_importance': pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    }
```

## 3. CATBOOST (CATEGORICAL BOOSTING)

### When to Use CatBoost
- **Best for**: High categorical cardinality, automatic feature engineering
- **Strength**: Handles categories without encoding, overfitting resistant
- **Use cases**: Customer segmentation, recommendation systems, text-heavy data

### CatBoost Implementation
```python
def catboost_analysis(X, y, categorical_features=None, problem_type='regression'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if problem_type == 'regression':
        model = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
    else:
        model = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )

    # CatBoost can handle categorical features automatically
    model.fit(X_train, y_train, cat_features=categorical_features)

    y_pred = model.predict(X_test)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'predictions': y_pred,
        'feature_importance': feature_importance
    }
```

## 4. ADVANCED TIME SERIES FORECASTING

### Prophet for Business Forecasting
```python
def prophet_business_forecasting(data, date_col, value_col, periods=30, include_holidays=True):
    # Use the advanced forecasting function
    forecasts = advanced_ml_forecast(data, date_col, value_col, periods)

    if 'Prophet' in forecasts:
        prophet_result = forecasts['Prophet']
        forecast_df = prophet_result['forecast']
        model = prophet_result['model']

        # Business insights
        trend_direction = 'increasing' if forecast_df['yhat'].iloc[-1] > forecast_df['yhat'].iloc[0] else 'decreasing'

        # Seasonal patterns
        components = prophet_result.get('components', pd.DataFrame())

        return {
            'forecast': forecast_df,
            'trend_direction': trend_direction,
            'seasonal_patterns': components,
            'model_summary': f"Prophet model with {trend_direction} trend",
            'business_insights': [
                f"Forecast shows {trend_direction} trend over {periods} periods",
                f"Confidence interval range: {forecast_df['yhat_upper'].mean() - forecast_df['yhat_lower'].mean():.2f}"
            ]
        }
    else:
        return {'error': 'Prophet forecasting not available'}

# Multi-method Time Series Ensemble
def ensemble_time_series_forecast(data, date_col, value_col, periods=30):
    # Get forecasts from multiple methods
    all_forecasts = advanced_ml_forecast(data, date_col, value_col, periods)

    # Combine forecasts using simple averaging
    forecast_methods = [method for method in all_forecasts.keys() if 'error' not in all_forecasts[method]]

    if len(forecast_methods) > 1:
        ensemble_forecast = []
        forecast_dates = None

        for method in forecast_methods:
            method_forecast = all_forecasts[method]['forecast']
            if forecast_dates is None:
                forecast_dates = method_forecast['ds'] if 'ds' in method_forecast.columns else method_forecast.index

            forecast_values = method_forecast['yhat'] if 'yhat' in method_forecast.columns else method_forecast.iloc[:, 1]
            ensemble_forecast.append(forecast_values)

        # Average forecasts
        ensemble_values = np.mean(ensemble_forecast, axis=0)

        return {
            'ensemble_forecast': pd.DataFrame({
                'date': forecast_dates,
                'ensemble_prediction': ensemble_values
            }),
            'individual_forecasts': all_forecasts,
            'methods_used': forecast_methods,
            'ensemble_method': 'simple_average'
        }
    else:
        return all_forecasts
```

## 5. AUTOMATED HYPERPARAMETER OPTIMIZATION

### Using Optuna for Optimization
```python
def optimize_model_automatically(X, y, model_types=['xgboost', 'lightgbm'], problem_type='regression'):
    # Compare multiple optimized models
    results = {}

    for model_type in model_types:
        optimization_result = advanced_ml_optimize(X, y, model_type, problem_type)

        if 'error' not in optimization_result:
            results[model_type] = {
                'best_model': optimization_result['best_model'],
                'best_score': optimization_result['best_score'],
                'best_params': optimization_result['best_params'],
                'optimization_history': optimization_result['optimization_history']
            }

    # Find best performing model
    if results:
        best_model_type = max(results.keys(), key=lambda x: results[x]['best_score'])
        best_model = results[best_model_type]['best_model']

        return {
            'best_model': best_model,
            'best_model_type': best_model_type,
            'all_results': results,
            'performance_comparison': {model: results[model]['best_score'] for model in results}
        }
    else:
        return {'error': 'No models could be optimized'}
```

## 6. MODEL EXPLAINABILITY

### SHAP and LIME Explanations
```python
def explain_model_predictions(model, X_train, X_test, feature_names, explain_instances=5):
    # Use the advanced explainability function
    explanations = advanced_ml_explain(model, X_train, X_test, feature_names)

    if 'error' not in explanations:
        # Format for business consumption
        business_insights = []

        # SHAP insights
        if 'shap' in explanations:
            shap_importance = explanations['shap']['feature_importance']
            top_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            business_insights.append("Top 5 Most Important Features (SHAP Analysis):")
            for feature, importance in top_features:
                business_insights.append(f"  - {feature}: {importance:.4f}")

        # LIME insights
        if 'lime' in explanations:
            lime_explanations = explanations['lime']['explanations']
            business_insights.append(f"\\nLocal Explanations for {len(lime_explanations)} instances:")

            for i, lime_exp in enumerate(lime_explanations):
                business_insights.append(f"  Instance {i+1}: Top factors")
                for feature, weight in lime_exp['explanation'][:3]:
                    direction = "increases" if weight > 0 else "decreases" 
                    business_insights.append(f"    - {feature} {direction} prediction by {abs(weight):.3f}")

        return {
            'explanations': explanations,
            'business_insights': business_insights,
            'summary': "Model explanations provide insights into feature importance and prediction drivers"
        }
    else:
        return explanations
```

## 7. ANOMALY DETECTION

### Comprehensive Anomaly Detection
```python
def detect_business_anomalies(data, contamination=0.1, methods=['isolation_forest', 'lof']):
    # Use the advanced anomaly detection function
    anomaly_results = advanced_ml_anomaly(data, contamination)

    if 'error' not in anomaly_results:
        # Business interpretation
        business_summary = []

        if 'Consensus' in anomaly_results:
            consensus = anomaly_results['Consensus']
            business_summary.append(f"Detected {consensus['outliers_count']} consensus anomalies ({consensus['outlier_percentage']:.2f}% of data)")

        # Method comparison
        for method, results in anomaly_results.items():
            if method != 'Consensus' and method != 'data_with_outliers':
                business_summary.append(f"{method}: {results['outliers_count']} anomalies ({results['outlier_percentage']:.2f}%)")

        # Get anomalous records
        if 'data_with_outliers' in anomaly_results:
            data_with_flags = anomaly_results['data_with_outliers']
            consensus_anomalies = data_with_flags[data_with_flags.get('Consensus_outlier', 0) == 1]

            return {
                'summary': business_summary,
                'anomaly_results': anomaly_results,
                'consensus_anomalies': consensus_anomalies,
                'total_anomalies': len(consensus_anomalies) if not consensus_anomalies.empty else 0
            }
        else:
            return {
                'summary': business_summary,
                'anomaly_results': anomaly_results
            }
    else:
        return anomaly_results
```

CRITICAL ADVANCED ML RULES:

1. **Model Selection Guidelines**:
   - XGBoost: High accuracy, competition-grade results
   - LightGBM: Large datasets, speed requirements
   - CatBoost: High-cardinality categorical data
   - Prophet: Business time series with seasonality

2. **Hyperparameter Optimization**:
   - Always use optimization for production models
   - Set reasonable trial limits (50-200 trials)
   - Use cross-validation for robust evaluation

3. **Model Explainability**:
   - Always explain model predictions for business stakeholders
   - Use SHAP for global importance, LIME for local explanations
   - Translate technical insights into business language

4. **Time Series Best Practices**:
   - Use multiple forecasting methods when possible
   - Validate on out-of-sample data
   - Consider business cycles and seasonality
   - Apply ensemble methods for robustness

5. **Anomaly Detection Applications**:
   - Fraud detection: Financial transactions
   - Quality control: Manufacturing processes  
   - System monitoring: IT operations
   - Customer behavior: Marketing analytics

6. **Performance Monitoring**:
   - Track model performance over time
   - Detect data drift in production
   - Retrain models when performance degrades
   - Maintain model versioning and rollback capabilities
"""

def get_comprehensive_visualization_guide():
    """Return comprehensive guide for business analytics visualizations"""
    return """
# VISUALIZATION GUIDE
Data-driven chart selection regardless of domain

## CORE VALIDATION (MANDATORY FOR ALL CHARTS)
```python
# Required before any visualization
print(f"Shape: {data.shape}, Columns: {list(data.columns)}")
print(f"Missing: {data.isnull().sum().sum()}, Data types: {data.dtypes.to_dict()}")
data = data.dropna()
for col in data.select_dtypes(include=[np.number]).columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
```
## 1. LINE CHARTS

### Example Code
```python
def line_chart(data, x_col, y_col, group_col=None):
    if group_col:
        fig = px.line(data, x=x_col, y=y_col, color=group_col, markers=True)
    else:
        fig = px.line(data, x=x_col, y=y_col, markers=True)
    return fig.update_layout(title=f"{y_col} over {x_col}")
```
### Rules
- X-axis: Continuous or ordered categorical (time, sequence, rank)
- Y-axis: Continuous numerical data
- Data points: Minimum 3 points for meaningful trend
- Best for: Showing trends, changes over time/sequence
- Avoid when: X-axis categories are unordered or when showing parts of whole

## 2. BAR CHARTS (Vertical/Horizontal)

### Example Code
```python
def bar_chart(data, x_col, y_col, orientation='v', group_col=None):
    if orientation == 'h':
        fig = px.bar(data, x=y_col, y=x_col, color=group_col, orientation='h')
    else:
        fig = px.bar(data, x=x_col, y=y_col, color=group_col)
    return fig.update_layout(title=f"{y_col} by {x_col}")
```

### Rules
- X-axis: Categorical data (discrete categories)
- Y-axis: Numerical data (counts, amounts, percentages)
- Use horizontal: When category names are long or many categories
- Best for: Comparing quantities across categories
- Avoid when: Too many categories (>20) or showing trends over time

## 3. SCATTER PLOTS

### Example Code
```python
def scatter_plot(data, x_col, y_col, size_col=None, color_col=None):
    fig = px.scatter(data, x=x_col, y=y_col, size=size_col, color=color_col,
                    hover_data=data.columns.tolist())
    return fig.update_layout(title=f"{y_col} vs {x_col}")
```

### Rules
- Both axes: Continuous numerical data
- Best for: Showing correlation, relationships between variables
- Add size: For 3rd dimension (bubble chart effect)
- Add color: For categorical grouping or 4th dimension
- Avoid when: One variable is categorical or time-based sequence matters

## 4. BUBBLE CHARTS

### Example Code
```python
def bubble_chart(data, x_col, y_col, size_col, color_col=None):
    fig = px.scatter(data, x=x_col, y=y_col, size=size_col, color=color_col,
                    hover_name=data.index if color_col is None else color_col,
                    size_max=60)
    return fig.update_layout(title=f"{y_col} vs {x_col} (sized by {size_col})")
```

### Rules
- X & Y axes: Continuous numerical data
- Size: Positive numerical data (represents magnitude)
- Best for: 3-dimensional relationships, portfolio analysis
- Minimum: 10+ data points for meaningful patterns
- Avoid when: Size variable has negative values or limited range

## 5. HEATMAPS

### Example Code
```python
def heatmap(data, x_col, y_col, value_col=None):
    if value_col:
        pivot_data = data.pivot(index=y_col, columns=x_col, values=value_col)
    else:
        pivot_data = data.corr()  # Correlation heatmap
    return px.imshow(pivot_data, text_auto=True, aspect="auto",
                    title="Heatmap Analysis")
```

### Rules
- Data structure: Matrix format or pivot-able data
- Best for: Correlation matrices, pattern detection, density visualization
- Values: Continuous numerical data
- Grid size: Manageable dimensions (<50x50 for readability)
- Avoid when: Sparse data or when individual values need precise reading

## 6. PIE/DONUT CHARTS

### Example Code
```python
def pie_chart(data, category_col, value_col, donut=False):
    fig = px.pie(data, names=category_col, values=value_col,
                hole=0.4 if donut else 0)
    return fig.update_layout(title=f"Distribution of {value_col}")
```

### Rules
- Categories: Discrete categorical data (5-7 categories max)
- Values: Positive numerical data that sum to meaningful total
- Best for: Parts of a whole, percentage distributions
- Must sum to 100%: Or represent complete dataset
- Avoid when: Too many small slices, negative values, or comparing multiple datasets

## 7. BOX PLOTS

### Example Code
```python
def box_plot(data, category_col, value_col, group_col=None):
    fig = px.box(data, x=category_col, y=value_col, color=group_col)
    return fig.update_layout(title=f"Distribution of {value_col} by {category_col}")
```

### Rules
- X-axis: Categorical data (groups for comparison)
- Y-axis: Continuous numerical data
- Best for: Distribution comparison, outlier detection, statistical summaries
- Minimum: 5+ data points per category for meaningful statistics
- Avoid when: Data is not normally distributed or very few data points

## 8. HISTOGRAMS

### Example Code
```python
def histogram(data, value_col, bins=30, group_col=None):
    fig = px.histogram(data, x=value_col, nbins=bins, color=group_col,
                      marginal="box", opacity=0.7)
    return fig.update_layout(title=f"Distribution of {value_col}")
```

### Rules
- Data: Single continuous numerical variable
- Best for: Understanding data distribution, identifying patterns, outliers
- Bin selection: Use sqrt(n) or 10-50 bins depending on data size
- Minimum: 30+ data points for meaningful distribution
- Avoid when: Data is categorical or when comparing multiple groups (use box plots)

## 9. AREA CHARTS

### Example Code
```python
def area_chart(data, x_col, y_col, group_col=None, stacked=True):
    if stacked and group_col:
        fig = px.area(data, x=x_col, y=y_col, color=group_col)
    else:
        fig = px.area(data, x=x_col, y=y_col, color=group_col, line_group=group_col)
    return fig.update_layout(title=f"{y_col} over {x_col}")
```

### Rules
- X-axis: Continuous or time-based data
- Y-axis: Continuous numerical data (preferably positive)
- Best for: Cumulative values, part-to-whole over time, volume emphasis
- Stacked: When showing components of total
- Avoid when: Negative values, unordered categories, or precise value reading needed

## 10. VIOLIN PLOTS

### Example Code
```python
def violin_plot(data, category_col, value_col, group_col=None):
    fig = px.violin(data, x=category_col, y=value_col, color=group_col,
                   box=True, points="all")
    return fig.update_layout(title=f"Distribution Shape of {value_col} by {category_col}")
```

### Critical Rules
- X-axis: Categorical data
- Y-axis: Continuous numerical data
- Best for: Distribution shape comparison, density visualization
- Minimum: 20+ data points per category for smooth density curves
- Avoid when: Small datasets, when simple summary statistics are sufficient

## 11. WATERFALL CHARTS

### Example Code
```python
def waterfall_chart(categories, values):
    cumulative = [0] + [sum(values[:i+1]) for i in range(len(values)-1)]
    fig = go.Figure()
    for i, (cat, val) in enumerate(zip(categories[:-1], values[:-1])):
        fig.add_bar(x=[cat], y=[val], base=cumulative[i],
                   marker_color='green' if val >= 0 else 'red')
    fig.add_bar(x=[categories[-1]], y=[values[-1]], marker_color='blue')
    return fig.update_layout(title="Waterfall Analysis")
```

### Rules
- Data: Sequential components that build to a total
- Values: Can be positive or negative (gains/losses)
- Best for: Showing step-by-step changes, cumulative effects
- Order matters: Categories must be in logical sequence
- Avoid when: Components are independent or don't build to total

## 12. GAUGE CHARTS

### Example Code
```python
def gauge_chart(value, target, title, max_value=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value, title={'text': title},
        delta={'reference': target},
        gauge={'axis': {'range': [0, max_value or target*1.5]},
               'bar': {'color': 'green' if value >= target else 'red'},
               'threshold': {'line': {'color': "red", 'width': 4}, 'value': target}}))
    return fig
```

### Rules
- Single metric: One numerical value with target/benchmark
- Range: Clear minimum and maximum bounds
- Best for: KPI monitoring, performance against targets, progress tracking
- Target required: Meaningful reference point needed
- Avoid when: Multiple metrics, no clear target, or trend analysis needed

## 13. RADAR/SPIDER CHARTS

### Example Code
```python
def radar_chart(data, metrics, category_col):
    fig = go.Figure()
    for category in data[category_col].unique():
        cat_data = data[data[category_col] == category]
        values = [cat_data[m].mean() for m in metrics] + [cat_data[metrics[0]].mean()]
        fig.add_scatterpolar(r=values, theta=metrics + [metrics[0]],
                            fill='toself', name=category)
    return fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                            title="Multi-Dimensional Comparison")
```

### Rules
- Multiple metrics: 3-10 comparable numerical dimensions
- Same scale: All metrics should be normalized or comparable
- Best for: Multi-dimensional comparison, profile analysis
- Categories: 2-5 entities being compared
- Avoid when: Metrics have very different scales, too many dimensions

## 14. SANKEY DIAGRAMS

### Example Code
```python
def sankey_diagram(data, source_col, target_col, value_col):
    all_nodes = list(set(data[source_col].tolist() + data[target_col].tolist()))
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    fig = go.Figure(go.Sankey(
        node=dict(label=all_nodes),
        link=dict(source=[node_dict[src] for src in data[source_col]],
                 target=[node_dict[tgt] for tgt in data[target_col]],
                 value=data[value_col])))
    return fig.update_layout(title="Flow Analysis")
```

### Rules
- Flow data: Source-target relationships with quantities
- Positive values: Flow quantities must be positive
- Best for: Process flows, resource allocation, pathway analysis
- Clear nodes: Distinct source and target categories
- Avoid when: Circular flows, negative values, or simple comparisons

## 15. TREEMAPS

### Example Code
```python
def treemap(data, hierarchy_cols, value_col):
    fig = px.treemap(data, path=hierarchy_cols, values=value_col,
                    title="Hierarchical Analysis")
    return fig
```

### Rules
- Hierarchical data: Parent-child relationships, multiple levels
- Positive values: Sizes must be positive numbers
- Best for: Part-to-whole with hierarchy, nested comparisons
- Clear hierarchy: Logical parent-child relationships
- Avoid when: Flat data structure, negative values, time series

## CHART SELECTION DECISION TREE

```
1. TIME SERIES DATA? → Line Chart, Area Chart
2. CATEGORICAL COMPARISON? → Bar Chart, Box Plot
3. CORRELATION/RELATIONSHIP? → Scatter Plot, Bubble Chart, Heatmap
4. DISTRIBUTION ANALYSIS? → Histogram, Violin Plot, Box Plot
5. PART-TO-WHOLE? → Donut Chart, Treemap, Stacked Bar
6. MULTI-DIMENSIONAL? → Radar Chart, Heatmap, Bubble Chart
7. FLOW/PROCESS? → Sankey, Waterfall
8. SINGLE KPI? → Gauge Chart
9. HIERARCHICAL? → Treemap, Sunburst
10. DENSITY/PATTERN? → Heatmap, Contour Plot
```

## UNIVERSAL FORMATTING
```python
def format_chart(fig, title, x_label="", y_label=""):
    return fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=x_label, yaxis_title=y_label,
        font=dict(size=12), plot_bgcolor='white'
    )
```

# CRITICAL VISUALIZATION RULES:

1. DATA VALIDATION ALWAYS:
```python
# MANDATORY validation before any visualization
print(f"Data shape: {grouped_data.shape}")
print(f"Columns: {grouped_data.columns.tolist()}")
print(f"Data types: {grouped_data.dtypes}")
print(f"Value ranges: {grouped_data.select_dtypes(include=[np.number]).describe()}")
print(f"Missing values: {grouped_data.isnull().sum()}")

# Handle missing values
grouped_data = grouped_data.dropna()

# Validate data types
numeric_cols = grouped_data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    grouped_data[col] = pd.to_numeric(grouped_data[col], errors='coerce')
```
2. BUSINESS/DOMAIN CONTEXT:
- Always add meaningful titles, axis labels, and legends
- Use business-appropriate color schemes
- Include target lines, benchmarks, or thresholds where relevant
- Format numbers appropriately (currency, percentages, thousands separators)

3. INTERACTIVITY:
- Add hover information with business context
- Include drill-down capabilities where possible
- Use annotations for key insights or outliers

4. Plotly Line Instructions
- Never use add_vline() or add_hline()
- Use add_shape() with add_annotation() instead

INCORRECT - Use of add_vline() or add_hline():
```python
fig.add_vline(x=value, row=1, col=1) 
fig.add_hline(y=value, row=1, col=1)
```

CORRECT - Use add_shape() instead:
```python
# Vertical line in subplot
fig.add_shape(
    type="line", x0=value, x1=value, y0=0, y1=1,
    yref="y domain", line=dict(color="gray", width=2, dash="dot"),
    row=1, col=1
)

# Horizontal line in subplot  
fig.add_shape(
    type="line", x0=0, x1=1, y0=value, y1=value,
    xref="x domain", line=dict(color="gray", width=2, dash="dot"),
    row=1, col=1
)

# Add annotation separately
fig.add_annotation(
    x=value, y=1, yref="y domain", text="Text",
    showarrow=False, row=1, col=1
)
```
"""

def get_data_preparation_patterns():
    """Return advanced data preparation patterns for business analytics"""
    return """
# DATA PREPARATION GUIDE

## 1. DATA STRUCTURE & FORMAT

### Wide to Long Format Conversion
```python
# Convert wide format to long format for analysis
def wide_to_long(data, id_vars, value_vars=None, var_name='metric', value_name='value'):
    return pd.melt(data, id_vars=id_vars, value_vars=value_vars, 
                   var_name=var_name, value_name=value_name)

# Convert long to wide format
def long_to_wide(data, index_cols, columns_col, values_col):
    return data.pivot_table(index=index_cols, columns=columns_col, 
                           values=values_col, aggfunc='first').reset_index()
```

### Data Aggregation Strategies
```python
# Multi-level aggregation
def create_aggregations(data, group_cols, agg_dict):
    return data.groupby(group_cols).agg(agg_dict).round(2)

# Time-based aggregation
def time_aggregate(data, date_col, freq, agg_dict):
    return data.set_index(date_col).resample(freq).agg(agg_dict).reset_index()
```

## 2. CORRELATION & RELATIONSHIP ANALYSIS PREP

### Correlation-Ready Data Preparation
```
def prep_for_correlation(data, target_col=None, method='pearson', min_threshold=0.1):
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number]).copy()
    
    # Remove columns with all NaN or constant values
    numeric_data = numeric_data.dropna(axis=1, how='all')
    constant_cols = numeric_data.columns[numeric_data.std() == 0]
    if len(constant_cols) > 0:
        print(f"Removing constant columns: {list(constant_cols)}")
        numeric_data = numeric_data.drop(columns=constant_cols)
    
    # Handle missing values (drop rows with any NaN for correlation)
    clean_data = numeric_data.dropna()
    
    if len(clean_data) == 0:
        raise ValueError("No complete cases available for correlation analysis")
    
    # Calculate correlation matrix
    corr_matrix = clean_data.corr(method=method)
    
    # Extract meaningful correlations (avoid duplicates and self-correlations)
    def get_correlation_pairs(corr_matrix, min_threshold=min_threshold):
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)
        
        # Convert to long format safely
        corr_pairs = []
        for i in range(len(upper_triangle.index)):
            for j in range(len(upper_triangle.columns)):
                value = upper_triangle.iloc[i, j]
                if pd.notna(value) and abs(value) >= min_threshold:
                    corr_pairs.append({
                        'variable_1': upper_triangle.index[i],
                        'variable_2': upper_triangle.columns[j],
                        'correlation': value,
                        'abs_correlation': abs(value)
                    })
        
        return pd.DataFrame(corr_pairs).sort_values('abs_correlation', ascending=False)
    
    correlation_pairs = get_correlation_pairs(corr_matrix, min_threshold)
    
    # If target specified, get correlations with target
    target_correlations = None
    if target_col and target_col in corr_matrix.columns:
        target_correlations = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        target_correlations = target_correlations[target_correlations >= min_threshold]
    
    return {
        'clean_data': clean_data,
        'correlation_matrix': corr_matrix,
        'correlation_pairs': correlation_pairs,
        'target_correlations': target_correlations
    }

# Enhanced multicollinearity detection
def detect_multicollinearity(data, threshold=0.8, return_pairs=True):
    # Prepare data for correlation
    corr_result = prep_for_correlation(data, min_threshold=0.0)
    corr_matrix = corr_result['correlation_matrix']
    
    # Find high correlation pairs
    high_corr_pairs = corr_result['correlation_pairs']
    high_corr_pairs = high_corr_pairs[high_corr_pairs['abs_correlation'] >= threshold]
    
    if return_pairs:
        return high_corr_pairs
    else:
        # Return columns to potentially remove (keep first occurrence)
        cols_to_remove = set()
        for _, row in high_corr_pairs.iterrows():
            if row['variable_1'] not in cols_to_remove:
                cols_to_remove.add(row['variable_2'])
        return list(cols_to_remove)

# Safe correlation matrix visualization prep
def prep_correlation_heatmap(data, method='pearson', figsize=(10, 8)):
    corr_result = prep_for_correlation(data, method=method)
    corr_matrix = corr_result['correlation_matrix']
    
    # Create mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    return {
        'correlation_matrix': corr_matrix,
        'mask': mask,
        'figsize': figsize,
        'clean_data_shape': corr_result['clean_data'].shape
    }
```
## 3. FEATURE ENGINEERING PATTERNS

### Ratio & Rate Calculations
```python
def calculate_ratios(data, numerator_cols, denominator_cols, suffix='_ratio'):
    ratio_data = data.copy()
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col != den_col:
                ratio_data[f'{num_col}_{den_col}{suffix}'] = (
                    ratio_data[num_col] / ratio_data[den_col].replace(0, np.nan)
                )
    return ratio_data

# Growth rate calculations
def calculate_growth_rates(data, value_col, group_col=None, periods=[1, 12]):
    growth_data = data.copy()
    for period in periods:
        if group_col:
            growth_data[f'{value_col}_growth_{period}'] = (
                growth_data.groupby(group_col)[value_col].pct_change(periods=period)
            )
        else:
            growth_data[f'{value_col}_growth_{period}'] = (
                growth_data[value_col].pct_change(periods=period)
            )
    return growth_data
```

### Time-Based Features
```python
def create_time_features(data, date_col):
    time_data = data.copy()
    time_data[date_col] = pd.to_datetime(time_data[date_col])
    
    # Basic time components
    time_data['year'] = time_data[date_col].dt.year
    time_data['month'] = time_data[date_col].dt.month
    time_data['quarter'] = time_data[date_col].dt.quarter
    time_data['day_of_week'] = time_data[date_col].dt.dayofweek
    time_data['is_weekend'] = time_data['day_of_week'].isin([5, 6])
    time_data['is_month_end'] = time_data[date_col].dt.is_month_end
    time_data['is_quarter_end'] = time_data[date_col].dt.is_quarter_end
    
    return time_data

# Rolling window calculations
def add_rolling_features(data, value_col, windows=[7, 30, 90]):
    rolling_data = data.copy()
    for window in windows:
        rolling_data[f'{value_col}_rolling_mean_{window}'] = (
            rolling_data[value_col].rolling(window=window).mean()
        )
        rolling_data[f'{value_col}_rolling_std_{window}'] = (
            rolling_data[value_col].rolling(window=window).std()
        )
    return rolling_data
```

## 4. CATEGORICAL DATA PREPARATION

### Encoding Strategies
```python
def encode_categorical(data, method='auto', high_cardinality_threshold=50):
    encoded_data = data.copy()
    
    for col in data.select_dtypes(include=['object']).columns:
        unique_count = data[col].nunique()
        
        if method == 'auto':
            if unique_count <= 2:
                # Binary encoding
                encoded_data[f'{col}_encoded'] = pd.Categorical(data[col]).codes
            elif unique_count <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(data[col], prefix=col)
                encoded_data = pd.concat([encoded_data, dummies], axis=1)
            elif unique_count <= high_cardinality_threshold:
                # Label encoding
                encoded_data[f'{col}_encoded'] = pd.Categorical(data[col]).codes
            else:
                # Frequency encoding for high cardinality
                freq_map = data[col].value_counts().to_dict()
                encoded_data[f'{col}_frequency'] = data[col].map(freq_map)
    
    return encoded_data
```

## 5. SCALING & NORMALIZATION

### Feature Scaling
```python
def scale_features(data, numeric_cols, method='standard'):
    scaled_data = data.copy()
    
    if method == 'standard':
        # Z-score normalization
        scaled_data[numeric_cols] = (
            (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        )
    elif method == 'minmax':
        # Min-max scaling
        scaled_data[numeric_cols] = (
            (data[numeric_cols] - data[numeric_cols].min()) / 
            (data[numeric_cols].max() - data[numeric_cols].min())
        )
    elif method == 'robust':
        # Robust scaling (using median and IQR)
        median = data[numeric_cols].median()
        q75 = data[numeric_cols].quantile(0.75)
        q25 = data[numeric_cols].quantile(0.25)
        scaled_data[numeric_cols] = (data[numeric_cols] - median) / (q75 - q25)
    
    return scaled_data
```

## 6. OUTLIER & ANOMALY DETECTION

### Multi-Method Outlier Detection
```python
def detect_outliers(data, numeric_cols, methods=['iqr', 'zscore']):
    outlier_flags = pd.DataFrame(index=data.index)
    
    for col in numeric_cols:
        for method in methods:
            if method == 'iqr':
                Q1, Q3 = data[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outlier_flags[f'{col}_outlier_iqr'] = (
                    (data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)
                )
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_flags[f'{col}_outlier_zscore'] = z_scores > 3
    
    return outlier_flags
```

## 7. DATA QUALITY & VALIDATION

### Comprehensive Data Validation
```python
def validate_data_quality(data):
    # Data type validation and conversion
    for col in data.columns:
        if data[col].dtype == 'object':
            # Try numeric conversion
            numeric_test = pd.to_numeric(data[col], errors='coerce')
            if not numeric_test.isna().all():
                data[col] = numeric_test
            # Try datetime conversion
            elif any(keyword in col.lower() for keyword in ['date', 'time']):
                data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Missing value assessment
    missing_summary = data.isnull().sum()
    missing_pct = (missing_summary / len(data)) * 100
    
    # Duplicate detection
    duplicate_count = data.duplicated().sum()
    
    return {
        'missing_summary': missing_summary[missing_summary > 0],
        'missing_percentages': missing_pct[missing_pct > 0],
        'duplicate_count': duplicate_count,
        'data_types': data.dtypes
    }

# Strategic missing value handling
def handle_missing_values(data, strategy_map=None):
    if strategy_map is None:
        strategy_map = {
            'numeric': 'median',
            'categorical': 'mode',
            'datetime': 'forward_fill'
        }
    
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype in ['int64', 'float64']:
                if strategy_map['numeric'] == 'median':
                    data[col] = data[col].fillna(data[col].median())
                elif strategy_map['numeric'] == 'mean':
                    data[col] = data[col].fillna(data[col].mean())
            elif data[col].dtype == 'object':
                if strategy_map['categorical'] == 'mode':
                    data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown')
            elif 'datetime' in str(data[col].dtype):
                if strategy_map['datetime'] == 'forward_fill':
                    data[col] = data[col].fillna(method='ffill')
    
    return data
```

## 8. PERFORMANCE METRICS & BENCHMARKING

### Business Metrics Calculation
```python
def calculate_performance_metrics(data, actual_col, target_col=None, group_col=None):
    metrics = {}
    
    if group_col:
        grouped_data = data.groupby(group_col)
        metrics['group_totals'] = grouped_data[actual_col].sum()
        metrics['group_averages'] = grouped_data[actual_col].mean()
        metrics['group_growth'] = grouped_data[actual_col].pct_change()
    
    if target_col:
        metrics['variance'] = data[actual_col] - data[target_col]
        metrics['variance_pct'] = (metrics['variance'] / data[target_col]) * 100
        metrics['accuracy'] = 1 - (abs(metrics['variance']) / data[target_col])
    
    # Trend analysis
    data_sorted = data.sort_index()
    metrics['trend'] = data_sorted[actual_col].pct_change().mean()
    
    return metrics
```

## CRITICAL DATA PREPARATION RULES

1. Always validate data types first: Ensure numeric data is numeric, dates are datetime objects
2. Handle missing values strategically: Use business logic, not just statistical defaults
3. Check for duplicates: Remove or flag duplicate records before analysis
4. Detect outliers: Flag unusual values that might skew analysis
5. Normalize/scale when comparing: Use appropriate scaling for cross-metric analysis
6. Convert wide data to long: Most analytics work better with long format data
7. Create time-based features: Extract relevant time components for temporal analysis
8. Engineer ratio metrics: Create meaningful ratios and rates for business context
9. Encode categoricals appropriately: Choose encoding method based on cardinality
10. Prepare correlation matrices: Remove constants, handle missing values for relationship analysis
11. CORRELATION ANALYSIS SPECIFIC RULES:
	- NEVER use corr_matrix.unstack().reset_index() directly - it creates duplicates and includes self-correlations
	- Always remove constant columns before correlation analysis
	- Use upper triangle masking to avoid duplicate correlations
	- Set minimum correlation thresholds to filter noise
	- Handle missing values by using complete cases only
	- Check for multicollinearity before running regression analysis
	- Use the prep_for_correlation() function to get clean, actionable correlation results
"""

def get_pandas_safety_guide():
    """Return prompt to handle pandas robustly"""
    return """
# PANDAS CODE GENERATION RULES FOR 99% SUCCESS

## FUNDAMENTAL SAFETY RULES (ALWAYS APPLY)

### Data Type Safety
- **ALWAYS** cast to string before .str operations: `df['col'].astype(str).str.method()`
- **ALWAYS** check dtypes after read_csv and fix before operations
- **ALWAYS** use pd.to_numeric(errors='coerce') for numeric conversions
- **NEVER** compare floats directly - use np.isclose() or round first
- **NEVER** use == np.nan, use pd.isna() or df.isna()

### Memory & Assignment Safety  
- **ALWAYS** use .copy() when creating DataFrame subsets
- **ALWAYS** use .loc[mask, col] for assignments, never chained indexing
- **ALWAYS** assign results back: `df = df.operation()` not just `df.operation()`
- **NEVER** modify a DataFrame you're iterating over

## GROUPBY AGGREGATION RULES

### One-Line Rules
- **ALWAYS** use `as_index=False` or `.reset_index()` after groupby
- **ALWAYS** specify `dropna=False` to preserve groups with NaN keys
- **ALWAYS** name aggregations: `.agg(total=('value', 'sum'))` not `.agg({'value': 'sum'})`
- **ALWAYS** flatten MultiIndex columns after complex aggregations

### Critical Pattern (LLMs often miss this):
```python
# Safe groupby aggregation that preserves all columns
result = (df.groupby(['cat1', 'cat2'], as_index=False, dropna=False)
          .agg(total=('value', 'sum'), 
               avg=('value', 'mean'),
               count=('value', 'size'))
          .sort_values('total', ascending=False))
```

## MERGE/JOIN RULES

### One-Line Rules
- **ALWAYS** remove duplicates from merge keys first
- **ALWAYS** specify `how=` and `validate=` parameters explicitly  
- **ALWAYS** check merge key dtypes match exactly
- **ALWAYS** use `indicator=True` to track merge success
- **ALWAYS** verify row count after merge: `assert len(result) <= len(df1) * 1.1`

### Critical Pattern:
```python
# Safe merge with validation
df1_clean = df1.drop_duplicates(subset=['key'])
df2_clean = df2.drop_duplicates(subset=['key']) 
result = pd.merge(df1_clean, df2_clean, 
                  on='key', 
                  how='left', 
                  validate='1:1',
                  indicator=True)
print(f"Merge results: {result['_merge'].value_counts()}")
```

## RESHAPE OPERATIONS (PIVOT/MELT)

### One-Line Rules
- **ALWAYS** use pivot_table() not pivot() - handles duplicates gracefully
- **ALWAYS** check for duplicate (index, columns) pairs before pivot
- **ALWAYS** specify fill_value in pivot_table to avoid NaN
- **ALWAYS** reset_index() after pivot operations
- **ALWAYS** specify var_name and value_name in melt()

### Critical Pattern:
```python
# Safe pivot with duplicate handling
if df.duplicated(subset=['idx_col', 'col_col']).any():
    df = df.groupby(['idx_col', 'col_col'])['val_col'].first().reset_index()

result = pd.pivot_table(df, 
                       index='idx_col',
                       columns='col_col', 
                       values='val_col',
                       fill_value=0,
                       aggfunc='sum')
result.columns.name = None  # Clean column name
result = result.reset_index()
```

## DATETIME OPERATIONS

### One-Line Rules
- **ALWAYS** parse dates in read_csv: `parse_dates=['date_col']`
- **ALWAYS** specify format in to_datetime when known: `pd.to_datetime(df['date'], format='%Y-%m-%d')`
- **ALWAYS** standardize timezones before operations: `.dt.tz_localize(None)`
- **ALWAYS** sort by date before time-series operations
- **NEVER** compare string dates - convert to datetime first
- When working with pandas datetime operations, NEVER perform direct arithmetic between timestamps/DatetimeIndex and integers or arrays. This will cause errors in modern pandas versions.
- If you need to do mathematical operations (addition, subtraction, indexing) for forecasting or analysis, work with numeric indices or position-based indexing, then map back to dates separately. Keep datetime operations purely for date manipulation using pandas datetime methods.
- Never set datetime columns as DataFrame index if you plan to use shift(), rolling(), or similar positional operations

### Timestamp Arithmetic Error Prevention Guide

FORBIDDEN Operations:
# DON'T DO THESE:
timestamp + 5                    # Adding integer to timestamp
timestamp - 10                   # Subtracting integer from timestamp  
date_index + np.array([1,2,3])   # Adding array to DatetimeIndex
pd.date_range('2024-01-01', periods=10) + 7  # Direct arithmetic

CORRECT Approaches:
1. Use pd.Timedelta for date arithmetic:
timestamp + pd.Timedelta(days=5)
timestamp + pd.Timedelta(hours=3)
date_index + pd.Timedelta(weeks=1)
2. Use numeric indices for mathematical operations:
# For regression/forecasting, use numeric indices:
X = np.arange(len(time_series)).reshape(-1, 1)  # Not timestamps
future_indices = np.arange(len(data), len(data) + forecast_horizon)
3. Use proper pandas datetime methods:
# For resampling with gaps, use reindex instead:
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df_reindexed = df.reindex(full_range, fill_value=0)
# Use .dt accessor for datetime properties:
df['day_of_week'] = df['date'].dt.dayofweek  # Not arithmetic
4. For time-based calculations:
# Generate future dates properly:
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n, freq='D')
# OR
future_dates = pd.date_range(start='2024-06-01', end='2024-06-10', freq='D')

## MISSING VALUE HANDLING

### One-Line Rules
- **ALWAYS** check missing patterns before choosing fill strategy
- **ALWAYS** use `subset=` parameter in dropna to target specific columns
- **ALWAYS** fillna() before string operations to prevent NaN propagation
- **NEVER** use fillna(0) for non-numeric columns
- **PREFER** `dropna()` over `fillna()` when less than 5% missing

### Critical Pattern:
```python
# Intelligent missing value handling
missing_pct = df.isnull().sum() / len(df) * 100
for col in df.columns:
    if missing_pct[col] < 5:
        df[col] = df[col].dropna()
    elif df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna('Unknown')
```

## FILTERING & BOOLEAN OPERATIONS

### One-Line Rules
- **ALWAYS** use parentheses in compound conditions: `(df['A'] > 0) & (df['B'] < 10)`
- **ALWAYS** use &, |, ~ for boolean operations, never and/or/not
- **ALWAYS** use .isin() for multiple value checks instead of multiple ORs
- **ALWAYS** verify mask length matches DataFrame before applying
- **NEVER** chain boolean operations without parentheses

## OUTPUT VALIDATION

### One-Line Rules
- **ALWAYS** check shape after operations: `assert len(result) > 0`
- **ALWAYS** verify expected columns exist: `assert 'column' in df.columns`
- **ALWAYS** check for unintended NaN introduction
- **ALWAYS** reset_index(drop=True) before saving/exporting
- **ALWAYS** validate numeric columns are actually numeric after operations

## PERFORMANCE PATTERNS

### One-Line Rules
- **NEVER** use iterrows() - use vectorization or apply()
- **ALWAYS** use categorical dtype for low-cardinality strings (< 50% unique)
- **ALWAYS** specify chunksize for files > 100MB
- **ALWAYS** drop unnecessary columns before merge/groupby operations
- **PREFER** query() method for complex filtering on large DataFrames

## COLUMN OPERATIONS

### One-Line Rules
- **ALWAYS** use column names as they are present in the dataset
- **ALWAYS** use bracket notation for columns with spaces/special chars
- **NEVER** create column names that start with numbers
- **ALWAYS** flatten MultiIndex columns: `['_'.join(col).strip() for col in df.columns]`

## THE 10 GOLDEN RULES (MEMORIZE THESE)

1. **reset_index()** after EVERY groupby operation
2. **astype(str)** before EVERY .str operation
3. **copy()** when creating DataFrame subsets
4. **Specify how= and validate=** in EVERY merge
5. **Check shape and dtypes** after major operations
6. **Use .loc for assignments**, never chained indexing
7. **Parentheses** in ALL boolean conditions
8. **fillna() or dropna()** before string/numeric operations
9. **parse_dates** in read_csv, not after
10. **pd.isna()** not == np.nan for missing checks

## DEBUGGING CHECKLIST (WHEN THINGS GO WRONG)

1. Print `df.shape`, `df.dtypes`, `df.columns` 
2. Check `df.index` - is it what you expect?
3. Look for MultiIndex in columns or index
4. Check `df.isnull().sum()` for unexpected NaNs
5. Verify dtypes match between DataFrames before operations

"""

def get_numpy_scipy_statsmodels_guide():
    return """
## NUMPY CODE GENERATION RULES

### FUNDAMENTAL SAFETY RULES (ALWAYS APPLY)

#### Array Type Safety
- **ALWAYS** check array dimensions with `.shape` before operations
- **ALWAYS** use `np.asarray()` to ensure inputs are arrays, not lists
- **ALWAYS** specify dtype explicitly for precision: `np.array(data, dtype=np.float64)`
- **NEVER** compare floats directly - use `np.allclose()` or `np.isclose()`
- **NEVER** modify arrays you're iterating over - use `.copy()`

#### Broadcasting & Shape Safety
- **ALWAYS** verify shapes are compatible before operations: `assert a.shape[-1] == b.shape[0]`
- **ALWAYS** use `keepdims=True` in reductions to maintain broadcastable shapes
- **ALWAYS** reshape with `-1` for unknown dimensions: `arr.reshape(-1, 3)`
- **NEVER** use `np.resize()` - use `reshape()` or `np.pad()`
- **NEVER** rely on implicit broadcasting without checking shapes

### ARRAY CREATION & INITIALIZATION

#### One-Line Rules
- **ALWAYS** use `np.zeros()`, `np.ones()`, `np.full()` instead of manual loops
- **ALWAYS** specify dtype in array creation: `np.zeros(shape, dtype=np.float32)`
- **ALWAYS** use `np.linspace()` for evenly spaced floats, `np.arange()` for integers
- **ALWAYS** use `np.empty()` only when immediately filling all values
- **NEVER** use `np.empty()` expecting zeros - it contains garbage values

#### Critical Pattern:
```python
# Safe array creation with validation
shape = (100, 50)
arr = np.zeros(shape, dtype=np.float64)  # Explicit dtype
assert arr.shape == shape
assert arr.dtype == np.float64
```

### INDEXING & SLICING RULES

#### One-Line Rules
- **ALWAYS** use boolean masks with same shape: `arr[mask]` where `mask.shape == arr.shape`
- **ALWAYS** use `np.where()` for conditional indexing with fallback values
- **ALWAYS** check bounds before fancy indexing: `assert all(idx < arr.shape[0])`
- **ALWAYS** use `np.take()` with `mode='clip'` for safe out-of-bounds handling
- **NEVER** chain indexing - use tuple indexing: `arr[i, j]` not `arr[i][j]`

#### Critical Pattern:
```python
# Safe conditional operations
mask = arr > threshold
result = np.where(mask, arr * 2, arr / 2)  # Safe conditional operation
# Or for setting values
arr_copy = arr.copy()
arr_copy[mask] = new_value
```

### MATHEMATICAL OPERATIONS

#### One-Line Rules
- **ALWAYS** use `np.clip()` to prevent overflow: `np.clip(arr, -1e10, 1e10)`
- **ALWAYS** check for NaN/Inf after operations: `assert np.all(np.isfinite(result))`
- **ALWAYS** use `np.divide()` with `where` parameter to handle division by zero
- **ALWAYS** use `axis` parameter explicitly in reductions
- **NEVER** use `arr ** 0.5` for square root - use `np.sqrt()`

#### Critical Pattern:
```python
# Safe division with zero handling
denominator = arr2.copy()
denominator[denominator == 0] = 1e-10  # Avoid division by zero
result = np.divide(arr1, denominator, out=np.zeros_like(arr1), where=denominator!=0)
```

### AGGREGATION & STATISTICS

#### One-Line Rules
- **ALWAYS** use `np.nanmean()`, `np.nanstd()` for data with missing values
- **ALWAYS** specify `axis` explicitly: `arr.mean(axis=0)` not just `arr.mean()`
- **ALWAYS** use `ddof=1` for sample statistics: `np.std(arr, ddof=1)`
- **ALWAYS** check array size before statistics: `if arr.size > 0:`
- **NEVER** use statistics on empty arrays without checking

#### Critical Pattern:
```python
# Robust statistics calculation
if arr.size > 0:
    mean_val = np.nanmean(arr)
    std_val = np.nanstd(arr, ddof=1)
    median_val = np.nanmedian(arr)
else:
    mean_val = std_val = median_val = np.nan
```

### LINEAR ALGEBRA OPERATIONS

#### One-Line Rules
- **ALWAYS** verify matrix dimensions before multiplication: `assert A.shape[1] == B.shape[0]`
- **ALWAYS** use `@` operator or `np.dot()` for matrix multiplication, not `*`
- **ALWAYS** check condition number before inversion: `np.linalg.cond(matrix) < 1e10`
- **ALWAYS** use `np.linalg.lstsq()` instead of inverse for solving systems
- **NEVER** invert a matrix unless absolutely necessary

#### Critical Pattern:
```python
# Safe linear algebra operations
if np.linalg.cond(A) < 1e10:  # Check condition number
    # Solve Ax = b without inversion
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
else:
    raise ValueError("Matrix is ill-conditioned")
```

### RANDOM NUMBER GENERATION

#### One-Line Rules
- **ALWAYS** use `np.random.Generator` for new code: `rng = np.random.default_rng(seed)`
- **ALWAYS** set seed for reproducibility: `rng = np.random.default_rng(42)`
- **ALWAYS** specify size parameter explicitly in random functions
- **NEVER** use global `np.random.seed()` in production code
- **NEVER** modify the global random state

### THE 10 NUMPY GOLDEN RULES

1. **Check shapes** before EVERY operation with `.shape`
2. **Use `.copy()`** when modifying arrays you'll reuse
3. **Specify axis** explicitly in ALL reductions
4. **Use nan-functions** for data with missing values
5. **Verify dtypes** match before operations
6. **Use np.allclose()** for float comparisons
7. **Check for NaN/Inf** after mathematical operations
8. **Use broadcasting** but verify shapes first
9. **Prefer vectorization** over loops always
10. **Handle empty arrays** explicitly

---

## SCIPY CODE GENERATION RULES FOR 99% SUCCESS

### FUNDAMENTAL SAFETY RULES

#### Input Validation
- **ALWAYS** validate input array shapes and types before scipy functions
- **ALWAYS** check for NaN/Inf values: `np.all(np.isfinite(data))`
- **ALWAYS** ensure sufficient data points for statistical tests
- **NEVER** pass pandas objects directly - convert with `.values` or `.to_numpy()`
- **NEVER** ignore convergence warnings from optimization

### STATISTICAL TESTS

#### One-Line Rules
- **ALWAYS** check sample size requirements: t-test needs n≥3, shapiro needs 3≤n≤5000
- **ALWAYS** verify test assumptions (normality, equal variance) before parametric tests
- **ALWAYS** use `nan_policy='omit'` for real-world data with missing values
- **ALWAYS** report both test statistic and p-value
- **NEVER** use pearsonr on non-continuous data - use spearmanr

#### Critical Pattern:
```python
# Safe statistical testing with validation
data_clean = data[~np.isnan(data)]
if len(data_clean) >= 3:
    if len(data_clean) <= 5000:
        stat, p_value = stats.shapiro(data_clean)
    else:
        stat, p_value = stats.normaltest(data_clean)  # For large samples
else:
    raise ValueError("Insufficient data for normality test")
```

### OPTIMIZATION ROUTINES

#### One-Line Rules
- **ALWAYS** provide bounds for bounded optimization problems
- **ALWAYS** set `maxiter` to prevent infinite loops
- **ALWAYS** check optimization result: `result.success` before using solution
- **ALWAYS** try multiple initial guesses for non-convex problems
- **NEVER** ignore optimization warnings - they indicate problems

#### Critical Pattern:
```python
# Robust optimization with validation
from scipy.optimize import minimize
result = minimize(objective_func, x0=initial_guess, 
                 bounds=bounds, method='L-BFGS-B',
                 options={'maxiter': 1000})
if result.success:
    solution = result.x
else:
    raise RuntimeError(f"Optimization failed: {result.message}")
```

### INTERPOLATION

#### One-Line Rules
- **ALWAYS** check if interpolation points are within data range
- **ALWAYS** sort data by x before 1D interpolation
- **ALWAYS** use `bounds_error=False, fill_value='extrapolate'` carefully
- **ALWAYS** validate interpolated values are reasonable
- **NEVER** extrapolate far beyond data range

#### Critical Pattern:
```python
# Safe interpolation
x_sorted_idx = np.argsort(x)
x_sorted = x[x_sorted_idx]
y_sorted = y[x_sorted_idx]
# Remove duplicates
unique_mask = np.diff(x_sorted, prepend=-np.inf) > 0
f_interp = interp1d(x_sorted[unique_mask], y_sorted[unique_mask], 
                   kind='linear', bounds_error=False, 
                   fill_value=(y_sorted[0], y_sorted[-1]))
```

### THE 8 SCIPY GOLDEN RULES

1. **Validate inputs** before EVERY scipy function
2. **Check sample sizes** for statistical tests
3. **Verify convergence** in optimization results
4. **Sort data** before 1D interpolation
5. **Use nan_policy='omit'** for missing data
6. **Set maxiter** in iterative algorithms
7. **Check assumptions** before statistical tests
8. **Handle edge cases** explicitly

---

## STATSMODELS CODE GENERATION RULES

### FUNDAMENTAL SAFETY RULES

#### Data Preparation
- **ALWAYS** add constant for intercept: `X = sm.add_constant(X)`
- **ALWAYS** check for multicollinearity with VIF before regression
- **ALWAYS** ensure no missing values in endog/exog
- **ALWAYS** verify data types are numeric before modeling
- **NEVER** include the target variable in features

### REGRESSION MODELING

#### One-Line Rules
- **ALWAYS** check model convergence: `results.mle_retvals['converged']`
- **ALWAYS** examine residual plots for assumptions
- **ALWAYS** use robust standard errors when heteroscedasticity detected
- **ALWAYS** check condition number: `results.condition_number < 1000`
- **NEVER** interpret coefficients without checking p-values

#### Critical Pattern:
```python
# Safe regression with diagnostics
X_clean = X.dropna()
y_clean = y[X_clean.index]
X_with_const = sm.add_constant(X_clean)

# Check multicollinearity
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
              for i in range(X_with_const.shape[1])]
if all(vif["VIF"][1:] < 10):  # Exclude constant
    model = sm.OLS(y_clean, X_with_const)
    results = model.fit()
    if results.condition_number < 1000:
        # Safe to use results
        pass
```

### TIME SERIES ANALYSIS

#### One-Line Rules
- **ALWAYS** check stationarity before ARIMA: `adfuller(series).pvalue < 0.05`
- **ALWAYS** ensure datetime index with proper frequency
- **ALWAYS** have at least 50 observations for ARIMA
- **ALWAYS** check residual autocorrelation after fitting
- **NEVER** forecast beyond 30% of series length

#### Critical Pattern:
```python
# Safe time series modeling
ts_data = data.set_index('date').sort_index()
ts_data = ts_data.asfreq('M')  # Set frequency explicitly

# Check stationarity
adf_p = adfuller(ts_data['value'].dropna())[1]
if adf_p > 0.05:
    # Difference the series
    ts_diff = ts_data['value'].diff().dropna()
    d = 1
else:
    ts_diff = ts_data['value']
    d = 0

# Fit with validation
if len(ts_diff) >= 50:
    model = ARIMA(ts_diff, order=(1,d,1))
    results = model.fit()
    # Check residuals
    residuals_acf = acf(results.resid, nlags=20)
```

### THE 8 STATSMODELS GOLDEN RULES

1. **Add constant** for intercept in every model
2. **Check VIF** before regression modeling
3. **Verify convergence** in all models
4. **Test residuals** for all assumptions
5. **Use datetime index** for time series
6. **Check stationarity** before ARIMA
7. **Validate predictions** out-of-sample
8. **Report diagnostics** with results

### CRITICAL DEBUGGING CHECKLIST

**When NumPy operations fail:**
1. Print `arr.shape`, `arr.dtype`, `arr.size`
2. Check `np.isfinite(arr).all()`
3. Verify array contiguity: `arr.flags['C_CONTIGUOUS']`
4. Look for shape mismatches in operations

**When SciPy functions fail:**
1. Check input array properties
2. Verify sample size requirements
3. Look for convergence warnings
4. Check if data is sorted (for interpolation)

**When Statsmodels fails:**
1. Check for perfect multicollinearity
2. Verify no missing values
3. Check datetime index frequency
4. Examine condition number and VIF    
"""

def get_predictive_modeling_guide():
    return """
ADVANCED PREDICTIVE MODELING & MACHINE LEARNING GUIDE

1. REGRESSION ANALYSIS (Continuous Target Prediction):
```python
# Comprehensive Regression Analysis
def perform_regression_analysis(data, target_col, feature_cols):
    # Prepare data
    X = data[feature_cols].dropna()
    y = data[target_col].dropna()
    
    # Align X and y (remove rows with missing target)
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Multiple models comparison
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        if 'Forest' in name:
            model.fit(X_train, y_train)  # Tree models don't need scaling
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        results[name] = {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
    
    return pd.DataFrame(results).T

# Business Applications:
# - Sales forecasting: predict revenue based on marketing spend, seasonality
# - Price optimization: predict optimal pricing based on demand factors
# - Cost prediction: predict operational costs based on volume metrics
```

2. CLASSIFICATION ANALYSIS (Category Prediction):
```python
# Customer Churn Prediction Example
def perform_classification_analysis(data, target_col, feature_cols):
    # Prepare data
    X = data[feature_cols].dropna()
    y = data[target_col].dropna()
    
    # Align data
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Encode categorical target if needed
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Classification models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        if 'Forest' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
            'ROC-AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y)) == 2 else 'N/A'
        }
    
    return pd.DataFrame(results).T

# Business Applications:
# - Customer churn prediction
# - Lead scoring and qualification
# - Product recommendation (will customer buy?)
# - Risk assessment (loan default, fraud detection)
```

3. CLUSTERING ANALYSIS:
```python
def perform_clustering_analysis(data, feature_cols, n_clusters=4):
    # Prepare data
    cluster_data = data[feature_cols].dropna()
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(10, len(cluster_data)//10))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))
    
    # Use provided n_clusters or best silhouette score
    optimal_k = n_clusters
    if len(silhouette_scores) > 0:
        optimal_k = K_range[np.argmax(silhouette_scores)]
    
    # Final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to original data
    cluster_data['Cluster'] = cluster_labels
    
    # Cluster profiling
    cluster_profile = cluster_data.groupby('Cluster')[feature_cols].agg(['mean', 'std']).round(2)
    
    # Create cluster summary
    cluster_summary = {}
    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels == cluster_id
        cluster_summary[f'Cluster_{cluster_id}'] = {
            'size': cluster_mask.sum(),
            'percentage': (cluster_mask.sum() / len(cluster_data)) * 100
        }
        
        # Add feature means for this cluster
        for feature in feature_cols:
            cluster_summary[f'Cluster_{cluster_id}'][f'avg_{feature}'] = cluster_data[cluster_data['Cluster'] == cluster_id][feature].mean()
    
    return cluster_data, cluster_profile, pd.DataFrame(cluster_summary).T

# Business Applications:
# - Customer segmentation by behavior, value, demographics
# - Product portfolio analysis
# - Market segmentation
# - Supplier categorization
```

4. TIME SERIES FORECASTING:
```python
# Advanced Time Series Forecasting
def perform_time_series_forecast(data, date_col, value_col, forecast_periods=12):
    # Prepare time series
    ts_data = data[[date_col, value_col]].copy()
    ts_data[date_col] = pd.to_datetime(ts_data[date_col])
    ts_data = ts_data.sort_values(date_col).set_index(date_col)
    
    # Resample to monthly frequency
    monthly_data = ts_data.resample('M')[value_col].sum()
    
    # Check for sufficient data
    if len(monthly_data) < 24:
        return "Insufficient data for time series forecasting (need at least 24 months)"
    
    # Seasonal decomposition
    decomposition = seasonal_decompose(monthly_data, model='additive')
    
    # Simple forecasting methods
    forecasts = {}
    
    # 1. Moving Average
    ma_forecast = monthly_data.rolling(window=12).mean().iloc[-1]
    forecasts['Moving Average'] = [ma_forecast] * forecast_periods
    
    # 2. Linear Trend
    X = np.arange(len(monthly_data)).reshape(-1, 1)
    y = monthly_data.values
    trend_model = LinearRegression().fit(X, y)
    
    future_X = np.arange(len(monthly_data), len(monthly_data) + forecast_periods).reshape(-1, 1)
    trend_forecast = trend_model.predict(future_X)
    forecasts['Linear Trend'] = trend_forecast.tolist()
    
    # 3. Seasonal Naive (repeat last year's pattern)
    if len(monthly_data) >= 12:
        last_year_seasonal = decomposition.seasonal.iloc[-12:].values
        seasonal_forecast = []
        for i in range(forecast_periods):
            seasonal_forecast.append(monthly_data.iloc[-1] + last_year_seasonal[i % 12])
        forecasts['Seasonal Naive'] = seasonal_forecast
    
    # Create forecast DataFrame
    future_dates = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), 
                                periods=forecast_periods, freq='M')
    
    forecast_df = pd.DataFrame(forecasts, index=future_dates)
    
    # Calculate confidence intervals (simple approach)
    historical_std = monthly_data.std()
    for method in forecasts.keys():
        forecast_df[f'{method}_lower'] = forecast_df[method] - 1.96 * historical_std
        forecast_df[f'{method}_upper'] = forecast_df[method] + 1.96 * historical_std
    
    return forecast_df

# Business Applications:
# - Sales forecasting
# - Demand planning
# - Budget planning
# - Inventory optimization
```

5. FEATURE IMPORTANCE & SELECTION:
```python
# Feature Importance Analysis
def analyze_feature_importance(data, target_col, feature_cols):
    X = data[feature_cols].dropna()
    y = data[target_col].dropna()
    
    # Align data
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Random Forest for feature importance
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Correlation analysis
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    importance_df['correlation'] = importance_df['feature'].map(corr_with_target)
    
    return importance_df

# Business Applications:
# - Identify key drivers of performance
# - Optimize data collection efforts
# - Reduce model complexity
# - Focus on most impactful variables
```

6. A/B TESTING & STATISTICAL SIGNIFICANCE:
```python
# A/B Test Analysis
def perform_ab_test_analysis(data, group_col, metric_col, confidence_level=0.95):
    # Separate groups
    group_a = data[data[group_col] == data[group_col].unique()[0]][metric_col]
    group_b = data[data[group_col] == data[group_col].unique()[1]][metric_col]
    
    # Remove missing values
    group_a = group_a.dropna()
    group_b = group_b.dropna()
    
    # Calculate statistics
    stats_summary = {
        'Group A': {
            'mean': group_a.mean(),
            'std': group_a.std(),
            'count': len(group_a)
        },
        'Group B': {
            'mean': group_b.mean(),
            'std': group_b.std(),
            'count': len(group_b)
        }
    }
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group_a) - 1) * group_a.var() + (len(group_b) - 1) * group_b.var()) / 
                        (len(group_a) + len(group_b) - 2))
    cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
    
    # Results
    alpha = 1 - confidence_level
    is_significant = p_value < alpha
    lift = ((group_b.mean() - group_a.mean()) / group_a.mean()) * 100
    
    test_results = {
        'p_value': p_value,
        'is_significant': is_significant,
        'confidence_level': confidence_level,
        'lift_percentage': lift,
        'cohens_d': cohens_d,
        'effect_size': 'small' if abs(cohens_d) < 0.3 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }
    
    return pd.DataFrame(stats_summary).T, test_results

# Business Applications:
# - Marketing campaign effectiveness
# - Product feature testing
# - Pricing strategy validation
# - UI/UX optimization
```

7. MODEL SCORING:
```python
# Model Scoring and Prediction
def score_new_data(trained_model, scaler, new_data, feature_cols):
    #Apply trained model to score new data
    
    # Prepare new data
    X_new = new_data[feature_cols].copy()
    
    # Handle missing values (use training strategy)
    for col in feature_cols:
        if X_new[col].dtype in ['int64', 'float64']:
            X_new[col] = X_new[col].fillna(X_new[col].median())
        else:
            X_new[col] = X_new[col].fillna('Unknown')
    
    # Scale features if needed
    if scaler is not None:
        X_new_scaled = scaler.transform(X_new)
        predictions = trained_model.predict(X_new_scaled)
    else:
        predictions = trained_model.predict(X_new)
    
    # Add predictions to original data
    scored_data = new_data.copy()
    scored_data['prediction'] = predictions
    
    # Add prediction probability for classification
    if hasattr(trained_model, 'predict_proba'):
        if scaler is not None:
            probabilities = trained_model.predict_proba(X_new_scaled)
        else:
            probabilities = trained_model.predict_proba(X_new)
        
        if probabilities.shape[1] == 2:  # Binary classification
            scored_data['probability'] = probabilities[:, 1]
        else:  # Multi-class
            for i, class_name in enumerate(trained_model.classes_):
                scored_data[f'prob_{class_name}'] = probabilities[:, i]
    
    return scored_data

# Business Applications:
# - Score new leads for sales priority
# - Predict customer churn risk
# - Forecast demand for new products
# - Risk assessment for new loans
```

PREDICTIVE MODELING BEST PRACTICES:

1. DATA PREPARATION:
```python
# Always validate data quality before modeling
def prepare_modeling_data(data, target_col, feature_cols):
    # Check data quality
    quality_issues = []
    
    # Check missing values
    missing_pct = (data[feature_cols + [target_col]].isnull().sum() / len(data)) * 100
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    if high_missing:
        quality_issues.append(f"High missing values in: {high_missing}")
    
    # Check data types
    for col in feature_cols:
        if data[col].dtype == 'object':
            unique_vals = data[col].nunique()
            if unique_vals > 50:
                quality_issues.append(f"High cardinality categorical: {col} ({unique_vals} unique values)")
    
    # Check target variable
    if data[target_col].nunique() < 2:
        quality_issues.append("Target variable has insufficient variation")
    
    if quality_issues:
        print("Data Quality Issues:")
        for issue in quality_issues:
            print(f"- {issue}")
    
    return data, quality_issues
```

2. MODEL VALIDATION:
```python
# Cross-validation for robust model evaluation
def validate_model_performance(X, y, model, cv_folds=5):
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    
    validation_results = {
        'mean_r2': cv_scores.mean(),
        'std_r2': cv_scores.std(),
        'min_r2': cv_scores.min(),
        'max_r2': cv_scores.max(),
        'cv_scores': cv_scores.tolist()
    }
    
    return validation_results
```
"""

def get_statistics_guide():
    return """
# STATISTICAL TESTS & TIME SERIES ANALYSIS GUIDE

## 1. DESCRIPTIVE STATISTICS
```python
def comprehensive_descriptive_stats(data, numeric_cols, categorical_cols=None):
    results = {}
    
    # Numeric statistics
    numeric_stats = data[numeric_cols].describe()
    numeric_stats.loc['skewness'] = data[numeric_cols].skew()
    numeric_stats.loc['kurtosis'] = data[numeric_cols].kurtosis()
    numeric_stats.loc['cv'] = data[numeric_cols].std() / data[numeric_cols].mean()
    results['numeric_summary'] = numeric_stats
    
    # Distribution tests
    normality_tests = {}
    for col in numeric_cols:
        clean_data = data[col].dropna()
        if len(clean_data) > 3:
            shapiro_stat, shapiro_p = stats.shapiro(clean_data[:5000])  # Limit for large datasets
            jb_stat, jb_p = stats.jarque_bera(clean_data)
            normality_tests[col] = {
                'shapiro_p': shapiro_p,
                'jarque_bera_p': jb_p,
                'is_normal': (shapiro_p > 0.05) and (jb_p > 0.05)
            }
    results['normality_tests'] = pd.DataFrame(normality_tests).T
    
    # Categorical statistics
    if categorical_cols:
        cat_summary = {}
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            cat_summary[col] = {
                'unique_count': data[col].nunique(),
                'mode': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                'top_categories': value_counts.head(5).to_dict()
            }
        results['categorical_summary'] = pd.DataFrame(cat_summary).T
    
    return results
```

## 2. HYPOTHESIS TESTING
```python
def perform_hypothesis_tests(data, group_col=None, numeric_cols=None, categorical_cols=None):
    test_results = {}
    
    # Two-sample t-test
    if group_col and numeric_cols:
        groups = data.groupby(group_col)
        if len(groups) == 2:
            group_names = list(groups.groups.keys())
            for col in numeric_cols:
                group1 = groups.get_group(group_names[0])[col].dropna()
                group2 = groups.get_group(group_names[1])[col].dropna()
                
                # Check for equal variances
                levene_stat, levene_p = stats.levene(group1, group2)
                equal_var = levene_p > 0.05
                
                # Perform t-test
                t_stat, t_p = stats.ttest_ind(group1, group2, equal_var=equal_var)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1)-1)*group1.var() + (len(group2)-1)*group2.var()) / 
                                   (len(group1)+len(group2)-2))
                cohens_d = (group2.mean() - group1.mean()) / pooled_std
                
                test_results[f'ttest_{col}'] = {
                    't_statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < 0.05,
                    'cohens_d': cohens_d,
                    'equal_variances': equal_var,
                    'levene_p': levene_p
                }
    
    # Chi-square test for categorical variables
    if categorical_cols and len(categorical_cols) >= 2:
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                contingency_table = pd.crosstab(data[col1], data[col2])
                chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Cramér's V (effect size)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                
                test_results[f'chi2_{col1}_vs_{col2}'] = {
                    'chi2_statistic': chi2,
                    'p_value': chi2_p,
                    'degrees_of_freedom': dof,
                    'significant': chi2_p < 0.05,
                    'cramers_v': cramers_v
                }
    
    # One-way ANOVA
    if group_col and numeric_cols:
        groups = data.groupby(group_col)
        if len(groups) > 2:
            for col in numeric_cols:
                group_data = [group[col].dropna() for name, group in groups]
                group_data = [g for g in group_data if len(g) > 0]
                
                if len(group_data) > 2:
                    f_stat, f_p = stats.f_oneway(*group_data)
                    test_results[f'anova_{col}'] = {
                        'f_statistic': f_stat,
                        'p_value': f_p,
                        'significant': f_p < 0.05
                    }
    
    return pd.DataFrame(test_results).T
```

## 3. CORRELATION & COVARIANCE ANALYSIS
```python
def analyze_correlations(data, numeric_cols, method='pearson'):
    correlation_results = {}
    
    # Correlation matrix
    corr_matrix = data[numeric_cols].corr(method=method)
    correlation_results['correlation_matrix'] = corr_matrix
    
    # Significant correlations with p-values
    n = len(data[numeric_cols].dropna())
    correlation_tests = {}
    
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            clean_data = data[[col1, col2]].dropna()
            if len(clean_data) > 3:
                if method == 'pearson':
                    corr_coef, p_value = stats.pearsonr(clean_data[col1], clean_data[col2])
                elif method == 'spearman':
                    corr_coef, p_value = stats.spearmanr(clean_data[col1], clean_data[col2])
                else:  # kendall
                    corr_coef, p_value = stats.kendalltau(clean_data[col1], clean_data[col2])
                
                correlation_tests[f'{col1}_vs_{col2}'] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'sample_size': len(clean_data)
                }
    
    correlation_results['correlation_tests'] = pd.DataFrame(correlation_tests).T
    
    # Partial correlations (if more than 2 variables)
    if len(numeric_cols) > 2:
        try:
            from pingouin import partial_corr
            partial_corr_results = {}
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    control_vars = [col for col in numeric_cols if col not in [col1, col2]]
                    if control_vars:
                        pc_result = partial_corr(data, x=col1, y=col2, covar=control_vars[:3])  # Limit control vars
                        partial_corr_results[f'{col1}_vs_{col2}'] = {
                            'partial_r': pc_result['r'].iloc[0],
                            'p_value': pc_result['p-val'].iloc[0]
                        }
            correlation_results['partial_correlations'] = pd.DataFrame(partial_corr_results).T
        except ImportError:
            pass  # Skip if pingouin not available
    
    return correlation_results
```

## 4. TIME SERIES DECOMPOSITION & ANALYSIS
```python
def analyze_time_series(data, date_col, value_col, freq='M'):
    ts_data = data[[date_col, value_col]].copy()
    ts_data[date_col] = pd.to_datetime(ts_data[date_col])
    ts_data = ts_data.sort_values(date_col).set_index(date_col)
    
    # Resample to specified frequency
    if freq:
        ts_series = ts_data.resample(freq)[value_col].sum()
    else:
        ts_series = ts_data[value_col]
    
    results = {}
    
    # Basic time series statistics
    results['basic_stats'] = {
        'length': len(ts_series),
        'start_date': str(ts_series.index.min()),
        'end_date': str(ts_series.index.max()),
        'frequency': freq,
        'missing_values': ts_series.isnull().sum()
    }
    
    # Stationarity tests
    if len(ts_series.dropna()) > 10:
        # Augmented Dickey-Fuller test
        adf_result = adfuller(ts_series.dropna())
        results['stationarity'] = {
            'adf_statistic': adf_result[0],
            'adf_p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'critical_values': adf_result[4]
        }
        
        # KPSS test
        try:
            kpss_result = kpss(ts_series.dropna())
            results['stationarity']['kpss_statistic'] = kpss_result[0]
            results['stationarity']['kpss_p_value'] = kpss_result[1]
            results['stationarity']['kpss_stationary'] = kpss_result[1] > 0.05
        except:
            pass
    
    # Seasonal decomposition
    if len(ts_series) >= 24:  # Need sufficient data for decomposition
        try:
            decomposition = seasonal_decompose(ts_series, model='additive', period=12)
            results['decomposition'] = {
                'trend': decomposition.trend.dropna(),
                'seasonal': decomposition.seasonal.dropna(),
                'residual': decomposition.resid.dropna()
            }
            
            # Seasonality strength
            seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.resid.var() + decomposition.seasonal.var()))
            results['seasonality_strength'] = seasonal_strength
            
        except:
            results['decomposition'] = "Insufficient data for seasonal decomposition"
    
    # Autocorrelation analysis
    if len(ts_series.dropna()) > 10:
        # Calculate autocorrelation and partial autocorrelation
        max_lags = min(40, len(ts_series) // 4)
        acf_values, acf_confint = acf(ts_series.dropna(), nlags=max_lags, alpha=0.05)
        pacf_values, pacf_confint = pacf(ts_series.dropna(), nlags=max_lags, alpha=0.05)
        
        results['autocorrelation'] = {
            'acf': acf_values,
            'acf_confint': acf_confint,
            'pacf': pacf_values,
            'pacf_confint': pacf_confint
        }
        
        # Ljung-Box test for autocorrelation
        lb_stat, lb_p = acorr_ljungbox(ts_series.dropna(), lags=10, return_df=False)
        results['ljung_box'] = {
            'statistic': lb_stat[-1],
            'p_value': lb_p[-1],
            'has_autocorrelation': lb_p[-1] < 0.05
        }
    
    return results
```

## 5. TIME SERIES MODELING (ARIMA)
```python
def fit_arima_models(data, date_col, value_col, max_p=3, max_d=2, max_q=3):
    ts_data = data[[date_col, value_col]].copy()
    ts_data[date_col] = pd.to_datetime(ts_data[date_col])
    ts_data = ts_data.sort_values(date_col).set_index(date_col)
    ts_series = ts_data[value_col].dropna()
    
    if len(ts_series) < 50:
        return "Insufficient data for ARIMA modeling (need at least 50 observations)"
    
    # Split data
    train_size = int(len(ts_series) * 0.8)
    train_data = ts_series[:train_size]
    test_data = ts_series[train_size:]
    
    # Grid search for best ARIMA parameters
    best_aic = np.inf
    best_params = None
    best_model = None
    model_results = []
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # Model evaluation
                    aic = fitted_model.aic
                    bic = fitted_model.bic
                    
                    # Forecast on test set
                    forecast = fitted_model.forecast(steps=len(test_data))
                    rmse = np.sqrt(mean_squared_error(test_data, forecast))
                    mae = mean_absolute_error(test_data, forecast)
                    
                    model_results.append({
                        'order': (p, d, q),
                        'aic': aic,
                        'bic': bic,
                        'rmse': rmse,
                        'mae': mae
                    })
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except:
                    continue
    
    if best_model is None:
        return "Could not fit any ARIMA models"
    
    # Fit best model on full data
    final_model = ARIMA(ts_series, order=best_params).fit()
    
    # Generate forecasts
    forecast_periods = min(12, len(ts_series) // 4)
    forecast = final_model.forecast(steps=forecast_periods)
    forecast_ci = final_model.get_forecast(steps=forecast_periods).conf_int()
    
    results = {
        'best_model_order': best_params,
        'model_comparison': pd.DataFrame(model_results),
        'model_summary': final_model.summary(),
        'forecast': forecast,
        'forecast_confidence_intervals': forecast_ci,
        'residual_diagnostics': {
            'ljung_box_p': acorr_ljungbox(final_model.resid, lags=10, return_df=False)[1][-1],
            'jarque_bera_p': stats.jarque_bera(final_model.resid)[1],
            'residual_autocorr': acf(final_model.resid, nlags=20)
        }
    }
    
    return results
```

## 6. REGRESSION WITH STATISTICAL INFERENCE
```python
def statistical_regression_analysis(data, target_col, feature_cols, model_type='ols'):
    # Prepare data
    regression_data = data[feature_cols + [target_col]].dropna()
    X = regression_data[feature_cols]
    y = regression_data[target_col]
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    results = {}
    
    # Fit model
    if model_type == 'ols':
        model = sm.OLS(y, X_with_const)
    elif model_type == 'logit':
        model = sm.Logit(y, X_with_const)
    elif model_type == 'wls':
        # Weighted least squares (equal weights for simplicity)
        weights = np.ones(len(y))
        model = sm.WLS(y, X_with_const, weights=weights)
    
    fitted_model = model.fit()
    
    # Model summary
    results['model_summary'] = fitted_model.summary()
    results['parameters'] = fitted_model.params
    results['p_values'] = fitted_model.pvalues
    results['confidence_intervals'] = fitted_model.conf_int()
    results['r_squared'] = fitted_model.rsquared if hasattr(fitted_model, 'rsquared') else None
    results['adj_r_squared'] = fitted_model.rsquared_adj if hasattr(fitted_model, 'rsquared_adj') else None
    
    # Diagnostic tests
    if model_type == 'ols':
        # Residual analysis
        residuals = fitted_model.resid
        fitted_values = fitted_model.fittedvalues
        
        # Normality of residuals
        jb_stat, jb_p = stats.jarque_bera(residuals)
        sw_stat, sw_p = stats.shapiro(residuals[:5000])  # Limit for large datasets
        
        # Homoscedasticity (Breusch-Pagan test)
        bp_test = het_breuschpagan(residuals, X_with_const)
        
        # Autocorrelation (Durbin-Watson)
        dw_stat = durbin_watson(residuals)
        
        results['diagnostics'] = {
            'jarque_bera_p': jb_p,
            'shapiro_p': sw_p,
            'breusch_pagan_p': bp_test[1],
            'durbin_watson': dw_stat,
            'residuals_normal': (jb_p > 0.05) and (sw_p > 0.05),
            'homoscedastic': bp_test[1] > 0.05,
            'no_autocorrelation': 1.5 < dw_stat < 2.5
        }
        
        # Multicollinearity (VIF)
        if len(feature_cols) > 1:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_with_const.columns
            vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                             for i in range(X_with_const.shape[1])]
            results['vif'] = vif_data
    
    return results
```

## 7. DISTRIBUTION FITTING & GOODNESS OF FIT
```python
def fit_distributions(data, column, distributions=['norm', 'lognorm', 'expon', 'gamma']):
    sample_data = data[column].dropna()
    
    if len(sample_data) < 30:
        return "Insufficient data for distribution fitting"
    
    results = {}
    
    for dist_name in distributions:
        try:
            # Get distribution
            dist = getattr(stats, dist_name)
            
            # Fit distribution
            params = dist.fit(sample_data)
            
            # Goodness of fit tests
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(sample_data, lambda x: dist.cdf(x, *params))
            
            # Anderson-Darling test (if available)
            try:
                ad_stat, ad_critical, ad_significance = stats.anderson(sample_data, dist=dist_name)
                ad_p = None  # AD test doesn't provide p-value directly
            except:
                ad_stat, ad_critical, ad_significance = None, None, None
            
            # AIC/BIC for model comparison
            log_likelihood = np.sum(dist.logpdf(sample_data, *params))
            k = len(params)  # Number of parameters
            n = len(sample_data)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            results[dist_name] = {
                'parameters': params,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'ad_statistic': ad_stat,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'good_fit': ks_p > 0.05
            }
            
        except Exception as e:
            results[dist_name] = {'error': str(e)}
    
    # Rank distributions by AIC
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_dist = min(valid_results.keys(), key=lambda x: valid_results[x]['aic'])
        results['best_distribution'] = best_dist
    
    return results
```

## 8. CONFIDENCE INTERVALS & BOOTSTRAP
```python
def calculate_confidence_intervals(data, column, confidence_level=0.95, method='bootstrap', n_bootstrap=1000):
    sample_data = data[column].dropna()
    alpha = 1 - confidence_level
    
    results = {}
    
    # Mean confidence interval
    if method == 'bootstrap':
        # Bootstrap method
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(sample_data, size=len(sample_data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        mean_ci = np.percentile(bootstrap_means, [100 * alpha/2, 100 * (1 - alpha/2)])
        
        # Median confidence interval
        bootstrap_medians = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(sample_data, size=len(sample_data), replace=True)
            bootstrap_medians.append(np.median(bootstrap_sample))
        
        median_ci = np.percentile(bootstrap_medians, [100 * alpha/2, 100 * (1 - alpha/2)])
        
    else:  # t-distribution method
        mean = sample_data.mean()
        se = stats.sem(sample_data)
        t_critical = stats.t.ppf(1 - alpha/2, len(sample_data) - 1)
        margin_error = t_critical * se
        mean_ci = [mean - margin_error, mean + margin_error]
        median_ci = None  # T-method not applicable for median
    
    results['mean_ci'] = mean_ci
    results['median_ci'] = median_ci
    results['method'] = method
    results['confidence_level'] = confidence_level
    
    return results
```
"""

def get_code_generation_prompt(data_context, user_query, result_type, needs_multiple_sheets, 
                              actual_columns, numeric_columns, categorical_columns, 
                              worksheet_info, worksheet_context, active_worksheet):
    
    column_examples = ""
    if actual_columns:
        column_examples = f"""
Available columns in active worksheet ({active_worksheet}):
- All columns: {actual_columns}
- Numeric columns: {numeric_columns}
- Text/Categorical columns: {categorical_columns}

CRITICAL: Use ONLY these exact column names. Never use placeholders or generic names.
"""
    # Get comprehensive guides
    visualization_guide = get_comprehensive_visualization_guide()
    data_prep_patterns = get_data_preparation_patterns()
    pandas_safety_guide = get_pandas_safety_guide()
    predictive_modeling_guide = get_predictive_modeling_guide()
    statistics_guide = get_statistics_guide()
    advanced_ml_guide = get_advanced_ml_guide()

    return f"""
You are an advanced business analytics assistant proficient in Python Programming, Statistics & Machine Learning.
You have deep domain knowledge of various corporate functions like finance, operations, marketing, supply chain and strategy.
Your job is to generate robust, sophisticated code that answers user's question. This is typically a two step process:
 
Step 1. Analyze the user's query and data context to determine the most appropriate analytical approach. Some guidance on this step is provided below:
        As part of enhanced query analysis, determine:
            a. Business domain (Finance, Sales, Marketing, Operations, Supply Chain, Procurement, Safety,Health & Environment, Quality etc.)
            b. Analysis type (Descriptive, Diagnostic, Predictive, Prescriptive)
            c. Output format (KPI/Metric, Dashboard, Statistical Report, Predictive Model, Business Intelligence etc.)
            d. Temporal analysis needs (Time series, Seasonality, Forecasting, Trend analysis etc.)
            e. Statistical complexity (Basic stats, Hypothesis testing, Regression, Classification, Clustering etc.)
            f. Visualization requirements (Executive summary, Operational dashboard, Analytical deep-dive etc.)

Step 2. Use the analysis conducted in step 1 to generate required code. Adapt to available libraries and support multi-worksheet Excel analysis.
        Use the guidance

Here are some key code generation guides to help you in your task.

{data_prep_patterns}

{pandas_safety_guide}

{visualization_guide}

{predictive_modeling_guide}

{statistics_guide}

{advanced_ml_guide}

CRITICAL IMPORT RULES - NEVER IMPORT ANYTHING:
- ALL required functions are pre-imported and available
- Available Core functions: pd, np, px, go, current_data, worksheet_data, merge_worksheets
- Available Machine Learning functions imported as:
    - from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
    - from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, IsolationForest
    - from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    - from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
    - from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
    - from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                                   accuracy_score, precision_score, recall_score, f1_score,
                                   roc_auc_score, roc_curve, confusion_matrix, classification_report,
                                   silhouette_score)
    - from sklearn.decomposition import PCA, TruncatedSVD
    - from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
    - from sklearn.svm import SVC, SVR
    - from sklearn.naive_bayes import GaussianNB
    - from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    - from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
- Available Scipy Statistics functions imported as: 
    - from scipy import stats, signal, optimize, interpolate
    - from scipy.stats import (pearsonr, spearmanr, kendalltau, ttest_ind, ttest_rel, ttest_1samp, friedmanchisquare, chi2_contingency, fisher_exact, kstest, shapiro, normaltest, 
                               anderson, levene, bartlett, f_oneway, zscore, boxcox, probplot, rankdata, trim_mean, kurtosis, skew, mode, entropy, mutual_info_score, pointbiserialr)
    - from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    - from scipy.spatial.distance import euclidean, cosine, correlation
- Available Statsmodels functions imported as: 
    - import statsmodels.api as sm
    - from statsmodels.tsa.seasonal import seasonal_decompose, STL
    - from statsmodels.tsa.statespace.sarimax import SARIMAX
    - from statsmodels.tsa.arima.model import ARIMA
    - from statsmodels.tsa.holtwinters import ExponentialSmoothing
    - from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss, coint
    - from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   
UNIVERSAL CODE GENERATION RULES:

1. BUSINESS CONTEXT FIRST:
   - Always consider the business implications of the analysis
   - Use business-appropriate metrics and KPIs
   - Format results for executive consumption

2. ADVANCED ANALYTICS PATTERNS:
   - Apply statistical rigor to business problems
   - Use predictive modeling for forward-looking insights
   - Implement proper validation and testing

3. VISUALIZATION EXCELLENCE:
   - Create publication-ready charts with proper titles, labels, formatting
   - Use business-appropriate color schemes and layouts
   - Include contextual information (targets, benchmarks, trends)

4. ERROR PREVENTION & DATA SAFETY:
   - ALWAYS cast to string before using .str() methods: df['col'].astype('str').str.method()
   - ALWAYS reset_index() after groupby operations
   - ALWAYS validate data types and handle missing values
   - Use try-except blocks for robust error handling

5. PERFORMANCE & SCALABILITY:
   - Limit results to 50,000 rows for large datasets
   - Use efficient pandas operations
   - Provide progress indicators for long-running analysis

6. RESULT FORMAT:
   For 'value': Return business metrics, KPIs, statistical results
   For 'table': Return formatted DataFrames with business context
   For 'chart': Create advanced business visualizations following the comprehensive guide

7. MANDATORY TEMPORAL DATA HANDLING RULES:
    - **ALWAYS SORT** temporal value columns chronologically before creating time series visualizations
    - **USE CUSTOM SORTING** logic for business date formats (Q1 22, 2024-26, etc.)
    - **CONSIDER RESHAPING** data if temporal column names need to become a time axis
    - **VERIFY TIME ORDER** in line charts, forecasting, and trend analysis
    - **CONVERT TO DATETIME** where possible for proper temporal operations
    - following are some common temporal patterns used in our business:
        - Quarter patterns (Q1 22): Sort by (year, quarter_number)  
        - Year-week (2024-26): Sort by (year, week_number)
        - Year-month (2024-01): Sort by (year, month_number)
        - Before ANY time series visualization: df = df.sort_values(temporal_col, key=custom_sort_function)
    - There may be cases where column name matches a time period (Q1 22, 2024W17, 2024-01 etc.)
        - Temporal column names: Consider melting to time series format   

8. MULTI-WORKSHEET INTELLIGENCE:
    - Use current_data for single-worksheet analysis
    - Use merge_worksheets(worksheet_data) for comprehensive analysis across all sheets
    - Use worksheet_data dictionary for sheet-by-sheet comparison analysis
    
This is the Data Context you will work with:
{data_context}

{column_examples}

{worksheet_context}

User Query: {user_query}
Expected Result Type: {result_type}
Multi-Worksheet Query: {needs_multiple_sheets}

CRITICAL EXECUTION RULES:
- ONLY use exact column names from the data - never use placeholders
- NEVER include import statements - all functions are pre-imported
- ALWAYS assign final results to 'result' variable
- For visualizations, assign to 'fig' and set descriptive result message
- Use comprehensive error handling and data validation
- DO NOT alter or sanitize column names or any string or object values contained within columns! Always us exact column names & values as they appear in the dataframe.
- If the result can be best shown as a visualization/chart/plot, please prefer that
- Multi time period analyses, if they result in multiperiod output should be shown as charts
- NEVER show results in complex python formats like dictionary or json! Charts and/or tables or lists are always preferable
- Apply business intelligence best practices throughout

CRITICAL RESULT ASSIGNMENT RULES:
1. ALWAYS assign meaningful results to variables, NEVER use print() for final results:
   CORRECT:
   result = summary_stats
   result = correlation_matrix
   result = top_customers.head(10)
   
   INCORRECT:
   result = print("Analysis complete")
   result = print(summary_stats)
   print(correlation_matrix)  # without assignment

2. PRIORITIZE data objects over text summaries:
   PREFER: result = filtered_data
   AVOID: result = "Found 50 records"

3. NEVER end with print statements:
   INCORRECT:
   ```python 
   df.groupby('category').sum().plot()
   result = print("Chart created")
   ```
   CORRECT:
   ```python
   grouped_data = df.groupby('category').sum()
   fig = px.bar(grouped_data.reset_index(), x='category', y='sales')
   result = grouped_data
    ```
    
4. Use descriptive variable names for intermediate steps:
   For e.g.,
   ```python
   analysis_summary = current_data.groupby('category')['value_col'].agg(['sum', 'mean', 'std])
   result = analysis_summary
   ```
   
REMEMBER: The 'result' variable will be displayed to the user, so make it meaningful!
Generate Python code that produces actionable business insights using advanced analytics.
Only return the Python code, no explanations.
"""
