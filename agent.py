"""
Enhanced agent.py with intelligent wide format data handling, mixed measure type analysis, and automated commentary/insights generation
FEATURES:
- Automatic detection of wide format time-series data
- Smart handling of mixed measure types (absolute, percentage, ratio)
- Data transformation utilities for statistical analysis
- Enhanced correlation analysis for business metrics
- Configuration Management for all settings
- Enhanced Error Handling and Logging
- NEW: Automated commentary and insights generation for all outputs
"""

import sys
import os
import io
import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

#from llm_factory import create_universal_llm, UniversalLLMFactory, LLMConfig
try:
    from llm_factory import create_universal_llm, UniversalLLMFactory, LLMConfig, LLMProvider
    UNIVERSAL_LLM_AVAILABLE = True
except ImportError as e:
    print(f"Universal LLM factory not available: {e}")
    UNIVERSAL_LLM_AVAILABLE = False

# LangChain and LangGraph imports
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict, List, Dict, Any, Optional
import httpx
import openai
import requests
import threading
import time
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

# For result formatting
from result_formatter import format_result_for_user, validate_result_user_friendliness

# Import analytics libraries with error handling
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, IsolationForest
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                                 accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, roc_curve, confusion_matrix, classification_report,
                                 silhouette_score, davies_bouldin_score)
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, mutual_info_regression, mutual_info_classif
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    SKLEARN_AVAILABLE = True
    print("sklearn imports completed successfully!")
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats, signal, optimize, interpolate
    from scipy.stats import (pearsonr, spearmanr, kendalltau, ttest_ind, ttest_rel, ttest_1samp,
                             mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
                             chi2_contingency, fisher_exact, kstest, shapiro, normaltest,
                             anderson, levene, bartlett, f_oneway, zscore, boxcox,
                             probplot, rankdata, trim_mean, kurtosis, skew, mode,
                             entropy, pointbiserialr)
    from scipy.signal import find_peaks, savgol_filter, butter, filtfilt, welch
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import euclidean, cosine, correlation
    SCIPY_AVAILABLE = True
    print("scipy imports completed successfully!")
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss, coint
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.proportion import proportions_ztest
    from statsmodels.stats.power import TTestPower
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.filters.hp_filter import hpfilter
    from statsmodels.tsa.filters.cf_filter import cffilter
    STATSMODELS_AVAILABLE = True
    print("statsmodels imports completed successfully!")
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.switch_backend('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import advanced ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

try:
    import pycaret
    from pycaret.regression import *
    from pycaret.classification import *
    from pycaret.time_series import *
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import shap
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

try:
    import category_encoders as ce
    from feature_engine.selection import SelectByShuffling
    from feature_engine.creation import MathematicalCombination
    FEATURE_ENG_AVAILABLE = True
except ImportError:
    FEATURE_ENG_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.knn import KNN
    from pyod.models.ocsvm import OCSVM
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False

from prompts import get_code_generation_prompt

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ===========================
# CONFIGURATION MANAGEMENT
# ===========================

@dataclass
class AgentConfig:
    """Configuration management for DataAnalysisAgent with all settings centralized"""
    
    # Execution Settings
    max_execution_time: int = 300
    max_retry_attempts: int = 2
    max_memory_mb: int = 4096
    
    # Data Processing Limits
    max_dataframe_rows: int = 1000000
    max_dataframe_cols: int = 250
    sample_size_threshold: int = 100000
    sample_size_limit: int = 10000
    large_dataset_threshold: int = 1000000
    
    # Wide Format Detection Settings
    min_time_columns_for_wide_format: int = 3
    min_measure_columns_for_wide_format: int = 1
    correlation_min_threshold: float = 0.1
    high_cardinality_threshold: int = 100
    
    # Statistical Analysis Settings
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    high_variance_cv_threshold: float = 2.0
    correlation_significance_threshold: float = 0.05
    
    # Display and Output Settings
    max_display_columns: int = 100
    max_correlation_pairs_display: int = 100
    max_list_items_display: int = 10000
    max_error_message_length: int = 500
    
    # LLM Settings
    llm_temperature: float = 0.1
    llm_max_retries: int = 3
    token_refresh_buffer_minutes: int = 5
    
    # Commentary Settings (NEW)
    enable_commentary: bool = True
    max_commentary_length: int = 500
    max_insights_count: int = 5
    commentary_temperature: float = 0.3
    
    # Time Series Patterns
    time_patterns: List[str] = field(default_factory=lambda: [
        # Core Patterns
        r'^Q[1-4]\s*\d{2,4}$',  # Q1 22, Q2 2023, etc.
        r'^\d{4}[_-]?Q[1-4]$',  # 2023Q1, 2023_Q1, etc.
        r'^\d{1,2}[/_-]\d{2,4}$',  # 01/23, 1-2023, etc.
        r'^\w{3}\s*\d{2,4}$',   # Jan 23, Mar 2023, etc.
        r'^\d{4}[_-]?\d{2}$',   # 202301, 2023_01, etc.
        r'^FY\d{2,4}$',         # FY23, FY2023, etc.
        r'^\d{4}$',             # 2023, 2024, etc.
        # Year Week Patterns
        r'^\d{4}[-_]?W?\d{1,2}$',  # 2024-13, 2024W13, 2024_13, 202413
        r'^W?\d{1,2}[-_]?\d{4}$',  # W13-2024, 13_2024, 132024
        r'^\d{4}[-_]?week[-_]?\d{1,2}$',  # 2024-week-13, 2024_week_13
        r'^week[-_]?\d{1,2}[-_]?\d{4}$',  # week-13-2024, week_13_2024
        # Quarter Year Patterns
        r'^Q[1-4][-_\s]?\d{2,4}$',  # Q1 22, Q1-2022, Q1_2022, Q12022
        r'^\d{2,4}[-_\s]?Q[1-4]$',  # 2022-Q1, 2022_Q1, 2022 Q1, 2022Q1
        r'^(Q|q)(tr|uarter)?[-_\s]?[1-4][-_\s]?\d{2,4}$',  # Quarter1-2022, Qtr1_2022
        r'^\d{2,4}[-_\s]?(Q|q)(tr|uarter)?[-_\s]?[1-4]$',  # 2022-Quarter1, 2022_Qtr1
        # Year Month Patterns
        r'^\d{4}[-_]?\d{1,2}$',  # 2024-01, 2024_01, 202401
        r'^\d{1,2}[-_]?\d{4}$',  # 01-2024, 01_2024, 012024
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_\s]?\d{2,4}$',  # Jan-2024, Jan 24
        r'^\d{2,4}[-_\s]?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$',  # 2024-Jan, 24 Jan
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)[-_\s]?\d{2,4}$',
        r'^\d{2,4}[-_\s]?(January|February|March|April|May|June|July|August|September|October|November|December)$',
        # Year Patterns
        r'^\d{4}$',  # 2024
        r'^(FY|fy|CY|cy)[-_\s]?\d{4}$',  # FY2024, FY-2024, CY_2024
        r'^\d{4}[-_\s]?(FY|fy|CY|cy)$',  # 2024-FY, 2024_CY
        # Month Year Patterns
        r'^(0?[1-9]|1[0-2])[-/]\d{4}$',  # 1/2024, 01/2024, 12/2024
        r'^\d{4}[-/](0?[1-9]|1[0-2])$',  # 2024/1, 2024/01, 2024/12
    ])

    # Business Period Patterns (NEW)
    business_period_patterns: List[str] = field(default_factory=lambda: [
        # Moving Annual Total patterns
        r'^MAT$',  # MAT
        r'^MAT[-_]?\d+$',  # MAT-1, MAT_2, MAT3, etc.
        r'^MAT[-_]?[Pp]?\d+$',  # MAT-P1, MAT_P2, etc.
        r'^[Mm]oving\s*[Aa]nnual\s*[Tt]otal$',  # Moving Annual Total
        r'^[Mm]oving\s*[Aa]nnual\s*[Tt]otal[-_]?\d+$',  # Moving Annual Total-1

        # Year To Date patterns
        r'^YTD$',  # YTD
        r'^YTD[-_]?\d+$',  # YTD-1, YTD_2, etc.
        r'^[Yy]ear\s*[Tt]o\s*[Dd]ate$',  # Year To Date
        r'^[Yy]ear\s*[Tt]o\s*[Dd]ate[-_]?\d+$',  # Year To Date-1

        # Last N Weeks patterns
        r'^L\d+W$',  # L4W, L12W, L52W, etc.
        r'^L\d+[Ww]eeks?$',  # L4Weeks, L12Week, etc.
        r'^[Ll]ast\s*\d+\s*[Ww]eeks?$',  # Last 4 Weeks, Last 12 weeks
        r'^[Ll]ast\s*\d+[Ww]$',  # Last4W, Last12W
        r'^L\d+W[-_]?\d+$',  # L4W-1, L12W_2 (previous periods)

        # Last N Months patterns
        r'^L\d+M$',  # L3M, L6M, L12M, etc.
        r'^L\d+[Mm]onths?$',  # L3Months, L6Month, etc.
        r'^[Ll]ast\s*\d+\s*[Mm]onths?$',  # Last 3 Months, Last 6 months
        r'^[Ll]ast\s*\d+[Mm]$',  # Last3M, Last6M
        r'^L\d+M[-_]?\d+$',  # L3M-1, L6M_2 (previous periods)

        # Quarter To Date patterns
        r'^QTD$',  # QTD
        r'^QTD[-_]?\d+$',  # QTD-1, QTD_2, etc.
        r'^[Qq]uarter\s*[Tt]o\s*[Dd]ate$',  # Quarter To Date
        r'^[Qq]uarter\s*[Tt]o\s*[Dd]ate[-_]?\d+$',  # Quarter To Date-1

        # Rolling/Moving periods
        r'^R\d+[WMDQY]$',  # R4W, R3M, R1Q, R1Y (Rolling)
        r'^[Rr]olling\s*\d+\s*[Ww]eeks?$',  # Rolling 4 Weeks
        r'^[Rr]olling\s*\d+\s*[Mm]onths?$',  # Rolling 3 Months
        r'^[Mm]oving\s*\d+\s*[Ww]eeks?$',  # Moving 4 Weeks
        r'^[Mm]oving\s*\d+\s*[Mm]onths?$',  # Moving 3 Months

        # Fiscal patterns
        r'^FY\d{2,4}[-_]?YTD$',  # FY24-YTD, FY2024_YTD
        r'^FY\d{2,4}[-_]?MAT$',  # FY24-MAT, FY2024_MAT

        # Custom business cycles
        r'^H[12]$',  # H1, H2 (Half years)
        r'^H[12][-_]?\d{2,4}$',  # H1-24, H2_2024

    ])

    # Business Period Keywords (NEW)
    business_period_keywords: List[str] = field(default_factory=lambda: [
        'ytd', 'mtd', 'qtd', 'wtd', 'mat', 'rolling', 'moving', 'last', 'prev', 'previous',
        'sply', 'splq', 'splm', 'performance', 'perf', 'kpi', 'season', 'cycle'
    ])

    # Measure Keywords
    measure_keywords: List[str] = field(default_factory=lambda: [
        'measure', 'metric', 'kpi', 'indicator', 'variable', 'parameter', 'item'
    ])
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_detailed_logging: bool = True
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Create configuration from environment variables with fallback to defaults"""
        return cls(
            max_execution_time=int(os.getenv('AGENT_MAX_EXEC_TIME', cls.max_execution_time)),
            max_retry_attempts=int(os.getenv('AGENT_MAX_RETRIES', cls.max_retry_attempts)),
            max_memory_mb=int(os.getenv('AGENT_MAX_MEMORY_MB', cls.max_memory_mb)),
            max_dataframe_rows=int(os.getenv('AGENT_MAX_DF_ROWS', cls.max_dataframe_rows)),
            max_dataframe_cols=int(os.getenv('AGENT_MAX_DF_COLS', cls.max_dataframe_cols)),
            llm_temperature=float(os.getenv('AGENT_LLM_TEMPERATURE', cls.llm_temperature)),
            log_level=os.getenv('AGENT_LOG_LEVEL', cls.log_level),
            enable_detailed_logging=os.getenv('AGENT_DETAILED_LOGGING', 'true').lower() == 'true',
            enable_commentary=os.getenv('AGENT_ENABLE_COMMENTARY', 'true').lower() == 'true',
            commentary_temperature=float(os.getenv('AGENT_COMMENTARY_TEMPERATURE', cls.commentary_temperature))
        )

# ===========================
# ERROR HANDLING & LOGGING
# ===========================

class ErrorCategory(Enum):
    """Categorization of errors for better handling and logging"""
    DATA_VALIDATION = "data_validation"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "execution"
    AUTHENTICATION = "authentication"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    MEMORY = "memory"

class AgentError(Exception):
    """Custom exception with categorization and context"""
    def __init__(self, message: str, category: ErrorCategory, details: dict = None, original_error: Exception = None):
        self.message = message
        self.category = category
        self.details = details or {}
        self.original_error = original_error
        super().__init__(message)

class EnhancedLogger:
    """Enhanced logging with structured information and error categorization"""
    
    def __init__(self, config: AgentConfig, name: str = "DataAnalysisAgent"):
        self.config = config
        self.logger = logging.getLogger(name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:  # Avoid duplicate handlers
            # Set log level
            level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            self.logger.setLevel(level)
            
            # Create console handler
            handler = logging.StreamHandler()
            handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(self.config.log_format)
            handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
    
    def log_operation_start(self, operation: str, context: dict = None):
        """Log the start of an operation"""
        if self.config.enable_detailed_logging:
            context_str = f" | Context: {context}" if context else ""
            self.logger.info(f"🚀 Starting operation: {operation}{context_str}")
    
    def log_operation_success(self, operation: str, duration: float = None, details: dict = None):
        """Log successful completion of an operation"""
        duration_str = f" | Duration: {duration:.2f}s" if duration else ""
        details_str = f" | Details: {details}" if details and self.config.enable_detailed_logging else ""
        self.logger.info(f"✅ Completed operation: {operation}{duration_str}{details_str}")
    
    def log_operation_warning(self, operation: str, warning: str, details: dict = None):
        """Log a warning during an operation"""
        details_str = f" | Details: {details}" if details and self.config.enable_detailed_logging else ""
        self.logger.warning(f"⚠️ Warning in {operation}: {warning}{details_str}")
    
    def log_error(self, error: Exception, operation: str = None, category: ErrorCategory = None):
        """Log an error with full context"""
        if isinstance(error, AgentError):
            category = error.category
            error_details = error.details
            original_error = error.original_error
        else:
            category = category or self._categorize_error(error)
            error_details = {}
            original_error = error
        
        operation_str = f" in {operation}" if operation else ""
        details_str = f" | Details: {error_details}" if error_details and self.config.enable_detailed_logging else ""
        
        self.logger.error(f"❌ {category.value.upper()} ERROR{operation_str}: {str(error)}{details_str}")
        
        if original_error and original_error != error and self.config.enable_detailed_logging:
            self.logger.error(f"📋 Original error: {type(original_error).__name__}: {str(original_error)}")
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Automatically categorize errors based on type and message"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if 'auth' in error_str or '401' in error_str or 'token' in error_str:
            return ErrorCategory.AUTHENTICATION
        elif 'memory' in error_str or 'memoryerror' in error_type:
            return ErrorCategory.MEMORY
        elif 'network' in error_str or 'connection' in error_str or 'timeout' in error_str:
            return ErrorCategory.NETWORK
        elif 'keyerror' in error_type or 'column' in error_str:
            return ErrorCategory.DATA_VALIDATION
        elif 'syntax' in error_str or 'invalid syntax' in error_str:
            return ErrorCategory.CODE_GENERATION
        else:
            return ErrorCategory.SYSTEM

class ErrorHandler:
    """Centralized error handling with recovery strategies"""
    
    def __init__(self, config: AgentConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
    
    def handle_error(self, error: Exception, operation: str = None, context: dict = None) -> dict:
        """Handle errors with appropriate categorization and response"""
        # Log the error
        self.logger.log_error(error, operation)
        
        # Determine error category
        if isinstance(error, AgentError):
            category = error.category
            details = error.details
        else:
            category = self.logger._categorize_error(error)
            details = context or {}
        
        # Create error response
        error_response = {
            'error_type': type(error).__name__,
            'category': category.value,
            'message': self._format_error_message(error),
            'operation': operation,
            'recoverable': self._is_recoverable(category),
            'suggestions': self._get_suggestions(category, error),
            'details': details
        }
        
        return error_response
    
    def _format_error_message(self, error: Exception) -> str:
        """Format error message for user consumption"""
        message = str(error)
        if len(message) > self.config.max_error_message_length:
            message = message[:self.config.max_error_message_length] + "..."
        return message
    
    def _is_recoverable(self, category: ErrorCategory) -> bool:
        """Determine if an error category is typically recoverable"""
        recoverable_categories = {
            ErrorCategory.CODE_GENERATION,
            ErrorCategory.CODE_EXECUTION,
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.NETWORK
        }
        return category in recoverable_categories
    
    def _get_suggestions(self, category: ErrorCategory, error: Exception) -> List[str]:
        """Generate helpful suggestions based on error category"""
        error_str = str(error).lower()
        
        suggestions = []
        
        if category == ErrorCategory.DATA_VALIDATION:
            if 'column' in error_str:
                suggestions.append("Check column names - use exact names from the data overview")
                suggestions.append("Try asking 'What columns are available?' first")
            elif 'missing' in error_str:
                suggestions.append("Handle missing values before analysis")
            else:
                suggestions.append("Verify data format and structure")
        
        elif category == ErrorCategory.CODE_EXECUTION:
            suggestions.append("Try simplifying your query")
            suggestions.append("Ask for a basic summary first to understand the data")
        
        elif category == ErrorCategory.AUTHENTICATION:
            suggestions.append("Authentication token may have expired")
            suggestions.append("Please retry - token will be automatically refreshed")
        
        elif category == ErrorCategory.MEMORY:
            suggestions.append("Try filtering your data to a smaller subset")
            suggestions.append("Use sampling for large datasets")
        
        elif category == ErrorCategory.NETWORK:
            suggestions.append("Check your internet connection")
            suggestions.append("Retry the operation")
        
        return suggestions

# Configure logging globally
def setup_global_logging(config: AgentConfig):
    """Setup global logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format=config.log_format
    )

# ===========================
# ENHANCED CLASSES WITH CONFIG AND LOGGING
# ===========================

class DataFormatAnalyzer:
    """Intelligent analyzer for detecting and handling wide format business data"""
    
    def __init__(self, config: AgentConfig = None, logger: EnhancedLogger = None):
        self.config = config or AgentConfig()
        self.logger = logger or EnhancedLogger(self.config, "DataFormatAnalyzer")

    def detect_business_period_columns(self, df):
        """Detect business period columns (MAT, YTD, L4W, etc.)"""
        operation = "business_period_detection"
        start_time = time.time()

        try:
            self.logger.log_operation_start(operation, {'shape': df.shape if df is not None else None})

            if df is None or df.empty:
                self.logger.log_operation_warning(operation, "Empty or None DataFrame provided")
                return False, {}

            columns = df.columns.tolist()
            business_period_columns = []
            business_period_details = {}

            # Check each column against business period patterns
            for col in columns:
                col_str = str(col).strip()
                period_info = self._analyze_business_period_column(col_str)

                if period_info['is_business_period']:
                    business_period_columns.append(col)
                    business_period_details[col] = period_info

            has_business_periods = len(business_period_columns) > 0

            result = {
                'business_period_columns': business_period_columns,
                'business_period_details': business_period_details,
                'count': len(business_period_columns),
                'types_found': list(set(info['period_type'] for info in business_period_details.values()))
            }

            duration = time.time() - start_time
            self.logger.log_operation_success(
                operation,
                duration,
                {
                    'has_business_periods': has_business_periods,
                    'business_periods_found': len(business_period_columns),
                    'types_found': result['types_found']
                }
            )

            return has_business_periods, result

        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, operation)
            return False, {}

    def _analyze_business_period_column(self, col_str: str) -> dict:
        """Analyze a single column to determine if it's a business period"""
        col_lower = col_str.lower()

        # Check against business period patterns
        for pattern in self.config.business_period_patterns:
            if re.match(pattern, col_str, re.IGNORECASE):
                period_info = self._extract_period_info(col_str, pattern)
                period_info['is_business_period'] = True
                period_info['matched_pattern'] = pattern
                return period_info

        # Check for business period keywords as fallback
        if any(keyword in col_lower for keyword in self.config.business_period_keywords):
            return {
                'is_business_period': True,
                'period_type': 'keyword_match',
                'period_category': 'unknown',
                'aggregation_type': 'cumulative',
                'comparison_safe': False,
                'notes': f"Contains business period keyword: {col_str}"
            }

        return {
            'is_business_period': False,
            'period_type': None,
            'period_category': None,
            'aggregation_type': None,
            'comparison_safe': False,
            'notes': ''
        }

    def _extract_period_info(self, col_str: str, pattern: str) -> dict:
        """Extract detailed information about the business period"""
        col_lower = col_str.lower()

        # Determine period type and category
        if 'mat' in col_lower:
            return {
                'period_type': 'MAT',
                'period_category': 'cumulative_annual',
                'aggregation_type': 'cumulative',
                'comparison_safe': True,
                'time_horizon': 'annual',
                'notes': 'Moving Annual Total - cumulative value from start of year',
                'analysis_guidance': [
                    'Use for year-over-year growth analysis',
                    'Safe to compare across MAT periods',
                    'Represents cumulative performance'
                ]
            }
        elif 'ytd' in col_lower:
            return {
                'period_type': 'YTD',
                'period_category': 'cumulative_annual',
                'aggregation_type': 'cumulative',
                'comparison_safe': True,
                'time_horizon': 'annual',
                'notes': 'Year To Date - cumulative from start of current year',
                'analysis_guidance': [
                    'Compare YTD across years for growth trends',
                    'Shows progressive performance within year',
                    'Good for tracking against annual targets'
                ]
            }
        elif re.search(r'l\d+w', col_lower):
            weeks = re.search(r'l(\d+)w', col_lower).group(1)
            return {
                'period_type': f'L{weeks}W',
                'period_category': 'rolling_weeks',
                'aggregation_type': 'rolling',
                'comparison_safe': True,
                'time_horizon': f'{weeks}_weeks',
                'notes': f'Last {weeks} Weeks - rolling {weeks}-week period',
                'analysis_guidance': [
                    f'Shows {weeks}-week rolling trends',
                    'Good for short-term trend analysis',
                    'Smooths out weekly volatility'
                ]
            }
        elif re.search(r'l\d+m', col_lower):
            months = re.search(r'l(\d+)m', col_lower).group(1)
            return {
                'period_type': f'L{months}M',
                'period_category': 'rolling_months',
                'aggregation_type': 'rolling',
                'comparison_safe': True,
                'time_horizon': f'{months}_months',
                'notes': f'Last {months} Months - rolling {months}-month period',
                'analysis_guidance': [
                    f'Shows {months}-month rolling trends',
                    'Good for medium-term trend analysis',
                    'Smooths out monthly seasonality'
                ]
            }
        elif 'qtd' in col_lower:
            return {
                'period_type': 'QTD',
                'period_category': 'cumulative_quarterly',
                'aggregation_type': 'cumulative',
                'comparison_safe': True,
                'time_horizon': 'quarterly',
                'notes': 'Quarter To Date - cumulative from start of current quarter',
                'analysis_guidance': [
                    'Compare QTD across quarters for trends',
                    'Track quarterly target achievement',
                    'Shows within-quarter progression'
                ]
            }
        elif 'mtd' in col_lower:
            return {
                'period_type': 'MTD',
                'period_category': 'cumulative_monthly',
                'aggregation_type': 'cumulative',
                'comparison_safe': True,
                'time_horizon': 'monthly',
                'notes': 'Month To Date - cumulative from start of current month',
                'analysis_guidance': [
                    'Compare MTD across months for trends',
                    'Track monthly target achievement',
                    'Shows within-month progression'
                ]
            }
        elif any(word in col_lower for word in ['sply', 'prev', 'previous']):
            return {
                'period_type': 'COMPARATIVE',
                'period_category': 'comparative_period',
                'aggregation_type': 'comparative',
                'comparison_safe': True,
                'time_horizon': 'comparative',
                'notes': 'Comparative period (previous year/quarter/month)',
                'analysis_guidance': [
                    'Use for period-over-period comparisons',
                    'Calculate growth rates vs current period',
                    'Identify seasonal patterns'
                ]
            }
        else:
            return {
                'period_type': 'CUSTOM',
                'period_category': 'custom_business',
                'aggregation_type': 'unknown',
                'comparison_safe': False,
                'time_horizon': 'unknown',
                'notes': f'Custom business period: {col_str}',
                'analysis_guidance': [
                    'Verify period definition with business users',
                    'Understand aggregation method before analysis',
                    'Check if comparable across periods'
                ]
            }

    def detect_wide_format_timeseries_enhanced(self, df):
        """Enhanced wide format detection including business periods"""
        operation = "enhanced_wide_format_detection"
        start_time = time.time()

        try:
            self.logger.log_operation_start(operation, {'shape': df.shape if df is not None else None})

            if df is None or df.empty:
                self.logger.log_operation_warning(operation, "Empty or None DataFrame provided")
                return False, {}

            # Detect traditional time columns
            is_traditional_wide, traditional_info = self.detect_wide_format_timeseries(df)

            # Detect business period columns
            has_business_periods, business_period_info = self.detect_business_period_columns(df)

            # Combined analysis
            all_time_columns = traditional_info.get('time_columns', []) + business_period_info.get(
                'business_period_columns', [])
            measure_columns = traditional_info.get('measure_columns', [])

            # Determine if this is wide format (either traditional or business periods)
            is_wide_format = (
                    len(all_time_columns) >= self.config.min_time_columns_for_wide_format and
                    len(measure_columns) >= self.config.min_measure_columns_for_wide_format
            )

            result = {
                'is_wide_format': is_wide_format,
                'has_traditional_time': is_traditional_wide,
                'has_business_periods': has_business_periods,
                'time_columns': traditional_info.get('time_columns', []),
                'business_period_columns': business_period_info.get('business_period_columns', []),
                'all_time_columns': all_time_columns,
                'measure_columns': measure_columns,
                'business_period_details': business_period_info.get('business_period_details', {}),
                'business_period_types': business_period_info.get('types_found', []),
                'other_columns': [col for col in df.columns if
                                  col not in all_time_columns and col not in measure_columns]
            }

            duration = time.time() - start_time
            self.logger.log_operation_success(
                operation,
                duration,
                {
                    'is_wide_format': is_wide_format,
                    'traditional_time_cols': len(traditional_info.get('time_columns', [])),
                    'business_period_cols': len(business_period_info.get('business_period_columns', [])),
                    'business_period_types': business_period_info.get('types_found', [])
                }
            )

            return is_wide_format, result

        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, operation)
            return False, {}
    
    def detect_wide_format_timeseries(self, df):
        """Detect if DataFrame is in wide format with time-series columns"""
        operation = "wide_format_detection"
        start_time = time.time()
        
        try:
            self.logger.log_operation_start(operation, {'shape': df.shape if df is not None else None})
            
            if df is None or df.empty:
                self.logger.log_operation_warning(operation, "Empty or None DataFrame provided")
                return False, {}
            
            columns = df.columns.tolist()
            time_columns = []
            
            # Use configurable time patterns
            for col in columns:
                col_str = str(col).strip()
                for pattern in self.config.time_patterns:
                    if re.match(pattern, col_str, re.IGNORECASE):
                        time_columns.append(col)
                        break
            
            # Check if we have measure/KPI column
            measure_columns = []
            for col in columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in self.config.measure_keywords):
                    measure_columns.append(col)
            
            is_wide_format = (
                len(time_columns) >= self.config.min_time_columns_for_wide_format and 
                len(measure_columns) >= self.config.min_measure_columns_for_wide_format
            )
            
            result = {
                'time_columns': time_columns,
                'measure_columns': measure_columns,
                'time_column_count': len(time_columns),
                'measure_column_count': len(measure_columns),
                'other_columns': [col for col in columns if col not in time_columns and col not in measure_columns]
            }
            
            duration = time.time() - start_time
            self.logger.log_operation_success(
                operation, 
                duration, 
                {
                    'is_wide_format': is_wide_format,
                    'time_columns_found': len(time_columns),
                    'measure_columns_found': len(measure_columns)
                }
            )
            
            return is_wide_format, result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, operation)
            # Return safe defaults to maintain functionality
            return False, {}

    def detect_measure_types_enhanced(self, df, measure_col, value_cols):
        """Enhanced detection of measure types with detailed analysis and operation guidance"""
        operation = "enhanced_measure_type_detection"

        try:
            self.logger.log_operation_start(operation,
                                            {'measure_col': measure_col, 'value_cols_count': len(value_cols)})

            if measure_col not in df.columns:
                self.logger.log_operation_warning(operation, f"Measure column '{measure_col}' not found")
                return {}

            measure_analysis = {}

            # Import the patterns from enhanced_data_profiler
            from enhanced_data_profiler import detect_special_numeric_columns

            for measure in df[measure_col].unique():
                if pd.isna(measure):
                    continue

                # Extract data for this measure across all time periods
                measure_data = df[df[measure_col] == measure][value_cols]
                measure_values = measure_data.values.flatten()
                measure_values = measure_values[~pd.isna(measure_values)]

                if len(measure_values) == 0:
                    continue

                # Create a temporary DataFrame for this measure to leverage existing detection
                temp_df = pd.DataFrame({str(measure): measure_values})

                # Use the special numeric detection logic
                special_detection = detect_special_numeric_columns(temp_df)

                analysis = {
                    'measure_name': str(measure),
                    'value_count': len(measure_values),
                    'min_value': float(np.min(measure_values)),
                    'max_value': float(np.max(measure_values)),
                    'mean_value': float(np.mean(measure_values)),
                    'median_value': float(np.median(measure_values)),
                    'std_value': float(np.std(measure_values)),
                    'cv': float(np.std(measure_values) / np.mean(measure_values)) if np.mean(measure_values) != 0 else 0,
                    'has_negatives': bool(np.any(measure_values < 0)),
                    'has_decimals': bool(np.any(measure_values != measure_values.astype(int))),
                    'unique_value_count': len(np.unique(measure_values)),
                    'zero_count': int(np.sum(measure_values == 0)),
                    'zero_ratio': float(np.sum(measure_values == 0) / len(measure_values))
                }

                # Enhanced type detection based on multiple factors
                measure_lower = str(measure).lower()

                # Check if detected by special numeric columns logic
                detected_type = None
                safe_operations = []
                avoid_operations = []
                aggregation_method = 'mean'  # default

                # Check each category from special_detection
                for category in ['percentage_columns', 'ratio_columns', 'rate_columns',
                                 'currency_columns', 'coordinate_columns', 'year_columns',
                                 'id_like_columns', 'score_columns', 'index_columns']:
                    if special_detection.get(category):
                        col_info = special_detection[category][0]  # Get first (only) column info
                        detected_type = category.replace('_columns', '')
                        safe_operations = col_info.get('safe_operations', [])
                        avoid_operations = col_info.get('avoid_operations', [])
                        break

                # Determine aggregation method based on type
                if detected_type == 'percentage':
                    aggregation_method = 'mean'

                elif detected_type == 'ratio' or detected_type == 'rate':
                    aggregation_method = 'mean'

                elif detected_type == 'index':
                    aggregation_method = 'mean'

                elif detected_type == 'score':
                    aggregation_method = 'mean'

                elif detected_type == 'currency':
                    aggregation_method = 'sum'
                
                elif detected_type == 'count' or detected_type == 'absolute':
                    aggregation_method = 'sum'

                # Add all analysis results
                analysis.update({
                    'detected_type': detected_type,
                    'aggregation_method': aggregation_method,
                    'safe_operations': safe_operations,
                    'avoid_operations': avoid_operations,
                    'can_be_summed': 'sum' not in avoid_operations,
                    'can_be_averaged': 'simple_mean' not in avoid_operations,
                    'requires_special_handling': len(avoid_operations) > 2,
                })

                measure_analysis[measure] = analysis

            self.logger.log_operation_success(operation, details={
                'measures_analyzed': len(measure_analysis),
                'types_found': list(set(m['detected_type'] for m in measure_analysis.values()))
            })

            return measure_analysis

        except Exception as e:
            self.logger.log_error(e, operation)
            return {}


class TokenManager:
    """Manages Azure AD token lifecycle with automatic renewal"""
    
    def __init__(self, token_url, client_id, client_secret, scope, config: AgentConfig = None, logger: EnhancedLogger = None):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.token = None
        self.token_expires_at = None
        self.lock = threading.Lock()
        self.config = config or AgentConfig()
        self.logger = logger or EnhancedLogger(self.config, "TokenManager")

        # FIXED: retry state tracking
        self.last_refresh_attempt = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        
        # Initialize with first token
        self._refresh_token()
    
    def get_valid_token(self):
        """Get a valid token, refreshing if necessary"""
        with self.lock:
            # FIXED: Check for too many consecutive failures
            if self.consecutive_failures >= self.max_consecutive_failures:
                # Wait before trying again to avoid rapid retries
                if self.last_refresh_attempt:
                    time_since_last = datetime.now() - self.last_refresh_attempt
                    if time_since_last < timedelta(minutes=1):
                        self.logger.log_operation_warning("token_get",
                                                          f"Too many consecutive failures ({self.consecutive_failures}), waiting before retry")
                        raise AgentError(
                            "Token service temporarily unavailable due to repeated failures",
                            ErrorCategory.AUTHENTICATION,
                            {"consecutive_failures": self.consecutive_failures}
                        )

                # Reset failure count after waiting period
                self.consecutive_failures = 0
                
            if self._is_token_expired():
                self.logger.log_operation_start("token_refresh", {"reason": "expired_or_expiring"})
                self._refresh_token()
            return self.token
    
    def _is_token_expired(self):
        """Check if token is expired or will expire within configured buffer time"""
        if not self.token or not self.token_expires_at:
            return True
        
        # Use configurable buffer time
        buffer_time = timedelta(minutes=self.config.token_refresh_buffer_minutes)
        return datetime.now() >= (self.token_expires_at - buffer_time)
    
    def _refresh_token(self):
        """Refresh the Azure AD token"""
        operation = "token_refresh"
        start_time = time.time()
        
        try:
            response = requests.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials", 
                    "scope": self.scope
                },
                auth=(self.client_id, self.client_secret),
                timeout=self.config.max_execution_time,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                }
            )
            # FIXED: Better status code handling
            if response.status_code == 401:
                raise AgentError(
                    "Authentication failed - invalid client credentials",
                    ErrorCategory.AUTHENTICATION,
                    {"status_code": response.status_code}
                )
            elif response.status_code == 429:
                raise AgentError(
                    "Rate limited by authentication service",
                    ErrorCategory.AUTHENTICATION,
                    {"status_code": response.status_code, "retry_after": response.headers.get('Retry-After')}
                )
            elif not response.ok:
                raise AgentError(
                    f"Token request failed with status {response.status_code}: {response.text[:200]}",
                    ErrorCategory.AUTHENTICATION,
                    {"status_code": response.status_code}
                )
                        
            token_data = response.json()
            
            # FIXED: Validate token response structure
            if 'access_token' not in token_data:
                raise AgentError(
                    "Invalid token response - missing access_token",
                    ErrorCategory.AUTHENTICATION,
                    {"response_keys": list(token_data.keys())}
                )
            
            self.token = token_data['access_token']
            
            # Calculate expiration time (default to 1 hour if not provided)
            expires_in = token_data.get('expires_in', 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            # FIXED: Reset failure count on success
            self.consecutive_failures = 0
            
            duration = time.time() - start_time
            self.logger.log_operation_success(
                operation, 
                duration, 
                {"expires_at": self.token_expires_at.isoformat()}
            )

        except requests.exceptions.Timeout as e:
            duration = time.time() - start_time
            self.consecutive_failures += 1
            raise AgentError(
                f"Token refresh timeout after {duration:.1f}s",
                ErrorCategory.NETWORK,
                {"duration": duration, "consecutive_failures": self.consecutive_failures},
                e
            )
        except requests.exceptions.ConnectionError as e:
            duration = time.time() - start_time
            self.consecutive_failures += 1
            raise AgentError(
                f"Token refresh connection failed: {e}",
                ErrorCategory.NETWORK,
                {"duration": duration, "consecutive_failures": self.consecutive_failures},
                e
            )
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            self.consecutive_failures += 1
            raise AgentError(
                f"Token refresh failed: {e}",
                ErrorCategory.AUTHENTICATION,
                {"duration": duration, "consecutive_failures": self.consecutive_failures},
                e
            )
        except KeyError as e:
            self.consecutive_failures += 1
            raise AgentError(
                f"Invalid token response format: {e}",
                ErrorCategory.AUTHENTICATION,
                {"response_keys": list(token_data.keys()) if 'token_data' in locals() else [],
                 "consecutive_failures": self.consecutive_failures},
                e
            )
        except Exception as e:
            self.consecutive_failures += 1
            raise AgentError(
                f"Unexpected error during token refresh: {e}",
                ErrorCategory.SYSTEM,
                {"consecutive_failures": self.consecutive_failures},
                e
            )
    
    def force_refresh(self):
        """Force refresh the token (useful for auth failures)"""
        with self.lock:
            self.logger.log_operation_start("force_token_refresh", {"reason": "authentication_failure"})
            
            # FIXED: Clear current token to force refresh
            self.token = None
            self.token_expires_at = None

            try:
                self._refresh_token()
                return self.token
            except Exception as e:
                self.logger.log_error(e, "force_token_refresh")
                # Don't re-raise here, let the caller handle it
                raise

    def is_healthy(self) -> bool:
        """Check if the token manager is in a healthy state"""
        return (
                self.consecutive_failures < self.max_consecutive_failures and
                self.token is not None and
                not self._is_token_expired()
        )

class AutoRefreshHTTPClient:
    """HTTP client that automatically refreshes tokens on auth failures"""
    
    def __init__(self, token_manager: TokenManager, subscription_key: str, llm_endpoint: str, config: AgentConfig = None, logger: EnhancedLogger = None):
        self.token_manager = token_manager
        self.subscription_key = subscription_key
        self.llm_endpoint = llm_endpoint
        self.config = config or AgentConfig()
        self.logger = logger or EnhancedLogger(self.config, "HTTPClient")
        self._client = None
        self._create_client()
    
    def _create_client(self):
        """Create HTTP client with current token"""
        operation = "http_client_creation"
        
        try:
            self.logger.log_operation_start(operation)
            
            token = self.token_manager.get_valid_token()
            
            def update_request_with_fresh_token(request: httpx.Request) -> None:
                """Update request with fresh token and correct endpoint"""
                # Update endpoint
                if "/chat/completions" in request.url.path:
                    request.url = request.url.copy_with(path=self.llm_endpoint)
                
                # Update token (get fresh token for each request)
                fresh_token = self.token_manager.get_valid_token()
                request.headers['Authorization'] = f"Bearer {fresh_token}"
                request.headers['Ocp-Apim-Subscription-Key'] = self.subscription_key
            
            def handle_auth_failure(response: httpx.Response) -> None:
                """Handle authentication failures by refreshing token"""
                if response.status_code == 401:
                    self.logger.log_operation_warning(operation, "Received 401, attempting token refresh")
                    try:
                        self.token_manager.force_refresh()
                    except Exception as e:
                        self.logger.log_error(e, "token_refresh_after_401")
            
            self._client = httpx.Client(
                event_hooks={
                    "request": [update_request_with_fresh_token],
                    "response": [handle_auth_failure],
                },
                timeout=float(self.config.max_execution_time)
            )
            
            self.logger.log_operation_success(operation)
            
        except Exception as e:
            raise AgentError(
                f"Failed to create HTTP client: {e}",
                ErrorCategory.NETWORK,
                original_error=e
            )
    
    def get_client(self):
        """Get the HTTP client"""
        return self._client
    
    def close(self):
        """Close the HTTP client"""
        if self._client:
            try:
                self._client.close()
                self.logger.log_operation_success("http_client_close")
            except Exception as e:
                self.logger.log_error(e, "http_client_close")

# Global instances with configuration
_global_config = None
_global_logger = None
token_manager = None
http_client = None

def get_global_config() -> AgentConfig:
    """Get or create global configuration"""
    global _global_config
    if _global_config is None:
        _global_config = AgentConfig.from_env()
        setup_global_logging(_global_config)
    return _global_config

def get_global_logger() -> EnhancedLogger:
    """Get or create global logger"""
    global _global_logger
    if _global_logger is None:
        config = get_global_config()
        _global_logger = EnhancedLogger(config)
    return _global_logger

# Configuration constants
scope = os.getenv('SCOPE')
spn = os.getenv('SPN')
client_id = os.getenv('AZURE_CLIENT_ID')
client_secret_value = os.getenv('AZURE_CLIENT_SECRET')
subs_key = os.getenv('SUBSCRIPTION_KEY')
oauth_token_url = os.getenv('TOKEN_URL')
openai_base_url = os.getenv('OPENAI_BASE_URL')
model_version = os.getenv('MODEL_VERSION')

def initialize_token_manager():
    """Initialize the global token manager"""
    global token_manager
    
    if token_manager is None:
        config = get_global_config()
        logger = get_global_logger()
        
        token_url = os.getenv('TOKEN_URL')
        token_manager = TokenManager(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret_value,
            scope=scope,
            config=config,
            logger=logger
        )
        logger.log_operation_success("token_manager_initialization")
    
    return token_manager

# new create_langchain_llm_with_auto_refresh function
def create_langchain_llm_with_auto_refresh(endpoint_path="/openai4/az_openai_gpt-4o_chat"):
    """Create LangChain LLM with automatic token refresh capability

    This function now supports multiple LLM providers through the universal factory,
    with fallback to the original Azure OpenAI implementation.
    """
    global http_client

    config = get_global_config()
    logger = get_global_logger()
    operation = "llm_creation"

    try:
        logger.log_operation_start(operation, {"endpoint": endpoint_path})

        # ATTEMPT 1: Try using the universal LLM factory if available
        if UNIVERSAL_LLM_AVAILABLE:
            try:
                # Check which provider is configured
                llm_config = LLMConfig.from_env()
                provider_name = llm_config.provider.value

                logger.log_operation_start(f"{operation}_universal", {
                    "provider": provider_name,
                    "model": llm_config.model_name
                })

                # For Azure OpenAI, pass the token manager and http client
                if llm_config.provider == LLMProvider.AZURE_OPENAI:
                    # Initialize token manager if needed
                    tm = initialize_token_manager()

                    # Create or get http_client
                    if http_client is None or (hasattr(http_client, '_client') and http_client._client.is_closed):
                        if http_client:
                            http_client.close()

                        http_client = AutoRefreshHTTPClient(
                            token_manager=tm,
                            subscription_key=subs_key,
                            llm_endpoint=endpoint_path,
                            config=config,
                            logger=logger
                        )

                    # Use universal factory with Azure-specific components
                    llm = create_universal_llm(endpoint_path)
                else:
                    # For other providers (Claude, OpenAI, Gemini)
                    llm = create_universal_llm(endpoint_path)

                # Validate that we got a valid LLM
                if llm is None:
                    raise ValueError(f"Universal LLM factory returned None for provider {provider_name}")

                # Test the LLM with a simple ping to ensure it's working
                try:
                    from langchain.schema import SystemMessage, HumanMessage

                    # Use HumanMessage for Claude compatibility, SystemMessage for others
                    try:
                        # Try to detect if this is Claude by checking the provider
                        llm_config = LLMConfig.from_env()
                        if llm_config.provider == LLMProvider.CLAUDE:
                            test_response = llm.invoke([HumanMessage(content="Respond with 'OK'")])
                        else:
                            test_response = llm.invoke([SystemMessage(content="ping")])
                    except Exception:
                        # Fallback: always use HumanMessage if provider detection fails
                        test_response = llm.invoke([HumanMessage(content="Respond with 'OK'")])

                    if test_response is None:
                        raise ValueError("LLM test invocation returned None")
                except Exception as test_error:
                    print(f"LLM test invocation failed: {test_error}")
                    # Don't fail here - let it fail later if there's really an issue

                logger.log_operation_success(operation, details={
                    "method": "universal",
                    "provider": provider_name
                })

                return llm

            except Exception as universal_error:
                print(f"Universal LLM creation failed, will try fallback: {universal_error}")
                # Continue to fallback

        # ATTEMPT 2: Fallback to original Azure OpenAI implementation
        print("Using original Azure OpenAI implementation (fallback)")

        # Initialize token manager
        tm = initialize_token_manager()

        # Create auto-refresh HTTP client
        if http_client:
            try:
                http_client.close()  # Close previous client if exists
            except:
                pass

        http_client = AutoRefreshHTTPClient(
            token_manager=tm,
            subscription_key=subs_key,
            llm_endpoint=endpoint_path,
            config=config,
            logger=logger
        )

        # Set OpenAI base URL
        openai.api_base = openai_base_url

        # Get initial token
        initial_token = tm.get_valid_token()

        # Create LangChain LLM with auto-refresh HTTP client
        llm = AzureChatOpenAI(
            azure_endpoint=f"{openai.api_base}{endpoint_path}",
            temperature=config.llm_temperature,
            api_version=model_version,
            azure_ad_token=initial_token,
            default_headers={
                'Ocp-Apim-Subscription-Key': subs_key,
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {initial_token}"
            },
            http_client=http_client.get_client(),
            max_retries=config.llm_max_retries,
        )

        logger.log_operation_success(operation, details={
            "method": "fallback_azure",
            "provider": "azure_openai"
        })

        return llm

    except Exception as e:
        logger.log_error(e, operation)
        raise AgentError(
            f"Failed to create LLM: {e}",
            ErrorCategory.SYSTEM,
            {"endpoint": endpoint_path, "method": "all_attempts_failed"},
            e
        )

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: List[str]
    data_context: str
    user_query: str
    generated_code: str
    execution_result: str
    error: str
    raw_exception: str  # Store the raw exception for error correction
    plot_data: Dict[str, Any]
    result_type: str
    needs_multiple_sheets: bool
    current_data: Any
    worksheet_data: Dict[str, Any]
    merge_worksheets_func: Any
    # Add retry tracking
    retry_count: int
    correction_history: List[str]  # Track what corrections were attempted
    original_error: str 
    # Wide format data intelligence
    data_format_info: Dict[str, Any]
    measure_analysis: Dict[str, Any]
    transformation_suggestions: Dict[str, Any]
    # NEW: Commentary and insights
    commentary: str
    insights: List[str]

class DataAnalysisAgent:
    """Enhanced AI Agent for data analysis with wide format intelligence, auto-refresh tokens, and automated commentary generation"""
    
    def __init__(self, config: AgentConfig = None):
        # Initialize configuration and logging
        self.config = config or get_global_config()
        self.logger = get_global_logger()
        self.error_handler = ErrorHandler(self.config, self.logger)
        
        # Log initialization
        self.logger.log_operation_start("agent_initialization", {
            "sklearn_available": SKLEARN_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
            "statsmodels_available": STATSMODELS_AVAILABLE,
            "matplotlib_available": MATPLOTLIB_AVAILABLE,
            "commentary_enabled": self.config.enable_commentary
        })
        
        # Library availability
        self.sklearn_available = SKLEARN_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        self.statsmodels_available = STATSMODELS_AVAILABLE
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        self.advanced_ml_available = ADVANCED_ML_AVAILABLE
        self.prophet_available = PROPHET_AVAILABLE
        self.anomaly_detection_available = ANOMALY_DETECTION_AVAILABLE
        self.explainability_available = EXPLAINABILITY_AVAILABLE
        self.optuna_available = OPTUNA_AVAILABLE
        self.pycaret_available = PYCARET_AVAILABLE

        # Initialize data format analyzers with configuration
        self.format_analyzer = DataFormatAnalyzer(self.config, self.logger)

        # Create LLM with auto-refresh capability
        self.llm = create_langchain_llm_with_auto_refresh()
        self.graph = self._create_graph()
        
        self.logger.log_operation_success("agent_initialization", details={
            "config_source": "environment" if config is None else "provided",
            "wide_format_detection": True,
            "commentary_generation": self.config.enable_commentary
        })

    def refresh_llm_connection(self):
        """Manually refresh the LLM connection (useful for long-running agents)"""
        global token_manager, http_client
        operation = "llm_refresh"
        try:
            self.logger.log_operation_start(operation)

            # Check if we're using universal LLM
            if UNIVERSAL_LLM_AVAILABLE:
                try:
                    llm_config = LLMConfig.from_env()
                    provider_name = llm_config.provider.value

                    # For Azure OpenAI, force refresh the token
                    if llm_config.provider == LLMProvider.AZURE_OPENAI:
                        if token_manager:
                            token_manager.force_refresh()
                            self.logger.log_operation_success("token_refresh")

                    # For all providers, recreate the LLM
                    self.llm = create_langchain_llm_with_auto_refresh()

                    self.logger.log_operation_success(operation, details={
                        "provider": provider_name,
                        "method": "universal"
                    })
                    return

                except Exception as universal_error:
                    self.logger.log_operation_warning(operation, f"Universal LLM refresh failed, trying fallback: {universal_error}")

            # Fallback: Original Azure OpenAI refresh
            if token_manager:
                token_manager.force_refresh()

            self.llm = create_langchain_llm_with_auto_refresh()
            self.logger.log_operation_success(operation, details={
                "method": "fallback_azure"
            })

        except Exception as e:
            self.logger.log_error(e, operation)
            raise AgentError(
                f"Failed to refresh LLM connection: {e}",
                ErrorCategory.SYSTEM,
                {"operation": operation},
                e
            )

    def _analyze_data_format(self, state: AgentState) -> AgentState:
            """Enhanced data format analysis with comprehensive period detection"""
            operation = "enhanced_data_format_analysis"
            start_time = time.time()

            try:
                current_data = state.get('current_data')

                if current_data is None or current_data.empty:
                    self.logger.log_operation_warning(operation, "No data available for format analysis")
                    return state

                self.logger.log_operation_start(operation, {"data_shape": current_data.shape})

                # Use enhanced detection method that detects both types
                is_wide_format, format_info = self.format_analyzer.detect_wide_format_timeseries_enhanced(current_data)

                state['data_format_info'] = format_info
                state['data_format_info']['is_wide_format'] = is_wide_format

                # Analyze measure types if wide format detected
                if is_wide_format and format_info['measure_columns']:
                    # Use ALL time columns (traditional + business periods) for measure analysis
                    all_time_cols = format_info['all_time_columns']
                    measure_col = format_info['measure_columns'][0]

                    if all_time_cols:
                        measure_analysis = self.format_analyzer.detect_measure_types_enhanced(
                            current_data, measure_col, all_time_cols
                        )
                        state['measure_analysis'] = measure_analysis

                duration = time.time() - start_time

                # Enhanced logging with both period types
                log_details = {
                    "is_wide_format": is_wide_format,
                    "traditional_time_columns": len(format_info.get('time_columns', [])),
                    "business_period_columns": len(format_info.get('business_period_columns', [])),
                    "total_time_columns": len(format_info.get('all_time_columns', [])),
                    "business_period_types": format_info.get('business_period_types', []),
                    "measures_analyzed": len(state.get('measure_analysis', {})),
                    "has_mixed_periods": (len(format_info.get('time_columns', [])) > 0 and
                                          len(format_info.get('business_period_columns', [])) > 0)
                }

                self.logger.log_operation_success(operation, duration, log_details)

                return state

            except Exception as e:
                error_response = self.error_handler.handle_error(e, operation)
                self.logger.log_error(e, operation)
                return state

    def _format_business_period_context(self, data_format_info, measure_analysis):
        """Enhanced context formatting with business period intelligence"""

        context_parts = []
        context_parts.append("\n" + "=" * 70)
        context_parts.append("ENHANCED WIDE FORMAT WITH BUSINESS PERIODS INTELLIGENCE")
        context_parts.append("=" * 70)

        # Traditional time columns
        time_cols = data_format_info.get('time_columns', [])
        if time_cols:
            context_parts.append(f"\nTRADITIONAL TIME COLUMNS ({len(time_cols)}):")
            display_time_cols = time_cols[:10]
            time_cols_suffix = f" ... and {len(time_cols) - 10} more" if len(time_cols) > 10 else ""
            context_parts.append(f"  {', '.join(display_time_cols)}{time_cols_suffix}")

        # Business period columns
        business_period_cols = data_format_info.get('business_period_columns', [])
        business_period_details = data_format_info.get('business_period_details', {})

        if business_period_cols:
            context_parts.append(f"\nBUSINESS PERIOD COLUMNS ({len(business_period_cols)}):")

            # Group by period type
            periods_by_type = {}
            for col in business_period_cols:
                if col in business_period_details:
                    period_type = business_period_details[col]['period_type']
                    if period_type not in periods_by_type:
                        periods_by_type[period_type] = []
                    periods_by_type[period_type].append((col, business_period_details[col]))

            for period_type, periods in periods_by_type.items():
                context_parts.append(f"\n  {period_type} PERIODS:")
                for col, details in periods:
                    context_parts.append(f"    • {col}: {details['notes']}")
                    if details.get('analysis_guidance'):
                        context_parts.append(f"      - {details['analysis_guidance'][0]}")

        # Measure columns
        measure_cols = data_format_info.get('measure_columns', [])
        context_parts.append(f"\nMEASURE COLUMNS: {', '.join(measure_cols)}")

        # Analysis patterns for business periods
        context_parts.append(f"\nBUSINESS PERIOD ANALYSIS PATTERNS:")

        context_parts.append("\n1. CUMULATIVE PERIODS (MAT, YTD, QTD, MTD):")
        context_parts.append("   # These represent cumulative values - safe to compare")
        context_parts.append("   # Calculate growth rates")
        if measure_cols:
            context_parts.append(
                f"   current_ytd = current_data[current_data['{measure_cols[0]}'] == 'Revenue']['YTD'].values[0]")
            context_parts.append(
                f"   previous_ytd = current_data[current_data['{measure_cols[0]}'] == 'Revenue']['YTD-1'].values[0]")
        context_parts.append("   ytd_growth = ((current_ytd - previous_ytd) / previous_ytd) * 100")

        context_parts.append("\n2. ROLLING PERIODS (L4W, L12W, L3M, etc.):")
        context_parts.append("   # These show rolling averages - good for trend analysis")
        context_parts.append("   # Transpose for time series analysis")
        context_parts.append(
            "   rolling_cols = [col for col in current_data.columns if 'L' in str(col) and 'W' in str(col)]")
        if measure_cols:
            context_parts.append(f"   rolling_data = current_data.set_index('{measure_cols[0]}')[rolling_cols].T")

        context_parts.append("\n3. COMPARATIVE ANALYSIS:")
        context_parts.append("   # Mix different period types for comprehensive analysis")
        context_parts.append("   # Compare cumulative vs rolling vs traditional periods")
        if measure_cols:
            context_parts.append("   comparison_df = pd.DataFrame({")
            context_parts.append(
                f"       'Current_YTD': current_data[current_data['{measure_cols[0]}'] == 'Revenue']['YTD'],")
            context_parts.append(
                f"       'Previous_YTD': current_data[current_data['{measure_cols[0]}'] == 'Revenue']['YTD-1'],")
            context_parts.append(
                f"       'Rolling_12W': current_data[current_data['{measure_cols[0]}'] == 'Revenue']['L12W']")
            context_parts.append("   })")

        # Critical warnings for business periods
        context_parts.append(f"\nCRITICAL WARNINGS FOR BUSINESS PERIODS:")
        context_parts.append("1. MAT/YTD are CUMULATIVE - do not sum across measures")
        context_parts.append("2. Rolling periods (L4W, L12W) show averages/totals over time windows")
        context_parts.append("3. Different period types have different aggregation rules")
        context_parts.append("4. Always check period definitions before comparing")
        context_parts.append("5. Use growth rates for period-over-period comparisons")
        context_parts.append("6. Business periods may not align with calendar periods")
        context_parts.append("7. DO NOT alter or sanitize period column names - use exactly as they appear")

        return "\n".join(context_parts)

    def _format_wide_format_context(self, data_format_info, measure_analysis):
        """Enhanced context formatting that handles both traditional and business periods"""

        # Get information about both period types
        has_traditional_time = data_format_info.get('has_traditional_time', False)
        has_business_periods = data_format_info.get('has_business_periods', False)
        time_cols = data_format_info.get('time_columns', [])
        business_period_cols = data_format_info.get('business_period_columns', [])
        business_period_details = data_format_info.get('business_period_details', {})
        measure_cols = data_format_info.get('measure_columns', [])

        context_parts = []
        context_parts.append("\n" + "=" * 70)

        # Dynamic header based on what's detected
        if has_traditional_time and has_business_periods:
            context_parts.append("MIXED PERIOD WIDE FORMAT DATA INTELLIGENCE")
            context_parts.append("(Traditional Time Periods + Business Periods)")
        elif has_business_periods:
            context_parts.append("BUSINESS PERIODS WIDE FORMAT DATA INTELLIGENCE")
        else:
            context_parts.append("WIDE FORMAT TIME-SERIES DATA INTELLIGENCE")

        context_parts.append("=" * 70)

        # Section 1: Traditional Time Columns (if present)
        if has_traditional_time and time_cols:
            context_parts.append(f"\nTRADITIONAL TIME COLUMNS ({len(time_cols)}):")
            display_time_cols = time_cols[:15]
            time_cols_suffix = f" ... and {len(time_cols) - 15} more" if len(time_cols) > 15 else ""
            context_parts.append(f"  Examples: {', '.join(display_time_cols)}{time_cols_suffix}")
            context_parts.append(f"  • Calendar-based periods (quarters, months, years)")
            context_parts.append(f"  • Sequential time progression")
            context_parts.append(f"  • Good for: trend analysis, seasonality, forecasting")

        # Section 2: Business Period Columns (if present)
        if has_business_periods and business_period_cols:
            context_parts.append(f"\nBUSINESS PERIOD COLUMNS ({len(business_period_cols)}):")

            # Group business periods by type for better organization
            periods_by_category = {}
            for col in business_period_cols:
                if col in business_period_details:
                    category = business_period_details[col]['period_category']
                    period_type = business_period_details[col]['period_type']
                    if category not in periods_by_category:
                        periods_by_category[category] = []
                    periods_by_category[category].append((col, period_type, business_period_details[col]))

            # Display each category
            for category, periods in periods_by_category.items():
                category_display = category.replace('_', ' ').title()
                context_parts.append(f"\n  {category_display}:")
                for col, period_type, details in periods[:5]:  # Limit display
                    context_parts.append(f"    • {col} ({period_type}): {details['notes']}")
                if len(periods) > 5:
                    context_parts.append(f"    ... and {len(periods) - 5} more {category_display.lower()} columns")

        # Section 3: Measure Columns
        if measure_cols:
            context_parts.append(f"\nMEASURE COLUMNS: {', '.join(measure_cols)}")
            context_parts.append(f"Total measures analyzed: {len(measure_analysis)}")

        # Section 4: Analysis Patterns Based on What's Available
        context_parts.append(f"\nANALYSIS PATTERNS FOR YOUR DATA:")

        if has_traditional_time and has_business_periods:
            # Mixed period analysis patterns
            context_parts.append("\n1. MIXED PERIOD ANALYSIS (Traditional + Business Periods):")
            context_parts.append("   # Separate analysis by period type first, then compare insights")

            context_parts.append("\n   A) Traditional Time Series Analysis:")
            if measure_cols:
                context_parts.append(f"   traditional_cols = {time_cols[:5]}")  # Show first 5 as example
                context_parts.append(
                    f"   traditional_data = current_data.set_index('{measure_cols[0]}')[traditional_cols].T")
                context_parts.append("   # Good for: trend analysis, seasonality, forecasting")

            context_parts.append("\n   B) Business Period Analysis:")
            if business_period_cols and measure_cols:
                context_parts.append(f"   business_cols = {business_period_cols[:3]}")  # Show first 3 as example
                context_parts.append(f"   business_data = current_data.set_index('{measure_cols[0]}')[business_cols]")
                context_parts.append("   # Good for: performance tracking, target achievement, growth rates")

            context_parts.append("\n   C) Cross-Period Insights:")
            context_parts.append("   # Compare insights from both analyses")
            context_parts.append("   # Example: How does Q1 2024 performance compare to YTD and MAT?")
            if measure_cols:
                context_parts.append(
                    f"   revenue_q1 = current_data[current_data['{measure_cols[0]}'] == 'Revenue']['Q1 2024'].values[0]")
                context_parts.append(
                    f"   revenue_ytd = current_data[current_data['{measure_cols[0]}'] == 'Revenue']['YTD'].values[0]")
                context_parts.append("   # Analyze relationship between quarterly and cumulative performance")

        elif has_business_periods:
            # Business periods only
            context_parts.append("\n1. BUSINESS PERIOD ANALYSIS:")

            # Cumulative periods
            cumulative_cols = [col for col in business_period_cols
                               if col in business_period_details and
                               business_period_details[col]['aggregation_type'] == 'cumulative']
            if cumulative_cols:
                context_parts.append("\n   A) Cumulative Periods (MAT, YTD, QTD, MTD):")
                context_parts.append("   # Calculate growth rates between periods")
                if measure_cols:
                    context_parts.append(
                        f"   current_val = current_data[current_data['{measure_cols[0]}'] == 'Revenue']['YTD'].values[0]")
                    context_parts.append(
                        f"   previous_val = current_data[current_data['{measure_cols[0]}'] == 'Revenue']['YTD-1'].values[0]")
                    context_parts.append("   growth_rate = ((current_val - previous_val) / previous_val) * 100")

            # Rolling periods
            rolling_cols = [col for col in business_period_cols
                            if col in business_period_details and
                            business_period_details[col]['aggregation_type'] == 'rolling']
            if rolling_cols:
                context_parts.append("\n   B) Rolling Periods (L4W, L12W, L3M, etc.):")
                context_parts.append("   # Trend analysis over rolling windows")
                if measure_cols:
                    context_parts.append(
                        f"   rolling_data = current_data.set_index('{measure_cols[0]}')[{rolling_cols[:3]}].T")
                    context_parts.append("   # Shows smoothed trends without seasonality")

        else:
            # Traditional time only (existing logic)
            context_parts.append("\n1. TRADITIONAL TIME SERIES ANALYSIS:")
            if measure_cols and time_cols:
                context_parts.append(
                    f"   time_series_data = current_data.set_index('{measure_cols[0]}')[time_columns].T")
                context_parts.append("   # Transpose for time series analysis")
                context_parts.append("   correlation_matrix = time_series_data.corr()")

        # Section 5: Column Selection Patterns
        context_parts.append(f"\n2. SMART COLUMN SELECTION:")

        if has_traditional_time and has_business_periods:
            context_parts.append("   # MIXED PERIOD DATASETS - Be strategic about column selection")
            context_parts.append("   # Don't mix traditional time and business periods in same analysis")

            context_parts.append("\n   # For trend analysis - use traditional time columns:")
            context_parts.append(f"   trend_columns = {time_cols[:5] if time_cols else []}")

            context_parts.append("\n   # For performance tracking - use business periods:")
            context_parts.append(f"   performance_columns = {business_period_cols[:5] if business_period_cols else []}")

            context_parts.append("\n   # For correlation analysis - standardize first, then choose one type:")
            if measure_cols:
                context_parts.append(f"   # Option A: Traditional periods only")
                context_parts.append(
                    f"   traditional_corr = current_data.set_index('{measure_cols[0]}')[trend_columns].T.corr()")
                context_parts.append(f"   # Option B: Business periods only")
                context_parts.append(
                    f"   business_corr = current_data.set_index('{measure_cols[0]}')[performance_columns].T.corr()")

        elif has_business_periods:
            context_parts.append("   # BUSINESS PERIOD DATASETS - Group by period type")
            cumulative_examples = [col for col in business_period_cols[:3]
                                   if col in business_period_details and
                                   'cumulative' in business_period_details[col]['aggregation_type']]
            rolling_examples = [col for col in business_period_cols[:3]
                                if col in business_period_details and
                                'rolling' in business_period_details[col]['aggregation_type']]

            if cumulative_examples:
                context_parts.append(f"   cumulative_cols = {cumulative_examples}  # MAT, YTD, QTD, MTD")
            if rolling_examples:
                context_parts.append(f"   rolling_cols = {rolling_examples}  # L4W, L12W, L3M")

        # Section 6: Critical Warnings (Enhanced for Mixed Datasets)
        context_parts.append(f"\nCRITICAL WARNINGS:")

        if has_traditional_time and has_business_periods:
            context_parts.append("1. NEVER mix traditional time periods and business periods in same calculation")
            context_parts.append(
                "2. Traditional periods (Q1, Q2) and business periods (YTD, MAT) have different meanings")
            context_parts.append("3. Choose analysis approach based on business question:")
            context_parts.append("   - Trend/seasonality analysis → use traditional time columns")
            context_parts.append("   - Performance tracking → use business period columns")
            context_parts.append("   - Growth analysis → compare same period types only")
            context_parts.append("4. When correlating measures, use consistent period types")
        else:
            context_parts.append("1. NEVER aggregate across all measures without filtering - they have different units")
            context_parts.append("2. ALWAYS filter by measure before mathematical operations")
            if has_business_periods:
                context_parts.append("3. Business periods have specific aggregation rules - respect cumulative vs rolling")
                context_parts.append("4. Use growth rates for business period comparisons")

        context_parts.append("5. DO NOT alter or sanitize column names - use exactly as they appear")
        context_parts.append("6. For correlation analysis, standardize measures separately first")
        context_parts.append("7. Handle missing values appropriately for each period type")

        # Section 7: Recommended Analysis Approaches
        if has_traditional_time and has_business_periods:
            context_parts.append(f"\nRECOMMENDED ANALYSIS APPROACHES:")
            context_parts.append("1. SEPARATE ANALYSIS: Analyze traditional and business periods separately")
            context_parts.append("2. COMPLEMENTARY INSIGHTS: Use both analyses to answer different business questions")
            context_parts.append("3. MEASURE-SPECIFIC: Some measures work better with traditional time, others with business periods")
            context_parts.append("4. STAKEHOLDER-DRIVEN: Choose approach based on what business users need to know")

        return "\n".join(context_parts)

    def _generate_commentary(self, state: AgentState) -> AgentState:
        """NEW: Generate brief commentary and insights for the analysis results"""
        operation = "commentary_generation"
        start_time = time.time()
        
        # Skip commentary generation if disabled or if there was an error
        if not self.config.enable_commentary or state.get('error'):
            state['commentary'] = ""
            state['insights'] = []
            return state
        
        try:
            self.logger.log_operation_start(operation, {
                "result_type": state.get('result_type', 'unknown'),
                "has_plot": bool(state.get('plot_data', {})),
                "query_preview": state['user_query'][:50]
            })
            
            # Create commentary prompt
            commentary_prompt = self._create_commentary_prompt(state)

            # FIXED: Use the existing LLM instance instead of creating a new one
            # Store original temperature and restore it after commentary generation
            original_temperature = self.llm.temperature

            try:
                # Temporarily set temperature for commentary generation
                self.llm.temperature = self.config.commentary_temperature

                # Use the existing LLM with retry logic instead of creating new instance
                response = self._invoke_llm_with_retry([SystemMessage(content=commentary_prompt)])
                commentary_content = response.content

            finally:
                # Always restore original temperature
                self.llm.temperature = original_temperature
            
            response = self._invoke_llm_with_retry([SystemMessage(content=commentary_prompt)])
            commentary_content = response.content
            
            # Parse the response to extract commentary and insights
            commentary, insights = self._parse_commentary_response(commentary_content)

            print("commentary: \n", commentary)
            print("\ninsights: \n", insights)
            
            state['commentary'] = commentary
            state['insights'] = insights
            
            duration = time.time() - start_time
            self.logger.log_operation_success(operation, duration, {
                "commentary_length": len(commentary),
                "insights_count": len(insights)
            })
            
        except Exception as e:
            self.logger.log_error(e, operation)
            # Set default values on error
            state['commentary'] = ""
            state['insights'] = []
        
        return state
    
    def _create_commentary_prompt(self, state: AgentState) -> str:
        """Create a prompt for generating commentary and insights"""
        
        # Get result information
        result_type = state.get('result_type', 'unknown')
        user_query = state['user_query']
        execution_result = state.get('execution_result', '')
        has_plot = bool(state.get('plot_data', {}))
        
        # Get data format information for context
        data_format_info = state.get('data_format_info', {})
        measure_analysis = state.get('measure_analysis', {})
        is_wide_format = data_format_info.get('is_wide_format', False)
        has_business_periods = data_format_info.get('has_business_periods', False)
        
        # Create context about the data format
        data_context = ""
        if is_wide_format:
            if has_business_periods:
                data_context = "This analysis uses wide format data with business periods (YTD, MAT, rolling periods, etc.)."
            else:
                data_context = "This analysis uses wide format time-series data with traditional time periods."
        
        # Determine result summary for context
        result_summary = ""
        if isinstance(execution_result, list) and len(execution_result) > 0:
            if isinstance(execution_result[0], dict):
                result_summary = f"Table with {len(execution_result)} rows and {len(execution_result[0])} columns"
            else:
                result_summary = f"List with {len(execution_result)} items"
        elif isinstance(execution_result, dict):
            result_summary = f"Dictionary with {len(execution_result)} key-value pairs"
        elif isinstance(execution_result, str):
            if len(execution_result) < 100:
                result_summary = f"Single value: {execution_result}"
            else:
                result_summary = "Text result"
        else:
            result_summary = f"Result of type: {type(execution_result).__name__}"
        
        prompt = f"""
Generate brief commentary and insights for a data analysis result.

USER QUERY: {user_query}

ANALYSIS CONTEXT:
- Result Type: {result_type}
- Has Visualization: {has_plot}
- Result Summary: {result_summary}
- Data Context: {data_context}

TASK:
Provide a brief commentary explaining what the user is seeing, followed by 3-4 key insights.

GUIDELINES:
1. Keep commentary to 1-2 sentences maximum (under {self.config.max_commentary_length} characters)
2. Focus on what the user is seeing, not how it was calculated
3. Use plain language, avoid technical jargon
4. Be specific about the data when possible
5. For charts, explain what the visualization shows
6. For tables, summarize key patterns or findings
7. For single values, provide context about what the number means

RESPONSE FORMAT:
COMMENTARY: [Brief 1-2 sentence explanation of what the user is seeing]

INSIGHTS:
- [Insight 1: Key finding or pattern]
- [Insight 2: Notable observation or implication]
- [Insight 3: Actionable takeaways or business implication. Break this down into multiple lines if needed.] (optional)
- [Insight 4: Any other insight]

Focus on actionable observations, hidden connections & interesting facts & factoids pertinent to the data & output. Do not make generic statements. 
"""

        return prompt
    
    def _parse_commentary_response(self, response_content: str) -> tuple[str, List[str]]:
        """Parse the LLM response to extract commentary and insights"""
        try:
            lines = response_content.strip().split('\n')
            
            commentary = ""
            insights = []
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.upper().startswith('COMMENTARY:'):
                    current_section = 'commentary'
                    commentary_text = line.replace('COMMENTARY:', '').strip()
                    if commentary_text:
                        commentary = commentary_text
                elif line.upper().startswith('INSIGHTS:'):
                    current_section = 'insights'
                elif current_section == 'commentary' and not commentary:
                    commentary = line
                elif current_section == 'insights' and line.startswith('-'):
                    insight = line.replace('-', '').strip()
                    if insight and len(insights) < self.config.max_insights_count:
                        insights.append(insight)
                elif current_section == 'insights' and line.startswith('•'):
                    insight = line.replace('•', '').strip()
                    if insight and len(insights) < self.config.max_insights_count:
                        insights.append(insight)
            
            # Fallback parsing if structured format not found
            if not commentary and not insights:
                # Try to extract first sentence as commentary
                sentences = response_content.split('.')
                if sentences:
                    commentary = sentences[0].strip() + '.'
                
                # Try to extract bullet points as insights
                lines = response_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•'):
                        insight = line.replace('-', '').replace('•', '').strip()
                        if insight and len(insights) < self.config.max_insights_count:
                            insights.append(insight)
            
            # Ensure commentary is within length limit
            if len(commentary) > self.config.max_commentary_length:
                commentary = commentary[:self.config.max_commentary_length-3] + "..."
            
            return commentary, insights
            
        except Exception as e:
            self.logger.log_error(e, "commentary_parsing")
            return "", []

    def _create_graph(self) -> CompiledStateGraph:
        """Create the LangGraph workflow with LLM-based error correction, data format intelligence, and commentary generation"""
        operation = "workflow_creation"
        
        try:
            self.logger.log_operation_start(operation)
            
            workflow = StateGraph(AgentState)

            # Add nodes
            workflow.add_node("analyze_data_format", self._analyze_data_format)
            workflow.add_node("generate_code", self._generate_code_with_parsing)
            workflow.add_node("execute_code", self._execute_code)
            workflow.add_node("llm_error_correction", self._llm_error_correction)
            workflow.add_node("generate_commentary", self._generate_commentary)  # NEW: Commentary node

            # Add edges
            workflow.set_entry_point("analyze_data_format")
            workflow.add_edge("analyze_data_format", "generate_code")
            workflow.add_edge("generate_code", "execute_code")

            # CONDITIONAL EDGE: Only retry on fixable errors, max retries from config
            workflow.add_conditional_edges(
                "execute_code",
                self._should_retry_execution,
                {
                    "retry": "llm_error_correction",
                    "commentary": "generate_commentary",  # Go to commentary on success
                    "end": END  # End on max retries or unfixable errors
                }
            )

            # After LLM correction, go back to execute_code for retry evaluation
            workflow.add_edge("llm_error_correction", "execute_code")
            
            # After commentary generation, end the workflow
            workflow.add_edge("generate_commentary", END)

            graph = workflow.compile()
            self.logger.log_operation_success(operation)
            return graph
            
        except Exception as e:
            error_response = self.error_handler.handle_error(e, operation)
            raise AgentError(
                f"Failed to create workflow graph: {e}",
                ErrorCategory.SYSTEM,
                error_response,
                e
            )

    def _should_retry_execution(self, state: AgentState) -> str:
        """Decide whether to retry execution with LLM error correction or proceed to commentary"""
        operation = "retry_decision"
        
        try:
            # Use raw_exception for error correction logic
            raw_error = state.get('raw_exception', '')
            formatted_error = state.get('error', '')
            retry_count = state.get('retry_count', 0)

            self.logger.log_operation_start(operation, {
                "retry_count": retry_count,
                "max_retries": self.config.max_retry_attempts,
                "has_raw_error": bool(raw_error)
            })

            if not raw_error:
                # No error - proceed to commentary generation
                self.logger.log_operation_success(operation, details={"decision": "commentary", "reason": "no_error"})
                return "commentary"

            if retry_count >= self.config.max_retry_attempts:
                self.logger.log_operation_success(operation, details={"decision": "end", "reason": "max_retries_reached"})
                return "commentary"

            # Check if RAW error is fixable
            is_fixable = self._is_likely_fixable_error(raw_error)

            decision = "retry" if is_fixable else "commentary"
            self.logger.log_operation_success(operation, details={
                "decision": decision, 
                "is_fixable": is_fixable,
                "error_preview": raw_error[:100]
            })
            
            return decision
            
        except Exception as e:
            self.logger.log_error(e, operation)
            return "commentary"  # Safe fallback

    def _is_likely_fixable_error(self, error_str: str) -> bool:
        """Check if this is likely a fixable code error (not data issues)"""
        error_lower = error_str.lower()

        unfixable_patterns = [
            "no data loaded", "file not found", "connection",
            "authentication", "permission denied", "disk space",
            "out of memory", "timeout"
        ]

        for pattern in unfixable_patterns:
            if pattern in error_lower:
                return False

        return True

    def _llm_error_correction(self, state: AgentState) -> AgentState:
        """Use LLM to analyze and correct the error with data format awareness"""
        operation = "llm_error_correction"
        start_time = time.time()
        
        try:
            # Increment retry counter
            old_retry_count = state.get('retry_count', 0)
            state['retry_count'] = old_retry_count + 1
            
            self.logger.log_operation_start(operation, {
                "retry_attempt": state['retry_count'],
                "max_retries": self.config.max_retry_attempts
            })

            # Store original error if this is first retry
            if state['retry_count'] == 1:
                state['original_error'] = state['error']
                state['correction_history'] = []

            state['messages'].append(f"Attempt {state['retry_count']}: Analyzing error with AI correction...")

            # Create error correction prompt with data format intelligence
            correction_prompt = self._create_error_correction_prompt(state)

            # Get corrected code from LLM
            response = self._invoke_llm_with_retry([SystemMessage(content=correction_prompt)])
            corrected_code = self._extract_code_from_response(response.content)

            if corrected_code and corrected_code != state['generated_code']:
                state['correction_history'].append(f"Attempt {state['retry_count']}: {state['error'][:100]}...")
                state['generated_code'] = corrected_code
                state['messages'].append(f"LLM provided intelligent code correction. Will re-execute...")
                
                duration = time.time() - start_time
                self.logger.log_operation_success(operation, duration, {
                    "correction_provided": True,
                    "code_length": len(corrected_code)
                })
                
            else:
                # LLM couldn't provide a meaningful correction
                state['messages'].append(f"LLM could not provide a code correction.")
                state['execution_result'] = self._create_helpful_error_response(state)
                state['error'] = ""  # Clear error to prevent further retries
                
                duration = time.time() - start_time
                self.logger.log_operation_warning(operation, "LLM could not provide meaningful correction")

        except Exception as correction_error:
            self.logger.log_error(correction_error, operation)
            state['messages'].append(f"Error correction process failed: {str(correction_error)}")
            state['execution_result'] = self._create_helpful_error_response(state)
            state['error'] = ""  # Clear error to prevent further retries

        return state

    def _create_error_correction_prompt(self, state: AgentState) -> str:
        """Create an enhanced error correction prompt with data format intelligence"""
        # Use the RAW pandas error for correction, not the formatted one
        raw_error = state.get('raw_exception', state.get('error', ''))
        
        # Get data format information
        data_format_info = state.get('data_format_info', {})
        measure_analysis = state.get('measure_analysis', {})

        # Add specific guidance based on error type and data format
        error_specific_guidance = ""
        error_lower = raw_error.lower()

        # Add data format specific context
        data_format_context = ""
        if data_format_info.get('is_wide_format', False):
            time_cols = data_format_info.get('time_columns', [])
            measure_cols = data_format_info.get('measure_columns', [])
            
            # Limit display in prompt
            display_time_cols = time_cols[:20]
            time_cols_suffix = '...' if len(time_cols) > 20 else ''
            
            data_format_context = f"""
DETECTED DATA FORMAT - WIDE FORMAT TIME-SERIES:
- Time columns ({len(time_cols)}): {display_time_cols}{time_cols_suffix}
- Measure columns: {measure_cols}
- Mixed measure types detected: {len(measure_analysis)} unique measures
"""

        # Limit context lengths
        correction_history_preview = chr(10).join(state.get('correction_history', [])[:3])  # Last 3 attempts

        prompt = f"""
ENHANCED CODE ERROR CORRECTION TASK

CONTEXT:
User Query: {state['user_query']}
Data Context: {state['data_context']}

{data_format_context}

FAILED CODE:
```python
{state['generated_code']}
```

ERROR ENCOUNTERED:
{raw_error[:self.config.max_error_message_length]}

{error_specific_guidance}

CORRECTION HISTORY:
{correction_history_preview if correction_history_preview else 'First attempt'}

TASK:
Analyze the error and provide a corrected version that:
1. Fixes the specific error mentioned above
2. Uses wide format best practices for time-series data
3. Applies proper transformations for mixed measure types
4. Handles duplicate measures appropriately
5. Uses standardization for correlation analysis when needed

RESPONSE FORMAT:
```python
# Brief comment explaining the fix and transformation applied
[corrected code here]
```

Provide ONLY the corrected Python code block with wide format intelligence applied.
"""

        return prompt

    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from LLM response"""
        # Handle different code block formats
        if "```python" in response_content:
            try:
                code = response_content.split("```python")[1].split("```")[0]
            except IndexError:
                code = response_content.replace("```python", "").replace("```", "")
        elif "```" in response_content:
            try:
                code = response_content.split("```")[1]
            except IndexError:
                code = response_content.replace("```", "")
        else:
            # No code blocks found, assume entire response is code
            code = response_content

        return code.strip()

    def _create_helpful_error_response(self, state: AgentState) -> str:
        """Create a helpful response when correction fails, with data format suggestions"""
        original_error = state.get('original_error', state['error'])
        retry_count = state.get('retry_count', 0)
        data_format_info = state.get('data_format_info', {})

        response = f"Analysis failed after {retry_count} correction attempt(s).\n\n"
        response += f"Original Error: {original_error}\n\n"

        # Add data format specific suggestions
        if data_format_info.get('is_wide_format', False):
            response += "DETECTED: Wide format time-series data with mixed measures\n\n"
            response += "SUGGESTIONS FOR WIDE FORMAT DATA:\n"
            response += "• Try: 'Standardize measures for correlation analysis'\n"
            response += "• Try: 'Show measure types and their scales'\n"
            response += "• Try: 'Convert to percentage change for comparison'\n"
            response += "• Try: 'Group similar measures together'\n\n"

        # Provide helpful suggestions based on error type
        error_lower = original_error.lower()

        if "column" in error_lower or "key" in error_lower:
            response += "Suggestion: Try asking about a different column, or ask 'What columns are available?' first."
        elif "plot" in error_lower or "chart" in error_lower:
            response += "Suggestion: Try a simpler chart request, like 'Show me a basic summary of the data' first."
        elif "correlation" in error_lower:
            response += "Suggestion: For wide format data, try 'Show correlations between standardized measures' instead."
        elif "merge" in error_lower or "join" in error_lower:
            response += "Suggestion: Try analyzing individual worksheets first before combining data."
        else:
            response += "Suggestion: Try rephrasing your question or asking for a simpler analysis first."

        return response

    def _generate_code_with_parsing(self, state: AgentState) -> AgentState:
        """Enhanced code generation with wide format data intelligence"""
        operation = "code_generation"
        start_time = time.time()
        
        try:
            self.logger.log_operation_start(operation, {"query": state['user_query'][:100]})
            
            # Get data format information
            data_format_info = state.get('data_format_info', {})
            measure_analysis = state.get('measure_analysis', {})

            # STEP 1: Parse Query (enhanced with data format awareness)
            query_lower = state['user_query'].lower()
            worksheet_keywords = ['worksheet', 'sheet', 'tab', 'workbook', 'all sheets', 'combine', 'merge', 'join', 'across sheets']
            comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'between sheets', 'across worksheets']
            
            # Enhanced query analysis for wide format data
            correlation_keywords = ['correlation', 'correlate', 'relationship', 'related', 'connection']
            is_correlation_query = any(keyword in query_lower for keyword in correlation_keywords)
            
            # Determine if multiple worksheets are needed
            needs_multiple_sheets = any(keyword in query_lower for keyword in worksheet_keywords + comparison_keywords)
            state['needs_multiple_sheets'] = needs_multiple_sheets
            
            # Determine result type based on query keywords
            if any(word in query_lower for word in ['plot', 'chart', 'graph', 'bar chart', 'histogram', 'scatter', 'visualize', 'visualization']):
                state['result_type'] = 'chart'
            elif any(word in query_lower for word in ['table', 'dataframe', 'rows', 'columns', 'list', 'summary', 'dict' ,'dictionary', 'top-n', 'rank', 'top-k']):
                state['result_type'] = 'table'
            else:
                state['result_type'] = 'value'
            
            # STEP 2: Check if we should use wide format specific code generation
            use_wide_format_intelligence = (
                data_format_info.get('is_wide_format', False) and
                data_format_info.get('time_columns') and 
                data_format_info.get('measure_columns') and
                measure_analysis
            )
            
            if use_wide_format_intelligence:
                # Format the measure intelligence for the data context
                measure_context = self._format_wide_format_context(data_format_info, measure_analysis)

                # APPEND to existing data context instead of replacing
                state['data_context'] = state['data_context'] + "\n" + measure_context
                
                # Log that we've enhanced the context
                self.logger.log_operation_success("enhance_data_context", details={
                    "wide_format_detected": True,
                    "measures_analyzed": len(measure_analysis),
                    "context_enhanced": True
                })

            # STEP 3: Regular code generation (existing logic)
            # Extract actual column names and sample data from context (existing logic)
            context_lines = state['data_context'].split('\n')
            actual_columns = []
            numeric_columns = []
            categorical_columns = []
            worksheet_info = {}
            active_worksheet = None

            current_worksheet = None
            for line in context_lines:
                if line.startswith('=== Worksheet:'):
                    current_worksheet = line.replace('=== Worksheet:', '').strip()
                    worksheet_info[current_worksheet] = {'columns': [], 'numeric': [], 'categorical': []}
                elif line.startswith('CURRENT ANALYSIS SCOPE:'):
                    # Look for active worksheet in the following lines
                    for next_line in context_lines[context_lines.index(line):]:
                        if 'active worksheet (' in next_line:
                            active_worksheet = next_line.split('(')[1].split(')')[0]
                            break
                elif line.startswith('Columns:') and current_worksheet:
                    cols_text = line.replace('Columns:', '').strip()
                    cols = [col.strip() for col in cols_text.split(',') if col.strip()]
                    worksheet_info[current_worksheet]['columns'] = cols
                    if current_worksheet == active_worksheet:
                        actual_columns = cols
                elif line.startswith('Numeric columns:') and current_worksheet:
                    num_text = line.replace('Numeric columns:', '').strip()
                    nums = [col.strip() for col in num_text.split(',') if col.strip()]
                    worksheet_info[current_worksheet]['numeric'] = nums
                    if current_worksheet == active_worksheet:
                        numeric_columns = nums
                elif line.startswith('Categorical columns:') and current_worksheet:
                    cat_text = line.replace('Categorical columns:', '').strip()
                    cats = [col.strip() for col in cat_text.split(',') if col.strip()]
                    worksheet_info[current_worksheet]['categorical'] = cats
                    if current_worksheet == active_worksheet:
                        categorical_columns = cats
            
            # If no worksheet info found, fall back to old format
            if not worksheet_info:
                for line in context_lines:
                    if line.startswith('Columns:'):
                        cols_text = line.replace('Columns:', '').strip()
                        actual_columns = [col.strip() for col in cols_text.split(',') if col.strip()]
                    elif line.startswith('Numeric columns:'):
                        num_text = line.replace('Numeric columns:', '').strip()
                        numeric_columns = [col.strip() for col in num_text.split(',') if col.strip()]
                    elif line.startswith('Categorical columns:'):
                        cat_text = line.replace('Categorical columns:', '').strip()
                        categorical_columns = [col.strip() for col in cat_text.split(',') if col.strip()]
            
            # Add multi-worksheet information if available
            worksheet_context = ""
            if worksheet_info and len(worksheet_info) > 1:
                worksheet_context = f"""
MULTI-WORKSHEET INFORMATION:
Available worksheets: {list(worksheet_info.keys())}
Active worksheet: {active_worksheet}

Worksheet details:
"""
                for ws_name, ws_info in worksheet_info.items():
                    worksheet_context += f"- {ws_name}: {len(ws_info['columns'])} columns ({len(ws_info['numeric'])} numeric, {len(ws_info['categorical'])} text)\n"
                
                worksheet_context += f"""
WORKSHEET MERGING:
- If the query requires data from multiple worksheets, use: merged_data = merge_worksheets(worksheet_data)
- The merge_worksheets() function is available and will combine all worksheets intelligently
- After merging, use 'merged_data' instead of 'current_data' for analysis
- If the query is specific to one worksheet, use 'current_data' (active worksheet: {active_worksheet})
"""

            # Generate the system prompt for code generation
            system_prompt = get_code_generation_prompt(
                state['data_context'],
                state['user_query'],
                state['result_type'],
                state.get('needs_multiple_sheets', False),
                actual_columns,
                numeric_columns,
                categorical_columns,
                worksheet_info,
                worksheet_context,
                active_worksheet)

            print("full system prompt: \n", system_prompt)

            # Use LLM with retry logic for token refresh to generate code
            response = self._invoke_llm_with_retry([SystemMessage(content=system_prompt)])

            # Extract code from response with better error handling
            code = response.content

            # Clean and extract code
            if isinstance(code, str):
                if "```python" in code:
                    try:
                        code = code.split("```python")[1].split("```")[0]
                    except IndexError:
                        code = code.replace("```python", "").replace("```", "")
                elif "```" in code:
                    try:
                        code = code.split("```")[1]
                    except IndexError:
                        code = code.replace("```", "")

                # Additional cleaning
                code = code.strip()

                # Check if code looks corrupted (has comma separation)
                if code and ',' in code and len(code.split(',')) > len(code.split('\n')) * 2:
                    if code.startswith('#,') or code.startswith('r,e,s,u,l,t'):
                        code = code.replace(',', '')
            else:
                code = str(code)
            
            state['generated_code'] = code
            state['messages'].append(f"Generated Code: {code}")

            duration = time.time() - start_time
            self.logger.log_operation_success(operation, duration, {
                "approach": "standard",
                "result_type": state['result_type'],
                "multi_sheet": needs_multiple_sheets,
                "code_length": len(code)
            })

            return state
            
        except Exception as e:
            error_response = self.error_handler.handle_error(e, operation)
            state['error'] = error_response['message']
            return state

    def _invoke_llm_with_retry(self, messages, max_retries=None):
        """Invoke LLM with retry logic for authentication failures"""
        max_retries = max_retries or self.config.llm_max_retries
        operation = "llm_invocation"

        # Determine which provider we're using
        provider_type = "azure_openai"  # default
        if UNIVERSAL_LLM_AVAILABLE:
            try:
                llm_config = LLMConfig.from_env()
                provider_type = llm_config.provider.value
            except:
                pass

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.log_operation_start(f"{operation}_retry", {
                        "attempt": attempt + 1,
                        "provider": provider_type
                    })

                response = self.llm.invoke(messages)

                if attempt > 0:
                    self.logger.log_operation_success(f"{operation}_retry", details={
                        "attempt": attempt + 1,
                        "provider": provider_type
                    })

                return response

            except Exception as e:
                error_str = str(e).lower()

                # Enhanced error detection patterns for different providers
                is_auth_error = any(pattern in error_str for pattern in [
                    '401', 'unauthorized', 'authentication', 'token', 'expired',
                    'invalid api key', 'invalid_api_key', 'permission denied'
                ])

                is_connection_error = any(pattern in error_str for pattern in [
                    'connection', 'timeout', 'network', 'refused', 'unreachable'
                ])

                # Handle authentication errors
                if attempt < max_retries and is_auth_error:
                    self.logger.log_operation_warning(
                        operation,
                        f"Authentication error on attempt {attempt + 1} for {provider_type}, refreshing connection"
                    )
                    try:
                        self.refresh_llm_connection()
                        continue  # Retry with refreshed connection
                    except Exception as refresh_error:
                        if attempt == max_retries:
                            raise AgentError(
                                f"Authentication failed and could not refresh connection: {refresh_error}",
                                ErrorCategory.AUTHENTICATION,
                                {
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries,
                                    "provider": provider_type
                                },
                                refresh_error
                            )

                # Handle connection errors with exponential backoff
                elif attempt < max_retries and is_connection_error:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                    self.logger.log_operation_warning(
                        operation,
                        f"Connection error on attempt {attempt + 1} for {provider_type}, waiting {wait_time}s before retry"
                    )

                    import time
                    time.sleep(wait_time)

                    # Try refreshing connection on connection errors too
                    try:
                        self.refresh_llm_connection()
                        continue
                    except Exception as refresh_error:
                        self.logger.log_operation_warning(
                            operation,
                            f"Connection refresh failed: {refresh_error}, continuing with retry"
                        )
                        continue

                # If it's the last attempt or an unrecoverable error, raise it
                if attempt == max_retries:
                    raise AgentError(
                        f"LLM invocation failed after {max_retries + 1} attempts: {e}",
                        ErrorCategory.SYSTEM,
                        {
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error_type": type(e).__name__,
                            "provider": provider_type
                        },
                        e
                    )

        raise AgentError(
            "Max retries exceeded for LLM invocation",
            ErrorCategory.SYSTEM,
            {"max_retries": max_retries, "provider": provider_type}
        )
    
    def _execute_code(self, state: AgentState) -> AgentState:
        """Execute the generated code safely with enhanced wide format support"""
        operation = "code_execution"
        start_time = time.time()
        
        retry_count = state.get('retry_count', 0)
        if retry_count > 0:
            operation = f"code_execution_retry_{retry_count}"
        
        try:
            self.logger.log_operation_start(operation, {
                "retry_count": retry_count,
                "code_length": len(state['generated_code'])
            })
            
            # Get the data from the state
            current_data = state.get('current_data')
            worksheet_data = state.get('worksheet_data', {})
            merge_worksheets_func = state.get('merge_worksheets_func')
            
            if current_data is None:
                raise AgentError(
                    "No data available for analysis. Please upload a file first.",
                    ErrorCategory.DATA_VALIDATION
                )
            
            # Check data size limits
            if hasattr(current_data, 'shape'):
                rows, cols = current_data.shape
                if rows > self.config.max_dataframe_rows or cols > self.config.max_dataframe_cols:
                    self.logger.log_operation_warning(operation, f"Large dataset detected: {rows}x{cols}")
                    if rows > self.config.large_dataset_threshold:
                        # Sample large datasets
                        current_data = current_data.sample(n=min(self.config.sample_size_limit, rows))
                        self.logger.log_operation_warning(operation, f"Dataset sampled to {len(current_data)} rows")
            
            # Create execution environment with core libraries (always available)
            exec_globals = {
                # Core libraries (always available)
                'current_data': current_data,
                'worksheet_data': worksheet_data,
                'merge_worksheets': merge_worksheets_func,
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'result': None,
                'fig': None,
                # Add wide format utilities
                'DataFormatAnalyzer': DataFormatAnalyzer,
            }
            
            # Add advanced analytics libraries only if available
            if self.sklearn_available:
                exec_globals.update({
                    'LinearRegression': LinearRegression,
                    'Ridge': Ridge,
                    'Lasso': Lasso,
                    'ElasticNet': ElasticNet,
                    'LogisticRegression': LogisticRegression,
                    'RandomForestRegressor': RandomForestRegressor,
                    'RandomForestClassifier': RandomForestClassifier,
                    'GradientBoostingRegressor': GradientBoostingRegressor,
                    'GradientBoostingClassifier': GradientBoostingClassifier,
                    'SVC': SVC,
                    'SVR': SVR,
                    'GaussianNB': GaussianNB,
                    'DecisionTreeClassifier': DecisionTreeClassifier,
                    'DecisionTreeRegressor': DecisionTreeRegressor,
                    'KNeighborsClassifier': KNeighborsClassifier,
                    'KNeighborsRegressor': KNeighborsRegressor,
                    'KMeans': KMeans,
                    'DBSCAN': DBSCAN,
                    'AgglomerativeClustering': AgglomerativeClustering,
                    'IsolationForest': IsolationForest,
                    'StandardScaler': StandardScaler,
                    'MinMaxScaler': MinMaxScaler,
                    'LabelEncoder': LabelEncoder,
                    'OneHotEncoder': OneHotEncoder,
                    'PolynomialFeatures': PolynomialFeatures,
                    'PCA': PCA,
                    'TruncatedSVD': TruncatedSVD,
                    'SelectKBest': SelectKBest,
                    'RFE': RFE,
                    'f_classif': f_classif,
                    'mutual_info_regression': mutual_info_regression,
                    'mutual_info_classif': mutual_info_classif,
                    'f_regression': f_regression,
                    'train_test_split': train_test_split,
                    'cross_val_score': cross_val_score,
                    'GridSearchCV': GridSearchCV,
                    'TimeSeriesSplit': TimeSeriesSplit,
                    'mean_squared_error': mean_squared_error,
                    'mean_absolute_error': mean_absolute_error,
                    'r2_score': r2_score,
                    'accuracy_score': accuracy_score,
                    'precision_score': precision_score,
                    'recall_score': recall_score,
                    'f1_score': f1_score,
                    'roc_auc_score': roc_auc_score,
                    'roc_curve': roc_curve,
                    'confusion_matrix': confusion_matrix,
                    'classification_report': classification_report,
                    'silhouette_score': silhouette_score,
                    'davies_bouldin_score': davies_bouldin_score
                })
                
            if self.scipy_available:
                exec_globals.update({
                    'stats': stats,
                    'signal': signal,
                    'optimize': optimize,
                    'interpolate': interpolate,
                    'pearsonr': pearsonr,
                    'spearmanr': spearmanr,
                    'kendalltau': kendalltau,
                    'ttest_ind': ttest_ind,
                    'ttest_rel': ttest_rel,
                    'ttest_1samp': ttest_1samp,
                    'mannwhitneyu': mannwhitneyu,
                    'wilcoxon': wilcoxon,
                    'kruskal': kruskal,
                    'friedmanchisquare': friedmanchisquare,
                    'chi2_contingency': chi2_contingency,
                    'fisher_exact': fisher_exact,
                    'kstest': kstest,
                    'shapiro': shapiro,
                    'normaltest': normaltest,
                    'anderson': anderson,
                    'levene': levene,
                    'bartlett': bartlett,
                    'f_oneway': f_oneway,
                    'zscore': zscore,
                    'boxcox': boxcox,
                    'probplot': probplot,
                    'rankdata': rankdata,
                    'trim_mean': trim_mean,
                    'kurtosis': kurtosis,
                    'skew': skew,
                    'mode': mode,
                    'entropy': entropy,
                    'pointbiserialr': pointbiserialr,
                    'find_peaks': find_peaks,
                    'savgol_filter': savgol_filter,
                    'butter': butter,
                    'filtfilt': filtfilt,
                    'welch': welch,
                    'dendrogram': dendrogram,
                    'linkage': linkage,
                    'fcluster': fcluster,
                    'euclidean': euclidean,
                    'cosine': cosine,
                    'correlation': correlation
                })
                
            if self.statsmodels_available:
                exec_globals.update({
                    'sm': sm,
                    'seasonal_decompose': seasonal_decompose,
                    'STL': STL,
                    'SARIMAX': SARIMAX,
                    'ARIMA': ARIMA,
                    'ExponentialSmoothing': ExponentialSmoothing,
                    'adfuller': adfuller,
                    'acf': acf,
                    'pacf': pacf,
                    'kpss': kpss,
                    'coint': coint,
                    'acorr_ljungbox': acorr_ljungbox,
                    'variance_inflation_factor': variance_inflation_factor,
                    'proportions_ztest': proportions_ztest,
                    'TTestPower': TTestPower,
                    'pairwise_tukeyhsd': pairwise_tukeyhsd,
                    'plot_acf': plot_acf,
                    'plot_pacf': plot_pacf,
                })
                
            if self.matplotlib_available:
                exec_globals.update({
                    'plt': plt,
                    'sns': sns,
                })

            if self.advanced_ml_available:
                exec_globals.update({
                    'xgb': xgb,
                    'lgb': lgb,
                    'cb': cb,
                    'XGBRegressor': xgb.XGBRegressor,
                    'XGBClassifier': xgb.XGBClassifier,
                    'LGBMRegressor': lgb.LGBMRegressor,
                    'LGBMClassifier': lgb.LGBMClassifier,
                    'CatBoostRegressor': cb.CatBoostRegressor,
                    'CatBoostClassifier': cb.CatBoostClassifier
                })

            if self.prophet_available:
                exec_globals.update({
                    'Prophet': Prophet
                })

            if self.anomaly_detection_available:
                exec_globals.update({
                    'IForest': IForest,
                    'LOF': LOF,
                    'OCSVM': OCSVM,
                    'KNN': KNN
                })

            if self.explainability_available:
                exec_globals.update({
                    'shap': shap,
                    'LimeTabularExplainer': LimeTabularExplainer
                })

            if self.optuna_available:
                exec_globals.update({
                    'optuna': optuna
                })

            # Add import capability for dynamic imports
            exec_globals['__builtins__'] = __builtins__
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Validate data availability before execution
                if current_data is None or current_data.empty:
                    raise AgentError("No data available for analysis", ErrorCategory.DATA_VALIDATION)

                # Execute the code
                exec(state['generated_code'], exec_globals)

                # Restore stdout
                sys.stdout = old_stdout
            
                # Get results
                result = exec_globals.get('result')
                fig = exec_globals.get('fig')
                output = captured_output.getvalue()

                # Handle common empty result cases
                if result is None and fig is None:
                    # Check if code assigned to a variable but didn't assign to 'result'
                    potential_results = []
                    for var_name, var_value in exec_globals.items():
                        if not var_name.startswith('_') and var_name not in ['current_data', 'worksheet_data', 'pd', 'np', 'px', 'go', 'DataFormatAnalyzer']:
                            if isinstance(var_value, (str, int, float, list, dict, pd.DataFrame, pd.Series)):
                                potential_results.append((var_name, var_value))

                    if potential_results:
                        # Use the last assigned variable as result
                        result = potential_results[-1][1]
                        state['messages'].append(f"Auto-detected result from variable: {potential_results[-1][0]}")

            except IndexError as idx_error:
                sys.stdout = old_stdout
                state['raw_exception'] = str(idx_error)
                if "list index out of range" in str(idx_error):
                    state['error'] = "No data found matching your criteria. The result was empty."
                else:
                    state['error'] = f"Index error: {str(idx_error)}"
                state['execution_result'] = ""
                return state

            # Safety check for empty results
            if result is not None:
                if isinstance(result, (list, tuple)) and len(result) == 0:
                    result = "No data found matching your criteria"
                elif isinstance(result, pd.DataFrame) and result.empty:
                    result = "No data found matching your criteria"
                elif isinstance(result, pd.Series) and result.empty:
                    result = "No data found matching your criteria"

            # Enhanced result processing with user-friendly formatting
            if result is not None:
                # print unformatted result
                print("unformatted result: \n", result)
                state['execution_result'] = result

                # Validate if result needs formatting
                #is_user_friendly, reason = validate_result_user_friendliness(result)

                #if not is_user_friendly:
                #    self.logger.log_operation_warning(operation, f"Result needs formatting: {reason}")

                # Format result for user consumption
                #try:
                #    formatted_result = format_result_for_user(result, state.get('result_type', 'auto'))

                #    # Additional validation for very large results
                #    if isinstance(formatted_result, list) and len(formatted_result) > self.config.max_list_items_display:
                #        # Truncate very large lists
                #        truncated_result = formatted_result[:self.config.max_list_items_display]
                #        truncated_result.append({
                #            "_note": f"Showing first {self.config.max_list_items_display:,} of {len(formatted_result):,} items",
                #            "_type": "system_message"
                #        })
                #        state['execution_result'] = truncated_result
                #    else:
                #        state['execution_result'] = formatted_result

                #    # Log formatting success
                #    self.logger.log_operation_success("result_formatting", details={
                #        "original_type": type(result).__name__,
                #        "formatted_type": type(formatted_result).__name__,
                #        "user_friendly": True
                #    })

                #except Exception as format_error:
                #    self.logger.log_error(format_error, "result_formatting")
                #    # Fallback to safe string representation
                #    state['execution_result'] = f"Analysis completed successfully. Result: {str(result)[:500]}..."

            else:
                state['execution_result'] = output or "Code executed successfully"

            # Handle plots
            if fig is not None:
                state['plot_data'] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

                #try:
                #    from result_formatter import format_plot_result

                #    # If we have both plot and result, format result for plot context
                #    if result is not None:
                #        plot_optimized_result = format_plot_result(result, state.get('plot_data', {}))
                #        state['execution_result'] = plot_optimized_result
                #
                #    state['plot_data'] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
                #
                #except (TypeError, ValueError) as plot_error:
                #    self.logger.log_operation_warning(operation, f"Plot serialization error: {plot_error}")
                #    # Convert figure to dict if direct serialization fails
                #    try:
                #        state['plot_data'] = fig.to_dict()
                #    except:
                #        state['plot_data'] = {}
                #        state['execution_result'] = str(
                #            result) if result is not None else "Plot generated but could not be serialized"
            
            state['error'] = ""
            
            duration = time.time() - start_time
            self.logger.log_operation_success(operation, duration, {
                "result_type": type(result).__name__ if result is not None else "None",
                "has_plot": fig is not None,
                "result_size": len(str(state['execution_result']))
            })
            
        except Exception as e:
            sys.stdout = old_stdout if 'old_stdout' in locals() else sys.stdout
            duration = time.time() - start_time
            print("Erroneous code: \n", state['generated_code'])
            state['raw_exception'] = str(e)  # Raw pandas error for correction logic
            state['error'] = self._format_error_message(e, current_data)
            state['execution_result'] = ""
            
            self.logger.log_error(e, operation)
        
        return state
    
    def _format_error_message(self, error, current_data):
        """Format error messages with helpful context including wide format guidance"""
        error_str = str(error).lower()
        
        if isinstance(error, KeyError):
            # Handle missing column errors specifically
            column_name = str(error).strip("'\"")
            available_columns = list(current_data.columns) if current_data is not None else []
            
            # Try to suggest similar column names using fuzzy matching
            suggestions = []
            if available_columns:
                for col in available_columns:
                    # Check for partial matches (case insensitive)
                    if (column_name.lower() in col.lower() or 
                        col.lower() in column_name.lower() or
                        column_name.replace('_', ' ').lower() in col.replace('_', ' ').lower()):
                        suggestions.append(col)
            
            error_msg = f"Column '{column_name}' not found in your data."
            if suggestions:
                error_msg += f"\n\n Did you mean one of these?\n- {chr(10).join(['* ' + s for s in suggestions[:3]])}"
            
            if available_columns:
                display_cols = available_columns[:self.config.max_display_columns]
                cols_suffix = f"\n... and {len(available_columns) - self.config.max_display_columns} more" if len(available_columns) > self.config.max_display_columns else ""
                error_msg += f"\n\n Available columns:\n{chr(10).join(['* ' + col for col in display_cols])}{cols_suffix}"
            else:
                error_msg += "\n\n No data available. Please upload a file first."
                
            error_msg += "\n\n Tips:\n• Use exact column names from the Data Overview panel\n• Try 'List all column names' to see available columns\n• Copy-paste column names to avoid typos"
            
            return error_msg
            
        elif isinstance(error, ValueError) and "no data available" in error_str:
            return "❌ No data available for analysis. Please upload a file first before asking questions about your data."
            
        elif isinstance(error, AttributeError) and 'str' in error_str and 'attribute' in error_str:
            error_msg = f"❌ String method error: {str(error)}"
            error_msg += "\n\n Fix: Cast column to string first before using .str() methods:"
            error_msg += "\n Use: df['column'].astype('str').str.method()"
            error_msg += "\n Don't use: df['column'].str.method() directly"
            error_msg += "\n\n Example fixes:"
            error_msg += "\n• df['col'].astype('str').str.contains('text')"
            error_msg += "\n• df['col'].astype('str').str.upper()"
            error_msg += "\n• df['col'].astype('str').str.len()"
            return error_msg
            
        elif isinstance(error, NameError):
            error_msg = f"❌ Library or function not available: {str(error)}"
            
            # Check which library is missing and provide specific guidance
            if 'linearregression' in error_str or 'sklearn' in error_str:
                error_msg += "\n\n Missing scikit-learn library for machine learning."
                error_msg += "\n Install with: pip install scikit-learn"
                error_msg += "\n For basic analysis, try: 'Show correlation' or 'Create histogram'"
            elif 'stats' in error_str or 'scipy' in error_str:
                error_msg += "\n\n Missing scipy library for statistical analysis."
                error_msg += "\n Install with: pip install scipy"
            elif 'sm' in error_str or 'statsmodels' in error_str:
                error_msg += "\n\n Missing statsmodels library for advanced statistics."
                error_msg += "\n Install with: pip install statsmodels"
            elif 'plt' in error_str or 'sns' in error_str:
                error_msg += "\n\n Missing matplotlib/seaborn for advanced plotting."
                error_msg += "\n Install with: pip install matplotlib seaborn"
            else:
                error_msg += "\n\n For basic analysis without advanced libraries, try:"
                error_msg += "\n• 'Show summary statistics'\n• 'Create histogram'\n• 'Count missing values'"
            
            return error_msg
            
        elif isinstance(error, ImportError):
            error_msg = f"❌ Missing library: {str(error)}"
            error_msg += "\n\n Install missing analytics libraries:"
            error_msg += "\npip install scikit-learn scipy statsmodels matplotlib seaborn"
            error_msg += "\n\n For now, try basic operations like:"
            error_msg += "\n• 'Show summary statistics'\n• 'Create simple chart'\n• 'Count rows'"
            return error_msg
        
        else:
            # Limit error message length using configuration
            message = str(error)
            if len(message) > self.config.max_error_message_length:
                message = message[:self.config.max_error_message_length] + "..."
            
            error_msg = f"❌ Error executing code: {message}"
            
            # Add helpful context for common errors
            if "typeerror" in error_str:
                error_msg += "\n\n This might be a data type issue. Check for missing values or incompatible operations."
            elif "valueerror" in error_str:
                error_msg += "\n\n This might be a data format issue. Ensure numeric columns for mathematical operations."
            elif "linalgerror" in error_str:
                error_msg += "\n\n Linear algebra error. Check for multicollinearity or insufficient data."
            elif "'nonetype' object has no attribute" in error_str:
                error_msg += "\n\n Data not available. Please upload a file first before asking questions."
            
            return error_msg

    def process_query(self, user_query: str, data_context: str, current_data=None, worksheet_data=None,
                      merge_worksheets_func=None) -> Dict[str, Any]:
        """Process a user query with enhanced wide format intelligence and commentary generation"""
        operation = "query_processing"
        start_time = time.time()
        
        try:
            self.logger.log_operation_start(operation, {
                "query_preview": user_query[:100],
                "has_data": current_data is not None,
                "data_shape": current_data.shape if current_data is not None else None,
                "commentary_enabled": self.config.enable_commentary
            })
            
            initial_state = AgentState(
                messages=[],
                data_context=data_context,
                user_query=user_query,
                generated_code="",
                execution_result="",
                error="",
                raw_exception="",  # Initialize raw exception
                plot_data={},
                result_type="",
                needs_multiple_sheets=False,
                current_data=current_data,
                worksheet_data=worksheet_data or {},
                merge_worksheets_func=merge_worksheets_func,
                retry_count=0,  # Initialize retry counter
                original_error="",  # Initialize original error storage
                correction_history=[],  # Initialize correction history
                # Wide format intelligence
                data_format_info={},
                measure_analysis={},
                transformation_suggestions={},
                # NEW: Initialize commentary fields
                commentary="",
                insights=[]
            )

            final_state = self.graph.invoke(initial_state)

            duration = time.time() - start_time
            success = not bool(final_state['error'])
            
            self.logger.log_operation_success(operation, duration, {
                "success": success,
                "retry_attempts": final_state.get('retry_count', 0),
                "wide_format_detected": final_state.get('data_format_info', {}).get('is_wide_format', False),
                "result_type": final_state.get('result_type', 'unknown'),
                "has_commentary": bool(final_state.get('commentary', '')),
                "insights_count": len(final_state.get('insights', []))
            })

            print("execution result: \n", final_state['execution_result'])

            return {
                'success': success,
                'result': final_state['execution_result'],
                'error': final_state['error'],
                'code': final_state['generated_code'],
                'messages': final_state['messages'],
                'plot_data': final_state.get('plot_data', {}),
                'result_type': final_state['result_type'],
                'retry_count': final_state.get('retry_count', 0),  # Include retry info
                'correction_history': final_state.get('correction_history', []),  # Include correction attempts
                # Wide format intelligence results
                'data_format_info': final_state.get('data_format_info', {}),
                'measure_analysis': final_state.get('measure_analysis', {}),
                'transformation_suggestions': final_state.get('transformation_suggestions', {}),
                # NEW: Include commentary and insights
                'commentary': final_state.get('commentary', ''),
                'insights': final_state.get('insights', [])
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_response = self.error_handler.handle_error(e, operation, {
                "query": user_query[:100],
                "duration": duration
            })
            
            return {
                'success': False,
                'result': "",
                'error': error_response['message'],
                'code': "",
                'messages': [f"Query processing failed: {error_response['message']}"],
                'plot_data': {},
                'result_type': 'error',
                'retry_count': 0,
                'correction_history': [],
                'data_format_info': {},
                'measure_analysis': {},
                'transformation_suggestions': {},
                # Empty commentary on error
                'commentary': '',
                'insights': [],
                'error_details': error_response
            }

# Cleanup function for graceful shutdown
def cleanup_connections():
    """Clean up HTTP connections gracefully"""
    global http_client
    logger = get_global_logger()
    
    if http_client:
        try:
            http_client.close()
            logger.log_operation_success("cleanup", details={"component": "http_client"})
        except Exception as e:
            logger.log_error(e, "cleanup")

# Register cleanup function for app shutdown
import atexit
atexit.register(cleanup_connections)

# Initialize global configuration on module import
_global_config = get_global_config()
_global_logger = get_global_logger()

# Log module initialization
_global_logger.log_operation_success("module_initialization", details={
    "config_source": "environment",
    "log_level": _global_config.log_level,
    "max_retries": _global_config.max_retry_attempts,
    "sklearn_available": SKLEARN_AVAILABLE,
    "scipy_available": SCIPY_AVAILABLE,
    "statsmodels_available": STATSMODELS_AVAILABLE,
    "commentary_enabled": _global_config.enable_commentary
})