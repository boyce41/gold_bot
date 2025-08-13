"""
Master AI Data Validation Utilities
Provides comprehensive validation functions for data quality, model integrity, and signal consistency.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import joblib
import os


def validate_data_quality(data: pd.DataFrame, min_rows: int = 60) -> bool:
    """
    Comprehensive data quality validation for Master AI processing.
    
    Args:
        data: DataFrame with OHLC and indicator data
        min_rows: Minimum required rows for analysis
        
    Returns:
        bool: True if data passes all quality checks
        
    Raises:
        ValueError: If critical data quality issues are found
    """
    if data is None or data.empty:
        raise ValueError("Data is None or empty")
    
    if len(data) < min_rows:
        raise ValueError(f"Insufficient data: {len(data)} rows, minimum {min_rows} required")
    
    # Check for required OHLC columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLC columns: {missing_columns}")
    
    # Check for excessive missing values
    missing_percentages = data[required_columns].isnull().sum() / len(data)
    high_missing = missing_percentages[missing_percentages > 0.1]
    if not high_missing.empty:
        raise ValueError(f"Excessive missing values in columns: {high_missing.to_dict()}")
    
    # Check for data consistency
    invalid_ohlc = data[(data['high'] < data['low']) | 
                       (data['high'] < data['open']) | 
                       (data['high'] < data['close']) |
                       (data['low'] > data['open']) | 
                       (data['low'] > data['close'])]
    
    if not invalid_ohlc.empty:
        st.warning(f"⚠️ Found {len(invalid_ohlc)} rows with invalid OHLC relationships")
    
    # Check for extreme price movements (>50% in one candle)
    price_changes = data['close'].pct_change().abs()
    extreme_moves = price_changes[price_changes > 0.5]
    if not extreme_moves.empty:
        st.warning(f"⚠️ Found {len(extreme_moves)} extreme price movements (>50%)")
    
    return True


def validate_model_integrity(models: Dict[str, Any]) -> bool:
    """
    Validate model loading and integrity for Master AI ensemble.
    
    Args:
        models: Dictionary containing all AI models
        
    Returns:
        bool: True if all models pass integrity checks
        
    Raises:
        ValueError: If critical model integrity issues are found
    """
    required_models = ['lstm', 'xgb', 'cnn', 'svc', 'nb', 'meta']
    missing_models = []
    
    for model_name in required_models:
        if model_name not in models or models[model_name] is None:
            missing_models.append(model_name)
    
    if missing_models:
        raise ValueError(f"Missing critical AI models: {missing_models}")
    
    # Check scalers for models that need them
    required_scalers = ['scaler', 'cnn_scaler', 'svc_scaler']
    missing_scalers = []
    
    for scaler_name in required_scalers:
        if scaler_name not in models or models[scaler_name] is None:
            missing_scalers.append(scaler_name)
    
    if missing_scalers:
        st.warning(f"⚠️ Missing scalers (may affect predictions): {missing_scalers}")
    
    # Validate meta learner has proper classes
    meta_model = models.get('meta')
    if meta_model and hasattr(meta_model, 'classes_'):
        expected_classes = {-1, 0, 1}
        actual_classes = set(meta_model.classes_)
        if not expected_classes.issubset(actual_classes):
            missing_classes = expected_classes - actual_classes
            st.warning(f"⚠️ Meta learner missing classes: {missing_classes}")
    
    return True


def validate_master_ai_signal(signal_data: Dict[str, Any]) -> bool:
    """
    Validate Master AI signal data consistency and completeness.
    
    Args:
        signal_data: Dictionary containing Master AI signal results
        
    Returns:
        bool: True if signal data is valid
        
    Raises:
        ValueError: If signal data is invalid
    """
    required_keys = ['signal', 'confidence', 'probability_distribution', 'individual_consensus']
    missing_keys = [key for key in required_keys if key not in signal_data]
    
    if missing_keys:
        raise ValueError(f"Missing required signal data keys: {missing_keys}")
    
    # Validate signal value
    signal = signal_data['signal']
    if signal not in [-1, 0, 1]:
        raise ValueError(f"Invalid signal value: {signal}. Must be -1, 0, or 1")
    
    # Validate confidence
    confidence = signal_data['confidence']
    if not (0 <= confidence <= 1):
        raise ValueError(f"Invalid confidence value: {confidence}. Must be between 0 and 1")
    
    # Validate probability distribution
    prob_dist = signal_data['probability_distribution']
    if not isinstance(prob_dist, (list, np.ndarray)) or len(prob_dist) != 3:
        raise ValueError("Probability distribution must be array-like with 3 elements")
    
    prob_sum = np.sum(prob_dist)
    if not (0.95 <= prob_sum <= 1.05):  # Allow small numerical errors
        raise ValueError(f"Probability distribution must sum to ~1.0, got {prob_sum}")
    
    # Validate individual consensus
    consensus = signal_data['individual_consensus']
    if not (0 <= consensus <= 1):
        raise ValueError(f"Invalid consensus score: {consensus}. Must be between 0 and 1")
    
    return True


def validate_technical_indicators(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate availability and quality of technical indicators.
    
    Args:
        data: DataFrame with technical indicators
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Essential indicators for Master AI
    essential_indicators = {
        'rsi': 'RSI indicator',
        'MACD_12_26_9': 'MACD indicator', 
        'ATR_14': 'ATR indicator',
        'EMA_10': 'EMA 10 indicator',
        'EMA_20': 'EMA 20 indicator'
    }
    
    for indicator, description in essential_indicators.items():
        if indicator not in data.columns:
            warnings.append(f"Missing {description} ({indicator})")
        elif data[indicator].isnull().sum() / len(data) > 0.2:
            warnings.append(f"High missing values in {description} ({indicator})")
    
    # Advanced indicators (nice to have)
    advanced_indicators = {
        'STOCHk_14_3_3': 'Stochastic indicator',
        'MACDh_12_26_9': 'MACD Histogram',
        'bb_percent': 'Bollinger Bands percentage'
    }
    
    missing_advanced = []
    for indicator, description in advanced_indicators.items():
        if indicator not in data.columns:
            missing_advanced.append(description)
    
    if missing_advanced:
        warnings.append(f"Missing advanced indicators: {', '.join(missing_advanced)}")
    
    # Check for reasonable indicator ranges
    if 'rsi' in data.columns:
        rsi_out_of_range = data[(data['rsi'] < 0) | (data['rsi'] > 100)]
        if not rsi_out_of_range.empty:
            warnings.append(f"RSI values out of range (0-100): {len(rsi_out_of_range)} instances")
    
    is_valid = len([w for w in warnings if 'Missing' in w and any(ess in w for ess in essential_indicators.values())]) == 0
    
    return is_valid, warnings


def get_data_quality_score(data: pd.DataFrame) -> float:
    """
    Calculate overall data quality score for Master AI processing.
    
    Args:
        data: DataFrame to evaluate
        
    Returns:
        float: Quality score between 0 and 1
    """
    score = 1.0
    
    try:
        # Check data completeness (40% of score)
        required_columns = ['open', 'high', 'low', 'close', 'rsi', 'MACD_12_26_9', 'ATR_14']
        available_columns = [col for col in required_columns if col in data.columns]
        completeness_score = len(available_columns) / len(required_columns)
        score *= (0.4 + 0.6 * completeness_score)
        
        # Check missing values (30% of score)
        if not data.empty:
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            missing_score = max(0, 1 - missing_ratio * 5)  # Penalize heavily for missing data
            score *= (0.3 + 0.7 * missing_score)
        
        # Check data consistency (30% of score)
        if 'close' in data.columns and len(data) > 1:
            price_changes = data['close'].pct_change().abs()
            extreme_moves = (price_changes > 0.1).sum() / len(data)  # >10% moves
            consistency_score = max(0, 1 - extreme_moves * 2)
            score *= (0.3 + 0.7 * consistency_score)
            
    except Exception as e:
        st.warning(f"⚠️ Error calculating data quality score: {e}")
        score = 0.5  # Default to medium quality if calculation fails
    
    return max(0, min(1, score))