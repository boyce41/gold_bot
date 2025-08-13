"""
Master AI Optimization Utilities
AI-enhanced entry price calculation and risk assessment using Master AI insights.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional
from utils.indicators import get_support_resistance


def calculate_ai_optimized_buy_entry(current_price: float, predicted_price: float, 
                                   supports: List[float], resistances: List[float],
                                   master_confidence: float, master_proba: np.ndarray,
                                   individual_consensus: float, rsi: float, 
                                   atr: float, macd: float, macd_signal: float) -> Tuple[float, List[str], str]:
    """
    AI-optimized BUY entry price calculation using Master AI insights.
    
    Args:
        current_price: Current market price
        predicted_price: LSTM predicted price
        supports: List of support levels
        resistances: List of resistance levels  
        master_confidence: Master AI confidence level (0-1)
        master_proba: Probability distribution [SELL, HOLD, BUY]
        individual_consensus: Individual model consensus score (0-1)
        rsi: RSI value
        atr: ATR value
        macd: MACD value
        macd_signal: MACD signal line value
        
    Returns:
        Tuple[float, List[str], str]: (entry_price, strategy_reasons, risk_level)
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Base entry price from current price
    entry_price = current_price
    
    # Master AI confidence adjustment (exponential weighting)
    confidence_multiplier = np.exp(master_confidence * 2) - 1  # Range: 0 to ~6.39
    confidence_adjustment = (confidence_multiplier / 6.39) * 0.005  # Max 0.5% adjustment
    
    buy_probability = master_proba[2] if len(master_proba) >= 3 else 0.33
    
    # AI-enhanced entry logic
    if master_confidence >= 0.8 and buy_probability >= 0.7:
        # Very confident BUY - can pay small premium
        entry_price = current_price * (1 + confidence_adjustment)
        strategy_reasons.append(f"🤖 Master AI Very Confident BUY (Conf: {master_confidence:.1%}, Prob: {buy_probability:.1%})")
        risk_level = "LOW"
        
    elif master_confidence >= 0.6 and individual_consensus >= 0.7:
        # Good consensus - modest premium acceptable
        consensus_adj = (individual_consensus - 0.5) * 0.003  # Max 0.15% adjustment
        entry_price = current_price * (1 + consensus_adj)
        strategy_reasons.append(f"🤝 Strong Model Consensus ({individual_consensus:.1%}) + AI Confidence ({master_confidence:.1%})")
        
    else:
        # Lower confidence - wait for better price
        entry_price = current_price * 0.9995  # Small discount
        strategy_reasons.append(f"⚠️ Moderate AI Confidence ({master_confidence:.1%}) - Conservative Entry")
        risk_level = "HIGH"
    
    # Technical factor integration with AI weighting
    technical_weight = 0.3 + (master_confidence * 0.4)  # Weight technical factors by AI confidence
    
    # Support level optimization
    nearest_support = max([s for s in supports if s < current_price], default=current_price * 0.95)
    support_distance = (current_price - nearest_support) / current_price
    
    if support_distance < 0.01:  # Very close to support
        support_adjustment = -0.0005 * technical_weight  # Small discount
        entry_price *= (1 + support_adjustment)
        strategy_reasons.append(f"📊 Near Support Level (${nearest_support:.2f}) - Entry Discount Applied")
    
    # Resistance level consideration
    nearest_resistance = min([r for r in resistances if r > current_price], default=current_price * 1.05)
    resistance_distance = (nearest_resistance - current_price) / current_price
    
    if resistance_distance < 0.02:  # Close to resistance
        entry_price *= 0.9995  # Extra caution near resistance
        strategy_reasons.append(f"⚠️ Near Resistance (${nearest_resistance:.2f}) - Cautious Entry")
        risk_level = "HIGH"
    
    # RSI optimization with AI confidence weighting
    if rsi < 30:  # Oversold
        rsi_discount = (30 - rsi) / 100 * 0.002 * technical_weight
        entry_price *= (1 - rsi_discount)
        strategy_reasons.append(f"📈 RSI Oversold ({rsi:.1f}) - Entry Discount Applied")
    elif rsi > 70:  # Overbought
        entry_price *= 1.001  # Small premium for momentum
        strategy_reasons.append(f"🚀 RSI Overbought ({rsi:.1f}) - Momentum Premium")
        risk_level = "HIGH"
    
    # MACD momentum factor
    macd_momentum = macd - macd_signal
    if macd_momentum > 0:
        momentum_adj = min(macd_momentum / abs(macd_signal) * 0.001, 0.002) * technical_weight
        entry_price *= (1 + momentum_adj)
        strategy_reasons.append(f"⚡ Positive MACD Momentum - Entry Adjustment")
    
    # Volatility adjustment using ATR
    atr_percent = atr / current_price
    if atr_percent > 0.02:  # High volatility
        volatility_discount = atr_percent * 0.5 * technical_weight  # Max ~1% discount
        entry_price *= (1 - volatility_discount)
        strategy_reasons.append(f"🌪️ High Volatility (ATR: {atr_percent:.1%}) - Entry Discount")
        risk_level = "HIGH"
    
    # AI prediction alignment
    prediction_direction = (predicted_price - current_price) / current_price
    if prediction_direction > 0.01:  # LSTM strongly bullish
        pred_adjustment = min(prediction_direction * 0.1, 0.005) * master_confidence
        entry_price *= (1 + pred_adjustment)
        strategy_reasons.append(f"🧠 LSTM Bullish Prediction (+{prediction_direction:.1%}) - AI Alignment")
    
    return entry_price, strategy_reasons, risk_level


def calculate_ai_optimized_sell_entry(current_price: float, predicted_price: float,
                                    supports: List[float], resistances: List[float],
                                    master_confidence: float, master_proba: np.ndarray,
                                    individual_consensus: float, rsi: float,
                                    atr: float, macd: float, macd_signal: float) -> Tuple[float, List[str], str]:
    """
    AI-optimized SELL entry price calculation using Master AI insights.
    
    Args:
        current_price: Current market price
        predicted_price: LSTM predicted price
        supports: List of support levels
        resistances: List of resistance levels
        master_confidence: Master AI confidence level (0-1)
        master_proba: Probability distribution [SELL, HOLD, BUY]
        individual_consensus: Individual model consensus score (0-1)
        rsi: RSI value
        atr: ATR value
        macd: MACD value
        macd_signal: MACD signal line value
        
    Returns:
        Tuple[float, List[str], str]: (entry_price, strategy_reasons, risk_level)
    """
    strategy_reasons = []
    risk_level = "MEDIUM"
    
    # Base entry price from current price
    entry_price = current_price
    
    # Master AI confidence adjustment (exponential weighting)
    confidence_multiplier = np.exp(master_confidence * 2) - 1
    confidence_adjustment = (confidence_multiplier / 6.39) * 0.005  # Max 0.5% adjustment
    
    sell_probability = master_proba[0] if len(master_proba) >= 3 else 0.33
    
    # AI-enhanced entry logic for SELL
    if master_confidence >= 0.8 and sell_probability >= 0.7:
        # Very confident SELL - can accept small discount
        entry_price = current_price * (1 - confidence_adjustment)
        strategy_reasons.append(f"🤖 Master AI Very Confident SELL (Conf: {master_confidence:.1%}, Prob: {sell_probability:.1%})")
        risk_level = "LOW"
        
    elif master_confidence >= 0.6 and individual_consensus >= 0.7:
        # Good consensus - modest discount acceptable
        consensus_adj = (individual_consensus - 0.5) * 0.003
        entry_price = current_price * (1 - consensus_adj)
        strategy_reasons.append(f"🤝 Strong Model Consensus ({individual_consensus:.1%}) + AI Confidence ({master_confidence:.1%})")
        
    else:
        # Lower confidence - wait for better price
        entry_price = current_price * 1.0005  # Small premium
        strategy_reasons.append(f"⚠️ Moderate AI Confidence ({master_confidence:.1%}) - Conservative Entry")
        risk_level = "HIGH"
    
    # Technical factor integration with AI weighting
    technical_weight = 0.3 + (master_confidence * 0.4)
    
    # Resistance level optimization
    nearest_resistance = min([r for r in resistances if r > current_price], default=current_price * 1.05)
    resistance_distance = (nearest_resistance - current_price) / current_price
    
    if resistance_distance < 0.01:  # Very close to resistance
        resistance_adjustment = 0.0005 * technical_weight  # Small premium
        entry_price *= (1 + resistance_adjustment)
        strategy_reasons.append(f"📊 Near Resistance Level (${nearest_resistance:.2f}) - Entry Premium Applied")
    
    # Support level consideration
    nearest_support = max([s for s in supports if s < current_price], default=current_price * 0.95)
    support_distance = (current_price - nearest_support) / current_price
    
    if support_distance < 0.02:  # Close to support
        entry_price *= 1.0005  # Extra caution near support
        strategy_reasons.append(f"⚠️ Near Support (${nearest_support:.2f}) - Cautious Entry")
        risk_level = "HIGH"
    
    # RSI optimization with AI confidence weighting
    if rsi > 70:  # Overbought
        rsi_premium = (rsi - 70) / 100 * 0.002 * technical_weight
        entry_price *= (1 + rsi_premium)
        strategy_reasons.append(f"📉 RSI Overbought ({rsi:.1f}) - Entry Premium Applied")
    elif rsi < 30:  # Oversold
        entry_price *= 0.999  # Small discount for counter-trend
        strategy_reasons.append(f"🚀 RSI Oversold ({rsi:.1f}) - Counter-trend Risk")
        risk_level = "HIGH"
    
    # MACD momentum factor for SELL
    macd_momentum = macd - macd_signal
    if macd_momentum < 0:
        momentum_adj = min(abs(macd_momentum) / abs(macd_signal) * 0.001, 0.002) * technical_weight
        entry_price *= (1 - momentum_adj)
        strategy_reasons.append(f"⚡ Negative MACD Momentum - Entry Adjustment")
    
    # Volatility adjustment using ATR
    atr_percent = atr / current_price
    if atr_percent > 0.02:  # High volatility
        volatility_premium = atr_percent * 0.5 * technical_weight
        entry_price *= (1 + volatility_premium)
        strategy_reasons.append(f"🌪️ High Volatility (ATR: {atr_percent:.1%}) - Entry Premium")
        risk_level = "HIGH"
    
    # AI prediction alignment
    prediction_direction = (predicted_price - current_price) / current_price
    if prediction_direction < -0.01:  # LSTM strongly bearish
        pred_adjustment = min(abs(prediction_direction) * 0.1, 0.005) * master_confidence
        entry_price *= (1 - pred_adjustment)
        strategy_reasons.append(f"🧠 LSTM Bearish Prediction ({prediction_direction:.1%}) - AI Alignment")
    
    return entry_price, strategy_reasons, risk_level


def calculate_master_ai_risk_assessment(signal_data: Dict[str, Any], 
                                      technical_data: Dict[str, float],
                                      market_conditions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced AI-based risk assessment using multi-factor analysis.
    
    Args:
        signal_data: Master AI signal data with confidence and probabilities
        technical_data: Technical indicator values
        market_conditions: Market condition data (volatility, trend, etc.)
        
    Returns:
        Dict[str, Any]: Comprehensive risk assessment
    """
    risk_factors = {}
    overall_risk_score = 0.0
    risk_warnings = []
    
    # Master AI Risk Factors (40% weight)
    master_confidence = signal_data.get('confidence', 0.5)
    probability_dist = signal_data.get('probability_distribution', [0.33, 0.34, 0.33])
    consensus = signal_data.get('individual_consensus', 0.5)
    
    # Confidence risk (lower confidence = higher risk)
    confidence_risk = 1 - master_confidence
    risk_factors['ai_confidence_risk'] = confidence_risk
    overall_risk_score += confidence_risk * 0.2
    
    if master_confidence < 0.6:
        risk_warnings.append(f"Low AI Confidence ({master_confidence:.1%})")
    
    # Probability distribution uncertainty
    max_prob = np.max(probability_dist)
    prob_uncertainty = 1 - max_prob
    risk_factors['probability_uncertainty'] = prob_uncertainty
    overall_risk_score += prob_uncertainty * 0.1
    
    if max_prob < 0.5:
        risk_warnings.append(f"Uncertain Probability Distribution (Max: {max_prob:.1%})")
    
    # Model consensus risk
    consensus_risk = 1 - consensus
    risk_factors['consensus_risk'] = consensus_risk
    overall_risk_score += consensus_risk * 0.1
    
    if consensus < 0.6:
        risk_warnings.append(f"Low Model Consensus ({consensus:.1%})")
    
    # Technical Risk Factors (35% weight)
    rsi = technical_data.get('rsi', 50)
    atr_percent = technical_data.get('atr_percent', 0.01)
    macd_strength = technical_data.get('macd_strength', 0)
    
    # RSI extreme risk
    rsi_risk = 0
    if rsi > 80 or rsi < 20:
        rsi_risk = min(abs(rsi - 50) / 50, 1)
        risk_warnings.append(f"Extreme RSI Level ({rsi:.1f})")
    elif rsi > 70 or rsi < 30:
        rsi_risk = min(abs(rsi - 50) / 50 * 0.5, 0.5)
        
    risk_factors['rsi_risk'] = rsi_risk
    overall_risk_score += rsi_risk * 0.15
    
    # Volatility risk
    volatility_risk = min(atr_percent / 0.05, 1)  # Normalize to 5% ATR = max risk
    risk_factors['volatility_risk'] = volatility_risk
    overall_risk_score += volatility_risk * 0.15
    
    if atr_percent > 0.03:
        risk_warnings.append(f"High Volatility (ATR: {atr_percent:.1%})")
    
    # MACD momentum risk
    momentum_risk = max(0, -macd_strength) if macd_strength < 0 else 0
    risk_factors['momentum_risk'] = momentum_risk
    overall_risk_score += momentum_risk * 0.05
    
    # Market Condition Risk Factors (25% weight)
    trend_strength = market_conditions.get('trend_strength', 0.5)
    support_resistance_quality = market_conditions.get('sr_quality', 0.5)
    market_volatility = market_conditions.get('market_volatility', 0.5)
    
    # Trend weakness risk
    trend_risk = 1 - trend_strength
    risk_factors['trend_risk'] = trend_risk
    overall_risk_score += trend_risk * 0.1
    
    if trend_strength < 0.4:
        risk_warnings.append(f"Weak Trend Strength ({trend_strength:.1%})")
    
    # Support/Resistance quality risk
    sr_risk = 1 - support_resistance_quality
    risk_factors['sr_risk'] = sr_risk
    overall_risk_score += sr_risk * 0.1
    
    # Market volatility risk
    market_vol_risk = market_volatility
    risk_factors['market_volatility_risk'] = market_vol_risk
    overall_risk_score += market_vol_risk * 0.05
    
    # Normalize overall risk score
    overall_risk_score = min(max(overall_risk_score, 0), 1)
    
    # Determine risk level
    if overall_risk_score <= 0.3:
        risk_level = "LOW"
        risk_color = "🟢"
    elif overall_risk_score <= 0.6:
        risk_level = "MEDIUM"
        risk_color = "🟡"
    else:
        risk_level = "HIGH"
        risk_color = "🔴"
    
    return {
        'overall_risk_score': overall_risk_score,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_factors': risk_factors,
        'risk_warnings': risk_warnings,
        'risk_analysis': {
            'ai_risk_contribution': sum([risk_factors.get(k, 0) for k in ['ai_confidence_risk', 'probability_uncertainty', 'consensus_risk']]) * 100 / 3,
            'technical_risk_contribution': sum([risk_factors.get(k, 0) for k in ['rsi_risk', 'volatility_risk', 'momentum_risk']]) * 100 / 3,
            'market_risk_contribution': sum([risk_factors.get(k, 0) for k in ['trend_risk', 'sr_risk', 'market_volatility_risk']]) * 100 / 3
        }
    }


def calculate_ai_position_sizing(signal_data: Dict[str, Any], 
                               risk_assessment: Dict[str, Any],
                               account_balance: float,
                               base_risk_percent: float) -> Dict[str, Any]:
    """
    AI-optimized position sizing based on Master AI confidence and risk assessment.
    
    Args:
        signal_data: Master AI signal data
        risk_assessment: Risk assessment results
        account_balance: Account balance
        base_risk_percent: Base risk percentage
        
    Returns:
        Dict[str, Any]: Optimized position sizing recommendations
    """
    master_confidence = signal_data.get('confidence', 0.5)
    overall_risk_score = risk_assessment.get('overall_risk_score', 0.5)
    
    # AI-enhanced risk adjustment
    confidence_multiplier = np.sqrt(master_confidence)  # Square root to be conservative
    risk_multiplier = 1 - overall_risk_score
    
    # Combined AI adjustment factor
    ai_adjustment = (confidence_multiplier * risk_multiplier) ** 0.5
    
    # Adjusted risk percentage
    adjusted_risk_percent = base_risk_percent * ai_adjustment
    adjusted_risk_percent = max(0.1, min(adjusted_risk_percent, base_risk_percent * 1.5))  # Bounds
    
    # Risk amount calculation
    risk_amount = account_balance * (adjusted_risk_percent / 100)
    
    return {
        'base_risk_percent': base_risk_percent,
        'adjusted_risk_percent': adjusted_risk_percent,
        'ai_adjustment_factor': ai_adjustment,
        'confidence_multiplier': confidence_multiplier,
        'risk_multiplier': risk_multiplier,
        'risk_amount': risk_amount,
        'max_position_value': risk_amount * 10,  # Conservative 10:1 max leverage equivalent
        'recommendation': f"AI-Optimized Risk: {adjusted_risk_percent:.2f}% (Base: {base_risk_percent:.1f}%)"
    }