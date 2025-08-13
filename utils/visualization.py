"""
Master AI Visualization Utilities
Rich visualization components for Master AI dashboard and insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Import the enhanced price formatting function
def format_price_for_display(symbol, price, data_source=None):
    """
    Enhanced price formatting for visualization displays
    """
    try:
        price = float(price)
        symbol_upper = symbol.upper().replace('/', '')
        
        # Gold/XAU formatting
        if any(gold in symbol_upper for gold in ['XAU', 'GOLD']):
            return f"${price:,.2f}"
        
        # USDT crypto pairs (check before individual crypto check)
        elif symbol_upper in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']:
            if price >= 100:
                return f"${price:,.2f}"
            elif price >= 1:
                return f"${price:.4f}"
            else:
                return f"${price:.6f}"
        
        # Cryptocurrency formatting
        elif any(crypto in symbol_upper for crypto in ['BTC', 'BITCOIN']):
            if price >= 1000:
                return f"${price:,.1f}"
            else:
                return f"${price:.2f}"
        
        elif any(crypto in symbol_upper for crypto in ['ETH', 'ETHEREUM']):
            return f"${price:,.2f}"
        
        # Forex pairs
        elif any(fx in symbol_upper for fx in ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']):
            return f"{price:.5f}"
        
        elif any(jpy in symbol_upper for jpy in ['JPY', 'USDJPY', 'EURJPY', 'GBPJPY']):
            return f"{price:.3f}"
        
        # Default formatting based on price range
        else:
            if price >= 1000:
                return f"${price:,.2f}"
            elif price >= 1:
                return f"${price:.4f}"
            else:
                return f"${price:.6f}"
                
    except (ValueError, TypeError):
        return str(price)


def create_confidence_visualization(confidence: float, title: str = "Master AI Confidence") -> go.Figure:
    """
    Create confidence gauge visualization for Master AI.
    
    Args:
        confidence: Confidence level (0-1)
        title: Gauge title
        
    Returns:
        go.Figure: Plotly gauge chart
    """
    # Color coding based on confidence level
    if confidence >= 0.8:
        color = "green"
        status = "Very High"
    elif confidence >= 0.6:
        color = "lightgreen" 
        status = "High"
    elif confidence >= 0.4:
        color = "yellow"
        status = "Medium"
    elif confidence >= 0.2:
        color = "orange"
        status = "Low"
    else:
        color = "red"
        status = "Very Low"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{title}<br><span style='font-size:12px;color:gray'>{status}</span>"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'lightgray'},
                {'range': [20, 40], 'color': 'lightyellow'},
                {'range': [40, 60], 'color': 'lightblue'},
                {'range': [60, 80], 'color': 'lightgreen'},
                {'range': [80, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number = {'suffix': "%", 'font': {'size': 20}}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_probability_distribution_chart(probabilities: np.ndarray, 
                                         chart_type: str = "bar") -> go.Figure:
    """
    Create probability distribution visualization for Master AI predictions.
    
    Args:
        probabilities: Array of probabilities [SELL, HOLD, BUY]
        chart_type: Type of chart ("bar" or "pie")
        
    Returns:
        go.Figure: Plotly chart
    """
    labels = ['SELL', 'HOLD', 'BUY']
    colors = ['#ff4444', '#ffaa00', '#44ff44']
    
    # Ensure probabilities sum to 1
    if len(probabilities) == 3:
        probabilities = probabilities / np.sum(probabilities)
    else:
        probabilities = np.array([0.33, 0.34, 0.33])
    
    if chart_type == "pie":
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=probabilities,
            hole=.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Probability: %{percent}<br><extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': "Master AI Probability Distribution",
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
    else:  # bar chart
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<br><extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': "Master AI Probability Distribution",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Signal",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 100]),
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )
    
    return fig


def create_consensus_heatmap(individual_predictions: Dict[str, Any],
                           individual_confidences: Dict[str, Any]) -> go.Figure:
    """
    Create model consensus heatmap visualization.
    
    Args:
        individual_predictions: Dictionary of individual model predictions
        individual_confidences: Dictionary of individual model confidences
        
    Returns:
        go.Figure: Plotly heatmap
    """
    models = ['LSTM', 'XGBoost', 'CNN', 'SVC', 'Naive Bayes']
    signals = ['SELL', 'HOLD', 'BUY']
    
    # Create consensus matrix
    consensus_matrix = np.zeros((len(models), len(signals)))
    
    # Map predictions to matrix
    pred_mapping = {-1: 0, 0: 1, 1: 2}  # SELL, HOLD, BUY
    
    for i, model in enumerate(['lstm', 'xgb', 'cnn', 'svc', 'nb']):
        pred = individual_predictions.get(model, 0)
        conf = individual_confidences.get(model, 0.5)
        
        # Use confidence as the intensity
        signal_idx = pred_mapping.get(pred, 1)  # Default to HOLD
        consensus_matrix[i, signal_idx] = conf
    
    # Create hover text
    hover_text = []
    for i, model in enumerate(models):
        hover_row = []
        for j, signal in enumerate(signals):
            confidence = consensus_matrix[i, j]
            if confidence > 0:
                hover_row.append(f'{model}<br>{signal}<br>Confidence: {confidence:.1%}')
            else:
                hover_row.append(f'{model}<br>{signal}<br>No prediction')
        hover_text.append(hover_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=consensus_matrix,
        x=signals,
        y=models,
        colorscale='RdYlGn',
        text=hover_text,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(
            title="Confidence",
            tickformat=".0%"
        ),
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Model Consensus Heatmap",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Signal Type",
        yaxis_title="AI Models",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_risk_assessment_chart(risk_assessment: Dict[str, Any]) -> go.Figure:
    """
    Create comprehensive risk assessment visualization.
    
    Args:
        risk_assessment: Risk assessment data
        
    Returns:
        go.Figure: Plotly risk breakdown chart
    """
    risk_factors = risk_assessment.get('risk_factors', {})
    risk_analysis = risk_assessment.get('risk_analysis', {})
    
    # Main risk categories
    categories = ['AI Risk', 'Technical Risk', 'Market Risk']
    values = [
        risk_analysis.get('ai_risk_contribution', 0),
        risk_analysis.get('technical_risk_contribution', 0),
        risk_analysis.get('market_risk_contribution', 0)
    ]
    
    # Color coding
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Risk Contribution: %{y:.1f}%<br><extra></extra>'
    )])
    
    overall_risk = risk_assessment.get('overall_risk_score', 0) * 100
    risk_level = risk_assessment.get('risk_level', 'MEDIUM')
    
    fig.update_layout(
        title={
            'text': f"Risk Assessment Breakdown<br><span style='font-size:12px'>Overall Risk: {overall_risk:.1f}% ({risk_level})</span>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Risk Categories",
        yaxis_title="Risk Contribution (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def create_ai_insights_timeline(signal_history: List[Dict[str, Any]], 
                              max_points: int = 20) -> go.Figure:
    """
    Create Master AI insights timeline visualization.
    
    Args:
        signal_history: List of historical signal data
        max_points: Maximum number of points to display
        
    Returns:
        go.Figure: Plotly timeline chart
    """
    if not signal_history:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        fig.update_layout(height=300, title="Master AI Signal Timeline")
        return fig
    
    # Limit data points
    recent_history = signal_history[-max_points:]
    
    timestamps = [entry.get('timestamp', i) for i, entry in enumerate(recent_history)]
    confidences = [entry.get('confidence', 0.5) for entry in recent_history]
    signals = [entry.get('signal', 0) for entry in recent_history]
    
    # Color mapping for signals
    signal_colors = {-1: 'red', 0: 'gray', 1: 'green'}
    colors = [signal_colors.get(s, 'gray') for s in signals]
    signal_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    signal_labels = [signal_names.get(s, 'HOLD') for s in signals]
    
    fig = go.Figure()
    
    # Add confidence line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[c * 100 for c in confidences],
        mode='lines+markers',
        name='Confidence',
        line=dict(color='blue', width=2),
        marker=dict(
            size=8,
            color=colors,
            line=dict(color='white', width=1)
        ),
        text=signal_labels,
        hovertemplate='<b>Signal: %{text}</b><br>Confidence: %{y:.1f}%<br>Time: %{x}<br><extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Master AI Signal Timeline",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Time",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    return fig


def create_performance_metrics_display(performance_data: Dict[str, Any]) -> None:
    """
    Create Master AI performance metrics display using Streamlit columns.
    
    Args:
        performance_data: Dictionary containing performance metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = performance_data.get('accuracy', 0) * 100
        st.metric(
            "Signal Accuracy",
            f"{accuracy:.1f}%",
            delta=f"+{performance_data.get('accuracy_improvement', 0):.1f}%",
            help="Master AI signal prediction accuracy"
        )
    
    with col2:
        avg_confidence = performance_data.get('avg_confidence', 0) * 100
        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.1f}%",
            delta=f"+{performance_data.get('confidence_trend', 0):.1f}%",
            help="Average Master AI confidence level"
        )
    
    with col3:
        consensus_rate = performance_data.get('consensus_rate', 0) * 100
        st.metric(
            "Model Consensus",
            f"{consensus_rate:.1f}%",
            delta=f"+{performance_data.get('consensus_improvement', 0):.1f}%",
            help="Rate of agreement between individual models"
        )
    
    with col4:
        risk_adjusted_return = performance_data.get('risk_adjusted_return', 0) * 100
        st.metric(
            "Risk-Adj Return",
            f"{risk_adjusted_return:.1f}%",
            delta=f"+{performance_data.get('return_improvement', 0):.1f}%",
            help="Risk-adjusted trading return"
        )


def display_master_ai_insights_panel(signal_data: Dict[str, Any],
                                    risk_assessment: Dict[str, Any],
                                    technical_data: Dict[str, Any]) -> None:
    """
    Display comprehensive Master AI insights panel.
    
    Args:
        signal_data: Master AI signal data
        risk_assessment: Risk assessment results
        technical_data: Technical analysis data
    """
    st.markdown("### 🤖 **Master AI Insights Panel**")
    
    # Main insights in three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = signal_data.get('confidence', 0.5)
        fig_confidence = create_confidence_visualization(confidence)
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    with col2:
        probabilities = signal_data.get('probability_distribution', [0.33, 0.34, 0.33])
        fig_prob = create_probability_distribution_chart(probabilities, "bar")
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with col3:
        fig_risk = create_risk_assessment_chart(risk_assessment)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Detailed insights in expander
    with st.expander("📊 Detailed Master AI Analysis", expanded=False):
        
        # Individual model predictions (mock data for now)
        st.markdown("#### Individual Model Predictions")
        individual_preds = {
            'lstm': signal_data.get('signal', 0),
            'xgb': signal_data.get('signal', 0), 
            'cnn': signal_data.get('signal', 0),
            'svc': signal_data.get('signal', 0),
            'nb': signal_data.get('signal', 0)
        }
        individual_confs = {
            'lstm': confidence,
            'xgb': confidence * 0.9,
            'cnn': confidence * 0.85,
            'svc': confidence * 0.95,
            'nb': confidence * 0.8
        }
        
        fig_consensus = create_consensus_heatmap(individual_preds, individual_confs)
        st.plotly_chart(fig_consensus, use_container_width=True)
        
        # Risk warnings
        risk_warnings = risk_assessment.get('risk_warnings', [])
        if risk_warnings:
            st.markdown("#### ⚠️ Risk Warnings")
            for warning in risk_warnings:
                st.warning(warning)
        
        # Technical factor analysis
        st.markdown("#### 📈 Technical Factor Analysis")
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            rsi = technical_data.get('rsi', 50)
            rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            st.info(f"**RSI:** {rsi:.1f} ({rsi_status})")
            
            atr = technical_data.get('atr_percent', 0.01)
            vol_status = "High" if atr > 0.03 else "Normal" if atr > 0.01 else "Low"
            st.info(f"**Volatility (ATR):** {atr:.2%} ({vol_status})")
        
        with tech_col2:
            macd_strength = technical_data.get('macd_strength', 0)
            macd_status = "Bullish" if macd_strength > 0 else "Bearish" if macd_strength < 0 else "Neutral"
            st.info(f"**MACD Momentum:** {macd_status}")
            
            consensus = signal_data.get('individual_consensus', 0.5)
            consensus_status = "High" if consensus > 0.7 else "Medium" if consensus > 0.5 else "Low"
            st.info(f"**Model Consensus:** {consensus:.1%} ({consensus_status})")


def create_entry_optimization_display(entry_data: Dict[str, Any],
                                    current_price: float,
                                    symbol: str = "UNKNOWN") -> None:
    """
    Display entry price optimization analysis.
    
    Args:
        entry_data: Entry price optimization data
        current_price: Current market price
        symbol: Trading symbol for proper price formatting
    """
    st.markdown("#### 🎯 **AI Entry Price Optimization**")
    
    entry_price = entry_data.get('entry_price', current_price)
    strategy_reasons = entry_data.get('strategy_reasons', [])
    fill_probability = entry_data.get('expected_fill_probability', 0.5)
    
    # Price comparison
    price_diff = (entry_price - current_price) / current_price * 100
    price_direction = "Premium" if price_diff > 0 else "Discount" if price_diff < 0 else "Market"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Optimized Entry",
            format_price_for_display(symbol, entry_price),
            delta=f"{price_diff:+.2f}% {price_direction}"
        )
    
    with col2:
        st.metric(
            "Fill Probability",
            f"{fill_probability:.1%}",
            help="Estimated probability of order execution"
        )
    
    with col3:
        entry_score = fill_probability * (1 - abs(price_diff/100))
        st.metric(
            "Entry Score",
            f"{entry_score:.2f}",
            help="Combined score of price optimization and fill probability"
        )
    
    # Strategy reasoning
    if strategy_reasons:
        st.markdown("**🧠 AI Optimization Strategy:**")
        for i, reason in enumerate(strategy_reasons, 1):
            st.markdown(f"**{i}.** {reason}")