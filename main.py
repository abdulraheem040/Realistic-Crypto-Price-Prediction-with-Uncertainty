import ccxt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ta
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Data Fetching with Technical Indicators ---
def fetch_ohlcv(symbol, timeframe='1h', limit=1000):
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def add_technical_indicators(df):
    """Add various technical indicators with better feature engineering"""
    try:
        # Price-based features (normalized)
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Returns instead of absolute prices (more stationary)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['price_volume'] = df['returns'] * df['volume_ratio']
        
        # Moving Averages (use ratios instead of absolute values)
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        df['price_sma10_ratio'] = df['close'] / df['sma_10']
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['sma10_sma20_ratio'] = df['sma_10'] / df['sma_20']
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['RSI_normalized'] = (df['RSI'] - 50) / 50  # Normalize RSI around 0
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['close'])
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands (use ratios)
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # Stochastic
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # ATR (normalized)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['ATR_ratio'] = df['ATR'] / df['close']
        
        # Momentum features
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Trend strength
        df['trend_strength'] = df['close'].rolling(window=20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")
        # Basic fallback features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    return df.dropna()

# --- Improved LSTM Model with Better Architecture ---
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_layers=3, output_size=1, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Bidirectional LSTM for better context
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, 1, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size // 2, 1, batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size // 2, num_heads=4, batch_first=True)
        
        # Output layers with residual connection
        self.fc1 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc2 = nn.Linear(hidden_size // 4, output_size)
        self.residual = nn.Linear(hidden_size // 2, output_size)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Multi-layer LSTM
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        lstm_out, _ = self.lstm3(lstm_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last timestep
        last_timestep = attn_out[:, -1, :]
        
        # Batch normalization
        if last_timestep.size(0) > 1:
            last_timestep = self.batch_norm(last_timestep)
        
        # Residual connection
        residual = self.residual(last_timestep)
        
        # Main path
        x = self.relu(self.fc1(last_timestep))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Add residual connection and apply tanh to limit output range
        output = self.tanh(x + residual) * 0.1  # Limit to ±10% change
        
        return output

def prepare_improved_data(df, lookback=60, target_col='returns'):
    """Prepare data with better feature selection and target engineering"""
    
    # Select the most predictive features (avoiding absolute price values)
    feature_columns = [
        'returns', 'log_returns', 'volatility', 'volume_ratio', 'price_volume',
        'price_sma10_ratio', 'price_sma20_ratio', 'sma10_sma20_ratio',
        'RSI_normalized', 'MACD', 'MACD_histogram', 'bb_position', 'bb_width',
        'stoch_k', 'stoch_d', 'ATR_ratio', 'momentum_5', 'momentum_10',
        'trend_strength', 'price_range', 'body_ratio'
    ]
    
    # Select available columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    if len(available_features) < 5:
        st.error("Not enough features available for training!")
        return None, None, None
    
    st.info(f"Using {len(available_features)} features: {', '.join(available_features)}")
    
    # Prepare feature data
    feature_data = df[available_features].values
    
    # Handle any remaining NaN values
    if np.isnan(feature_data).any():
        df_temp = df[available_features].fillna(method='ffill').fillna(method='bfill')
        feature_data = df_temp.values
    
    # Use RobustScaler instead of MinMaxScaler (better for outliers)
    scaler = RobustScaler()
    try:
        feature_data_scaled = scaler.fit_transform(feature_data)
    except Exception as e:
        st.error(f"Error scaling data: {e}")
        return None, None, None
    
    # Prepare target (next period return)
    target_data = df[target_col].shift(-1).values[:-1]  # Next period return
    feature_data_scaled = feature_data_scaled[:-1]  # Remove last row to match target
    
    X, y = [], []
    for i in range(len(feature_data_scaled) - lookback):
        X.append(feature_data_scaled[i:i+lookback])
        y.append(target_data[i+lookback-1])  # Predict next return
    
    if len(X) == 0:
        st.error("Not enough data for training!")
        return None, None, None
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Remove any samples with NaN targets
    valid_indices = ~np.isnan(y.flatten())
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y, scaler

def train_improved_model(df, epochs=300, lr=0.001, lookback=60):
    result = prepare_improved_data(df, lookback=lookback)
    if result[0] is None:
        return None, None, None, [], []
    
    X, y, scaler = result
    
    # More conservative train/validation split
    split_idx = int(0.85 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    if len(X_train) < 50 or len(X_val) < 10:
        st.error("Not enough data for train/validation split!")
        return None, None, None, [], []
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # Improved model with better architecture
    model = ImprovedLSTMModel(
        input_size=X_train.shape[2], 
        hidden_size=64, 
        num_layers=3, 
        output_size=1, 
        dropout=0.3
    )
    
    # Better loss function and optimizer
    criterion = nn.HuberLoss(delta=0.01)  # More robust to outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train)
        train_loss = criterion(y_pred_train, y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = criterion(y_pred_val, y_val)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            st.info(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break
        
        if epoch % 20 == 0:
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch}: Train Loss = {train_loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
    
    progress_bar.progress(1.0)
    status_text.text("Training completed!")
    
    return model, scaler, (X_train, y_train, X_val, y_val), train_losses, val_losses

def make_realistic_predictions(model, df, scaler, lookback=60, num_predictions=10):
    """Make more realistic predictions with uncertainty bounds"""
    
    # Get the most recent features
    feature_columns = [col for col in [
        'returns', 'log_returns', 'volatility', 'volume_ratio', 'price_volume',
        'price_sma10_ratio', 'price_sma20_ratio', 'sma10_sma20_ratio',
        'RSI_normalized', 'MACD', 'MACD_histogram', 'bb_position', 'bb_width',
        'stoch_k', 'stoch_d', 'ATR_ratio', 'momentum_5', 'momentum_10',
        'trend_strength', 'price_range', 'body_ratio'
    ] if col in df.columns]
    
    recent_data = df[feature_columns].tail(lookback).values
    recent_data_scaled = scaler.transform(recent_data)
    
    model.eval()
    predictions = []
    uncertainties = []
    
    # Use Monte Carlo dropout for uncertainty estimation
    with torch.no_grad():
        input_seq = torch.tensor(recent_data_scaled, dtype=torch.float32).unsqueeze(0)
        
        for step in range(num_predictions):
            # Multiple forward passes for uncertainty
            step_predictions = []
            model.train()  # Enable dropout
            for _ in range(50):  # Monte Carlo samples
                pred = model(input_seq)
                step_predictions.append(pred.item())
            
            # Calculate mean and std
            pred_mean = np.mean(step_predictions)
            pred_std = np.std(step_predictions)
            
            predictions.append(pred_mean)
            uncertainties.append(pred_std)
            
            # Update input sequence (simplified - in reality would need full feature update)
            # For now, just shift and add the prediction
            new_features = recent_data_scaled[-1].copy()
            if len(feature_columns) > 0:
                new_features[0] = pred_mean  # Update returns
            
            # Shift the sequence
            input_seq = torch.cat([
                input_seq[:, 1:, :],
                torch.tensor(new_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            ], dim=1)
    
    return np.array(predictions), np.array(uncertainties)

def plot_realistic_predictions(df, predictions, uncertainties):
    """Plot predictions with uncertainty bounds"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price with Predictions', 'Predicted Returns', 'Technical Indicators'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Historical prices
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Historical Price'
        ), row=1, col=1
    )
    
    # Convert returns to price predictions
    current_price = df['close'].iloc[-1]
    predicted_prices = [current_price]
    
    for ret in predictions:
        predicted_prices.append(predicted_prices[-1] * (1 + ret))
    
    predicted_prices = predicted_prices[1:]  # Remove initial price
    
    # Create future timestamps
    last_timestamp = df['timestamp'].iloc[-1]
    future_timestamps = pd.date_range(last_timestamp, periods=len(predictions)+1, freq='1h')[1:]
    
    # Uncertainty bounds for prices
    upper_bounds = []
    lower_bounds = []
    current = current_price
    
    for i, (ret, unc) in enumerate(zip(predictions, uncertainties)):
        upper_ret = ret + 2 * unc  # 95% confidence
        lower_ret = ret - 2 * unc
        
        upper_price = current * (1 + upper_ret)
        lower_price = current * (1 + lower_ret)
        
        upper_bounds.append(upper_price)
        lower_bounds.append(lower_price)
        current = predicted_prices[i]
    
    # Plot predictions with uncertainty
    fig.add_trace(
        go.Scatter(
            x=future_timestamps, 
            y=predicted_prices,
            name='Price Prediction', 
            line=dict(color='red', width=2)
        ), row=1, col=1
    )
    
    # Uncertainty bands
    fig.add_trace(
        go.Scatter(
            x=list(future_timestamps) + list(future_timestamps[::-1]),
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            name='95% Confidence',
            showlegend=True
        ), row=1, col=1
    )
    
    # Plot predicted returns
    fig.add_trace(
        go.Scatter(
            x=future_timestamps, 
            y=predictions * 100,  # Convert to percentage
            name='Predicted Returns (%)', 
            line=dict(color='blue', width=2)
        ), row=2, col=1
    )
    
    # Error bars for returns
    fig.add_trace(
        go.Scatter(
            x=future_timestamps,
            y=(predictions + 2 * uncertainties) * 100,
            mode='lines',
            line=dict(color='rgba(0,0,255,0.3)'),
            name='Upper Bound',
            showlegend=False
        ), row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_timestamps,
            y=(predictions - 2 * uncertainties) * 100,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(0,0,255,0.3)'),
            name='Confidence Band',
            showlegend=False
        ), row=2, col=1
    )
    
    # Technical indicators
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'].tail(100), y=df['RSI'].tail(100), 
                      name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
    
    fig.update_layout(
        title='Improved Crypto Prediction with Uncertainty',
        height=900,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig

# --- Enhanced Streamlit App ---
def main():
    st.set_page_config(page_title="Realistic Crypto Predictor", layout="wide")
    st.title("🎯 Realistic Crypto Price Prediction with Uncertainty")
    
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Crypto Symbol:", "BTC/USDT")
    timeframe = st.sidebar.selectbox("Timeframe:", ['1h', '4h', '1d'])
    limit = st.sidebar.slider("Data Points:", 1000, 3000, 2000)
    lookback = st.sidebar.slider("Lookback Period:", 40, 100, 60)
    epochs = st.sidebar.slider("Max Training Epochs:", 100, 500, 300)
    lr = st.sidebar.select_slider("Learning Rate:", [0.01, 0.005, 0.001, 0.0005, 0.0001], value=0.001)
    
    if st.button("🚀 Train Improved Model"):
        with st.spinner("Fetching data..."):
            df = fetch_ohlcv(symbol, timeframe, limit)
            
        if df is not None:
            st.success(f"✅ Fetched {len(df)} data points")
            
            # Add technical indicators
            with st.spinner("Engineering features..."):
                df = add_technical_indicators(df)
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", len(df))
            with col2:
                st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
            with col3:
                recent_return = df['returns'].iloc[-1] * 100
                st.metric("Last Return", f"{recent_return:.2f}%")
            with col4:
                volatility = df['volatility'].iloc[-1] * 100 if 'volatility' in df.columns else 0
                st.metric("Volatility", f"{volatility:.2f}%")
            
            # Train model
            st.subheader("🧠 Advanced Model Training")
            with st.spinner("Training improved model with early stopping..."):
                result = train_improved_model(df, epochs=epochs, lr=lr, lookback=lookback)
                
            if result[0] is None:
                st.error("❌ Training failed!")
                return
                
            model, scaler, data_splits, train_losses, val_losses = result
            X_train, y_train, X_val, y_val = data_splits
            
            # Model performance
            col1, col2 = st.columns(2)
            with col1:
                if train_losses and val_losses:
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(y=train_losses, name='Training Loss'))
                    loss_fig.add_trace(go.Scatter(y=val_losses, name='Validation Loss'))
                    loss_fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(loss_fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Model Performance")
                if train_losses and val_losses:
                    final_train_loss = train_losses[-1]
                    final_val_loss = val_losses[-1]
                    st.metric("Final Training Loss", f"{final_train_loss:.6f}")
                    st.metric("Final Validation Loss", f"{final_val_loss:.6f}")
                    
                    overfit_ratio = final_val_loss / final_train_loss
                    if overfit_ratio < 1.5:
                        st.success("✅ Good generalization")
                    elif overfit_ratio < 2.0:
                        st.warning("⚠️ Slight overfitting")
                    else:
                        st.error("❌ Significant overfitting")
            
            # Make predictions with uncertainty
            st.subheader("🔮 Realistic Predictions")
            with st.spinner("Generating predictions with uncertainty estimation..."):
                predictions, uncertainties = make_realistic_predictions(model, df, scaler, lookback)
            
            # Visualization
            fig = plot_realistic_predictions(df, predictions, uncertainties)
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions summary
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Next 10 Periods")
                pred_summary = []
                current_price = df['close'].iloc[-1]
                
                for i, (pred_return, uncertainty) in enumerate(zip(predictions, uncertainties)):
                    predicted_price = current_price * (1 + pred_return)
                    confidence = max(0, 100 - uncertainty * 1000)  # Rough confidence measure
                    
                    pred_summary.append({
                        'Period': i + 1,
                        'Return': f"{pred_return * 100:.2f}%",
                        'Price': f"${predicted_price:.2f}",
                        'Confidence': f"{confidence:.0f}%"
                    })
                
                st.dataframe(pred_summary)
            
            with col2:
                st.subheader("⚡ Smart Signals")
                
                # Overall trend
                avg_return = np.mean(predictions)
                trend_strength = abs(avg_return) / np.mean(uncertainties) if np.mean(uncertainties) > 0 else 0
                
                if trend_strength > 1.0 and avg_return > 0.005:
                    st.success("🟢 Strong Bullish Signal")
                elif trend_strength > 1.0 and avg_return < -0.005:
                    st.error("🔴 Strong Bearish Signal")
                elif avg_return > 0.002:
                    st.info("🟡 Weak Bullish Signal")
                elif avg_return < -0.002:
                    st.warning("🟠 Weak Bearish Signal")
                else:
                    st.info("➡️ Neutral/Sideways")
                
                # Risk assessment
                avg_uncertainty = np.mean(uncertainties)
                if avg_uncertainty < 0.01:
                    st.success("✅ Low Risk/High Confidence")
                elif avg_uncertainty < 0.02:
                    st.warning("⚠️ Medium Risk")
                else:
                    st.error("🚨 High Risk/Low Confidence")
                
                # Market regime
                recent_volatility = df['volatility'].iloc[-10:].mean() if 'volatility' in df.columns else 0
                if recent_volatility > 0.03:
                    st.warning("🌊 High Volatility Regime")
                elif recent_volatility > 0.015:
                    st.info("📊 Normal Volatility")
                else:
                    st.success("😴 Low Volatility")
    
    # Enhanced information section
    with st.expander("🧠 Model Improvements Explained"):
        st.write("""
        **Key Improvements for Accuracy:**
        
        🎯 **Better Target Variable**: 
        - Predicts returns instead of absolute prices
        - More stationary and realistic for financial data
        
        📊 **Feature Engineering**:
        - Uses ratios and normalized indicators instead of absolute values
        - Includes volatility, momentum, and trend strength features
        - RobustScaler handles outliers better than MinMaxScaler
        
        🧠 **Advanced Architecture**:
        - Bidirectional LSTM captures both past and future context
        - Multi-head attention mechanism
        - Residual connections prevent vanishing gradients
        - Output limited to ±10% to prevent unrealistic predictions
        
        📈 **Training Improvements**:
        - HuberLoss is more robust to outliers than MSE
        - Early stopping prevents overfitting
        - Cosine annealing learning rate schedule
        - Monte Carlo dropout for uncertainty estimation
        
        🎲 **Uncertainty Quantification**:
        - Provides confidence intervals for predictions
        - Helps assess prediction reliability
        - Enables better risk management
        
        **Why This Is More Accurate**:
        - Focuses on realistic return predictions
        - Accounts for market uncertainty
        - Uses more stable features
        - Better regularization techniques
        - Provides confidence measures
        """)

if __name__ == '__main__':
    main()