-- Stock Arbitrage Model Database Schema
-- PostgreSQL 14+

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- REFERENCE DATA TABLES
-- ============================================================================

-- Stocks reference table
CREATE TABLE IF NOT EXISTS stocks (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255),
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stocks_exchange ON stocks(exchange);
CREATE INDEX idx_stocks_sector ON stocks(sector);

-- ============================================================================
-- MARKET DATA TABLES
-- ============================================================================

-- Stock prices (OHLCV data)
-- Partitioned by date for performance
CREATE TABLE IF NOT EXISTS prices (
    id BIGSERIAL,
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    date DATE NOT NULL,
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(12, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
) PARTITION BY RANGE (date);

-- Create partitions for recent years (example for 2024-2026)
CREATE TABLE IF NOT EXISTS prices_2024 PARTITION OF prices
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS prices_2025 PARTITION OF prices
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS prices_2026 PARTITION OF prices
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX idx_prices_symbol_date ON prices(symbol, date);
CREATE INDEX idx_prices_date ON prices(date);

-- Technical indicators
CREATE TABLE IF NOT EXISTS technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    date DATE NOT NULL,
    rsi DECIMAL(8, 4),
    macd DECIMAL(12, 6),
    macd_signal DECIMAL(12, 6),
    macd_histogram DECIMAL(12, 6),
    ma_5 DECIMAL(12, 4),
    ma_20 DECIMAL(12, 4),
    ma_50 DECIMAL(12, 4),
    ma_200 DECIMAL(12, 4),
    volume_ma_5 BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE INDEX idx_technical_indicators_symbol_date ON technical_indicators(symbol, date);

-- Sentiment data
CREATE TABLE IF NOT EXISTS sentiment_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    date DATE NOT NULL,
    source VARCHAR(50) NOT NULL,  -- 'reddit', 'twitter', 'news', etc.
    mean_sentiment DECIMAL(6, 4),
    median_sentiment DECIMAL(6, 4),
    std_sentiment DECIMAL(6, 4),
    post_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date, source)
);

CREATE INDEX idx_sentiment_symbol_date ON sentiment_data(symbol, date);
CREATE INDEX idx_sentiment_source ON sentiment_data(source);

-- Options data
CREATE TABLE IF NOT EXISTS options_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    date DATE NOT NULL,
    total_volume BIGINT,
    calls_volume BIGINT,
    puts_volume BIGINT,
    contract_count INTEGER,
    implied_volatility DECIMAL(8, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE INDEX idx_options_symbol_date ON options_data(symbol, date);

-- ============================================================================
-- MODEL TABLES
-- ============================================================================

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'lstm', 'xgboost', etc.
    framework VARCHAR(50),  -- 'tensorflow', 'scikit-learn', etc.
    file_path TEXT NOT NULL,
    hyperparameters JSONB,
    training_date TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'inactive', 'testing'
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_training_date ON models(training_date);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    metric_type VARCHAR(50) NOT NULL,  -- 'train', 'validation', 'test', 'production'
    mse DECIMAL(12, 6),
    mae DECIMAL(12, 6),
    r_squared DECIMAL(8, 6),
    directional_accuracy DECIMAL(6, 4),
    sharpe_ratio DECIMAL(8, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, metric_date, metric_type)
);

CREATE INDEX idx_model_metrics_model_id ON model_metrics(model_id);
CREATE INDEX idx_model_metrics_metric_date ON model_metrics(metric_date);

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES models(id),
    prediction_date DATE NOT NULL,
    symbol_1 VARCHAR(10) NOT NULL,
    symbol_2 VARCHAR(10) NOT NULL,
    predicted_difference DECIMAL(12, 6) NOT NULL,
    actual_difference DECIMAL(12, 6),
    confidence DECIMAL(6, 4),
    features JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_model_id_date ON predictions(model_id, prediction_date);
CREATE INDEX idx_predictions_symbols ON predictions(symbol_1, symbol_2);
CREATE INDEX idx_predictions_date ON predictions(prediction_date);

-- ============================================================================
-- TRADING TABLES
-- ============================================================================

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_type VARCHAR(20) NOT NULL,  -- 'MARKET', 'LIMIT', 'STOP_LOSS'
    side VARCHAR(10) NOT NULL,  -- 'BUY', 'SELL'
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    quantity INTEGER NOT NULL,
    price DECIMAL(12, 4),
    status VARCHAR(20) NOT NULL,  -- 'PENDING', 'FILLED', 'CANCELLED', 'REJECTED'
    filled_quantity INTEGER DEFAULT 0,
    filled_price DECIMAL(12, 4),
    filled_at TIMESTAMP WITH TIME ZONE,
    signal VARCHAR(100),  -- Trading signal that generated this order
    prediction_id BIGINT REFERENCES predictions(id),
    paper_trading BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_orders_paper_trading ON orders(paper_trading);

-- Trades (filled orders)
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id),
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(12, 4) NOT NULL,
    commission DECIMAL(12, 4) DEFAULT 0,
    sec_fee DECIMAL(12, 4) DEFAULT 0,
    exchange_fee DECIMAL(12, 4) DEFAULT 0,
    slippage DECIMAL(12, 4) DEFAULT 0,
    total_cost DECIMAL(12, 4) NOT NULL,
    execution_time TIMESTAMP WITH TIME ZONE NOT NULL,
    paper_trading BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_order_id ON trades(order_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_execution_time ON trades(execution_time);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    quantity INTEGER NOT NULL,
    average_price DECIMAL(12, 4) NOT NULL,
    current_price DECIMAL(12, 4),
    unrealized_pnl DECIMAL(12, 4),
    realized_pnl DECIMAL(12, 4) DEFAULT 0,
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    closed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'OPEN',  -- 'OPEN', 'CLOSED'
    paper_trading BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_paper_trading ON positions(paper_trading);

-- ============================================================================
-- RISK MANAGEMENT TABLES
-- ============================================================================

-- Risk metrics (daily snapshots)
CREATE TABLE IF NOT EXISTS risk_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_date DATE NOT NULL UNIQUE,
    total_capital DECIMAL(18, 4) NOT NULL,
    available_capital DECIMAL(18, 4) NOT NULL,
    total_exposure DECIMAL(18, 4) NOT NULL,
    long_exposure DECIMAL(18, 4) NOT NULL,
    short_exposure DECIMAL(18, 4) NOT NULL,
    max_drawdown DECIMAL(8, 4),
    current_drawdown DECIMAL(8, 4),
    var_95 DECIMAL(18, 4),  -- Value at Risk (95%)
    var_99 DECIMAL(18, 4),  -- Value at Risk (99%)
    sharpe_ratio DECIMAL(8, 4),
    paper_trading BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_metrics_date ON risk_metrics(metric_date);

-- Circuit breaker events
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,  -- 'MAX_DRAWDOWN', 'DAILY_LOSS', 'API_ERROR', etc.
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL,
    reason TEXT NOT NULL,
    metric_value DECIMAL(18, 4),
    threshold_value DECIMAL(18, 4),
    actions_taken TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_circuit_breaker_triggered_at ON circuit_breaker_events(triggered_at);
CREATE INDEX idx_circuit_breaker_event_type ON circuit_breaker_events(event_type);

-- ============================================================================
-- AUDIT TABLES
-- ============================================================================

-- Audit log (immutable)
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,  -- 'TRADE', 'CONFIG_CHANGE', 'MODEL_UPDATE', etc.
    entity_type VARCHAR(50),  -- 'ORDER', 'POSITION', 'MODEL', etc.
    entity_id VARCHAR(100),
    user_id VARCHAR(100),
    action VARCHAR(50) NOT NULL,  -- 'CREATE', 'UPDATE', 'DELETE', 'EXECUTE'
    before_state JSONB,
    after_state JSONB,
    metadata JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_log_timestamp ON audit_log(event_timestamp);
CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Current positions summary
CREATE OR REPLACE VIEW v_current_positions AS
SELECT
    p.symbol,
    p.quantity,
    p.average_price,
    p.current_price,
    p.unrealized_pnl,
    p.realized_pnl,
    (p.quantity * p.current_price) as market_value,
    p.opened_at,
    p.paper_trading,
    s.name as stock_name,
    s.sector
FROM positions p
JOIN stocks s ON p.symbol = s.symbol
WHERE p.status = 'OPEN';

-- Daily P&L summary
CREATE OR REPLACE VIEW v_daily_pnl AS
SELECT
    DATE(t.execution_time) as trade_date,
    t.paper_trading,
    COUNT(*) as num_trades,
    SUM(CASE WHEN t.side = 'BUY' THEN t.total_cost ELSE 0 END) as total_buys,
    SUM(CASE WHEN t.side = 'SELL' THEN t.total_cost ELSE 0 END) as total_sells,
    SUM(t.commission + t.sec_fee + t.exchange_fee + t.slippage) as total_costs,
    SUM(CASE WHEN t.side = 'SELL' THEN t.total_cost ELSE -t.total_cost END) as net_pnl
FROM trades t
GROUP BY DATE(t.execution_time), t.paper_trading
ORDER BY trade_date DESC;

-- Model performance comparison
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    m.name,
    m.version,
    m.model_type,
    m.status,
    mm.metric_type,
    mm.sharpe_ratio,
    mm.directional_accuracy,
    mm.mse,
    mm.mae,
    mm.metric_date
FROM models m
JOIN model_metrics mm ON m.id = mm.model_id
WHERE m.status = 'active'
ORDER BY mm.metric_date DESC, mm.sharpe_ratio DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_stocks_updated_at BEFORE UPDATE ON stocks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert common stocks
INSERT INTO stocks (symbol, name, exchange, sector) VALUES
    ('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology'),
    ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology'),
    ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology'),
    ('AMZN', 'Amazon.com Inc.', 'NASDAQ', 'Consumer Cyclical'),
    ('TSLA', 'Tesla Inc.', 'NASDAQ', 'Consumer Cyclical'),
    ('META', 'Meta Platforms Inc.', 'NASDAQ', 'Technology'),
    ('NVDA', 'NVIDIA Corporation', 'NASDAQ', 'Technology')
ON CONFLICT (symbol) DO NOTHING;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_app;
