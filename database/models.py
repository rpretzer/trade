"""
SQLAlchemy ORM Models for Trading Database
"""

from sqlalchemy import (
    Column, String, Integer, BigInteger, Numeric, DateTime, Date,
    Boolean, Text, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class Stock(Base):
    """Stock reference data"""
    __tablename__ = 'stocks'

    symbol = Column(String(10), primary_key=True)
    name = Column(String(255))
    exchange = Column(String(50))
    sector = Column(String(100))
    industry = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    prices = relationship("Price", back_populates="stock")
    orders = relationship("Order", back_populates="stock")
    trades = relationship("Trade", back_populates="stock")
    positions = relationship("Position", back_populates="stock")


class Price(Base):
    """Stock OHLCV price data"""
    __tablename__ = 'prices'

    id = Column(BigInteger, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Numeric(12, 4), nullable=False)
    high = Column(Numeric(12, 4), nullable=False)
    low = Column(Numeric(12, 4), nullable=False)
    close = Column(Numeric(12, 4), nullable=False)
    volume = Column(BigInteger, nullable=False)
    adjusted_close = Column(Numeric(12, 4))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    stock = relationship("Stock", back_populates="prices")

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_price_symbol_date'),
        Index('idx_prices_symbol_date', 'symbol', 'date'),
    )


class TechnicalIndicator(Base):
    """Technical indicators"""
    __tablename__ = 'technical_indicators'

    id = Column(BigInteger, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    date = Column(Date, nullable=False)
    rsi = Column(Numeric(8, 4))
    macd = Column(Numeric(12, 6))
    macd_signal = Column(Numeric(12, 6))
    macd_histogram = Column(Numeric(12, 6))
    ma_5 = Column(Numeric(12, 4))
    ma_20 = Column(Numeric(12, 4))
    ma_50 = Column(Numeric(12, 4))
    ma_200 = Column(Numeric(12, 4))
    volume_ma_5 = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_technical_symbol_date'),
    )


class SentimentData(Base):
    """Sentiment data from various sources"""
    __tablename__ = 'sentiment_data'

    id = Column(BigInteger, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    date = Column(Date, nullable=False)
    source = Column(String(50), nullable=False)
    mean_sentiment = Column(Numeric(6, 4))
    median_sentiment = Column(Numeric(6, 4))
    std_sentiment = Column(Numeric(6, 4))
    post_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'date', 'source', name='uq_sentiment_symbol_date_source'),
    )


class OptionsData(Base):
    """Options volume data"""
    __tablename__ = 'options_data'

    id = Column(BigInteger, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    date = Column(Date, nullable=False)
    total_volume = Column(BigInteger)
    calls_volume = Column(BigInteger)
    puts_volume = Column(BigInteger)
    contract_count = Column(Integer)
    implied_volatility = Column(Numeric(8, 4))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_options_symbol_date'),
    )


class Model(Base):
    """ML model registry"""
    __tablename__ = 'models'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    framework = Column(String(50))
    file_path = Column(Text, nullable=False)
    hyperparameters = Column(JSONB)
    training_date = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(20), default='active')
    created_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    metrics = relationship("ModelMetric", back_populates="model")
    predictions = relationship("Prediction", back_populates="model")

    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
    )


class ModelMetric(Base):
    """Model performance metrics"""
    __tablename__ = 'model_metrics'

    id = Column(BigInteger, primary_key=True)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id', ondelete='CASCADE'), nullable=False)
    metric_date = Column(Date, nullable=False)
    metric_type = Column(String(50), nullable=False)
    mse = Column(Numeric(12, 6))
    mae = Column(Numeric(12, 6))
    r_squared = Column(Numeric(8, 6))
    directional_accuracy = Column(Numeric(6, 4))
    sharpe_ratio = Column(Numeric(8, 4))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    model = relationship("Model", back_populates="metrics")

    __table_args__ = (
        UniqueConstraint('model_id', 'metric_date', 'metric_type', name='uq_metric_model_date_type'),
    )


class Prediction(Base):
    """Model predictions"""
    __tablename__ = 'predictions'

    id = Column(BigInteger, primary_key=True)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    prediction_date = Column(Date, nullable=False)
    symbol_1 = Column(String(10), nullable=False)
    symbol_2 = Column(String(10), nullable=False)
    predicted_difference = Column(Numeric(12, 6), nullable=False)
    actual_difference = Column(Numeric(12, 6))
    confidence = Column(Numeric(6, 4))
    features = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    model = relationship("Model", back_populates="predictions")
    orders = relationship("Order", back_populates="prediction")


class Order(Base):
    """Trading orders"""
    __tablename__ = 'orders'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_type = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(12, 4))
    status = Column(String(20), nullable=False)
    filled_quantity = Column(Integer, default=0)
    filled_price = Column(Numeric(12, 4))
    filled_at = Column(DateTime(timezone=True))
    signal = Column(String(100))
    prediction_id = Column(BigInteger, ForeignKey('predictions.id'))
    paper_trading = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    stock = relationship("Stock", back_populates="orders")
    prediction = relationship("Prediction", back_populates="orders")
    trades = relationship("Trade", back_populates="order")


class Trade(Base):
    """Executed trades"""
    __tablename__ = 'trades'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey('orders.id'), nullable=False)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(12, 4), nullable=False)
    commission = Column(Numeric(12, 4), default=0)
    sec_fee = Column(Numeric(12, 4), default=0)
    exchange_fee = Column(Numeric(12, 4), default=0)
    slippage = Column(Numeric(12, 4), default=0)
    total_cost = Column(Numeric(12, 4), nullable=False)
    execution_time = Column(DateTime(timezone=True), nullable=False)
    paper_trading = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    order = relationship("Order", back_populates="trades")
    stock = relationship("Stock", back_populates="trades")


class Position(Base):
    """Current and historical positions"""
    __tablename__ = 'positions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    quantity = Column(Integer, nullable=False)
    average_price = Column(Numeric(12, 4), nullable=False)
    current_price = Column(Numeric(12, 4))
    unrealized_pnl = Column(Numeric(12, 4))
    realized_pnl = Column(Numeric(12, 4), default=0)
    opened_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True))
    status = Column(String(20), default='OPEN')
    paper_trading = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    stock = relationship("Stock", back_populates="positions")


class RiskMetric(Base):
    """Daily risk metrics"""
    __tablename__ = 'risk_metrics'

    id = Column(BigInteger, primary_key=True)
    metric_date = Column(Date, nullable=False, unique=True)
    total_capital = Column(Numeric(18, 4), nullable=False)
    available_capital = Column(Numeric(18, 4), nullable=False)
    total_exposure = Column(Numeric(18, 4), nullable=False)
    long_exposure = Column(Numeric(18, 4), nullable=False)
    short_exposure = Column(Numeric(18, 4), nullable=False)
    max_drawdown = Column(Numeric(8, 4))
    current_drawdown = Column(Numeric(8, 4))
    var_95 = Column(Numeric(18, 4))
    var_99 = Column(Numeric(18, 4))
    sharpe_ratio = Column(Numeric(8, 4))
    paper_trading = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CircuitBreakerEvent(Base):
    """Circuit breaker events"""
    __tablename__ = 'circuit_breaker_events'

    id = Column(BigInteger, primary_key=True)
    event_type = Column(String(50), nullable=False)
    triggered_at = Column(DateTime(timezone=True), nullable=False)
    reason = Column(Text, nullable=False)
    metric_value = Column(Numeric(18, 4))
    threshold_value = Column(Numeric(18, 4))
    actions_taken = Column(Text)
    resolved_at = Column(DateTime(timezone=True))
    resolved_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """Immutable audit log"""
    __tablename__ = 'audit_log'

    id = Column(BigInteger, primary_key=True)
    event_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    event_type = Column(String(50), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(100))
    user_id = Column(String(100))
    action = Column(String(50), nullable=False)
    before_state = Column(JSONB)
    after_state = Column(JSONB)
    metadata = Column(JSONB)
    ip_address = Column(INET)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
