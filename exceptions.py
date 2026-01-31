"""
Custom Exception Classes for Stock Arbitrage Model
Provides specific exception types for better error handling and debugging
"""


class TradingException(Exception):
    """Base exception for all trading-related errors."""
    pass


# Data Errors
class DataException(TradingException):
    """Base exception for data-related errors."""
    pass


class DataNotFoundError(DataException):
    """Raised when required data is not found."""
    pass


class DataValidationError(DataException):
    """Raised when data validation fails."""
    pass


class InsufficientDataError(DataException):
    """Raised when there is not enough data for analysis."""
    pass


class StaleDataError(DataException):
    """Raised when data is too old to be reliable."""
    pass


# API Errors
class APIException(TradingException):
    """Base exception for API-related errors."""
    pass


class APIConnectionError(APIException):
    """Raised when API connection fails."""
    pass


class APIRateLimitError(APIException):
    """Raised when API rate limit is exceeded."""
    pass


class APIAuthenticationError(APIException):
    """Raised when API authentication fails."""
    pass


class APIResponseError(APIException):
    """Raised when API returns unexpected response."""
    pass


# Model Errors
class ModelException(TradingException):
    """Base exception for ML model errors."""
    pass


class ModelLoadError(ModelException):
    """Raised when model cannot be loaded."""
    pass


class ModelPredictionError(ModelException):
    """Raised when model prediction fails."""
    pass


class ModelVersionMismatchError(ModelException):
    """Raised when model version doesn't match expected version."""
    pass


# Trading Errors
class OrderException(TradingException):
    """Base exception for order-related errors."""
    pass


class InsufficientFundsError(OrderException):
    """Raised when there are insufficient funds for an order."""
    pass


class InvalidOrderError(OrderException):
    """Raised when order parameters are invalid."""
    pass


class OrderRejectedError(OrderException):
    """Raised when broker rejects an order."""
    pass


class OrderTimeoutError(OrderException):
    """Raised when order times out without being filled."""
    pass


# Risk Management Errors
class RiskException(TradingException):
    """Base exception for risk management errors."""
    pass


class RiskLimitExceededError(RiskException):
    """Raised when a risk limit is exceeded."""
    pass


class PositionLimitExceededError(RiskException):
    """Raised when position limit is exceeded."""
    pass


class DrawdownExceededError(RiskException):
    """Raised when maximum drawdown is exceeded."""
    pass


class ConcentrationLimitExceededError(RiskException):
    """Raised when position concentration limit is exceeded."""
    pass


class CorrelationLimitExceededError(RiskException):
    """Raised when correlation limit is exceeded."""
    pass


# Circuit Breaker Errors
class CircuitBreakerException(TradingException):
    """Base exception for circuit breaker events."""
    pass


class CircuitBreakerOpenError(CircuitBreakerException):
    """Raised when circuit breaker is open and blocking operations."""
    pass


class CircuitBreakerTriggeredError(CircuitBreakerException):
    """Raised when circuit breaker is triggered by failures."""
    pass


# Configuration Errors
class ConfigurationException(TradingException):
    """Base exception for configuration errors."""
    pass


class MissingConfigurationError(ConfigurationException):
    """Raised when required configuration is missing."""
    pass


class InvalidConfigurationError(ConfigurationException):
    """Raised when configuration is invalid."""
    pass


# Security Errors
class SecurityException(TradingException):
    """Base exception for security-related errors."""
    pass


class CredentialError(SecurityException):
    """Raised when credential operation fails."""
    pass


class EncryptionError(SecurityException):
    """Raised when encryption/decryption fails."""
    pass


class AuditLogError(SecurityException):
    """Raised when audit logging fails."""
    pass


# Market Data Errors
class MarketDataException(TradingException):
    """Base exception for market data errors."""
    pass


class PriceNotAvailableError(MarketDataException):
    """Raised when current price is not available."""
    pass


class InvalidPriceError(MarketDataException):
    """Raised when price appears invalid or unrealistic."""
    pass


class ShortNotAvailableError(MarketDataException):
    """Raised when stock cannot be shorted."""
    pass
