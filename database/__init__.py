"""
Database package
Provides ORM models and connection management
"""

from database.models import (
    Base,
    Stock,
    Price,
    TechnicalIndicator,
    SentimentData,
    OptionsData,
    Model,
    ModelMetric,
    Prediction,
    Order,
    Trade,
    Position,
    RiskMetric,
    CircuitBreakerEvent,
    AuditLog
)

from database.connection import (
    DatabaseManager,
    get_db_manager,
    init_database
)

from database.data_access import (
    DataAccess,
    get_data_access
)

__all__ = [
    # Models
    'Base',
    'Stock',
    'Price',
    'TechnicalIndicator',
    'SentimentData',
    'OptionsData',
    'Model',
    'ModelMetric',
    'Prediction',
    'Order',
    'Trade',
    'Position',
    'RiskMetric',
    'CircuitBreakerEvent',
    'AuditLog',
    # Connection
    'DatabaseManager',
    'get_db_manager',
    'init_database',
    # Data Access
    'DataAccess',
    'get_data_access'
]
