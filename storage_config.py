"""
Storage Configuration

Centralized configuration for data storage backend.
Supports both CSV (legacy) and PostgreSQL (production) storage.
"""

import os
from enum import Enum


class StorageBackend(Enum):
    """Storage backend options"""
    CSV = "csv"
    DATABASE = "database"


class StorageConfig:
    """Global storage configuration"""

    def __init__(self):
        # Default to database if DATABASE_URL is set, otherwise CSV
        database_url = os.environ.get('DATABASE_URL')
        if database_url and database_url != 'postgresql://localhost:5432/trading_db':
            self._backend = StorageBackend.DATABASE
        else:
            # Check if we should use database
            use_db = os.environ.get('USE_DATABASE', 'false').lower() in ['true', '1', 'yes']
            self._backend = StorageBackend.DATABASE if use_db else StorageBackend.CSV

        self._database_url = database_url or 'postgresql://localhost:5432/trading_db'

    @property
    def backend(self) -> StorageBackend:
        """Get current storage backend"""
        return self._backend

    @backend.setter
    def backend(self, value: StorageBackend):
        """Set storage backend"""
        self._backend = value

    @property
    def database_url(self) -> str:
        """Get database URL"""
        return self._database_url

    @property
    def use_csv(self) -> bool:
        """Check if using CSV backend"""
        return self._backend == StorageBackend.CSV

    @property
    def use_database(self) -> bool:
        """Check if using database backend"""
        return self._backend == StorageBackend.DATABASE

    def set_csv_backend(self):
        """Switch to CSV backend"""
        self._backend = StorageBackend.CSV

    def set_database_backend(self):
        """Switch to database backend"""
        self._backend = StorageBackend.DATABASE


# Global configuration instance
_config = None


def get_storage_config() -> StorageConfig:
    """Get global storage configuration"""
    global _config
    if _config is None:
        _config = StorageConfig()
    return _config


def use_database():
    """Force use of database backend"""
    config = get_storage_config()
    config.set_database_backend()


def use_csv():
    """Force use of CSV backend (for backward compatibility)"""
    config = get_storage_config()
    config.set_csv_backend()
