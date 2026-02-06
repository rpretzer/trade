"""
Database Connection Management
Handles PostgreSQL connections with connection pooling
"""

import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging

from database.models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections with connection pooling.

    Features:
    - Connection pooling for performance
    - Automatic session management
    - Context managers for safe transactions
    - Health checks
    """

    def __init__(self, database_url=None, echo=False):
        """
        Initialize database manager.

        Args:
            database_url: PostgreSQL connection string
                         Format: postgresql://user:password@host:port/database
            echo: Whether to echo SQL queries (for debugging)
        """
        # Get database URL from environment or parameter
        self.database_url = database_url or os.environ.get(
            'DATABASE_URL',
            'postgresql://localhost:5432/trading_db'
        )

        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            echo=echo,
            poolclass=QueuePool,
            pool_size=10,  # Number of connections to maintain
            max_overflow=20,  # Additional connections if pool exhausted
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
        )

        # Create session factory
        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        )

        # Register event listeners
        self._register_listeners()

        logger.info(f"Database connection initialized: {self._mask_password(self.database_url)}")

    def _mask_password(self, url):
        """Mask password in database URL for logging."""
        if '@' in url and ':' in url:
            parts = url.split('@')
            credentials = parts[0].split('//')[-1]
            if ':' in credentials:
                user = credentials.split(':')[0]
                return url.replace(credentials, f"{user}:****")
        return url

    def _register_listeners(self):
        """Register SQLAlchemy event listeners."""
        # Log slow queries (> 1 second)
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            conn.info.setdefault('query_start_time', []).append(time.time())

        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, params, context, executemany):
            total_time = time.time() - conn.info['query_start_time'].pop()
            if total_time > 1.0:  # Log queries taking > 1 second
                logger.warning(f"Slow query ({total_time:.2f}s): {statement[:100]}...")

    def create_tables(self):
        """Create all tables in the database."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all tables in the database (USE WITH CAUTION!)."""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")

    def health_check(self):
        """
        Check database connectivity.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @contextmanager
    def get_session(self):
        """
        Get a database session with automatic cleanup.

        Usage:
            with db.get_session() as session:
                # Use session
                user = session.query(User).first()

        Yields:
            Session: SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def execute_sql_file(self, sql_file_path):
        """
        Execute SQL from a file.

        Args:
            sql_file_path: Path to SQL file

        Returns:
            bool: True if successful
        """
        try:
            with open(sql_file_path, 'r') as f:
                sql = f.read()

            with self.engine.connect() as conn:
                # Execute in transaction
                with conn.begin():
                    # Split by semicolon and execute each statement
                    for statement in sql.split(';'):
                        statement = statement.strip()
                        if statement:
                            conn.execute(text(statement))

            logger.info(f"SQL file executed successfully: {sql_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute SQL file {sql_file_path}: {e}")
            return False

    def close(self):
        """Close all database connections."""
        self.SessionLocal.remove()
        self.engine.dispose()
        logger.info("Database connections closed")


# Global database instance (singleton pattern)
_db_manager = None


def get_db_manager(database_url=None, echo=False):
    """
    Get global database manager instance.

    Args:
        database_url: PostgreSQL connection string
        echo: Whether to echo SQL queries

    Returns:
        DatabaseManager: Database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url, echo)
    return _db_manager


def init_database(database_url=None):
    """
    Initialize database with schema.

    Args:
        database_url: PostgreSQL connection string

    Returns:
        DatabaseManager: Initialized database manager
    """
    db = get_db_manager(database_url)

    # Create tables
    db.create_tables()

    # Execute schema.sql if it exists
    schema_file = os.path.join(
        os.path.dirname(__file__),
        'schema.sql'
    )
    if os.path.exists(schema_file):
        logger.info(f"Executing schema file: {schema_file}")
        db.execute_sql_file(schema_file)

    # Verify health
    if db.health_check():
        logger.info("Database initialized and healthy")
    else:
        logger.error("Database health check failed after initialization")

    return db


# Add time module import at top
import time


if __name__ == '__main__':
    # Example usage
    print("Database Connection Manager")
    print("=" * 70)

    # Initialize database
    db = init_database('postgresql://localhost:5432/trading_db')

    # Health check
    print(f"\nHealth check: {'✓ Healthy' if db.health_check() else '✗ Failed'}")

    # Example query
    print("\nExample query:")
    with db.get_session() as session:
        from database.models import Stock

        # Query stocks
        stocks = session.query(Stock).limit(5).all()
        print(f"Found {len(stocks)} stocks:")
        for stock in stocks:
            print(f"  - {stock.symbol}: {stock.name} ({stock.exchange})")

    # Close connections
    db.close()
    print("\nDatabase connections closed")
