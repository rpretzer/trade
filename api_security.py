"""
API Security Module
Implements request signing, signature verification, and replay attack prevention
"""

import hmac
import hashlib
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from exceptions import SecurityException

logger = logging.getLogger(__name__)


class SignatureVerificationError(SecurityException):
    """Raised when signature verification fails."""
    pass


class ReplayAttackError(SecurityException):
    """Raised when replay attack is detected."""
    pass


@dataclass
class SignedRequest:
    """
    Signed API request with HMAC signature.
    """
    method: str  # HTTP method
    endpoint: str  # API endpoint
    params: Dict  # Request parameters
    timestamp: str  # ISO format timestamp
    nonce: str  # Unique request identifier
    signature: str  # HMAC-SHA256 signature

    def to_dict(self) -> Dict:
        """Convert to dictionary for transmission."""
        return {
            'method': self.method,
            'endpoint': self.endpoint,
            'params': self.params,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'signature': self.signature
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SignedRequest':
        """Create from dictionary."""
        return cls(
            method=data['method'],
            endpoint=data['endpoint'],
            params=data['params'],
            timestamp=data['timestamp'],
            nonce=data['nonce'],
            signature=data['signature']
        )


class RequestSigner:
    """
    Signs API requests with HMAC-SHA256.

    Prevents:
    - Request tampering (signature changes if data modified)
    - Replay attacks (timestamp + nonce validation)
    - Man-in-the-middle attacks (secret key required)
    """

    def __init__(
        self,
        secret_key: str,
        max_timestamp_age_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize request signer.

        Args:
            secret_key: Secret key for HMAC signing
            max_timestamp_age_seconds: Maximum age of timestamp before rejecting
        """
        if not secret_key or len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")

        self.secret_key = secret_key.encode('utf-8')
        self.max_timestamp_age = max_timestamp_age_seconds
        self.used_nonces = set()  # Track used nonces (in-memory, would use Redis in prod)

    def sign_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> SignedRequest:
        """
        Sign an API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Request parameters

        Returns:
            SignedRequest with signature
        """
        if params is None:
            params = {}

        # Generate timestamp and nonce
        timestamp = datetime.now(timezone.utc).isoformat() + 'Z'
        nonce = self._generate_nonce()

        # Create signature
        signature = self._create_signature(method, endpoint, params, timestamp, nonce)

        return SignedRequest(
            method=method,
            endpoint=endpoint,
            params=params,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature
        )

    def verify_request(self, signed_request: SignedRequest) -> Tuple[bool, Optional[str]]:
        """
        Verify a signed request.

        Args:
            signed_request: Request to verify

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check timestamp freshness
        try:
            request_time = datetime.fromisoformat(signed_request.timestamp.replace('Z', ''))
            age = (datetime.now(timezone.utc) - request_time).total_seconds()

            if age > self.max_timestamp_age:
                return False, f"Request too old: {age:.0f}s > {self.max_timestamp_age}s"

            if age < -60:  # Clock skew tolerance: 1 minute
                return False, "Request timestamp in future (clock skew)"

        except (ValueError, AttributeError) as e:
            return False, f"Invalid timestamp format: {e}"

        # Check nonce uniqueness (prevent replay)
        if signed_request.nonce in self.used_nonces:
            logger.warning(f"Replay attack detected: nonce {signed_request.nonce} already used")
            return False, "Replay attack detected (nonce reused)"

        # Verify signature
        expected_signature = self._create_signature(
            signed_request.method,
            signed_request.endpoint,
            signed_request.params,
            signed_request.timestamp,
            signed_request.nonce
        )

        if not hmac.compare_digest(signed_request.signature, expected_signature):
            logger.error("Signature verification failed")
            return False, "Invalid signature"

        # Mark nonce as used
        self.used_nonces.add(signed_request.nonce)

        # Clean old nonces periodically (simple in-memory approach)
        if len(self.used_nonces) > 10000:
            # In production, would use TTL in Redis
            self.used_nonces.clear()
            logger.info("Cleared nonce cache")

        return True, None

    def _create_signature(
        self,
        method: str,
        endpoint: str,
        params: Dict,
        timestamp: str,
        nonce: str
    ) -> str:
        """
        Create HMAC-SHA256 signature for request.

        Signature covers:
        - HTTP method
        - Endpoint path
        - Parameters (sorted for consistency)
        - Timestamp
        - Nonce

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            timestamp: Request timestamp
            nonce: Unique nonce

        Returns:
            Hex-encoded signature
        """
        # Create canonical representation of request
        # Sort params for deterministic signature
        sorted_params = json.dumps(params, sort_keys=True, separators=(',', ':'))

        message = f"{method}|{endpoint}|{sorted_params}|{timestamp}|{nonce}"

        # Create HMAC signature
        signature = hmac.new(
            self.secret_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _generate_nonce(self) -> str:
        """
        Generate unique nonce for request.

        Uses timestamp + random bytes for uniqueness.

        Returns:
            Hex-encoded nonce
        """
        import os
        import time

        # Combine timestamp (microseconds) with random bytes
        timestamp_bytes = str(time.time_ns()).encode('utf-8')
        random_bytes = os.urandom(16)

        nonce = hashlib.sha256(timestamp_bytes + random_bytes).hexdigest()[:32]
        return nonce


class ResponseVerifier:
    """
    Verifies API responses haven't been tampered with.
    """

    def __init__(self, secret_key: str):
        """
        Initialize response verifier.

        Args:
            secret_key: Secret key for verification
        """
        if not secret_key or len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")

        self.secret_key = secret_key.encode('utf-8')

    def sign_response(self, response_data: Dict, request_nonce: str) -> Dict:
        """
        Sign a response payload.

        Args:
            response_data: Response data to sign
            request_nonce: Nonce from original request

        Returns:
            Signed response with signature field
        """
        # Create canonical representation
        data_str = json.dumps(response_data, sort_keys=True, separators=(',', ':'))
        timestamp = datetime.now(timezone.utc).isoformat() + 'Z'

        # Create signature
        message = f"{data_str}|{request_nonce}|{timestamp}"
        signature = hmac.new(
            self.secret_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Add signature metadata to response
        signed_response = response_data.copy()
        signed_response['_signature'] = signature
        signed_response['_timestamp'] = timestamp
        signed_response['_request_nonce'] = request_nonce

        return signed_response

    def verify_response(
        self,
        signed_response: Dict,
        request_nonce: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a signed response.

        Args:
            signed_response: Response with signature
            request_nonce: Original request nonce

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Extract signature metadata
        if '_signature' not in signed_response:
            return False, "Response not signed"

        signature = signed_response['_signature']
        timestamp = signed_response.get('_timestamp')
        response_nonce = signed_response.get('_request_nonce')

        # Check nonce matches
        if response_nonce != request_nonce:
            return False, "Nonce mismatch (possible tampering)"

        # Remove signature fields for verification
        response_data = {
            k: v for k, v in signed_response.items()
            if not k.startswith('_')
        }

        # Recreate signature
        data_str = json.dumps(response_data, sort_keys=True, separators=(',', ':'))
        message = f"{data_str}|{request_nonce}|{timestamp}"
        expected_signature = hmac.new(
            self.secret_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Verify
        if not hmac.compare_digest(signature, expected_signature):
            logger.error("Response signature verification failed")
            return False, "Invalid response signature"

        return True, None


def create_api_signer(secret_key: str) -> Tuple[RequestSigner, ResponseVerifier]:
    """
    Create matched request signer and response verifier.

    Args:
        secret_key: Shared secret key

    Returns:
        Tuple of (RequestSigner, ResponseVerifier)
    """
    return RequestSigner(secret_key), ResponseVerifier(secret_key)
