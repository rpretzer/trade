"""
Unit Tests for API Security
Tests request signing, verification, and replay attack prevention
"""

import pytest
import time
from datetime import datetime, timedelta
from api_security import (
    RequestSigner, ResponseVerifier, SignedRequest,
    SignatureVerificationError, ReplayAttackError,
    create_api_signer
)


class TestRequestSigner:
    """Test request signing functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.secret_key = "a" * 32  # 32 character secret
        self.signer = RequestSigner(self.secret_key, max_timestamp_age_seconds=300)

    def test_initialization_valid_key(self):
        """Test initialization with valid secret key."""
        signer = RequestSigner("b" * 32)
        assert signer is not None
        assert signer.max_timestamp_age == 300  # Default

    def test_initialization_short_key_fails(self):
        """Test that short secret key raises error."""
        with pytest.raises(ValueError, match="at least 32 characters"):
            RequestSigner("short_key")

    def test_sign_request_basic(self):
        """Test signing a basic request."""
        signed = self.signer.sign_request(
            method='GET',
            endpoint='/api/account',
            params={'user_id': '123'}
        )

        assert signed.method == 'GET'
        assert signed.endpoint == '/api/account'
        assert signed.params == {'user_id': '123'}
        assert signed.timestamp is not None
        assert signed.nonce is not None
        assert signed.signature is not None
        assert len(signed.signature) == 64  # SHA-256 hex digest

    def test_sign_request_no_params(self):
        """Test signing request without parameters."""
        signed = self.signer.sign_request(
            method='GET',
            endpoint='/api/balance'
        )

        assert signed.params == {}
        assert signed.signature is not None

    def test_different_requests_different_signatures(self):
        """Test that different requests produce different signatures."""
        signed1 = self.signer.sign_request('GET', '/api/v1', {'a': '1'})
        time.sleep(0.001)  # Ensure different timestamp/nonce
        signed2 = self.signer.sign_request('GET', '/api/v2', {'a': '1'})

        assert signed1.signature != signed2.signature

    def test_verify_valid_request(self):
        """Test verifying a valid signed request."""
        signed = self.signer.sign_request(
            method='POST',
            endpoint='/api/trade',
            params={'symbol': 'AAPL', 'quantity': 100}
        )

        is_valid, error = self.signer.verify_request(signed)
        assert is_valid is True
        assert error is None

    def test_verify_tampered_params_fails(self):
        """Test that tampering with params invalidates signature."""
        signed = self.signer.sign_request(
            method='POST',
            endpoint='/api/trade',
            params={'symbol': 'AAPL', 'quantity': 100}
        )

        # Tamper with params
        signed.params['quantity'] = 200

        is_valid, error = self.signer.verify_request(signed)
        assert is_valid is False
        assert "Invalid signature" in error

    def test_verify_tampered_endpoint_fails(self):
        """Test that changing endpoint invalidates signature."""
        signed = self.signer.sign_request('GET', '/api/account')

        # Tamper with endpoint
        signed.endpoint = '/api/admin'

        is_valid, error = self.signer.verify_request(signed)
        assert is_valid is False

    def test_verify_old_timestamp_fails(self):
        """Test that old requests are rejected."""
        signed = self.signer.sign_request('GET', '/api/test')

        # Manually age the timestamp
        old_time = datetime.utcnow() - timedelta(seconds=400)
        signed.timestamp = old_time.isoformat() + 'Z'

        # Need to re-sign with old timestamp
        signed.signature = self.signer._create_signature(
            signed.method, signed.endpoint, signed.params,
            signed.timestamp, signed.nonce
        )

        is_valid, error = self.signer.verify_request(signed)
        assert is_valid is False
        assert "too old" in error.lower()

    def test_verify_future_timestamp_fails(self):
        """Test that future timestamps are rejected (clock skew)."""
        signed = self.signer.sign_request('GET', '/api/test')

        # Set timestamp in future
        future_time = datetime.utcnow() + timedelta(seconds=120)
        signed.timestamp = future_time.isoformat() + 'Z'
        signed.signature = self.signer._create_signature(
            signed.method, signed.endpoint, signed.params,
            signed.timestamp, signed.nonce
        )

        is_valid, error = self.signer.verify_request(signed)
        assert is_valid is False
        assert "future" in error.lower() or "clock skew" in error.lower()

    def test_verify_replay_attack_fails(self):
        """Test that replaying a request fails."""
        signed = self.signer.sign_request('POST', '/api/transfer', {'amount': 1000})

        # First verification succeeds
        is_valid1, error1 = self.signer.verify_request(signed)
        assert is_valid1 is True

        # Replay attack - same request again
        is_valid2, error2 = self.signer.verify_request(signed)
        assert is_valid2 is False
        assert "replay" in error2.lower() or "nonce" in error2.lower()

    def test_nonce_uniqueness(self):
        """Test that nonces are unique."""
        nonce1 = self.signer._generate_nonce()
        time.sleep(0.001)
        nonce2 = self.signer._generate_nonce()

        assert nonce1 != nonce2
        assert len(nonce1) == 32
        assert len(nonce2) == 32

    def test_param_order_independence(self):
        """Test that parameter order doesn't affect signature."""
        # Manually create signatures with different param orders
        params1 = {'a': '1', 'b': '2', 'c': '3'}
        params2 = {'c': '3', 'a': '1', 'b': '2'}

        timestamp = datetime.utcnow().isoformat() + 'Z'
        nonce = self.signer._generate_nonce()

        sig1 = self.signer._create_signature('GET', '/test', params1, timestamp, nonce)
        sig2 = self.signer._create_signature('GET', '/test', params2, timestamp, nonce)

        # Should be identical due to JSON sorting
        assert sig1 == sig2

    def test_signed_request_to_dict(self):
        """Test converting SignedRequest to dictionary."""
        signed = self.signer.sign_request('GET', '/api/test', {'key': 'value'})
        data = signed.to_dict()

        assert data['method'] == 'GET'
        assert data['endpoint'] == '/api/test'
        assert data['params'] == {'key': 'value'}
        assert 'timestamp' in data
        assert 'nonce' in data
        assert 'signature' in data

    def test_signed_request_from_dict(self):
        """Test creating SignedRequest from dictionary."""
        data = {
            'method': 'POST',
            'endpoint': '/api/order',
            'params': {'symbol': 'MSFT'},
            'timestamp': '2024-01-15T10:00:00Z',
            'nonce': 'abc123',
            'signature': 'def456'
        }

        signed = SignedRequest.from_dict(data)
        assert signed.method == 'POST'
        assert signed.endpoint == '/api/order'
        assert signed.params == {'symbol': 'MSFT'}


class TestResponseVerifier:
    """Test response verification functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.secret_key = "a" * 32
        self.verifier = ResponseVerifier(self.secret_key)
        self.test_nonce = "test_nonce_123"

    def test_initialization_valid_key(self):
        """Test initialization with valid key."""
        verifier = ResponseVerifier("b" * 32)
        assert verifier is not None

    def test_initialization_short_key_fails(self):
        """Test that short key raises error."""
        with pytest.raises(ValueError, match="at least 32 characters"):
            ResponseVerifier("short")

    def test_sign_response(self):
        """Test signing a response."""
        response_data = {
            'status': 'success',
            'balance': 10000,
            'positions': ['AAPL', 'MSFT']
        }

        signed = self.verifier.sign_response(response_data, self.test_nonce)

        assert signed['status'] == 'success'
        assert signed['balance'] == 10000
        assert '_signature' in signed
        assert '_timestamp' in signed
        assert '_request_nonce' in signed
        assert signed['_request_nonce'] == self.test_nonce

    def test_verify_valid_response(self):
        """Test verifying a valid signed response."""
        response_data = {'result': 'ok', 'value': 42}
        signed = self.verifier.sign_response(response_data, self.test_nonce)

        is_valid, error = self.verifier.verify_response(signed, self.test_nonce)
        assert is_valid is True
        assert error is None

    def test_verify_unsigned_response_fails(self):
        """Test that unsigned response fails verification."""
        response_data = {'result': 'ok'}

        is_valid, error = self.verifier.verify_response(response_data, self.test_nonce)
        assert is_valid is False
        assert "not signed" in error.lower()

    def test_verify_tampered_response_fails(self):
        """Test that tampering with response fails verification."""
        response_data = {'balance': 1000}
        signed = self.verifier.sign_response(response_data, self.test_nonce)

        # Tamper with data
        signed['balance'] = 999999

        is_valid, error = self.verifier.verify_response(signed, self.test_nonce)
        assert is_valid is False
        assert "invalid" in error.lower()

    def test_verify_nonce_mismatch_fails(self):
        """Test that nonce mismatch fails verification."""
        response_data = {'status': 'ok'}
        signed = self.verifier.sign_response(response_data, 'nonce_1')

        # Verify with different nonce
        is_valid, error = self.verifier.verify_response(signed, 'nonce_2')
        assert is_valid is False
        assert "nonce mismatch" in error.lower()

    def test_verify_tampered_signature_fails(self):
        """Test that tampering with signature fails."""
        response_data = {'data': 'test'}
        signed = self.verifier.sign_response(response_data, self.test_nonce)

        # Tamper with signature
        signed['_signature'] = 'tampered_signature_' + signed['_signature'][:40]

        is_valid, error = self.verifier.verify_response(signed, self.test_nonce)
        assert is_valid is False


class TestIntegration:
    """Test integrated request/response flow."""

    def setup_method(self):
        """Set up test environment."""
        self.secret_key = "integration_test_secret_" + "x" * 10
        self.signer, self.verifier = create_api_signer(self.secret_key)

    def test_create_api_signer(self):
        """Test creating matched signer and verifier."""
        assert self.signer is not None
        assert self.verifier is not None
        assert self.signer.secret_key == self.verifier.secret_key

    def test_full_request_response_flow(self):
        """Test complete request/response signing and verification."""
        # Client signs request
        request = self.signer.sign_request(
            method='POST',
            endpoint='/api/trade',
            params={'symbol': 'AAPL', 'quantity': 100, 'side': 'BUY'}
        )

        # Server verifies request
        is_valid, error = self.signer.verify_request(request)
        assert is_valid is True

        # Server signs response
        response_data = {
            'order_id': '12345',
            'status': 'FILLED',
            'filled_quantity': 100
        }
        signed_response = self.verifier.sign_response(response_data, request.nonce)

        # Client verifies response
        is_valid, error = self.verifier.verify_response(signed_response, request.nonce)
        assert is_valid is True

    def test_different_keys_fail_verification(self):
        """Test that different keys cause verification to fail."""
        signer1 = RequestSigner("key1" + "a" * 28)
        signer2 = RequestSigner("key2" + "b" * 28)

        # Sign with one key
        request = signer1.sign_request('GET', '/api/test')

        # Verify with different key
        is_valid, error = signer2.verify_request(request)
        assert is_valid is False
