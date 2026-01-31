"""
Secure Credential Manager
Encrypts and stores sensitive credentials using Fernet symmetric encryption.
"""

import os
import sys
import base64
import getpass
import logging
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Platform-specific imports for file protection
if sys.platform == 'win32':
    try:
        import win32security
        import ntsecuritycon as con
        WINDOWS_ACL_AVAILABLE = True
    except ImportError:
        WINDOWS_ACL_AVAILABLE = False
else:
    WINDOWS_ACL_AVAILABLE = False

from audit_logging import audit_log, AuditEventType
from exceptions import CredentialError, EncryptionError

logger = logging.getLogger(__name__)

class CredentialManager:
    """Manages secure storage and retrieval of credentials."""
    
    def __init__(self, config_file='schwab_config.enc', key_file='.schwab_key'):
        """
        Initialize credential manager.
        
        Args:
            config_file: Encrypted config file path
            key_file: File to store encryption key (hidden file)
        """
        self.config_file = config_file
        self.key_file = key_file
        self._key = None
    
    def _get_or_create_key(self, master_password=None):
        """
        Get or create encryption key.

        Args:
            master_password: Optional master password for key derivation

        Returns:
            Encryption key (bytes)
        """
        # Try to load existing key
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, 'rb') as f:
                    return f.read()
            except Exception:
                pass

        # Generate new key
        if master_password:
            # FIXED: Generate random salt per user instead of hardcoded
            salt = os.urandom(32)  # 256-bit random salt

            # Derive key from master password using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))

            # Save salt alongside key for future use
            key_with_salt = salt + key
        else:
            # Generate random key
            key_with_salt = Fernet.generate_key()

        # Save key to file (with restricted permissions)
        try:
            with open(self.key_file, 'wb') as f:
                f.write(key_with_salt)

            # Restrict file permissions - cross-platform
            self._secure_file_permissions(self.key_file)

        except Exception as e:
            raise EncryptionError(f"Failed to save encryption key: {e}")

        return key_with_salt if not master_password else key

    def _secure_file_permissions(self, filepath: str):
        """
        Secure file permissions cross-platform.

        On Unix: chmod 600 (owner read/write only)
        On Windows: Set ACL to allow only current user
        """
        if sys.platform == 'win32':
            if WINDOWS_ACL_AVAILABLE:
                try:
                    # Get current user SID
                    user, domain, _ = win32security.LookupAccountName("", os.getlogin())

                    # Create security descriptor
                    sd = win32security.SECURITY_DESCRIPTOR()
                    dacl = win32security.ACL()

                    # Add ACE for current user (full control)
                    dacl.AddAccessAllowedAce(
                        win32security.ACL_REVISION,
                        con.FILE_ALL_ACCESS,
                        user
                    )

                    # Set DACL
                    sd.SetSecurityDescriptorDacl(1, dacl, 0)

                    # Apply to file
                    win32security.SetFileSecurity(
                        filepath,
                        win32security.DACL_SECURITY_INFORMATION,
                        sd
                    )

                    logger.info(f"Windows ACL protection applied to {filepath}")

                except Exception as e:
                    logger.warning(f"Failed to apply Windows ACL: {e}")
                    logger.warning("Credentials file may not be fully protected on Windows")
            else:
                logger.warning(
                    "pywin32 not available. Install with: pip install pywin32\n"
                    "Credentials file may not be fully protected on Windows"
                )
        else:
            # Unix/Linux/Mac - use chmod
            try:
                os.chmod(filepath, 0o600)
                logger.info(f"Unix permissions (600) applied to {filepath}")
            except Exception as e:
                logger.error(f"Failed to set file permissions: {e}")
    
    def _get_key(self, master_password=None):
        """Get encryption key, prompting for password if needed."""
        if self._key:
            return self._key

        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, 'rb') as f:
                    key_data = f.read()

                # Check if this is a password-derived key (has salt prefix)
                if len(key_data) > 32 and master_password:
                    # Extract salt (first 32 bytes) and regenerate key
                    salt = key_data[:32]
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=salt,
                        iterations=100000,
                    )
                    self._key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
                else:
                    # Direct key (Fernet generated)
                    self._key = key_data

                return self._key
            except Exception:
                pass

        # If no key file exists, create one
        if master_password is None:
            print("🔐 Setting up secure credential storage...")
            print("You can use a master password for extra security, or press Enter to use system-generated key.")
            master_password = getpass.getpass("Enter master password (or press Enter for auto-generated): ")
            if not master_password:
                master_password = None

        self._key = self._get_or_create_key(master_password)
        return self._key
    
    def save_credentials(self, app_key, app_secret, master_password=None, user='system'):
        """
        Encrypt and save credentials.

        Args:
            app_key: Schwab App Key
            app_secret: Schwab App Secret
            master_password: Optional master password
            user: User saving credentials (for audit logging)

        Returns:
            True if successful, False otherwise

        Raises:
            ImportError: If cryptography library not available
            EncryptionError: If encryption fails
        """
        if not CRYPTO_AVAILABLE:
            audit_log(
                event_type=AuditEventType.ERROR_OCCURRED,
                user=user,
                action="save_credentials",
                resource="credentials",
                status="FAILURE",
                details={'error': 'cryptography library not available'}
            )
            raise ImportError("cryptography library not available. Install with: pip install cryptography")

        try:
            key = self._get_key(master_password)
            fernet = Fernet(key)

            # Encrypt credentials
            credentials = f"{app_key}\n{app_secret}"
            encrypted = fernet.encrypt(credentials.encode())

            # Save encrypted file
            with open(self.config_file, 'wb') as f:
                f.write(encrypted)

            # Restrict file permissions - cross-platform
            self._secure_file_permissions(self.config_file)

            # Audit log - SUCCESS
            audit_log(
                event_type=AuditEventType.CREDENTIALS_ACCESSED,
                user=user,
                action="save_credentials",
                resource=self.config_file,
                status="SUCCESS",
                details={'encrypted': True, 'has_master_password': master_password is not None}
            )

            logger.info("Credentials saved and encrypted successfully")
            return True

        except Exception as e:
            # Audit log - FAILURE
            audit_log(
                event_type=AuditEventType.ERROR_OCCURRED,
                user=user,
                action="save_credentials",
                resource="credentials",
                status="FAILURE",
                details={'error': str(e)}
            )
            raise EncryptionError(f"Failed to save credentials: {e}")
    
    def load_credentials(self, master_password=None, user='system'):
        """
        Load and decrypt credentials.

        Args:
            master_password: Optional master password (if used during save)
            user: User loading credentials (for audit logging)

        Returns:
            Tuple of (app_key, app_secret) or (None, None) if not found

        Raises:
            CredentialError: If credentials cannot be loaded
        """
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available")
            return None, None

        if not os.path.exists(self.config_file):
            logger.warning(f"Credential file not found: {self.config_file}")
            return None, None

        try:
            key = self._get_key(master_password)
            fernet = Fernet(key)

            # Read and decrypt
            with open(self.config_file, 'rb') as f:
                encrypted = f.read()

            decrypted = fernet.decrypt(encrypted).decode()
            lines = decrypted.strip().split('\n')

            if len(lines) >= 2:
                # Audit log - SUCCESS
                audit_log(
                    event_type=AuditEventType.CREDENTIALS_ACCESSED,
                    user=user,
                    action="load_credentials",
                    resource=self.config_file,
                    status="SUCCESS",
                    details={'encrypted': True}
                )

                return lines[0].strip(), lines[1].strip()
            else:
                raise CredentialError("Invalid credential format")

        except Exception as e:
            # If decryption fails, might need password
            if "InvalidToken" in str(e) or "Invalid key" in str(e):
                if master_password is None:
                    print("🔐 Encrypted credentials require a master password.")
                    master_password = getpass.getpass("Enter master password: ")
                    try:
                        # Try with password
                        key = self._get_or_create_key(master_password)
                        fernet = Fernet(key)
                        with open(self.config_file, 'rb') as f:
                            encrypted = f.read()
                        decrypted = fernet.decrypt(encrypted).decode()
                        lines = decrypted.strip().split('\n')
                        if len(lines) >= 2:
                            # Audit log - SUCCESS (after password)
                            audit_log(
                                event_type=AuditEventType.CREDENTIALS_ACCESSED,
                                user=user,
                                action="load_credentials",
                                resource=self.config_file,
                                status="SUCCESS",
                                details={'encrypted': True, 'required_password': True}
                            )
                            return lines[0].strip(), lines[1].strip()
                    except Exception as retry_error:
                        # Audit log - FAILURE
                        audit_log(
                            event_type=AuditEventType.AUTH_FAILURE,
                            user=user,
                            action="load_credentials",
                            resource=self.config_file,
                            status="FAILURE",
                            details={'error': str(retry_error)}
                        )
                        logger.error(f"Failed to decrypt credentials: {retry_error}")

            # Audit log - FAILURE
            audit_log(
                event_type=AuditEventType.ERROR_OCCURRED,
                user=user,
                action="load_credentials",
                resource=self.config_file,
                status="FAILURE",
                details={'error': str(e)}
            )

            return None, None
    
    def credentials_exist(self):
        """Check if credentials file exists."""
        return os.path.exists(self.config_file)
    
    def delete_credentials(self):
        """Delete stored credentials (for reconfiguration)."""
        deleted = False
        if os.path.exists(self.config_file):
            try:
                os.remove(self.config_file)
                deleted = True
            except Exception:
                pass
        
        # Optionally delete key file (but keep it for future use)
        # if os.path.exists(self.key_file):
        #     try:
        #         os.remove(self.key_file)
        #     except Exception:
        #         pass
        
        return deleted
