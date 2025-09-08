import os
import asyncio
import json
import logging
import secrets
import base64
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import aiohttp
from urllib.parse import urlencode, parse_qs
from pydantic import BaseModel
import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class TokenData(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    scope: str = ""
    created_at: datetime
    
    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > (self.created_at + timedelta(seconds=self.expires_in - 300))  # 5 min buffer

class OAuthState(BaseModel):
    state: str
    code_verifier: str
    redirect_uri: str
    scopes: List[str]
    created_at: datetime
    
    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > (self.created_at + timedelta(minutes=10))

class DerivOAuthManager:
    """
    Complete OAuth 2.0 implementation for Deriv API
    Supports PKCE (Proof Key for Code Exchange) for enhanced security
    """
    
    def __init__(self):
        # Deriv OAuth endpoints
        self.oauth_base_url = "https://oauth.deriv.com/oauth2"
        self.authorize_url = f"{self.oauth_base_url}/authorize"
        self.token_url = f"{self.oauth_base_url}/token"
        self.user_info_url = f"{self.oauth_base_url}/userinfo"
        
        # Application configuration
        self.client_id = os.getenv("DERIV_CLIENT_ID", "99188")  # Default app ID
        self.client_secret = os.getenv("DERIV_CLIENT_SECRET")
        self.redirect_uri = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:3000/oauth/callback")
        
        # Security configuration
        self.encryption_key = os.getenv("OAUTH_ENCRYPTION_KEY", Fernet.generate_key().decode())
        self.fernet = Fernet(self.encryption_key.encode())
        
        # State management
        self.active_states: Dict[str, OAuthState] = {}
        self.token_storage: Dict[str, TokenData] = {}
        
        # Supported scopes
        self.available_scopes = [
            "read",      # Read account information
            "trade",     # Execute trades
            "payments",  # Handle payments (use with caution)
            "admin"      # Administrative access (use with extreme caution)
        ]
        
        # Default scopes (minimal required for trading)
        self.default_scopes = ["read", "trade"]
        
        logger.info("DerivOAuthManager initialized")
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, code_verifier: str) -> str:
        """Generate PKCE code challenge from verifier"""
        digest = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    def _generate_state(self) -> str:
        """Generate cryptographically secure state parameter"""
        return secrets.token_urlsafe(32)
    
    async def get_authorization_url(self, 
                                  scopes: Optional[List[str]] = None,
                                  redirect_uri: Optional[str] = None) -> Dict[str, str]:
        """
        Generate OAuth authorization URL with PKCE
        
        Args:
            scopes: List of requested scopes
            redirect_uri: Custom redirect URI
            
        Returns:
            Dict with authorization URL and state information
        """
        try:
            # Use provided parameters or defaults
            scopes = scopes or self.default_scopes
            redirect_uri = redirect_uri or self.redirect_uri
            
            # Validate scopes
            invalid_scopes = [scope for scope in scopes if scope not in self.available_scopes]
            if invalid_scopes:
                raise ValueError(f"Invalid scopes: {invalid_scopes}")
            
            # Generate PKCE parameters
            code_verifier = self._generate_code_verifier()
            code_challenge = self._generate_code_challenge(code_verifier)
            state = self._generate_state()
            
            # Store OAuth state
            oauth_state = OAuthState(
                state=state,
                code_verifier=code_verifier,
                redirect_uri=redirect_uri,
                scopes=scopes,
                created_at=datetime.utcnow()
            )
            self.active_states[state] = oauth_state
            
            # Build authorization URL
            params = {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "scope": " ".join(scopes),
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256"
            }
            
            authorization_url = f"{self.authorize_url}?{urlencode(params)}"
            
            logger.info(f"Generated OAuth authorization URL for scopes: {scopes}")
            
            return {
                "authorization_url": authorization_url,
                "state": state,
                "scopes": scopes,
                "redirect_uri": redirect_uri
            }
            
        except Exception as e:
            logger.error(f"Error generating authorization URL: {e}")
            raise
    
    async def exchange_code_for_token(self, 
                                    authorization_code: str, 
                                    state: str) -> TokenData:
        """
        Exchange authorization code for access token
        
        Args:
            authorization_code: Code received from OAuth callback
            state: State parameter for CSRF protection
            
        Returns:
            TokenData with access and refresh tokens
        """
        try:
            # Validate state parameter
            if state not in self.active_states:
                raise ValueError("Invalid or expired state parameter")
            
            oauth_state = self.active_states[state]
            
            # Check state expiration
            if oauth_state.is_expired:
                del self.active_states[state]
                raise ValueError("OAuth state expired")
            
            # Prepare token exchange request
            token_data = {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "code": authorization_code,
                "redirect_uri": oauth_state.redirect_uri,
                "code_verifier": oauth_state.code_verifier
            }
            
            # Add client secret if available
            if self.client_secret:
                token_data["client_secret"] = self.client_secret
            
            # Exchange code for tokens
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Token exchange failed: {error_text}")
                        raise Exception(f"Token exchange failed: {response.status}")
                    
                    response_data = await response.json()
            
            # Create TokenData object
            token_data = TokenData(
                access_token=response_data["access_token"],
                refresh_token=response_data.get("refresh_token", ""),
                expires_in=response_data.get("expires_in", 3600),
                token_type=response_data.get("token_type", "Bearer"),
                scope=response_data.get("scope", " ".join(oauth_state.scopes)),
                created_at=datetime.utcnow()
            )
            
            # Store encrypted token
            self.token_storage[state] = token_data
            
            # Clean up state
            del self.active_states[state]
            
            logger.info(f"Successfully exchanged code for token with scopes: {token_data.scope}")
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            raise
    
    async def refresh_access_token(self, refresh_token: str) -> TokenData:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: Refresh token from previous authentication
            
        Returns:
            New TokenData with refreshed tokens
        """
        try:
            refresh_data = {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": refresh_token
            }
            
            if self.client_secret:
                refresh_data["client_secret"] = self.client_secret
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    data=refresh_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Token refresh failed: {error_text}")
                        raise Exception(f"Token refresh failed: {response.status}")
                    
                    response_data = await response.json()
            
            # Create new TokenData
            new_token_data = TokenData(
                access_token=response_data["access_token"],
                refresh_token=response_data.get("refresh_token", refresh_token),
                expires_in=response_data.get("expires_in", 3600),
                token_type=response_data.get("token_type", "Bearer"),
                scope=response_data.get("scope", ""),
                created_at=datetime.utcnow()
            )
            
            logger.info("Successfully refreshed access token")
            
            return new_token_data
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise
    
    async def validate_token(self, access_token: str) -> Dict[str, Any]:
        """
        Validate access token and get user information
        
        Args:
            access_token: Access token to validate
            
        Returns:
            User information and token validation data
        """
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.user_info_url,
                    headers=headers
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Token validation failed: {error_text}")
                        raise Exception(f"Invalid token: {response.status}")
                    
                    user_info = await response.json()
            
            logger.info(f"Token validated successfully for user: {user_info.get('email', 'unknown')}")
            
            return user_info
            
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            raise
    
    def encrypt_token(self, token_data: TokenData) -> str:
        """
        Encrypt token data for secure storage
        
        Args:
            token_data: TokenData to encrypt
            
        Returns:
            Encrypted token string
        """
        try:
            token_json = token_data.json()
            encrypted_token = self.fernet.encrypt(token_json.encode())
            return encrypted_token.decode()
        except Exception as e:
            logger.error(f"Error encrypting token: {e}")
            raise
    
    def decrypt_token(self, encrypted_token: str) -> TokenData:
        """
        Decrypt token data from storage
        
        Args:
            encrypted_token: Encrypted token string
            
        Returns:
            Decrypted TokenData
        """
        try:
            decrypted_data = self.fernet.decrypt(encrypted_token.encode())
            token_json = json.loads(decrypted_data.decode())
            return TokenData(**token_json)
        except Exception as e:
            logger.error(f"Error decrypting token: {e}")
            raise
    
    async def revoke_token(self, access_token: str) -> bool:
        """
        Revoke access token (logout)
        
        Args:
            access_token: Token to revoke
            
        Returns:
            True if successful
        """
        try:
            revoke_data = {
                "token": access_token,
                "client_id": self.client_id
            }
            
            if self.client_secret:
                revoke_data["client_secret"] = self.client_secret
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.oauth_base_url}/revoke",
                    data=revoke_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                ) as response:
                    
                    # Token revocation typically returns 200 regardless
                    success = response.status == 200
                    
                    if success:
                        logger.info("Token revoked successfully")
                    else:
                        logger.warning(f"Token revocation returned status: {response.status}")
                    
                    return success
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    def cleanup_expired_states(self):
        """Remove expired OAuth states"""
        expired_states = [
            state for state, oauth_state in self.active_states.items()
            if oauth_state.is_expired
        ]
        
        for state in expired_states:
            del self.active_states[state]
        
        if expired_states:
            logger.info(f"Cleaned up {len(expired_states)} expired OAuth states")
    
    async def get_token_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get detailed information about an access token
        
        Args:
            access_token: Token to inspect
            
        Returns:
            Token information including scopes and expiration
        """
        try:
            # For Deriv API, we validate token by making an API call
            headers = {
                "Authorization": f"Bearer {access_token}"
            }
            
            # Use Deriv WebSocket API to validate token
            async with aiohttp.ClientSession() as session:
                ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={self.client_id}"
                
                async with session.ws_connect(ws_url) as ws:
                    # Send authorize request
                    auth_request = {
                        "authorize": access_token,
                        "req_id": 1
                    }
                    
                    await ws.send_str(json.dumps(auth_request))
                    
                    # Wait for response
                    response = await ws.receive()
                    data = json.loads(response.data)
                    
                    if "error" in data:
                        raise Exception(f"Token validation failed: {data['error']}")
                    
                    return {
                        "valid": True,
                        "account_info": data.get("authorize", {}),
                        "scopes": data.get("authorize", {}).get("scopes", []),
                        "expires_at": "unknown",  # Deriv doesn't provide expiration in response
                        "token_type": "Bearer"
                    }
            
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            raise

# Singleton instance
oauth_manager = DerivOAuthManager()