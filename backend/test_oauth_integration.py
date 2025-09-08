#!/usr/bin/env python3
"""
OAuth Integration Tests for Deriv API
Comprehensive testing of OAuth 2.0 flow with PKCE implementation
"""

import asyncio
import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from oauth_manager import DerivOAuthManager, TokenData, OAuthState
import json
import aiohttp
from aiohttp import web
import urllib.parse

class TestDerivOAuthManager:
    """Test suite for Deriv OAuth Manager"""
    
    @pytest.fixture
    def oauth_manager(self):
        """Create a fresh OAuth manager instance for each test"""
        return DerivOAuthManager()
    
    @pytest.fixture
    def mock_token_response(self):
        """Mock token response from Deriv OAuth server"""
        return {
            "access_token": "mock_access_token_12345",
            "refresh_token": "mock_refresh_token_67890", 
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "read trade"
        }
    
    @pytest.fixture
    def mock_user_info(self):
        """Mock user info response"""
        return {
            "user_id": "12345",
            "email": "test@example.com",
            "country": "BR",
            "currency": "USD",
            "scopes": ["read", "trade"]
        }

    def test_initialization(self, oauth_manager):
        """Test OAuth manager initialization"""
        assert oauth_manager.client_id == "99188"
        assert oauth_manager.oauth_base_url == "https://oauth.deriv.com/oauth2"
        assert oauth_manager.authorize_url == "https://oauth.deriv.com/oauth2/authorize"
        assert oauth_manager.token_url == "https://oauth.deriv.com/oauth2/token"
        assert oauth_manager.available_scopes == ["read", "trade", "payments", "admin"]
        assert oauth_manager.default_scopes == ["read", "trade"]
    
    def test_code_verifier_generation(self, oauth_manager):
        """Test PKCE code verifier generation"""
        verifier1 = oauth_manager._generate_code_verifier()
        verifier2 = oauth_manager._generate_code_verifier()
        
        # Each verifier should be unique
        assert verifier1 != verifier2
        
        # Should be proper length (32 bytes base64url encoded)
        assert len(verifier1) >= 43  # 32 bytes * 4/3 (base64) - padding
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" 
                  for c in verifier1)
    
    def test_code_challenge_generation(self, oauth_manager):
        """Test PKCE code challenge generation"""
        verifier = "test_code_verifier_123456789"
        challenge = oauth_manager._generate_code_challenge(verifier)
        
        # Challenge should be deterministic for same verifier
        challenge2 = oauth_manager._generate_code_challenge(verifier)
        assert challenge == challenge2
        
        # Should be base64url encoded SHA256
        assert len(challenge) == 43  # SHA256 is 32 bytes -> 43 chars base64url
    
    def test_state_generation(self, oauth_manager):
        """Test OAuth state generation"""
        state1 = oauth_manager._generate_state()
        state2 = oauth_manager._generate_state()
        
        # Each state should be unique
        assert state1 != state2
        assert len(state1) >= 32  # Should be cryptographically secure
    
    @pytest.mark.asyncio
    async def test_get_authorization_url_default_params(self, oauth_manager):
        """Test authorization URL generation with default parameters"""
        result = await oauth_manager.get_authorization_url()
        
        assert "authorization_url" in result
        assert "state" in result
        assert "scopes" in result
        assert "redirect_uri" in result
        
        # Check URL components
        url = result["authorization_url"]
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        
        assert parsed.hostname == "oauth.deriv.com"
        assert params["response_type"][0] == "code"
        assert params["client_id"][0] == "99188"
        assert params["scope"][0] == "read trade"
        assert "code_challenge" in params
        assert params["code_challenge_method"][0] == "S256"
        assert "state" in params
        
        # State should be stored
        state = result["state"]
        assert state in oauth_manager.active_states
    
    @pytest.mark.asyncio
    async def test_get_authorization_url_custom_params(self, oauth_manager):
        """Test authorization URL generation with custom parameters"""
        custom_scopes = ["read", "payments"]
        custom_redirect = "https://myapp.com/callback"
        
        result = await oauth_manager.get_authorization_url(
            scopes=custom_scopes,
            redirect_uri=custom_redirect
        )
        
        url = result["authorization_url"]
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        
        assert params["scope"][0] == "read payments"
        assert params["redirect_uri"][0] == custom_redirect
        assert result["scopes"] == custom_scopes
        assert result["redirect_uri"] == custom_redirect
    
    @pytest.mark.asyncio
    async def test_get_authorization_url_invalid_scopes(self, oauth_manager):
        """Test authorization URL generation with invalid scopes"""
        invalid_scopes = ["read", "invalid_scope"]
        
        with pytest.raises(ValueError) as exc_info:
            await oauth_manager.get_authorization_url(scopes=invalid_scopes)
        
        assert "Invalid scopes: ['invalid_scope']" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_exchange_code_for_token_success(self, oauth_manager, mock_token_response):
        """Test successful code-to-token exchange"""
        # Set up OAuth state
        state = "test_state_123"
        oauth_state = OAuthState(
            state=state,
            code_verifier="test_verifier",
            redirect_uri="http://localhost:3000/callback",
            scopes=["read", "trade"],
            created_at=datetime.utcnow()
        )
        oauth_manager.active_states[state] = oauth_state
        
        # Mock HTTP response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_token_response
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Execute token exchange
            token_data = await oauth_manager.exchange_code_for_token("auth_code_123", state)
            
            # Verify token data
            assert isinstance(token_data, TokenData)
            assert token_data.access_token == "mock_access_token_12345"
            assert token_data.refresh_token == "mock_refresh_token_67890"
            assert token_data.expires_in == 3600
            assert token_data.token_type == "Bearer"
            assert token_data.scope == "read trade"
            
            # Verify state cleanup
            assert state not in oauth_manager.active_states
            assert state in oauth_manager.token_storage
    
    @pytest.mark.asyncio
    async def test_exchange_code_invalid_state(self, oauth_manager):
        """Test code exchange with invalid state"""
        with pytest.raises(ValueError) as exc_info:
            await oauth_manager.exchange_code_for_token("auth_code_123", "invalid_state")
        
        assert "Invalid or expired state parameter" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_exchange_code_expired_state(self, oauth_manager):
        """Test code exchange with expired state"""
        state = "expired_state_123"
        expired_oauth_state = OAuthState(
            state=state,
            code_verifier="test_verifier",
            redirect_uri="http://localhost:3000/callback",
            scopes=["read"],
            created_at=datetime.utcnow() - timedelta(minutes=15)  # Expired
        )
        oauth_manager.active_states[state] = expired_oauth_state
        
        with pytest.raises(ValueError) as exc_info:
            await oauth_manager.exchange_code_for_token("auth_code_123", state)
        
        assert "OAuth state expired" in str(exc_info.value)
        assert state not in oauth_manager.active_states  # Should be cleaned up
    
    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self, oauth_manager, mock_token_response):
        """Test successful token refresh"""
        refresh_token = "refresh_token_123"
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                **mock_token_response,
                "access_token": "new_access_token_456"
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            new_token_data = await oauth_manager.refresh_access_token(refresh_token)
            
            assert new_token_data.access_token == "new_access_token_456"
            assert new_token_data.refresh_token == "mock_refresh_token_67890"
    
    @pytest.mark.asyncio
    async def test_refresh_access_token_failure(self, oauth_manager):
        """Test token refresh failure"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text.return_value = "Invalid refresh token"
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(Exception) as exc_info:
                await oauth_manager.refresh_access_token("invalid_refresh_token")
            
            assert "Token refresh failed: 400" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_token_success(self, oauth_manager, mock_user_info):
        """Test successful token validation"""
        access_token = "valid_token_123"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_user_info
            mock_get.return_value.__aenter__.return_value = mock_response
            
            user_info = await oauth_manager.validate_token(access_token)
            
            assert user_info["email"] == "test@example.com"
            assert user_info["user_id"] == "12345"
            assert user_info["scopes"] == ["read", "trade"]
    
    @pytest.mark.asyncio
    async def test_validate_token_failure(self, oauth_manager):
        """Test token validation failure"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.text.return_value = "Unauthorized"
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(Exception) as exc_info:
                await oauth_manager.validate_token("invalid_token")
            
            assert "Invalid token: 401" in str(exc_info.value)
    
    def test_token_encryption_decryption(self, oauth_manager):
        """Test token encryption and decryption"""
        original_token = TokenData(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_in=3600,
            token_type="Bearer",
            scope="read trade",
            created_at=datetime.utcnow()
        )
        
        # Encrypt token
        encrypted = oauth_manager.encrypt_token(original_token)
        assert isinstance(encrypted, str)
        assert encrypted != original_token.access_token
        
        # Decrypt token
        decrypted = oauth_manager.decrypt_token(encrypted)
        assert decrypted.access_token == original_token.access_token
        assert decrypted.refresh_token == original_token.refresh_token
        assert decrypted.expires_in == original_token.expires_in
    
    @pytest.mark.asyncio
    async def test_revoke_token_success(self, oauth_manager):
        """Test successful token revocation"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await oauth_manager.revoke_token("token_to_revoke")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_revoke_token_failure(self, oauth_manager):
        """Test token revocation failure"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await oauth_manager.revoke_token("invalid_token")
            assert result is False
    
    def test_cleanup_expired_states(self, oauth_manager):
        """Test cleanup of expired OAuth states"""
        # Add fresh state
        fresh_state = OAuthState(
            state="fresh_state",
            code_verifier="verifier",
            redirect_uri="http://localhost:3000",
            scopes=["read"],
            created_at=datetime.utcnow()
        )
        oauth_manager.active_states["fresh_state"] = fresh_state
        
        # Add expired state
        expired_state = OAuthState(
            state="expired_state",
            code_verifier="verifier",
            redirect_uri="http://localhost:3000",
            scopes=["read"],
            created_at=datetime.utcnow() - timedelta(minutes=15)
        )
        oauth_manager.active_states["expired_state"] = expired_state
        
        # Cleanup
        oauth_manager.cleanup_expired_states()
        
        # Fresh state should remain, expired should be removed
        assert "fresh_state" in oauth_manager.active_states
        assert "expired_state" not in oauth_manager.active_states
    
    def test_token_data_expiration(self):
        """Test TokenData expiration logic"""
        # Create token that expires in 1 hour, created 50 minutes ago (should not be expired with 5 min buffer)
        token = TokenData(
            access_token="test_token",
            refresh_token="refresh_token",
            expires_in=3600,
            created_at=datetime.utcnow() - timedelta(minutes=50)  # 50 minutes ago
        )
        
        # Should not be expired (5 minute buffer)
        assert not token.is_expired
        
        # Create token that should be expired (created more than expires_in - buffer ago)
        expired_token = TokenData(
            access_token="test_token",
            refresh_token="refresh_token", 
            expires_in=3600,
            created_at=datetime.utcnow() - timedelta(minutes=65)  # 65 minutes ago (exceeds 60-5 buffer)
        )
        
        assert expired_token.is_expired
    
    @pytest.mark.asyncio
    async def test_get_token_info_with_websocket(self, oauth_manager):
        """Test getting token info via WebSocket validation"""
        access_token = "valid_ws_token"
        
        # Mock WebSocket response
        mock_auth_response = {
            "authorize": {
                "account_list": [{"loginid": "CR123456", "currency": "USD"}],
                "balance": 1000.00,
                "country": "br",
                "email": "test@example.com",
                "scopes": ["read", "trade"]
            }
        }
        
        with patch('aiohttp.ClientSession.ws_connect') as mock_ws:
            mock_ws_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.data = json.dumps(mock_auth_response)
            
            mock_ws_instance.send_str = AsyncMock()
            mock_ws_instance.receive = AsyncMock(return_value=mock_response)
            mock_ws.return_value.__aenter__.return_value = mock_ws_instance
            
            token_info = await oauth_manager.get_token_info(access_token)
            
            assert token_info["valid"] is True
            assert token_info["account_info"] == mock_auth_response["authorize"]
            assert token_info["scopes"] == ["read", "trade"]
            assert token_info["token_type"] == "Bearer"
    
    @pytest.mark.asyncio
    async def test_get_token_info_websocket_error(self, oauth_manager):
        """Test token info retrieval with WebSocket error"""
        access_token = "invalid_ws_token"
        
        # Mock WebSocket error response
        mock_error_response = {
            "error": {
                "code": "InvalidToken",
                "message": "Invalid or expired token"
            }
        }
        
        with patch('aiohttp.ClientSession.ws_connect') as mock_ws:
            mock_ws_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.data = json.dumps(mock_error_response)
            
            mock_ws_instance.send_str = AsyncMock()
            mock_ws_instance.receive = AsyncMock(return_value=mock_response)
            mock_ws.return_value.__aenter__.return_value = mock_ws_instance
            
            with pytest.raises(Exception) as exc_info:
                await oauth_manager.get_token_info(access_token)
            
            assert "Token validation failed" in str(exc_info.value)

class TestOAuthIntegrationFlow:
    """Integration tests for complete OAuth flow"""
    
    @pytest.mark.asyncio
    async def test_complete_oauth_flow(self):
        """Test complete OAuth 2.0 flow from start to finish"""
        oauth_manager = DerivOAuthManager()
        
        # Step 1: Generate authorization URL
        auth_result = await oauth_manager.get_authorization_url(
            scopes=["read", "trade"]
        )
        
        state = auth_result["state"]
        assert state in oauth_manager.active_states
        
        # Step 2: Mock authorization code callback
        authorization_code = "mock_auth_code_12345"
        
        mock_token_response = {
            "access_token": "integration_access_token",
            "refresh_token": "integration_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "read trade"
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_token_response
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Step 3: Exchange code for token
            token_data = await oauth_manager.exchange_code_for_token(
                authorization_code, state
            )
            
            assert token_data.access_token == "integration_access_token"
            assert state not in oauth_manager.active_states
        
        # Step 4: Test token validation
        mock_user_info = {
            "user_id": "integration_user_123",
            "email": "integration@test.com",
            "scopes": ["read", "trade"]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_user_info
            mock_get.return_value.__aenter__.return_value = mock_response
            
            user_info = await oauth_manager.validate_token(token_data.access_token)
            assert user_info["email"] == "integration@test.com"
        
        # Step 5: Test token refresh
        new_token_response = {
            **mock_token_response,
            "access_token": "refreshed_access_token"
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = new_token_response
            mock_post.return_value.__aenter__.return_value = mock_response
            
            refreshed_token = await oauth_manager.refresh_access_token(
                token_data.refresh_token
            )
            assert refreshed_token.access_token == "refreshed_access_token"
        
        # Step 6: Test token revocation
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            revoke_success = await oauth_manager.revoke_token(
                refreshed_token.access_token
            )
            assert revoke_success is True

# Real environment test (commented out, enable for manual testing)
class TestOAuthRealEnvironment:
    """
    Real environment tests - ONLY run manually with valid credentials
    These tests require actual Deriv OAuth configuration
    """
    
    @pytest.mark.skip(reason="Requires real OAuth credentials - run manually")
    @pytest.mark.asyncio
    async def test_real_authorization_url_generation(self):
        """Test generating real authorization URL (manual verification)"""
        oauth_manager = DerivOAuthManager()
        
        result = await oauth_manager.get_authorization_url(
            scopes=["read"]  # Use minimal scopes for testing
        )
        
        print(f"Generated authorization URL: {result['authorization_url']}")
        print(f"State: {result['state']}")
        print("Manual verification: Visit the URL and check it loads correctly")
        
        # Verify URL structure
        assert "oauth.deriv.com" in result["authorization_url"]
        assert "code_challenge" in result["authorization_url"]
        assert result["state"] in oauth_manager.active_states

if __name__ == "__main__":
    print("🧪 Running OAuth Integration Tests...")
    print("=" * 60)
    
    # Install pytest if not available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not installed. Installing...")
        os.system("pip install pytest pytest-asyncio")
        import pytest
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])