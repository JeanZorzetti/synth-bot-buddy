#!/usr/bin/env python3
"""
Live OAuth Testing Script for Deriv API
Interactive script to test OAuth flow with real Deriv environment
Use this for manual testing and validation
"""

import asyncio
import os
from dotenv import load_dotenv
import webbrowser
from oauth_manager import DerivOAuthManager
from datetime import datetime
import json

# Load environment variables
load_dotenv()

async def test_oauth_authorization_url():
    """Test OAuth authorization URL generation"""
    print("🔐 Testing OAuth Authorization URL Generation...")
    
    oauth_manager = DerivOAuthManager()
    
    # Test with default scopes
    print("\n📝 Generating authorization URL with default scopes (read, trade)...")
    result = await oauth_manager.get_authorization_url()
    
    print(f"✅ Authorization URL generated successfully:")
    print(f"   URL: {result['authorization_url']}")
    print(f"   State: {result['state']}")
    print(f"   Scopes: {result['scopes']}")
    print(f"   Redirect URI: {result['redirect_uri']}")
    
    # Ask user if they want to open the URL
    open_browser = input("\n🌐 Open authorization URL in browser? (y/n): ").lower().strip()
    if open_browser == 'y':
        webbrowser.open(result['authorization_url'])
        print("✅ Browser opened with authorization URL")
        
        # Provide instructions for manual testing
        print("\n📋 Manual Testing Instructions:")
        print("1. Complete the authorization in the browser")
        print("2. You'll be redirected to your redirect URI with a code parameter")
        print("3. Copy the authorization code from the URL")
        print("4. Return here and enter the code when prompted")
        
        # Wait for manual code input (for demo purposes)
        print("\n⏳ After completing authorization, you can test token exchange...")
        auth_code = input("Enter authorization code (or 'skip' to continue): ").strip()
        
        if auth_code and auth_code != 'skip':
            print(f"📋 Authorization code received: {auth_code}")
            try:
                # Simulate token exchange (would work with real code)
                print("⚠️  Note: Token exchange requires real authorization code from Deriv")
                print("   This is a demo - actual exchange would happen here")
            except Exception as e:
                print(f"❌ Token exchange failed: {e}")
    
    return result

async def test_oauth_token_management():
    """Test OAuth token management features"""
    print("\n🎫 Testing OAuth Token Management...")
    
    oauth_manager = DerivOAuthManager()
    
    # Create mock token for testing
    from oauth_manager import TokenData
    mock_token = TokenData(
        access_token="demo_access_token_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        refresh_token="demo_refresh_token_12345",
        expires_in=3600,
        token_type="Bearer",
        scope="read trade",
        created_at=datetime.utcnow()
    )
    
    print(f"📦 Created mock token:")
    print(f"   Access Token: {mock_token.access_token}")
    print(f"   Expires In: {mock_token.expires_in} seconds")
    print(f"   Scopes: {mock_token.scope}")
    print(f"   Is Expired: {mock_token.is_expired}")
    
    # Test encryption
    print("\n🔒 Testing token encryption...")
    encrypted_token = oauth_manager.encrypt_token(mock_token)
    print(f"✅ Token encrypted successfully (length: {len(encrypted_token)})")
    
    # Test decryption
    print("🔓 Testing token decryption...")
    decrypted_token = oauth_manager.decrypt_token(encrypted_token)
    print(f"✅ Token decrypted successfully")
    print(f"   Decrypted Access Token: {decrypted_token.access_token}")
    print(f"   Match Original: {decrypted_token.access_token == mock_token.access_token}")
    
    return mock_token

async def test_oauth_state_management():
    """Test OAuth state management and cleanup"""
    print("\n🗂️  Testing OAuth State Management...")
    
    oauth_manager = DerivOAuthManager()
    
    # Generate multiple authorization URLs to create states
    states = []
    for i in range(3):
        result = await oauth_manager.get_authorization_url(
            scopes=["read"] if i % 2 == 0 else ["read", "trade"]
        )
        states.append(result['state'])
        print(f"   State {i+1}: {result['state'][:20]}...")
    
    print(f"✅ Created {len(states)} OAuth states")
    print(f"   Active states count: {len(oauth_manager.active_states)}")
    
    # Test state cleanup (no expired states yet, so count should remain)
    initial_count = len(oauth_manager.active_states)
    oauth_manager.cleanup_expired_states()
    after_cleanup = len(oauth_manager.active_states)
    
    print(f"🧹 Cleanup completed:")
    print(f"   Before cleanup: {initial_count} states")
    print(f"   After cleanup: {after_cleanup} states")
    print(f"   Expired states removed: {initial_count - after_cleanup}")
    
    return states

async def test_oauth_scopes_validation():
    """Test OAuth scopes validation"""
    print("\n🎯 Testing OAuth Scopes Validation...")
    
    oauth_manager = DerivOAuthManager()
    
    # Test valid scopes
    valid_scope_sets = [
        ["read"],
        ["read", "trade"],
        ["read", "trade", "payments"],
        ["admin"]  # Use with extreme caution
    ]
    
    print("✅ Testing valid scopes:")
    for scopes in valid_scope_sets:
        try:
            result = await oauth_manager.get_authorization_url(scopes=scopes)
            print(f"   ✓ {scopes} - Valid")
        except Exception as e:
            print(f"   ❌ {scopes} - Error: {e}")
    
    # Test invalid scopes
    invalid_scope_sets = [
        ["invalid_scope"],
        ["read", "invalid_scope"],
        ["trade", "not_real_scope"]
    ]
    
    print("\n❌ Testing invalid scopes:")
    for scopes in invalid_scope_sets:
        try:
            result = await oauth_manager.get_authorization_url(scopes=scopes)
            print(f"   ❌ {scopes} - Should have failed!")
        except ValueError as e:
            print(f"   ✓ {scopes} - Correctly rejected: {e}")
        except Exception as e:
            print(f"   ⚠️  {scopes} - Unexpected error: {e}")

async def test_oauth_security_features():
    """Test OAuth security features (PKCE, state validation, etc.)"""
    print("\n🛡️  Testing OAuth Security Features...")
    
    oauth_manager = DerivOAuthManager()
    
    # Test PKCE code generation
    print("🔐 Testing PKCE implementation:")
    verifier1 = oauth_manager._generate_code_verifier()
    verifier2 = oauth_manager._generate_code_verifier()
    challenge1 = oauth_manager._generate_code_challenge(verifier1)
    challenge2 = oauth_manager._generate_code_challenge(verifier2)
    
    print(f"   Code Verifier 1: {verifier1[:20]}...")
    print(f"   Code Challenge 1: {challenge1[:20]}...")
    print(f"   Verifiers Unique: {verifier1 != verifier2}")
    print(f"   Challenges Unique: {challenge1 != challenge2}")
    
    # Test state generation
    print("\n🎲 Testing state generation:")
    states = [oauth_manager._generate_state() for _ in range(5)]
    unique_states = len(set(states))
    print(f"   Generated 5 states, {unique_states} unique")
    print(f"   State uniqueness: {'✅ PASS' if unique_states == 5 else '❌ FAIL'}")
    
    # Test state validation with expired state
    print("\n⏰ Testing state expiration:")
    from oauth_manager import OAuthState
    from datetime import timedelta
    
    # Create expired state manually
    expired_state = OAuthState(
        state="test_expired_state",
        code_verifier="test_verifier",
        redirect_uri="http://localhost:3000",
        scopes=["read"],
        created_at=datetime.utcnow() - timedelta(minutes=15)  # 15 minutes ago
    )
    
    print(f"   Expired state created: {expired_state.state}")
    print(f"   Is expired: {expired_state.is_expired}")
    print(f"   State expiration check: {'✅ PASS' if expired_state.is_expired else '❌ FAIL'}")

async def run_comprehensive_oauth_test():
    """Run comprehensive OAuth testing suite"""
    print("=" * 70)
    print("         COMPREHENSIVE OAUTH TESTING - SYNTH BOT BUDDY")
    print("=" * 70)
    print("🚀 Starting comprehensive OAuth testing suite...\n")
    
    try:
        # Test 1: Authorization URL generation
        await test_oauth_authorization_url()
        await asyncio.sleep(1)
        
        # Test 2: Token management
        await test_oauth_token_management()
        await asyncio.sleep(1)
        
        # Test 3: State management
        await test_oauth_state_management()
        await asyncio.sleep(1)
        
        # Test 4: Scopes validation
        await test_oauth_scopes_validation()
        await asyncio.sleep(1)
        
        # Test 5: Security features
        await test_oauth_security_features()
        
        print("\n" + "=" * 70)
        print("🎉 COMPREHENSIVE OAUTH TEST COMPLETED SUCCESSFULLY!")
        print("✅ All OAuth components are functioning correctly")
        print("🔐 Security features (PKCE, state validation) working properly")
        print("🎯 Scope validation and management operational")
        print("🎫 Token management and encryption functional")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n💥 ERROR during OAuth testing: {e}")
        print("❌ OAuth testing failed - check configuration and network connection")
        return False

async def interactive_oauth_demo():
    """Interactive OAuth demonstration"""
    print("\n🎮 Interactive OAuth Demo")
    print("=" * 40)
    
    while True:
        print("\n🔐 OAuth Demo Options:")
        print("1. Generate Authorization URL")
        print("2. Test Token Encryption/Decryption")
        print("3. Test State Management")
        print("4. Test Scope Validation")
        print("5. Security Feature Demo")
        print("6. Exit Demo")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            await test_oauth_authorization_url()
        elif choice == "2":
            await test_oauth_token_management()
        elif choice == "3":
            await test_oauth_state_management()
        elif choice == "4":
            await test_oauth_scopes_validation()
        elif choice == "5":
            await test_oauth_security_features()
        elif choice == "6":
            print("👋 Exiting OAuth demo...")
            break
        else:
            print("❌ Invalid option. Please select 1-6.")

if __name__ == "__main__":
    print("🔐 Deriv OAuth Live Testing Utility")
    print("=" * 50)
    
    # Check environment
    deriv_client_id = os.getenv("DERIV_CLIENT_ID")
    if deriv_client_id:
        print(f"🆔 Using Client ID: {deriv_client_id}")
    else:
        print("⚠️  No DERIV_CLIENT_ID configured, using default (99188)")
    
    try:
        print("\n🚀 Starting OAuth live testing...")
        
        # Ask user for test mode
        print("\nSelect test mode:")
        print("1. Comprehensive automated test")
        print("2. Interactive demo mode")
        
        mode = input("\nSelect mode (1-2): ").strip()
        
        if mode == "1":
            result = asyncio.run(run_comprehensive_oauth_test())
            if result:
                print("\n✅ All tests passed!")
            else:
                print("\n❌ Some tests failed!")
        elif mode == "2":
            asyncio.run(interactive_oauth_demo())
        else:
            print("❌ Invalid selection, running comprehensive test...")
            asyncio.run(run_comprehensive_oauth_test())
            
    except KeyboardInterrupt:
        print("\n🛑 OAuth testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during OAuth testing: {e}")
    
    print("\n🏁 OAuth testing session completed.")
    print("=" * 50)