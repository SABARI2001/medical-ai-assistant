#!/usr/bin/env python3
"""
Deployment helper script for Medical AI Assistant
Supports multiple cloud platforms
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if required tools are installed"""
    tools = {
        'git': 'Git is required for deployment',
        'python': 'Python is required'
    }
    
    missing = []
    for tool, message in tools.items():
        if not subprocess.run(['which', tool], capture_output=True).returncode == 0:
            missing.append(message)
    
    if missing:
        print("❌ Missing requirements:")
        for req in missing:
            print(f"  - {req}")
        return False
    
    print("✅ All requirements met")
    return True

def setup_environment():
    """Setup environment variables for deployment"""
    print("\n🔧 Setting up environment...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please create one from env_template.txt")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required API keys
    required_keys = ['GROQ_API_KEY', 'GOOGLE_API_KEY', 'SERPER_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key) or os.getenv(key) == f'your_{key.lower()}_here':
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please update your .env file with actual API keys")
        return False
    
    print("✅ Environment configured")
    return True

def deploy_railway():
    """Deploy to Railway"""
    print("\n🚀 Deploying to Railway...")
    
    try:
        # Check if Railway CLI is installed
        result = subprocess.run(['railway', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Railway CLI not found. Install with: npm install -g @railway/cli")
            return False
        
        # Login to Railway
        print("Logging into Railway...")
        subprocess.run(['railway', 'login'], check=True)
        
        # Initialize project
        print("Initializing Railway project...")
        subprocess.run(['railway', 'init'], check=True)
        
        # Add PostgreSQL database
        print("Adding PostgreSQL database...")
        subprocess.run(['railway', 'add', 'postgresql'], check=True)
        
        # Deploy
        print("Deploying application...")
        subprocess.run(['railway', 'up'], check=True)
        
        print("✅ Successfully deployed to Railway!")
        print("Your app will be available at the URL shown above")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Railway deployment failed: {e}")
        return False

def deploy_heroku():
    """Deploy to Heroku"""
    print("\n🚀 Deploying to Heroku...")
    
    try:
        # Check if Heroku CLI is installed
        result = subprocess.run(['heroku', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Heroku CLI not found. Install from: https://devcenter.heroku.com/articles/heroku-cli")
            return False
        
        # Login to Heroku
        print("Logging into Heroku...")
        subprocess.run(['heroku', 'login'], check=True)
        
        # Create app
        app_name = input("Enter Heroku app name (or press Enter for auto-generated): ").strip()
        if app_name:
            subprocess.run(['heroku', 'create', app_name], check=True)
        else:
            subprocess.run(['heroku', 'create'], check=True)
        
        # Add PostgreSQL
        print("Adding PostgreSQL database...")
        subprocess.run(['heroku', 'addons:create', 'heroku-postgresql:mini'], check=True)
        
        # Set environment variables
        print("Setting environment variables...")
        env_vars = ['GROQ_API_KEY', 'GOOGLE_API_KEY', 'SERPER_API_KEY']
        for var in env_vars:
            value = os.getenv(var)
            if value:
                subprocess.run(['heroku', 'config:set', f'{var}={value}'], check=True)
        
        # Deploy
        print("Deploying application...")
        subprocess.run(['git', 'push', 'heroku', 'main'], check=True)
        
        print("✅ Successfully deployed to Heroku!")
        print("Your app will be available at the URL shown above")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Heroku deployment failed: {e}")
        return False

def main():
    """Main deployment function"""
    print("🏥 Medical AI Assistant - Deployment Helper")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Choose deployment platform
    print("\n🌐 Choose deployment platform:")
    print("1. Railway (Recommended)")
    print("2. Heroku")
    print("3. Manual instructions")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        deploy_railway()
    elif choice == '2':
        deploy_heroku()
    elif choice == '3':
        print("\n📖 Manual deployment instructions:")
        print("See deployment/README.md for detailed instructions")
    else:
        print("❌ Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main()
