#!/usr/bin/env python3
"""
Deployment helper script for Medical AI Assistant
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        '.streamlit/config.toml',
        'config/config.py',
        'models/llm.py',
        'models/database.py',
        'models/embeddings.py',
        'utils/error_handler.py',
        'utils/api_utilities.py',
        'utils/database_utilities.py',
        'utils/validation_utilities.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def check_git_status():
    """Check git status and provide deployment guidance"""
    try:
        # Check if git is initialized
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Git repository not initialized")
            print("Run: git init")
            return False
        
        # Check for uncommitted changes
        if "nothing to commit" not in result.stdout:
            print("⚠️  You have uncommitted changes")
            print("Consider committing them before deployment:")
            print("   git add .")
            print("   git commit -m 'Prepare for deployment'")
        
        print("✅ Git repository ready")
        return True
        
    except FileNotFoundError:
        print("❌ Git not found. Please install Git first")
        return False

def create_deployment_checklist():
    """Create a deployment checklist"""
    checklist = """
🚀 STREAMLIT CLOUD DEPLOYMENT CHECKLIST

1. ✅ Code Preparation:
   - All files committed to git
   - requirements.txt updated
   - .streamlit/config.toml configured
   - .gitignore excludes sensitive files

2. 🔑 API Keys Setup:
   - Groq API Key (recommended)
   - Google API Key (for Gemini)
   - OpenAI API Key (optional)
   - Serper API Key (for web search)

3. 📤 GitHub Upload:
   - Repository created on GitHub
   - Code pushed to main branch
   - Repository is public (for free Streamlit Cloud)

4. 🌐 Streamlit Cloud Deployment:
   - Go to https://share.streamlit.io/
   - Connect GitHub repository
   - Configure secrets with API keys
   - Deploy app

5. ✅ Post-Deployment:
   - Test all features
   - Verify API keys work
   - Check error handling
   - Monitor usage

📋 NEXT STEPS:
1. Run: git add .
2. Run: git commit -m "Prepare for Streamlit Cloud deployment"
3. Push to GitHub: git push origin main
4. Go to: https://share.streamlit.io/
5. Deploy your app!

🔗 Your app will be available at: https://YOUR_APP_NAME.streamlit.app
"""
    print(checklist)

def main():
    """Main deployment helper function"""
    print("🏥 Medical AI Assistant - Deployment Helper")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Deployment preparation failed")
        sys.exit(1)
    
    # Check git status
    check_git_status()
    
    # Show deployment checklist
    create_deployment_checklist()
    
    print("\n🎉 Ready for deployment!")

if __name__ == "__main__":
    main()
