"""
Helper script to set up Replit environment.
Run this once to install dependencies and set up the environment.
"""
import os
import subprocess
import sys

def main():
    print("Setting up Replit environment for AI Content Generator backend...")
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check for environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "FIREBASE_PRIVATE_KEY_ID",
        "FIREBASE_PRIVATE_KEY",
        "FIREBASE_CLIENT_EMAIL",
        "FIREBASE_CLIENT_ID",
        "FIREBASE_CLIENT_X509_CERT_URL",
        "FIREBASE_PROJECT_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("\n⚠️ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease add these in the Replit Secrets tab (lock icon in the sidebar).")
    else:
        print("\n✅ All required environment variables are set.")
    
    print("\nSetup complete! You can now run the backend with:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8080")

if __name__ == "__main__":
    main()

