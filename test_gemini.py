#!/usr/bin/env python3
"""
Quick test script to diagnose Gemini API issues (NEW google-genai package)
"""

import os
import sys

print("="*60)
print(" Gemini API Diagnostic Test (NEW API)")
print("="*60)

# Step 1: Check environment variable
print("\n1. Checking GEMINI_API_KEY environment variable...")
api_key = os.environ.get('GEMINI_API_KEY')
if api_key:
    print(f"   ✓ API key found (length: {len(api_key)} chars)")
    print(f"   First 10 chars: {api_key[:10]}...")
else:
    print("   ✗ GEMINI_API_KEY not set!")
    print("   Set it with: export GEMINI_API_KEY='your-key-here'")
    sys.exit(1)

# Step 2: Check package installation
print("\n2. Checking google-genai package...")
try:
    from google import genai
    from google.genai import types
    print("   ✓ Package imported successfully")
except ImportError as e:
    print(f"   ✗ Package not found: {e}")
    print("   Install with: pip install -q -U google-genai")
    sys.exit(1)

# Step 3: Try to create client
print("\n3. Creating Gemini client...")
try:
    client = genai.Client(api_key=api_key)
    print("   ✓ Client created successfully")
except Exception as e:
    print(f"   ✗ Client creation error: {e}")
    sys.exit(1)

# Step 4: Try a simple generation
print("\n4. Testing API call...")
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents='Respond with only: OK'
    )
    print(f"   ✓ API call successful!")
    print(f"   Response: {response.text.strip()}")
except Exception as e:
    print(f"   ✗ API call failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    # Try alternative model
    print("\n   Trying alternative model...")
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents='Respond with only: OK'
        )
        print(f"   ✓ API call successful with gemini-1.5-flash!")
        print(f"   Response: {response.text.strip()}")
    except Exception as e2:
        print(f"   ✗ Alternative model also failed: {e2}")
        sys.exit(1)

print("\n" + "="*60)
print(" ✓ All tests passed! Gemini is working correctly.")
print("="*60)
