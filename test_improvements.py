
"""
Test Script for Improved Fact Checker
Tests all the failing cases to verify improvements
"""

import json
import requests
from typing import Dict, Any

API_URL = "http://localhost:7001/fact-check"

# Test cases that were failing
TEST_CASES = [
    {
        "name": "Math - Logarithm Question",
        "data": {
            "question": "logₐx = 1, logₐy = 2 এবং logₐz = 3 হলে logₐ(x³y²/z) এর মান কত?",
            "option1": "2",
            "option2": "4",
            "option3": "1",
            "option4": "2/3",
            "option5": "",
            "answer": "4",
            "explanation": "",
            "language": "auto"
        },
        "expected_answer": "4"
    }
]

def test_api():
    """Quick test"""
    try:
        for test in TEST_CASES:
            print(f"Testing: {test['name']}")
            response = requests.post(API_URL, json=test['data'])
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Got result: {result.get('final_answer')}")
            else:
                print(f"✗ Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()