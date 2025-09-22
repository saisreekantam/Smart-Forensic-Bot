#!/usr/bin/env python3
"""Test script to check available dependencies for RAG system"""

def test_imports():
    """Test all required imports and show what's available"""
    results = {}
    
    # Test basic packages
    packages = [
        ('numpy', 'np'),
        ('json', None),
        ('pickle', None),
        ('logging', None),
        ('pathlib', 'Path'),
        ('datetime', 'datetime'),
        ('uuid', None),
        ('typing', None),
        ('dataclasses', None),
    ]
    
    for package, alias in packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            results[package] = "✅ Available"
        except ImportError as e:
            results[package] = f"❌ Missing: {e}"
    
    # Test optional packages
    optional_packages = [
        'openai',
        'sentence_transformers', 
        'chromadb',
        'faiss',
        'transformers',
        'torch',
        'sklearn',
        'spacy',
        'nltk',
        'rouge_score',
        'google.generativeai'
    ]
    
    for package in optional_packages:
        try:
            exec(f"import {package}")
            results[package] = "✅ Available"
        except ImportError as e:
            results[package] = f"❌ Missing: {e}"
    
    # Print results
    print("=== Dependency Check Results ===")
    for package, status in results.items():
        print(f"{package:20} {status}")
    
    return results

if __name__ == "__main__":
    test_imports()