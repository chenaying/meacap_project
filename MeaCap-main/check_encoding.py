#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to check encoding issues in Python files
"""
import os
import sys

def check_file_encoding(filepath):
    """Check if a file can be read as UTF-8"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read()
        return True, None
    except UnicodeDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def find_python_files(directory):
    """Find all Python files in directory"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

if __name__ == "__main__":
    # Check current directory and common subdirectories
    directories_to_check = [
        '.',
        'utils',
        'models',
        'dataset',
        'src',
        'language_models'
    ]
    
    all_files = []
    for directory in directories_to_check:
        if os.path.exists(directory):
            files = find_python_files(directory)
            all_files.extend(files)
    
    print("Checking Python files for encoding issues...")
    print("=" * 60)
    
    problematic_files = []
    for filepath in all_files:
        is_ok, error = check_file_encoding(filepath)
        if not is_ok:
            problematic_files.append((filepath, error))
            print(f"ERROR: {filepath}")
            print(f"  {error}")
            print()
    
    if problematic_files:
        print("=" * 60)
        print(f"Found {len(problematic_files)} file(s) with encoding issues:")
        for filepath, error in problematic_files:
            print(f"  - {filepath}")
        sys.exit(1)
    else:
        print("=" * 60)
        print("All checked files are UTF-8 compatible!")
        sys.exit(0)

