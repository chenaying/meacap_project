#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to fix encoding issues in all Python files
"""
import os
import sys
import glob

def fix_file_encoding(filepath):
    """Fix encoding of a single file"""
    print(f"Processing: {filepath}")
    
    # Try to read with different encodings
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1']
    content = None
    used_encoding = None
    
    for encoding in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            if encoding != 'utf-8':
                print(f"  Found encoding: {encoding}, converting to UTF-8...")
            else:
                print(f"  Already UTF-8")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"  Error reading with {encoding}: {e}")
            continue
    
    if content is None:
        print(f"  ERROR: Could not read file with any encoding!")
        return False
    
    # Write back as UTF-8
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        if used_encoding != 'utf-8':
            print(f"  SUCCESS: Converted from {used_encoding} to UTF-8")
        return True
    except Exception as e:
        print(f"  ERROR: Could not write file: {e}")
        return False

def find_python_files(directory):
    """Find all Python files in directory"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Fix specific files
        files_to_fix = sys.argv[1:]
    else:
        # Fix all Python files in common directories
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
        
        files_to_fix = all_files
    
    print("=" * 60)
    print(f"Fixing encoding for {len(files_to_fix)} file(s)...")
    print("=" * 60)
    print()
    
    fixed_count = 0
    failed_count = 0
    
    for filepath in files_to_fix:
        if fix_file_encoding(filepath):
            fixed_count += 1
        else:
            failed_count += 1
        print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Failed: {failed_count}")
    print("=" * 60)
    
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

