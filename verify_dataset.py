#!/usr/bin/env python3
"""
Chess Dataset Verification Script

This script verifies the integrity of the chess dataset by comparing the SHA-256 hash
of the downloaded file with the expected hash stored in a reference file.

Usage:
    python verify_dataset.py [--file FILEPATH] [--hash HASHFILE]

Arguments:
    --file FILEPATH    Path to the dataset file (default: "data/lichess_processed_1000000_games_first_15_moves.pkl")
    --hash HASHFILE    Path to the hash file (default: "lichess_processed_1000000_games_first_15_moves.sha256")

Example:
    python verify_dataset.py
    python verify_dataset.py --file path/to/dataset.pkl --hash path/to/hashfile.sha256
"""

import os
import sys
import hashlib
import argparse
import time

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'
BOLD = '\033[1m'

def calculate_hash(file_path, buffer_size=65536):
    """
    Calculate the SHA-256 hash of a file.
    
    Args:
        file_path (str): Path to the file
        buffer_size (int): Size of the buffer for reading the file
        
    Returns:
        str: Hexadecimal representation of the SHA-256 hash
    """
    # Initialize SHA-256 hash object
    sha256 = hashlib.sha256()
    
    # Open file in binary mode and read in chunks
    try:
        with open(file_path, 'rb') as f:
            # Show a simple progress indicator for large files
            file_size = os.path.getsize(file_path)
            total_chunks = file_size // buffer_size + (1 if file_size % buffer_size else 0)
            chunks_read = 0
            
            # Read and update hash in chunks to handle large files efficiently
            print(f"{BLUE}Calculating hash for {os.path.basename(file_path)}...{ENDC}")
            print(f"File size: {file_size / (1024*1024):.2f} MB")
            
            start_time = time.time()
            
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                sha256.update(data)
                chunks_read += 1
                
                # Print progress every 100 chunks or so
                if chunks_read % 100 == 0:
                    percent = min(100, int(chunks_read * 100 / total_chunks))
                    sys.stdout.write(f"\rProgress: {percent}% ({chunks_read}/{total_chunks} chunks)")
                    sys.stdout.flush()
                    
        elapsed_time = time.time() - start_time
        print(f"\rHash calculation completed in {elapsed_time:.2f} seconds.{' ' * 30}")
        
        # Return the hexadecimal representation of the hash
        return sha256.hexdigest()
    except FileNotFoundError:
        print(f"{RED}Error: File '{file_path}' not found.{ENDC}")
        return None
    except Exception as e:
        print(f"{RED}Error calculating hash: {str(e)}{ENDC}")
        return None

def read_expected_hash(hash_file_path):
    """
    Read the expected hash from a file.
    
    Args:
        hash_file_path (str): Path to the hash file
        
    Returns:
        str: The expected hash as read from the file
    """
    try:
        with open(hash_file_path, 'r') as f:
            # Read the first line and remove any whitespace
            return f.readline().strip()
    except FileNotFoundError:
        print(f"{RED}Error: Hash file '{hash_file_path}' not found.{ENDC}")
        return None
    except Exception as e:
        print(f"{RED}Error reading hash file: {str(e)}{ENDC}")
        return None

def verify_dataset(file_path, hash_file_path):
    """
    Verify the dataset by comparing its hash with the expected hash.
    
    Args:
        file_path (str): Path to the dataset file
        hash_file_path (str): Path to the hash file
        
    Returns:
        bool: True if the verification passed, False otherwise
    """
    # Read the expected hash
    expected_hash = read_expected_hash(hash_file_path)
    if not expected_hash:
        return False
    
    # Calculate the actual hash
    actual_hash = calculate_hash(file_path)
    if not actual_hash:
        return False
    
    # Print the hashes for comparison
    print(f"{BLUE}Expected hash: {ENDC}{expected_hash}")
    print(f"{BLUE}Calculated hash: {ENDC}{actual_hash}")
    
    # Compare the hashes
    if expected_hash.lower() == actual_hash.lower():
        print(f"\n{GREEN}{BOLD}✓ Verification successful: {ENDC}{GREEN}The dataset is valid!{ENDC}")
        return True
    else:
        print(f"\n{RED}{BOLD}✗ Verification failed: {ENDC}{RED}The dataset may be corrupted or incomplete.{ENDC}")
        print(f"{YELLOW}Consider re-downloading the dataset or checking if you have the correct hash file.{ENDC}")
        return False

def main():
    """
    Main function to parse arguments and run verification.
    """
    parser = argparse.ArgumentParser(description='Verify the integrity of the chess dataset.')
    parser.add_argument('--file', type=str, 
                        default="data/lichess_processed_1000000_games_first_15_moves.pkl",
                        help='Path to the dataset file')
    parser.add_argument('--hash', type=str, 
                        default="lichess_processed_1000000_games_first_15_moves.sha256",
                        help='Path to the hash file')
    
    args = parser.parse_args()
    
    print(f"{BOLD}{BLUE}Chess Dataset Verification{ENDC}")
    print(f"Dataset file: {args.file}")
    print(f"Hash file: {args.hash}")
    print("-" * 50)
    
    success = verify_dataset(args.file, args.hash)
    
    # Return an exit code based on verification result
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())