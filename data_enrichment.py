#!/usr/bin/env python3
"""
Data Merging and Enrichment
===========================
This module handles:
1. Merging email-password data with metadata using round-robin
2. Generating honeywords using substitution and LLM
"""

import json
import random
import os
from typing import Dict, List, Any
from collections import defaultdict


# =============================================================================
# Merging Email-Password Data and Metadata using Roundrobin
# =============================================================================

def merge_breaches_roundrobin(
    email_passwords: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    output_file: str = "trainData.json"
) -> List[Dict[str, Any]]:
    """
    Merge email-password data with breach metadata using round-robin distribution.
    
    Args:
        email_passwords: List of {email, password, source} dicts
        metadata: List of {email, breaches, breach_count} dicts
        output_file: Output JSON file path
        
    Returns:
        List of merged breach records
    """
    # Build lookup for metadata by email
    metadata_map = {item["email"]: item for item in metadata}
    
    # Group passwords by email
    email_to_passwords = defaultdict(list)
    for entry in email_passwords:
        email = entry["email"]
        password = entry["password"]
        source = entry.get("source", "unknown")
        email_to_passwords[email].append({"password": password, "source": source})
    
    # Merge data
    merged_data = []
    
    for email, passwords in email_to_passwords.items():
        if email not in metadata_map:
            continue
            
        meta = metadata_map[email]
        breaches = meta.get("breaches", [])
        
        if not breaches:
            continue
        
        # Round-robin distribution of passwords to breaches
        breach_names = [b.get("Name", "Unknown") for b in breaches]
        
        for i, pwd_entry in enumerate(passwords):
            breach_idx = i % len(breach_names)
            breach_name = breach_names[breach_idx]
            breach_info = breaches[breach_idx]
            
            merged_entry = {
                "email": email,
                "password": pwd_entry["password"],
                "breach_source": breach_name,
                "breach_date": breach_info.get("BreachDate", ""),
                "breach_description": breach_info.get("Description", ""),
                "pwn_count": breach_info.get("PwnCount", 0),
                "data_classes": breach_info.get("DataClasses", []),
                "original_source": pwd_entry["source"]
            }
            merged_data.append(merged_entry)
    
    # Save merged data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Created {len(merged_data)} breach records")
    return merged_data


# =============================================================================
# Generating Honeywords using Simple Substitution and LLM
# =============================================================================

def generate_honeywords_simple(password: str, count: int = 3) -> List[str]:
    """
    Generate honeywords using simple character substitution.
    
    Args:
        password: Original password
        count: Number of honeywords to generate
        
    Returns:
        List of honeyword candidates
    """
    substitutions = {
        'a': ['@', '4', 'A'],
        'e': ['3', 'E'],
        'i': ['1', '!', 'I'],
        'o': ['0', 'O'],
        's': ['$', '5', 'S'],
        'l': ['1', 'L'],
        't': ['7', 'T'],
        'g': ['9', 'G'],
        'b': ['8', 'B']
    }
    
    honeywords = set()
    
    # Generate variations
    for _ in range(count * 3):  # Generate extra to account for duplicates
        new_pwd = list(password.lower())
        
        # Apply random substitutions
        for i, char in enumerate(new_pwd):
            if char in substitutions and random.random() > 0.5:
                new_pwd[i] = random.choice(substitutions[char])
        
        # Random case changes
        new_pwd = [c.upper() if random.random() > 0.7 else c for c in new_pwd]
        
        honeyword = ''.join(new_pwd)
        if honeyword != password:
            honeywords.add(honeyword)
        
        if len(honeywords) >= count:
            break
    
    return list(honeywords)[:count]


def generate_honeywords_with_llm(password: str, count: int = 10, use_llm: bool = False) -> List[str]:
    """
    Generate honeywords for a password using substitution and optionally LLM.
    
    Args:
        password: Original password
        count: Number of honeywords to generate (default: 10)
        use_llm: Whether to use LLM for generation (requires API setup)
        
    Returns:
        List of honeywords
    """
    honeywords = []
    
    # Simple substitution honeywords
    honeywords.extend(generate_honeywords_simple(password, count // 2))
    
    # Add some random character modifications
    for _ in range(count // 2):
        chars = list(password)
        
        # Random modifications
        if len(chars) > 2:
            # Insert random character
            if random.random() > 0.5:
                pos = random.randint(0, len(chars))
                chars.insert(pos, random.choice('!@#$%^&*123456789'))
            
            # Replace random character
            if random.random() > 0.5 and len(chars) > 0:
                pos = random.randint(0, len(chars) - 1)
                chars[pos] = random.choice('!@#$%^&*abcdefghijklmnopqrstuvwxyz123456789')
        
        honeyword = ''.join(chars)
        if honeyword != password:
            honeywords.append(honeyword)
    
    # Remove duplicates and limit to count
    honeywords = list(set(honeywords))[:count]
    
    # If using LLM, this is where you'd integrate the API call
    # For now, we'll just pad with more simple variations if needed
    while len(honeywords) < count:
        honeywords.extend(generate_honeywords_simple(password, 1))
        honeywords = list(set(honeywords))[:count]
    
    return honeywords


def enrich_with_honeywords(
    input_file: str = "trainData.json",
    output_file: str = "trainData_enriched_with_honeywords.json",
    honeywords_per_password: int = 10
):
    """
    Generate honeywords for each breach password in the training data.
    
    Adds a "Honeywords" list (10 per password by default) to each entry.
    
    Args:
        input_file: Input JSON file with breach data
        output_file: Output JSON file with honeywords
        honeywords_per_password: Number of honeywords to generate per password
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    enriched_data = []
    
    for i, entry in enumerate(data):
        password = entry["password"]
        
        # Generate honeywords
        honeywords = generate_honeywords_with_llm(password, count=honeywords_per_password)
        
        # Add to entry
        enriched_entry = entry.copy()
        enriched_entry["honeywords"] = honeywords
        enriched_data.append(enriched_entry)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} passwords")
    
    # Save enriched data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2)
    
    print(f"\nEnriched {len(enriched_data)} passwords with honeywords")
    return enriched_data


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Data Merging and Enrichment Pipeline")
    print("=" * 80)
    
    # Example usage
    
    # Step 1: Merge breaches (requires data from data_collection.py)
    # email_passwords = json.load(open("email_passwords_filtered.json"))
    # metadata = json.load(open("metadata.json"))
    # merge_breaches_roundrobin(email_passwords, metadata)
    
    # Step 2: Enrich with honeywords
    # enrich_with_honeywords()
    
    print("\nNote: Uncomment and configure the steps above to run the pipeline")
