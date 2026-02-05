#!/usr/bin/env python3
"""
Data Collection and Initial Processing
======================================
This module handles:
1. Reading and consolidating email-password data
2. Filtering Gmail accounts with multiple breaches
3. Fetching metadata from haveibeenpwned.com
"""

import json
import time
import requests
import urllib.parse
from typing import Dict, List, Optional


# =============================================================================
# Step 1: Reading the Complete Email-Password Data
# =============================================================================

def read_and_consolidate_email_passwords(input_files: List[str], output_file: str = "email_passwords.json"):
    """
    Reading downloaded files store, selected only gmail emails.
    Storing selected emails into single file called email_passwords.json.
    """
    import json

    all_entries = []

    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    email, password = line.split(":", 1)
                    # Filter only Gmail accounts
                    if "@gmail.com" in email.lower():
                        all_entries.append({
                            "email": email,
                            "password": password,
                            "source": file_path
                        })
                except ValueError:
                    # Skip lines that don't have exactly one colon
                    continue

    # Write to output JSON file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(all_entries, out_f, indent=2)

    print(f"Total Gmail entries saved: {len(all_entries)}")
    return all_entries


# =============================================================================
# Step 2: Filtering out Gmail Accounts with Multiple Breaches
# =============================================================================

def filter_multiple_breach_accounts(input_file: str = "email_passwords.json", 
                                   output_file: str = "email_passwords_filtered.json"):
    """
    Filter out Gmail accounts that appear in multiple breaches.
    """
    import json

    # Path to your input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dictionary to count occurrences of each email
    email_count = {}
    for entry in data:
        email = entry["email"]
        email_count[email] = email_count.get(email, 0) + 1

    # Filter entries: keep only emails that appear more than once
    filtered_data = [entry for entry in data if email_count[entry["email"]] > 1]

    # Save the filtered data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Original entries: {len(data)}")
    print(f"Filtered entries (multiple breaches): {len(filtered_data)}")
    
    return filtered_data


# =============================================================================
# Step 3: Getting Metadata from haveibeenpwned.com
# =============================================================================

def get_hibp_metadata(email: str, api_key: str) -> Optional[List[Dict]]:
    """
    Fetch breach metadata for a given email from haveibeenpwned.com API.
    
    Args:
        email: Email address to check
        api_key: HIBP API key
        
    Returns:
        List of breach dictionaries or None if error
    """
    encoded_email = urllib.parse.quote(email)
    url = f"https://haveibeenpwned.com/api/v3/breachedaccount/{encoded_email}"
    
    headers = {
        "hibp-api-key": api_key,
        "user-agent": "BreachDataCollector"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # No breaches found
            return []
        else:
            print(f"Error {response.status_code} for {email}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception for {email}: {e}")
        return None


def enrich_with_metadata(input_file: str = "email_passwords_filtered.json",
                        output_file: str = "trainData.json",
                        api_key: str = None,
                        rate_limit_delay: float = 1.5):
    """
    Enrich email-password data with breach metadata from HIBP.
    
    Args:
        input_file: Input JSON file with email-password data
        output_file: Output JSON file with enriched data
        api_key: HIBP API key (required)
        rate_limit_delay: Delay between API calls in seconds
    """
    if not api_key:
        raise ValueError("HIBP API key is required")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    enriched_data = []
    processed_emails = set()
    
    for i, entry in enumerate(data):
        email = entry["email"]
        password = entry["password"]
        
        # Skip if we already processed this email
        if email in processed_emails:
            continue
        
        print(f"Processing {i+1}/{len(data)}: {email}")
        
        # Get metadata from HIBP
        breaches = get_hibp_metadata(email, api_key)
        
        if breaches is not None:
            enriched_entry = {
                "email": email,
                "password": password,
                "breaches": breaches,
                "breach_count": len(breaches)
            }
            enriched_data.append(enriched_entry)
            processed_emails.add(email)
        
        # Rate limiting
        time.sleep(rate_limit_delay)
    
    # Save enriched data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2)
    
    print(f"\nEnriched {len(enriched_data)} unique emails")
    return enriched_data


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Data Collection Pipeline")
    print("=" * 80)
    
    # Step 1: Read and consolidate (uncomment and provide file paths)
    # input_files = ["breach1.txt", "breach2.txt", "breach3.txt"]
    # read_and_consolidate_email_passwords(input_files)
    
    # Step 2: Filter multiple breaches
    # filter_multiple_breach_accounts()
    
    # Step 3: Enrich with metadata (requires API key)
    # HIBP_API_KEY = "your-api-key-here"
    # enrich_with_metadata(api_key=HIBP_API_KEY)
    
    print("\nNote: Uncomment and configure the steps above to run the pipeline")
