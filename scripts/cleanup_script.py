#!/usr/bin/env python3
"""
Cleanup script for hybrid-llm-router-workshop resources
Uses the utils.py cleanup_resources function to clean up Azure resources
"""

import sys
import os

# Add parent directory to path for module imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modules_dir = os.path.join(parent_dir, 'modules')
if modules_dir not in sys.path:
    sys.path.append(modules_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from modules.utils import cleanup_resources, print_info, print_warning

def main():
    """Main cleanup function"""
    deployment_name = "hybrid-llm-router-20251021-132358"  # Most recent deployment
    resource_group_name = "rg-hybridllm-workshop-poc-test"
    
    print_info(f"Starting cleanup for deployment: {deployment_name}")
    print_info(f"Resource group: {resource_group_name}")
    print_warning("This will delete all resources in the resource group!")
    
    # Confirm before proceeding
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print_info("Cleanup cancelled.")
        return
    
    # Execute cleanup
    cleanup_resources(deployment_name, resource_group_name)
    
    print_info("Cleanup script completed.")

if __name__ == "__main__":
    main()