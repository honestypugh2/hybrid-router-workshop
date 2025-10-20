#!/usr/bin/env python3
"""
Cleanup script for hybrid-llm-router-workshop resources
Uses the utils.py cleanup_resources function to clean up Azure resources
"""

import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.utils import cleanup_resources, print_info, print_warning

def main():
    """Main cleanup function"""
    deployment_name = "hybrid-llm-225420"  # Most recent deployment
    resource_group_name = "rg-hybridllm-workshop-dev"
    
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