#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Deploy Hybrid LLM Router infrastructure using Azure Developer CLI (azd)

.DESCRIPTION
    This script deploys the Hybrid LLM Router infrastructure using Azure Developer CLI (azd).
    It provides a modern, streamlined deployment experience compared to traditional az cli.

.PARAMETER EnvironmentName
    The environment name to deploy to (e.g., dev, test, prod). Defaults to 'dev'

.PARAMETER Location
    The Azure region to deploy to. Defaults to 'eastus2'

.PARAMETER AdminEmail
    Admin email for API Management service. Defaults to current user

.PARAMETER WhatIf
    Run in what-if mode to see what would be deployed without actually deploying

.PARAMETER Force
    Skip confirmation prompts and deploy immediately

.EXAMPLE
    .\deploy-azd.ps1
    Deploy to dev environment with default settings

.EXAMPLE
    .\deploy-azd.ps1 -EnvironmentName "prod" -Location "westus2" -AdminEmail "admin@company.com"
    Deploy to production environment in West US 2

.EXAMPLE
    .\deploy-azd.ps1 -WhatIf
    Preview what would be deployed without actually deploying
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$EnvironmentName = "dev",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus2",
    
    [Parameter(Mandatory=$false)]
    [string]$AdminEmail = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$WhatIf,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-Status {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Status "✅ $Message" "Green"
}

function Write-Warning {
    param([string]$Message)
    Write-Status "⚠️  $Message" "Yellow"
}

function Write-Error {
    param([string]$Message)
    Write-Status "❌ $Message" "Red"
}

function Write-Info {
    param([string]$Message)
    Write-Status "ℹ️  $Message" "Cyan"
}

# Banner
Write-Host @"
🚀 Hybrid LLM Router - Azure Developer CLI Deployment
====================================================
"@ -ForegroundColor Magenta

Write-Info "Environment: $EnvironmentName"
Write-Info "Location: $Location"
Write-Info "Mode: $(if ($WhatIf) { 'Preview (What-If)' } else { 'Deploy' })"

# Check prerequisites
Write-Status "🔍 Checking prerequisites..." "Yellow"

# Check if azd is installed
try {
    $azdVersion = azd version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Azure Developer CLI is installed: $($azdVersion -split "`n" | Select-Object -First 1)"
    } else {
        throw "azd command failed"
    }
} catch {
    Write-Error "Azure Developer CLI (azd) is not installed or not in PATH"
    Write-Info "Please install azd from: https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd"
    exit 1
}

# Check if user is logged in to azd
try {
    $authStatus = azd auth login --check-status 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Not logged in to Azure Developer CLI"
        Write-Info "Logging in to Azure Developer CLI..."
        azd auth login
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to login to Azure Developer CLI"
            exit 1
        }
    }
    Write-Success "Authenticated with Azure Developer CLI"
} catch {
    Write-Error "Failed to check authentication status"
    exit 1
}

# Set default admin email if not provided
if ([string]::IsNullOrEmpty($AdminEmail)) {
    try {
        $currentUser = az ad signed-in-user show --query userPrincipalName -o tsv 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrEmpty($currentUser)) {
            $AdminEmail = $currentUser
            Write-Info "Using current user email: $AdminEmail"
        } else {
            $AdminEmail = "brittanypugh@microsoft.com"
            Write-Warning "Could not detect current user, using default: $AdminEmail"
        }
    } catch {
        $AdminEmail = "brittanypugh@microsoft.com"
        Write-Warning "Could not detect current user, using default: $AdminEmail"
    }
}

# Set environment variables for azd
Write-Status "🔧 Setting environment variables..." "Yellow"
$env:AZURE_ENV_NAME = $EnvironmentName
$env:AZURE_LOCATION = $Location
$env:APIM_ADMIN_EMAIL = $AdminEmail

Write-Success "Environment variables set"

# Initialize azd environment if it doesn't exist
Write-Status "🏗️  Initializing azd environment..." "Yellow"
try {
    # Check if environment already exists
    $envExists = azd env list --output json 2>$null | ConvertFrom-Json | Where-Object { $_.Name -eq $EnvironmentName }
    
    if (-not $envExists) {
        Write-Info "Creating new environment: $EnvironmentName"
        azd env new $EnvironmentName --location $Location
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create azd environment"
            exit 1
        }
    } else {
        Write-Info "Using existing environment: $EnvironmentName"
        azd env select $EnvironmentName
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to select azd environment"
            exit 1
        }
    }
    Write-Success "Environment ready: $EnvironmentName"
} catch {
    Write-Error "Failed to initialize azd environment"
    exit 1
}

# Show confirmation unless Force is specified
if (-not $Force -and -not $WhatIf) {
    Write-Status @"

📋 Deployment Summary:
- Environment: $EnvironmentName
- Location: $Location
- Admin Email: $AdminEmail
- Resource Group: rg-hybridllm-workshop-$EnvironmentName

"@ "Cyan"

    $confirmation = Read-Host "Do you want to proceed with deployment? (y/N)"
    if ($confirmation -notmatch '^[Yy]$') {
        Write-Info "Deployment cancelled by user"
        exit 0
    }
}

# Run deployment
if ($WhatIf) {
    Write-Status "🔍 Running deployment preview..." "Yellow"
    try {
        azd provision --preview
        Write-Success "Preview completed successfully"
    } catch {
        Write-Error "Preview failed"
        exit 1
    }
} else {
    Write-Status "🚀 Starting deployment..." "Yellow"
    try {
        # Use azd up for first-time deployment or azd provision for infrastructure-only
        azd up --no-prompt
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Deployment completed successfully!"
            
            # Show deployment outputs
            Write-Status "📋 Deployment Information:" "Cyan"
            azd env get-values
            
            Write-Status @"

🎉 Deployment Complete!
=======================
Your Hybrid LLM Router infrastructure has been deployed successfully.

Next steps:
1. Review the deployment outputs above
2. Test your API endpoints
3. Configure any additional settings as needed

To manage this deployment:
- View resources: azd env get-values
- Update deployment: azd up
- Clean up resources: azd down

"@ "Green"
        } else {
            Write-Error "Deployment failed"
            Write-Info "Check the deployment logs above for details"
            exit 1
        }
    } catch {
        Write-Error "Deployment failed with exception: $($_.Exception.Message)"
        exit 1
    }
}

Write-Status "🎉 Script completed!" "Green"