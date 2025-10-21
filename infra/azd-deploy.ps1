#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Quick deployment script using azd up

.DESCRIPTION
    Simple wrapper for azd up deployment with sensible defaults.
    Handles authentication and subscription setup automatically.

.EXAMPLE
    .\azd-deploy.ps1
    Deploy with defaults

.EXAMPLE
    .\azd-deploy.ps1 -WhatIf
    Preview deployment

.EXAMPLE
    .\azd-deploy.ps1 -EnvironmentName "prod"
    Deploy to production environment
#>

param(
    [Parameter(Mandatory=$false)]
    [switch]$WhatIf,
    
    [Parameter(Mandatory=$false)]
    [string]$EnvironmentName = "poc-dev",
    
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId = ""
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Navigate to the project root if we're in the infra directory
if ((Get-Location).Path.EndsWith('\infra')) {
    Set-Location ..
}

Write-Host "🚀 Hybrid LLM Router - Quick azd Deployment" -ForegroundColor Magenta
Write-Host "=============================================" -ForegroundColor Magenta

# Function to write colored output
function Write-Status {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message); Write-Status "✅ $Message" "Green" }
function Write-Warning { param([string]$Message); Write-Status "⚠️  $Message" "Yellow" }
function Write-Error { param([string]$Message); Write-Status "❌ $Message" "Red" }
function Write-Info { param([string]$Message); Write-Status "ℹ️  $Message" "Cyan" }

Write-Info "Environment: $EnvironmentName"
Write-Info "Mode: $(if ($WhatIf) { 'Preview' } else { 'Deploy' })"

# Check if azd is installed
Write-Status "🔍 Checking prerequisites..." "Yellow"
try {
    $azdVersion = azd version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Azure Developer CLI is installed"
    } else {
        throw "azd command failed"
    }
} catch {
    Write-Error "Azure Developer CLI (azd) is not installed"
    Write-Info "Install with: winget install microsoft.azd"
    exit 1
}

# Check authentication status
Write-Status "🔐 Checking authentication..." "Yellow"
try {
    $authCheck = azd auth login --check-status 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Not logged in to Azure Developer CLI"
        Write-Info "Logging in to Azure..."
        azd auth login
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to login to Azure"
            exit 1
        }
    }
    Write-Success "Authenticated with Azure"
} catch {
    Write-Error "Failed to check authentication status"
    exit 1
}

# Handle subscription selection
if ([string]::IsNullOrEmpty($SubscriptionId)) {
    Write-Status "🔍 Checking subscription..." "Yellow"
    try {
        # Try to get current subscription from az cli
        $currentSub = az account show --query id -o tsv 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrEmpty($currentSub)) {
            $SubscriptionId = $currentSub
            Write-Info "Using current Azure CLI subscription: $SubscriptionId"
        } else {
            Write-Warning "No default subscription found"
            Write-Info "Please set a default subscription:"
            Write-Host "  az account set --subscription 'your-subscription-id'" -ForegroundColor Cyan
            Write-Host "  OR" -ForegroundColor White
            Write-Host "  .\azd-deploy.ps1 -SubscriptionId 'your-subscription-id'" -ForegroundColor Cyan
            exit 1
        }
    } catch {
        Write-Warning "Could not determine subscription"
    }
}

# Set environment variables
Write-Status "🔧 Setting environment variables..." "Yellow"
$env:AZURE_ENV_NAME = $EnvironmentName
$env:AZURE_LOCATION = "eastus2"
$env:APIM_ADMIN_EMAIL = "brittanypugh@microsoft.com"
if (-not [string]::IsNullOrEmpty($SubscriptionId)) {
    $env:AZURE_SUBSCRIPTION_ID = $SubscriptionId
}

Write-Success "Environment variables configured"

# Initialize or select environment
Write-Status "🏗️  Managing azd environment..." "Yellow"
try {
    # Check if environment exists
    $envList = azd env list --output json 2>$null
    if ($LASTEXITCODE -eq 0) {
        $environments = $envList | ConvertFrom-Json
        $envExists = $environments | Where-Object { $_.Name -eq $EnvironmentName }
        
        if (-not $envExists) {
            Write-Info "Creating new environment: $EnvironmentName"
            azd env new $EnvironmentName --location $env:AZURE_LOCATION --subscription $SubscriptionId
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to create environment"
                exit 1
            }
        } else {
            Write-Info "Selecting existing environment: $EnvironmentName"
            azd env select $EnvironmentName
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to select environment"
                exit 1
            }
        }
    }
    Write-Success "Environment ready: $EnvironmentName"
} catch {
    Write-Warning "Environment management failed, continuing with deployment..."
}

# Run deployment
if ($WhatIf) {
    Write-Status "🔍 Running preview mode..." "Yellow"
    try {
        azd provision --preview
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Preview completed successfully"
        } else {
            Write-Error "Preview failed"
            exit 1
        }
    } catch {
        Write-Error "Preview failed with exception"
        exit 1
    }
} else {
    Write-Status "🚀 Deploying infrastructure..." "Yellow"
    try {
        # Use azd up but allow prompts for subscription selection if needed
        azd up
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Deployment completed successfully!"
            Write-Status "📋 Environment values:" "Cyan"
            azd env get-values
        } else {
            Write-Error "Deployment failed"
            Write-Info "Check the output above for details"
            exit 1
        }
    } catch {
        Write-Error "Deployment failed with exception: $($_.Exception.Message)"
        exit 1
    }
}

Write-Success "Script completed successfully!"