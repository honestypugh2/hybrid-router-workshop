#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Setup script for Azure Developer CLI deployment

.DESCRIPTION
    Helps configure Azure CLI and azd for successful deployment

.EXAMPLE
    .\setup-azd.ps1
    Interactive setup

.EXAMPLE
    .\setup-azd.ps1 -SubscriptionId "your-subscription-id"
    Setup with specific subscription
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId = ""
)

Write-Host "🔧 Azure Developer CLI Setup" -ForegroundColor Magenta
Write-Host "=============================" -ForegroundColor Magenta

# Function to write colored output
function Write-Status {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message); Write-Status "✅ $Message" "Green" }
function Write-Warning { param([string]$Message); Write-Status "⚠️  $Message" "Yellow" }
function Write-Error { param([string]$Message); Write-Status "❌ $Message" "Red" }
function Write-Info { param([string]$Message); Write-Status "ℹ️  $Message" "Cyan" }

# Check Azure CLI
Write-Status "🔍 Checking Azure CLI..." "Yellow"
try {
    $azVersion = az version --query '"azure-cli"' -o tsv 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Azure CLI is installed (version: $azVersion)"
    } else {
        throw "az command failed"
    }
} catch {
    Write-Error "Azure CLI is not installed"
    Write-Info "Install from: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
}

# Check azd
Write-Status "🔍 Checking Azure Developer CLI..." "Yellow"
try {
    $azdVersion = azd version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Azure Developer CLI is installed"
    } else {
        throw "azd command failed"
    }
} catch {
    Write-Error "Azure Developer CLI is not installed"
    Write-Info "Install with: winget install microsoft.azd"
    Write-Info "Or from: https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd"
    exit 1
}

# Login to Azure CLI
Write-Status "🔐 Checking Azure CLI authentication..." "Yellow"
try {
    $account = az account show 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Not logged in to Azure CLI"
        Write-Info "Logging in to Azure CLI..."
        az login
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to login to Azure CLI"
            exit 1
        }
    }
    Write-Success "Authenticated with Azure CLI"
} catch {
    Write-Error "Failed to check Azure CLI authentication"
    exit 1
}

# Handle subscription
if ([string]::IsNullOrEmpty($SubscriptionId)) {
    Write-Status "📋 Available subscriptions:" "Cyan"
    az account list --query "[].{Name:name, SubscriptionId:id, IsDefault:isDefault}" --output table
    
    $currentSub = az account show --query id -o tsv 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrEmpty($currentSub)) {
        Write-Info "Current default subscription: $currentSub"
        $useDefault = Read-Host "Use current subscription? (Y/n)"
        if ($useDefault -match '^[Nn]$') {
            $SubscriptionId = Read-Host "Enter subscription ID"
        } else {
            $SubscriptionId = $currentSub
        }
    } else {
        $SubscriptionId = Read-Host "Enter subscription ID to use"
    }
}

# Set default subscription
if (-not [string]::IsNullOrEmpty($SubscriptionId)) {
    Write-Status "🔧 Setting default subscription..." "Yellow"
    az account set --subscription $SubscriptionId
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Default subscription set to: $SubscriptionId"
    } else {
        Write-Error "Failed to set subscription"
        exit 1
    }
}

# Login to azd
Write-Status "🔐 Checking azd authentication..." "Yellow"
try {
    $authCheck = azd auth login --check-status 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Not logged in to Azure Developer CLI"
        Write-Info "Logging in to azd..."
        azd auth login
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to login to azd"
            exit 1
        }
    }
    Write-Success "Authenticated with Azure Developer CLI"
} catch {
    Write-Error "Failed to check azd authentication"
    exit 1
}

Write-Success "Setup completed successfully!"
Write-Info "You can now run: .\infra\azd-deploy.ps1"
Write-Host ""
Write-Status "Next steps:" "Cyan"
Write-Host "1. Run deployment: .\infra\azd-deploy.ps1" -ForegroundColor White
Write-Host "2. Or preview first: .\infra\azd-deploy.ps1 -WhatIf" -ForegroundColor White
Write-Host "3. Manage environments: azd env list" -ForegroundColor White