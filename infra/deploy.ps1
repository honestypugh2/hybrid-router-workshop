#!/usr/bin/env pwsh

# Deploy new infrastructure based on existing resources
# This script deploys the new clean Bicep templates

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus2",
    
    [Parameter(Mandatory=$false)]
    [switch]$WhatIf
)

# Set variables
$TemplateFile = "main.bicep"
$ParametersFile = "main.bicepparam"
$DeploymentName = "hybrid-llm-router-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

Write-Host "🚀 Starting deployment of Hybrid LLM Router Infrastructure" -ForegroundColor Green
Write-Host "   Resource Group: $ResourceGroupName" -ForegroundColor Cyan
Write-Host "   Location: $Location" -ForegroundColor Cyan
Write-Host "   Deployment Name: $DeploymentName" -ForegroundColor Cyan

# Check if resource group exists
$rgExists = az group exists --name $ResourceGroupName --output tsv
if ($rgExists -eq "false") {
    Write-Host "   Creating resource group: $ResourceGroupName" -ForegroundColor Yellow
    az group create --name $ResourceGroupName --location $Location
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Failed to create resource group"
        exit 1
    }
}

# Validate the template
Write-Host "🔍 Validating Bicep template..." -ForegroundColor Yellow
az deployment group validate `
    --resource-group $ResourceGroupName `
    --template-file $TemplateFile `
    --parameters $ParametersFile `
    --output table

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Template validation failed"
    exit 1
}

Write-Host "✅ Template validation passed" -ForegroundColor Green

if ($WhatIf) {
    # Run what-if analysis
    Write-Host "🔍 Running what-if analysis..." -ForegroundColor Yellow
    az deployment group what-if `
        --resource-group $ResourceGroupName `
        --template-file $TemplateFile `
        --parameters $ParametersFile `
        --output table
} else {
    # Deploy the template
    Write-Host "🚀 Deploying infrastructure..." -ForegroundColor Yellow
    az deployment group create `
        --resource-group $ResourceGroupName `
        --template-file $TemplateFile `
        --parameters $ParametersFile `
        --name $DeploymentName `
        --output table

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
        
        # Get deployment outputs
        Write-Host "📋 Deployment Outputs:" -ForegroundColor Cyan
        az deployment group show `
            --resource-group $ResourceGroupName `
            --name $DeploymentName `
            --query "properties.outputs" `
            --output table
    } else {
        Write-Error "❌ Deployment failed"
        exit 1
    }
}

Write-Host "🎉 Script completed!" -ForegroundColor Green