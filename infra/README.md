# Azure Infrastructure for Hybrid LLM Router

This directory contains the Azure infrastructure as code (Bicep) templates for deploying the hybrid LLM router workshop resources.

## 🏗️ Architecture

The infrastructure includes the following Azure resources:

- **Azure OpenAI Service**: Provides cloud-based GPT models (GPT-4o, GPT-4o-mini)
- **Azure AI Foundry Project**: AI Services project workspace for hybrid routing models
- **Azure API Management (APIM)**: Gateway for routing and managing API calls
- **Azure Application Insights**: Telemetry and monitoring for the application
- **Azure Log Analytics**: Centralized logging and analytics

### Key Features

- **Simplified Architecture**: Uses CognitiveServices API for AI Foundry Project instead of complex Hub/Project architecture
- **Clean Dependencies**: No Key Vault or Storage Account dependencies that can cause deployment errors
- **Modular Design**: Each service is a separate module for easier maintenance
- **Conditional Deployment**: Feature flags to enable/disable services
- **Proper Error Handling**: Uses safe navigation operators for nullable module references
- **Role-Based Access Control**: Automatic role assignments using azure-roles.json for secure service-to-service communication

## 📁 Structure

```text
infra/
├── main.bicep                   # Main orchestration template
├── main.bicepparam             # Parameters file for deployment
├── deploy.ps1                 # PowerShell deployment script
├── README.md                  # This file
└── modules/
    ├── ai-foundry-project.bicep # Azure AI Foundry Project
    ├── apim.bicep              # API Management service
    ├── app-insights.bicep      # Application Insights
    ├── azure-roles.json        # Azure built-in role definitions
    ├── log-analytics.bicep     # Log Analytics workspace
    ├── openai.bicep           # Azure OpenAI service with multiple models
    └── role-assignment.bicep  # Role assignment module
```

## 🚀 Deployment

### Prerequisites

1. Azure CLI installed and configured
2. Bicep CLI installed
3. Appropriate Azure subscription access
4. Resource group created

### Option 1: Using the PowerShell Script

1. **Login to Azure**:

   ```bash
   az login
   ```

2. **Set your subscription**:

   ```bash
   az account set --subscription "your-subscription-id"
   ```

3. **Deploy using the script**:

   ```powershell
   # Validate template (what-if analysis)
   .\infra\deploy.ps1 -ResourceGroupName "your-resource-group" -WhatIf

   # Deploy to existing resource group
   .\infra\deploy.ps1 -ResourceGroupName "your-existing-resource-group"

   # Deploy to new resource group
   .\infra\deploy.ps1 -ResourceGroupName "new-resource-group" -Location "eastus2"
   ```

### Option 2: Manual Deployment

1. **Create a resource group**:

   ```bash
   az group create --name "rg-hybridllm-workshop-poc" --location "eastus2"
   ```

2. **Update parameters**: Edit `main.bicepparam` with your specific values:
   - Replace `apimAdminEmail` with your email address
   - Adjust location, environment name, and other parameters as needed

3. **Deploy the infrastructure**:

   ```bash
   az deployment group create \
     --resource-group "rg-hybridllm-workshop-poc" \
     --template-file main.bicep \
     --parameters main.bicepparam
   ```

### Manual Parameter Override

You can also deploy with inline parameters:

```bash
az deployment group create \
  --resource-group "rg-hybridllm-dev" \
  --template-file main.bicep \
  --parameters workloadName="hybridllm" \
               environmentName="dev" \
               location="eastus" \
               apimAdminEmail="your-email@domain.com"
```

## 🔧 Configuration

### Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `workloadName` | Name of the workload (max 10 chars) | `hybridllm` | Yes |
| `environmentName` | Environment name (dev/test/prod) | `dev` | Yes |
| `location` | Azure region for deployment | Resource group location | No |
| `apimAdminEmail` | Admin email for APIM | - | Yes |
| `apimOrganizationName` | Organization name for APIM | `Hybrid LLM Router` | No |
| `openAiSkuName` | OpenAI service pricing tier | `S0` | No |
| `apimSkuName` | APIM pricing tier | `Developer` | No |
| `enableTelemetry` | Enable Application Insights | `true` | No |
| `enableAiFoundry` | Enable Azure AI Foundry Project | `true` | No |

### Outputs

After deployment, the following outputs are available:

- `openAiEndpoint`: Azure OpenAI service endpoint
- `openAiServiceName`: Name of the OpenAI service
- `apimGatewayUrl`: API Management gateway URL
- `applicationInsightsConnectionString`: App Insights connection string
- `aiFoundryProjectEndpoint`: AI Foundry Project endpoint URL
- `aiFoundryProjectDiscoveryUrl`: AI Foundry Project discovery URL

## 🔐 Security

### Access Control

- OpenAI service is configured with managed identity support
- APIM is configured with secure protocols only
- AI Foundry Project uses system-assigned managed identity
- **Automatic Role Assignments**: Services are automatically assigned appropriate roles using azure-roles.json:
  - AI Foundry Project gets `CognitiveServicesOpenAIUser` role on OpenAI service
  - API Management gets `CognitiveServicesOpenAIUser` role on OpenAI service

### API Keys and Secrets

- OpenAI API keys are managed through Azure RBAC
- APIM subscription keys provide controlled access
- All services use secure HTTPS endpoints

## 📊 Monitoring

### Application Insights

When `enableTelemetry` is true, the deployment includes:

- Application Insights instance
- Log Analytics workspace
- Automatic telemetry collection setup

### Metrics and Logs

The infrastructure captures:

- API request/response metrics
- Model usage statistics
- Performance telemetry
- Error tracking and diagnostics

## 🛠️ Customization

### Adding Resources

To add new Azure resources:

1. Create a new module in the `modules/` directory
2. Reference it in `main.bicep`
3. Add required parameters to `main.bicepparam`

### Environment-Specific Configurations

Create additional parameter files for different environments:

```bash
# Development
main.dev.bicepparam

# Testing
main.test.bicepparam

# Production
main.prod.bicepparam
```

## 🔄 Updates and Maintenance

### Updating Infrastructure

1. Modify the Bicep templates as needed
2. Test in a development environment first
3. Deploy using the same commands with updated parameters

### Resource Cleanup

To remove all resources:

```bash
az group delete --name "rg-hybridllm-dev" --yes --no-wait
```

## 🆘 Troubleshooting

### Common Issues

1. **Deployment Failures**:

   - Check Azure CLI version and authentication
   - Verify resource quotas and region availability
   - Review error messages in Azure portal

2. **Permission Issues**:

   - Ensure sufficient permissions on the subscription
   - Check managed identity assignments if needed

3. **Parameter Validation**:

   - Verify all required parameters are provided
   - Check parameter value constraints (length, allowed values)

### Support

For infrastructure issues:

1. Check Azure Resource Manager deployment logs
2. Review the Azure portal for resource-specific errors
3. Consult Azure documentation for service-specific requirements

## 📝 Best Practices

1. **Environment Separation**: Use different resource groups for dev/test/prod
2. **Naming Conventions**: Follow consistent naming patterns
3. **Security**: Rotate keys regularly and use managed identities
4. **Monitoring**: Enable comprehensive logging and alerting
5. **Cost Management**: Monitor costs and set up budgets

## 🔗 Related Resources

- [Azure Bicep Documentation](https://docs.microsoft.com/azure/azure-resource-manager/bicep/)
- [Azure OpenAI Service](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Azure API Management](https://docs.microsoft.com/azure/api-management/)
- [Azure AI Foundry](https://docs.microsoft.com/azure/machine-learning/)