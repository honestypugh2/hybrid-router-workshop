# Azure Infrastructure for Hybrid LLM Router

This directory contains the Azure infrastructure as code (Bicep) templates for deploying the hybrid LLM router workshop resources.

## ⚙️ Prerequisites

Before deploying, you must configure environment variables for the deployment. Both parameter files now use environment variables instead of hardcoded values.

### Option 1: Set Environment Variables Directly

Set the required environment variable for APIM:

```powershell
# Required: APIM Admin Email
$env:APIM_ADMIN_EMAIL = "your-email@domain.com"

# Optional: Override defaults
$env:AZURE_LOCATION = "eastus2"
$env:ENVIRONMENT_NAME = "dev"
$env:WORKLOAD_NAME = "hybridllm"
```

### Option 2: Use Environment File (Recommended)

1. **Copy the sample environment file**:

   ```powershell
   copy infra\.env.sample infra\.env
   ```

2. **Edit `infra\.env`** with your values:

   ```bash
   APIM_ADMIN_EMAIL=your-email@domain.com
   AZURE_LOCATION=eastus2
   ENVIRONMENT_NAME=dev
   WORKLOAD_NAME=hybridllm
   ```

3. **Load environment variables** (for Azure CLI deployment):

   ```powershell
   # Load .env file (requires dotenv or manual loading)
   Get-Content infra\.env | ForEach-Object {
     if ($_ -match "^([^#][^=]+)=(.*)$") {
       [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
     }
   }
   ```

### For Azure Developer CLI (azd)

azd automatically handles environment variables. Just set them directly:

```powershell
azd env set APIM_ADMIN_EMAIL "your-email@domain.com"
azd env set AZURE_LOCATION "eastus2"
azd env set ENVIRONMENT_NAME "dev"
```

⚠️ **Important**: `APIM_ADMIN_EMAIL` is required and must be set before deployment.

## 🚀 Quick Start

### Deploy Infrastructure with Azure Developer CLI (Recommended)

The Azure Developer CLI (azd) provides the best deployment experience with automatic environment management, resource grouping, and modern DevOps practices.

```powershell
# 1. Initialize azd (first-time setup)
azd init

# 2. Simple deployment with defaults
azd up

# 3. Or use our custom deployment scripts
.\infra\azd-deploy.ps1

# 4. Preview what will be deployed
azd provision --preview
```

### Deploy Infrastructure with Azure CLI (Alternative)

```powershell
.\infra\deploy.ps1 -ResourceGroupName "your-resource-group"
```

### Cleanup Resources

```powershell
# Using azd (recommended)
azd down

# Using cleanup script
.venv\Scripts\activate.bat && python scripts/cleanup_script.py
```

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
├── modules/
│   ├── ai-foundry-project.bicep # Azure AI Foundry Project
│   ├── apim.bicep              # API Management service
│   ├── app-insights.bicep      # Application Insights
│   ├── azure-roles.json        # Azure built-in role definitions
│   ├── log-analytics.bicep     # Log Analytics workspace
│   ├── openai.bicep           # Azure OpenAI service with multiple models
│   ├── role-assignment.bicep  # Role assignment module
│   └── utils.py               # Azure utility functions for deployment and cleanup
└── scripts/
    └── cleanup_script.py      # Automated resource cleanup script
```

## 🚀 Deployment

### Prerequisites

**For Azure Developer CLI (azd) - Recommended:**
1. [Azure Developer CLI (azd)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) installed
2. Azure CLI installed and configured
3. Appropriate Azure subscription access

**For traditional deployment:**
1. Azure CLI installed and configured
2. Bicep CLI installed
3. Appropriate Azure subscription access
4. Resource group created

### Option 1: Using Azure Developer CLI (azd) - Recommended ⭐

Azure Developer CLI provides the best deployment experience with automatic environment management, resource grouping, and modern DevOps practices.

#### Step-by-Step azd Deployment

1. **Install Azure Developer CLI** (if not already installed):

   ```powershell
   # Install via PowerShell
   powershell -ex AllSigned -c "Invoke-RestMethod 'https://aka.ms/install-azd.ps1' | Invoke-Expression"
   
   # Or install via winget
   winget install microsoft.azd
   ```

2. **Run the setup script** (recommended for first-time users):

   ```powershell
   cd c:\Users\brittanypugh\hybrid-llm-router-workshop
   .\infra\setup-azd.ps1
   ```

   This script will:
   - Verify Azure CLI and azd installation
   - Handle authentication for both tools
   - Configure your default Azure subscription
   - Prepare the environment for deployment

3. **Deploy the infrastructure**:

   ```powershell
   # Preview deployment
   .\infra\azd-deploy.ps1 -WhatIf
   
   # Deploy with defaults
   .\infra\azd-deploy.ps1
   
   # Deploy to specific environment
   .\infra\azd-deploy.ps1 -EnvironmentName "prod"
   ```

#### Alternative: Using Custom azd Scripts

We've provided enhanced deployment scripts with additional features:

```powershell
# Quick deployment with sensible defaults
.\infra\azd-deploy.ps1

# Preview what will be deployed
.\infra\azd-deploy.ps1 -WhatIf

# Full-featured deployment with custom options
.\infra\deploy-azd.ps1 -EnvironmentName "dev" -Location "eastus2"

# Deploy to production environment
.\infra\deploy-azd.ps1 -EnvironmentName "prod" -Location "westus2" -AdminEmail "admin@company.com"

# Deploy without confirmation prompts
.\infra\deploy-azd.ps1 -Force
```

#### azd Environment Management

```powershell
# List environments
azd env list

# Select an environment
azd env select <environment-name>

# Set environment variables
azd env set APIM_ADMIN_EMAIL "your-email@domain.com"
azd env set AZURE_LOCATION "eastus2"

# View environment values
azd env get-values

# Deploy to specific environment
azd up --environment prod
```

#### azd Benefits

- ✅ **Automatic Environment Management**: Creates and manages dev/test/prod environments
- ✅ **Smart Resource Grouping**: Auto-generates resource group names (`rg-hybrid-router-workshop-{env}`)
- ✅ **Parameter Management**: Environment-specific configuration with variable substitution
- ✅ **Deployment History**: Built-in tracking, rollback, and state management
- ✅ **CI/CD Integration**: Ready for GitHub Actions and Azure DevOps pipelines
- ✅ **Modern CLI Experience**: Better error handling, progress indicators, and user feedback
- ✅ **Infrastructure as Code**: Full Bicep template support with parameter validation
- ✅ **Cost Management**: Easy cleanup with `azd down`

### Option 2: Using Azure CLI (Traditional)

#### Using the PowerShell Script

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

### Option 3: Manual Deployment

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
  --resource-group "rg-hybridllm-workshop-poc" \
  --template-file main.bicep \
  --parameters workloadName="hybridllm" \
               environmentName="dev" \
               location="eastus" \
               apimAdminEmail="your-email@domain.com"
```

## 🔧 Configuration

### azd Configuration Files

The project includes the following azd configuration files:

- **`azure.yaml`**: Main azd project configuration
- **`infra/main.parameters.json`**: Parameter file with environment variable substitution

#### Environment Variables

azd uses environment variables for configuration. Set these before deployment:

```powershell
# Required
azd env set APIM_ADMIN_EMAIL "your-email@domain.com"

# Optional (with defaults)
azd env set AZURE_LOCATION "eastus2"
azd env set AZURE_ENV_NAME "dev"
```

### Bicep Parameters

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

#### Option 1: Using Azure Developer CLI (azd) - Recommended

The simplest way to clean up resources deployed with azd:

```powershell
# Delete all resources in the current environment
azd down

# Delete resources and purge soft-deleted services
azd down --purge

# Force deletion without confirmation
azd down --force

# Delete specific environment
azd down --environment prod
```

#### Option 2: Using the Cleanup Script

The project includes a comprehensive cleanup script that uses the `modules/utils.py` functions to properly clean up Azure resources:

1. **Navigate to the project root**:

   ```bash
   cd hybrid-router-workshop
   ```

2. **Activate the Python virtual environment**:

   ```bash
   # Windows
   .venv\Scripts\activate.bat
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install required dependencies** (if not already installed):

   ```bash
   pip install requests
   ```

4. **Run the cleanup script**:

   ```bash
   python scripts/cleanup_script.py
   ```

   The script will:
   - Prompt for confirmation before proceeding
   - Show the deployment name and resource group to be cleaned
   - Use the `cleanup_resources` function from `modules/utils.py`
   - Properly delete and purge all Azure resources:
     - AI Foundry projects
     - Cognitive Services accounts (with purging)
     - API Management services (with purging)
     - Key Vault resources (with purging)
     - Finally delete the entire resource group

5. **Confirm deletion** when prompted:

   ```
   Are you sure you want to proceed? (yes/no): yes
   ```

#### Features of the Cleanup Script

- ✅ **Comprehensive**: Handles all resource types deployed by the infrastructure
- ✅ **Safe Purging**: Properly purges soft-deleted resources (Cognitive Services, APIM, Key Vault)
- ✅ **Progress Tracking**: Colored output with timestamps and duration tracking
- ✅ **Error Handling**: Graceful handling of missing resources or failed operations
- ✅ **Confirmation**: Requires explicit confirmation before deletion
- ✅ **Complete Cleanup**: Removes the entire resource group as the final step

#### Customizing the Cleanup Script

To cleanup a different resource group or deployment:

1. **Edit the cleanup script** (`scripts/cleanup_script.py`):

   ```python
   def main():
       deployment_name = "your-deployment-name"  # Change this
       resource_group_name = "your-resource-group"  # Change this
   ```

2. **Or create a custom cleanup script**:

   ```python
   from modules.utils import cleanup_resources
   
   # Cleanup specific deployment
   cleanup_resources("my-deployment", "my-resource-group")
   ```

#### Option 2: Manual Resource Group Deletion

For simple cases, you can delete the entire resource group:

```bash
az group delete --name "rg-hybridllm-dev" --yes --no-wait
```

**Note**: Manual deletion may leave soft-deleted resources that need separate purging.

#### Option 3: Using utils.py Functions Directly

For advanced users, you can use the utility functions directly:

```python
from modules.utils import cleanup_resources, delete_resource, get_resources

# Cleanup entire deployment
cleanup_resources("deployment-name", "resource-group-name")

# Or cleanup individual resources
resources = get_resources("resource-group-name", {})
for resource in resources:
    delete_resource(resource, "resource-group-name")
```

## 🆘 Troubleshooting

### Common Issues

#### azd-Specific Issues

1. **azd not found**:
   ```powershell
   # Install azd
   winget install microsoft.azd
   # Or
   powershell -ex AllSigned -c "Invoke-RestMethod 'https://aka.ms/install-azd.ps1' | Invoke-Expression"
   ```

2. **Authentication Issues**:
   ```powershell
   # Re-authenticate
   azd auth login
   
   # Check auth status
   azd auth login --check-status
   ```

3. **Environment Issues**:
   ```powershell
   # List environments
   azd env list
   
   # Create new environment
   azd env new <environment-name>
   
   # Reset environment
   azd env select <environment-name>
   azd provision
   ```

4. **Parameter Issues**:
   ```powershell
   # Check current environment values
   azd env get-values
   
   # Set missing parameters
   azd env set APIM_ADMIN_EMAIL "your-email@domain.com"
   ```

#### General Deployment Issues

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

4. **Cleanup Issues**:

   - **Python Dependencies**: Install required packages with `pip install requests`
   - **Virtual Environment**: Ensure `.venv` is activated before running cleanup script
   - **Azure CLI Authentication**: Run `az login` if you get authentication errors
   - **Soft-Deleted Resources**: The cleanup script handles purging automatically
   - **Resource Dependencies**: Let the script handle dependency order (AI Foundry → Cognitive Services → APIM → Key Vault → Resource Group)
   - **Permission Errors**: Ensure you have Contributor or Owner role on the subscription
   - **Script Path Issues**: Run the cleanup script from the project root directory

### Cleanup Troubleshooting

If the cleanup script encounters issues:

1. **Check Azure CLI connectivity**:

   ```bash
   az account show
   ```

2. **Verify resource group exists**:

   ```bash
   az group show --name "your-resource-group"
   ```

3. **List deployments to find the correct deployment name**:

   ```bash
   az deployment group list -g "your-resource-group" --query "[].{Name:name, State:properties.provisioningState}"
   ```

4. **Run cleanup with verbose logging**:

   ```python
   # Edit cleanup_script.py to add debug output
   from modules.utils import cleanup_resources, print_info
   print_info("Debug: Starting cleanup process...")
   ```

5. **Manual resource inspection**:

   ```bash
   # List all resources in the group
   az resource list -g "your-resource-group" -o table
   ```

### Support

For infrastructure issues:

1. Check Azure Resource Manager deployment logs
2. Review the Azure portal for resource-specific errors
3. Consult Azure documentation for service-specific requirements
4. For cleanup issues, check the colored output from `utils.py` functions for specific error details

## 📝 Best Practices

1. **Environment Separation**: Use different resource groups for dev/test/prod
2. **Naming Conventions**: Follow consistent naming patterns
3. **Security**: Rotate keys regularly and use managed identities
4. **Monitoring**: Enable comprehensive logging and alerting
5. **Cost Management**: Monitor costs and set up budgets

## 🔗 Related Resources

### Azure Developer CLI (azd)
- [Azure Developer CLI Documentation](https://learn.microsoft.com/azure/developer/azure-developer-cli/)
- [azd Installation Guide](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd)
- [azd Templates and Examples](https://github.com/Azure-Samples/azd-templates)

### Azure Infrastructure
- [Azure Bicep Documentation](https://docs.microsoft.com/azure/azure-resource-manager/bicep/)
- [Azure OpenAI Service](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Azure API Management](https://docs.microsoft.com/azure/api-management/)
- [Azure AI Foundry](https://docs.microsoft.com/azure/machine-learning/)
