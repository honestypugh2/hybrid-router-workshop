using 'main.bicep'

// Environment and naming parameters
param workloadName = 'hybridllm'
param environmentName = 'poc2'
param location = 'eastus2'

// Feature toggles
param enableTelemetry = true
param enableAiFoundry = true

// Azure OpenAI Configuration
param openAiSkuName = 'S0'

// API Management Configuration
param apimSkuName = 'Developer'
param apimAdminEmail = 'brittanypugh@microsoft.com'
param apimOrganizationName = 'Hybrid LLM Router'

// Tags
param tags = {
  project: 'hybrid-llm-router-workshop'
  owner: 'your_name'
}
