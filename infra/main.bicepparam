using 'main.bicep'

// Environment and naming parameters
param workloadName = readEnvironmentVariable('WORKLOAD_NAME', 'hybridllm')
param environmentName = readEnvironmentVariable('ENVIRONMENT_NAME', 'poc-dev')
param location = readEnvironmentVariable('AZURE_LOCATION', 'eastus2')

// Feature toggles
param enableTelemetry = bool(readEnvironmentVariable('ENABLE_TELEMETRY', 'true'))
param enableAiFoundry = bool(readEnvironmentVariable('ENABLE_AI_FOUNDRY', 'true'))

// Azure OpenAI Configuration
param openAiSkuName = readEnvironmentVariable('OPENAI_SKU_NAME', 'S0')

// API Management Configuration
param apimSkuName = readEnvironmentVariable('APIM_SKU_NAME', 'Developer')
param apimAdminEmail = readEnvironmentVariable('APIM_ADMIN_EMAIL', '')
param apimOrganizationName = readEnvironmentVariable('APIM_ORGANIZATION_NAME', 'Hybrid LLM Router')

// Tags
param tags = {
  project: 'hybrid-llm-router-workshop'
  owner: readEnvironmentVariable('AZURE_PRINCIPAL_ID', 'azd-user')
  environment: readEnvironmentVariable('ENVIRONMENT_NAME', 'poc-dev')
}
