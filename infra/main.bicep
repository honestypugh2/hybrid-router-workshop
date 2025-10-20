@description('The name of the workload that is being deployed. Up to 10 characters long.')
@minLength(1)
@maxLength(10)
param workloadName string = 'hybridllm'

@description('The name of the environment (e.g. "dev", "test", "prod").')
@maxLength(8)
param environmentName string = 'dev'

@description('The location where the resources will be created.')
param location string = resourceGroup().location

@description('Tags to apply to all resources.')
param tags object = {}

@description('The pricing tier for the OpenAI service.')
@allowed(['S0'])
param openAiSkuName string = 'S0'

@description('The pricing tier for the API Management service.')
@allowed(['Developer', 'Standard', 'Premium'])
param apimSkuName string = 'Developer'

@description('Administrator email for API Management service.')
param apimAdminEmail string

@description('Organization name for API Management service.')
param apimOrganizationName string = 'Hybrid LLM Router'

@description('Enable Application Insights for telemetry.')
param enableTelemetry bool = true

@description('Enable Azure AI Foundry workspace.')
param enableAiFoundry bool = true

// Azure role definition IDs (instead of loading JSON to avoid "content consumed" errors)
var azureRoles = {
  CognitiveServicesOpenAIUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  CognitiveServicesUser: 'a97b65f3-24c7-4388-baec-2e87135dc908'
}

// Define common naming convention
var resourceNames = {
  openAi: '${workloadName}-${environmentName}-openai'
  aiFoundryProject: '${workloadName}-${environmentName}-aiproject'
  apim: '${workloadName}-${environmentName}-apim'
  appInsights: '${workloadName}-${environmentName}-ai'
  logAnalytics: '${workloadName}-${environmentName}-logs'
}

// Common tags for all resources
var commonTags = union(tags, {
  workload: workloadName
  environment: environmentName
  deployedBy: 'bicep'
})

// Log Analytics Workspace for Application Insights
module logAnalytics 'modules/log-analytics.bicep' = if (enableTelemetry) {
  name: 'logAnalytics'
  params: {
    name: resourceNames.logAnalytics
    location: location
    tags: commonTags
  }
}

// Application Insights for telemetry
module applicationInsights 'modules/app-insights.bicep' = if (enableTelemetry) {
  name: 'applicationInsights'
  params: {
    name: resourceNames.appInsights
    location: location
    tags: commonTags
    workspaceResourceId: logAnalytics.?outputs.id ?? ''
  }
}

// Azure AI Foundry Project
module aiFoundryProject 'modules/ai-foundry-project.bicep' = if (enableAiFoundry) {
  name: 'aiFoundryProject'
  params: {
    name: resourceNames.aiFoundryProject
    location: location
    tags: commonTags
    cognitiveServicesUserRoleId: azureRoles.CognitiveServicesUser
  }
}

// Azure OpenAI Service
module openAi 'modules/openai.bicep' = {
  name: 'openAi'
  params: {
    name: resourceNames.openAi
    location: location
    tags: commonTags
    skuName: openAiSkuName
    userPrincipalIds: enableAiFoundry ? [aiFoundryProject.?outputs.principalId ?? ''] : []
    cognitiveServicesOpenAIUserRoleId: azureRoles.CognitiveServicesOpenAIUser
    cognitiveServicesUserRoleId: azureRoles.CognitiveServicesUser
  }
}

// Azure API Management
module apiManagement 'modules/apim.bicep' = {
  name: 'apiManagement'
  params: {
    name: resourceNames.apim
    location: location
    tags: commonTags
    skuName: apimSkuName
    adminEmail: apimAdminEmail
    organizationName: apimOrganizationName
    openAiServiceName: openAi.outputs.name
    aiFoundryProjectEndpoint: aiFoundryProject.?outputs.discoveryUrl ?? ''
    modelRouterEndpoint: '' // Can be updated later when model router is deployed
  }
}

// Additional role assignments for APIM to access OpenAI
module apimOpenAiRoleAssignment 'modules/role-assignment.bicep' = {
  name: 'apimOpenAiRoleAssignment'
  params: {
    roleDefinitionId: azureRoles.CognitiveServicesOpenAIUser
    principalId: apiManagement.outputs.principalId
    resourceId: openAi.outputs.id
  }
}

// Outputs for use in applications
@description('The endpoint URL of the OpenAI service.')
output openAiEndpoint string = openAi.?outputs.endpoint ?? ''

@description('The name of the OpenAI service.')
output openAiServiceName string = openAi.?outputs.name ?? ''

@description('The deployed models from OpenAI service.')
output openAiModels object = openAi.?outputs.deployedModels ?? {}

@description('The gateway URL of the API Management service.')
output apimGatewayUrl string = apiManagement.?outputs.gatewayUrl ?? ''

@description('The name of the API Management service.')
output apimServiceName string = apiManagement.?outputs.name ?? ''

@description('The connection string for Application Insights.')
output applicationInsightsConnectionString string = applicationInsights.?outputs.connectionString ?? ''

@description('The instrumentation key for Application Insights.')
output applicationInsightsInstrumentationKey string = applicationInsights.?outputs.instrumentationKey ?? ''

@description('The name of the AI Foundry Project.')
output aiFoundryProjectName string = aiFoundryProject.?outputs.name ?? ''

@description('The endpoint of the AI Foundry Project.')
output aiFoundryProjectEndpoint string = aiFoundryProject.?outputs.endpoint ?? ''

@description('The discovery URL of the AI Foundry Project.')
output aiFoundryProjectDiscoveryUrl string = aiFoundryProject.?outputs.discoveryUrl ?? ''

@description('The Log Analytics workspace name.')
output logAnalyticsWorkspaceName string = logAnalytics.?outputs.name ?? ''
