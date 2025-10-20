@description('The name of the AI Foundry Project.')
param name string

@description('The location where the AI Foundry Project will be created.')
param location string

@description('Tags to apply to the AI Foundry Project.')
param tags object = {}

@description('Network access configuration.')
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Enabled'

@description('The SKU for the AI Project.')
param sku object = {
  name: 'S0'
}

@description('Principal IDs to assign Cognitive Services User role to.')
param cognitiveServicesPrincipalIds array = []

@description('Role definition ID for Cognitive Services User.')
param cognitiveServicesUserRoleId string = 'a97b65f3-24c7-4388-baec-2e87135dc908'

// Variables - Store all references first
var projectEndpoint = aiFoundryProject.properties.endpoint
var projectId = aiFoundryProject.id
var projectName = aiFoundryProject.name
var projectPrincipalId = aiFoundryProject.identity.principalId

var deploymentNames = {
  gpt4o: 'gpt-4o'
  gpt4oMini: 'gpt-4o-mini'
  gpt41: 'gpt-4.1'
  modelRouter: 'model-router'
  embedding: 'text-embedding-ada-002'
}

var modelConfigurations = {
  gpt4o: {
    name: 'gpt-4o'
    version: '2024-11-20'
    capacity: 50
  }
  gpt4oMini: {
    name: 'gpt-4o-mini'
    version: '2024-07-18'
    capacity: 100
  }
  gpt41: {
    name: 'gpt-4.1'
    version: '2025-04-14'
    capacity: 50
  }
  modelRouter: {
    name: 'model-router'
    version: '2025-08-07'
    capacity: 50
  }
  embedding: {
    name: 'text-embedding-ada-002'
    version: '2'
    capacity: 50
  }
}

// Create AI Foundry Project using CognitiveServices API
resource aiFoundryProject 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  tags: tags
  kind: 'AIServices'
  sku: sku
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: publicNetworkAccess
    networkAcls: {
      defaultAction: 'Allow'
    }
    disableLocalAuth: false
  }
}

// Deploy GPT-4o model
resource gpt4oDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiFoundryProject
  name: deploymentNames.gpt4o
  properties: {
    model: {
      format: 'OpenAI'
      name: modelConfigurations.gpt4o.name
      version: modelConfigurations.gpt4o.version
    }
  }
  sku: {
    name: 'Standard'
    capacity: modelConfigurations.gpt4o.capacity
  }
}

// Deploy GPT-4o-mini model
resource gpt4oMiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiFoundryProject
  name: deploymentNames.gpt4oMini
  dependsOn: [gpt4oDeployment]
  properties: {
    model: {
      format: 'OpenAI'
      name: modelConfigurations.gpt4oMini.name
      version: modelConfigurations.gpt4oMini.version
    }
  }
  sku: {
    name: 'Standard'
    capacity: modelConfigurations.gpt4oMini.capacity
  }
}

// Deploy GPT-4.1 model
resource gpt41Deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiFoundryProject
  name: deploymentNames.gpt41
  dependsOn: [gpt4oMiniDeployment]
  properties: {
    model: {
      format: 'OpenAI'
      name: modelConfigurations.gpt41.name
      version: modelConfigurations.gpt41.version
    }
  }
  sku: {
    name: 'GlobalStandard'
    capacity: modelConfigurations.gpt41.capacity
  }
}

// Deploy model-router (using GPT-4o as the base model for routing)
resource modelRouterDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiFoundryProject
  name: deploymentNames.modelRouter
  dependsOn: [gpt41Deployment]
  properties: {
    model: {
      format: 'OpenAI'
      name: modelConfigurations.modelRouter.name
      version: modelConfigurations.modelRouter.version
    }
  }
  sku: {
    name: 'GlobalStandard'
    capacity: modelConfigurations.modelRouter.capacity
  }
}

// Deploy text-embedding-ada-002 model for embeddings
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiFoundryProject
  name: deploymentNames.embedding
  dependsOn: [modelRouterDeployment]
  properties: {
    model: {
      format: 'OpenAI'
      name: modelConfigurations.embedding.name
      version: modelConfigurations.embedding.version
    }
  }
  sku: {
    name: 'Standard'
    capacity: modelConfigurations.embedding.capacity
  }
}

@description('The resource ID of the AI Foundry Project.')
output id string = projectId

@description('The name of the AI Foundry Project.')
output name string = projectName

@description('The endpoint of the AI Foundry Project.')
output endpoint string = projectEndpoint

@description('The discovery URL of the AI Foundry Project.')
output discoveryUrl string = projectEndpoint

@description('The principal ID of the project managed identity.')
output principalId string = projectPrincipalId

@description('The deployed models information.')
output deployedModels object = {
  gpt4o: {
    name: deploymentNames.gpt4o
    endpoint: '${projectEndpoint}openai/deployments/${deploymentNames.gpt4o}'
  }
  gpt4oMini: {
    name: deploymentNames.gpt4oMini
    endpoint: '${projectEndpoint}openai/deployments/${deploymentNames.gpt4oMini}'
  }
  gpt41: {
    name: deploymentNames.gpt41
    endpoint: '${projectEndpoint}openai/deployments/${deploymentNames.gpt41}'
  }
  modelRouter: {
    name: deploymentNames.modelRouter
    endpoint: '${projectEndpoint}openai/deployments/${deploymentNames.modelRouter}'
  }
  embedding: {
    name: deploymentNames.embedding
    endpoint: '${projectEndpoint}openai/deployments/${deploymentNames.embedding}'
  }
}

// Role assignments for Cognitive Services User access
resource cognitiveServicesUserRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in cognitiveServicesPrincipalIds: {
  name: guid(projectId, principalId, cognitiveServicesUserRoleId)
  scope: aiFoundryProject
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesUserRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}]
