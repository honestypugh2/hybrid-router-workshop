@description('The name of the Azure OpenAI service.')
param name string

@description('The location where the Azure OpenAI service will be created.')
param location string

@description('Tags to apply to the Azure OpenAI service.')
param tags object = {}

@description('The pricing tier for the Azure OpenAI service.')
@allowed(['S0'])
param skuName string = 'S0'

@description('Custom subdomain name for the OpenAI service.')
param customSubDomainName string = name

@description('Whether to disable local authentication.')
param disableLocalAuth bool = false

@description('Network access configuration.')
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Enabled'

@description('Principal IDs to assign OpenAI User role to.')
param userPrincipalIds array = []

@description('Principal IDs to assign Cognitive Services User role to.')
param cognitiveServicesPrincipalIds array = []

@description('Role definition ID for Cognitive Services OpenAI User.')
param cognitiveServicesOpenAIUserRoleId string = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

@description('Role definition ID for Cognitive Services User.')
param cognitiveServicesUserRoleId string = 'a97b65f3-24c7-4388-baec-2e87135dc908'

// Variables - Store all references first
var accountId = openAiAccount.id
var accountName = openAiAccount.name
var accountEndpoint = openAiAccount.properties.endpoint
var accountPrincipalId = openAiAccount.identity.principalId

var deploymentNames = {
  gpt4o: 'gpt-4o'
  gpt4oMini: 'gpt-4o-mini'
  gpt41: 'gpt-4.1'
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
    capacity: 50
  }
  gpt41: {
    name: 'gpt-4.1'
    version: '2025-04-14'
    capacity: 50
  }
  embedding: {
    name: 'text-embedding-ada-002'
    version: '2'
    capacity: 50
  }
}

resource openAiAccount 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  tags: tags
  kind: 'OpenAI'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: customSubDomainName
    networkAcls: {
      defaultAction: publicNetworkAccess == 'Enabled' ? 'Allow' : 'Deny'
    }
    publicNetworkAccess: publicNetworkAccess
    disableLocalAuth: disableLocalAuth
  }
  sku: {
    name: skuName
  }
}

// Deploy GPT-4o model
resource gpt4oDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAiAccount
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
  parent: openAiAccount
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
  parent: openAiAccount
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

// Deploy text-embedding-ada-002 model for embeddings
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAiAccount
  name: deploymentNames.embedding
  dependsOn: [gpt41Deployment]
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

@description('The resource ID of the Azure OpenAI service.')
output id string = accountId

@description('The name of the Azure OpenAI service.')
output name string = accountName

@description('The endpoint URL of the Azure OpenAI service.')
output endpoint string = accountEndpoint

@description('The principal ID of the managed identity.')
output principalId string = accountPrincipalId

@description('The deployed models information.')
output deployedModels object = {
  gpt4o: {
    name: deploymentNames.gpt4o
    endpoint: '${accountEndpoint}openai/deployments/${deploymentNames.gpt4o}'
  }
  gpt4oMini: {
    name: deploymentNames.gpt4oMini
    endpoint: '${accountEndpoint}openai/deployments/${deploymentNames.gpt4oMini}'
  }
  gpt41: {
    name: deploymentNames.gpt41
    endpoint: '${accountEndpoint}openai/deployments/${deploymentNames.gpt41}'
  }
  embedding: {
    name: deploymentNames.embedding
    endpoint: '${accountEndpoint}openai/deployments/${deploymentNames.embedding}'
  }
}

// Role assignments for OpenAI User access
resource openAiUserRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in userPrincipalIds: {
  name: guid(accountId, principalId, cognitiveServicesOpenAIUserRoleId)
  scope: openAiAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}]

// Role assignments for Cognitive Services User access
resource cognitiveServicesUserRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in cognitiveServicesPrincipalIds: {
  name: guid(accountId, principalId, cognitiveServicesUserRoleId)
  scope: openAiAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesUserRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}]
