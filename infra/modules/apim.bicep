@description('The name of the API Management service.')
param name string

@description('The location where API Management will be created.')
param location string

@description('Tags to apply to the API Management service.')
param tags object = {}

@description('The pricing tier for the API Management service.')
@allowed(['Developer', 'Standard', 'Premium'])
param skuName string = 'Developer'

@description('Administrator email for API Management service.')
param adminEmail string

@description('Organization name for API Management service.')
param organizationName string = 'Hybrid LLM Router'

@description('The name of the OpenAI service to integrate with.')
param openAiServiceName string

@description('The endpoint URL of the AI Foundry project.')
param aiFoundryProjectEndpoint string = ''

@description('The endpoint URL of the model router.')
param modelRouterEndpoint string = ''

resource apimService 'Microsoft.ApiManagement/service@2023-09-01-preview' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: skuName
    capacity: skuName == 'Developer' ? 1 : 2
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publisherEmail: adminEmail
    publisherName: organizationName
    notificationSenderEmail: 'noreply@${name}.azure-api.net'
    hostnameConfigurations: [
      {
        type: 'Proxy'
        hostName: '${name}.azure-api.net'
        negotiateClientCertificate: false
        defaultSslBinding: true
        certificateSource: 'BuiltIn'
      }
    ]
    customProperties: {
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls11': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls10': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls11': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls10': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Ssl30': 'false'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Protocols.Server.Http2': 'false'
    }
    virtualNetworkType: 'None'
    disableGateway: false
    apiVersionConstraint: {}
    publicNetworkAccess: 'Enabled'
  }
}

// Create OpenAI API Backend
resource openAiBackend 'Microsoft.ApiManagement/service/backends@2023-09-01-preview' = {
  parent: apimService
  name: 'openai-backend'
  properties: {
    description: 'Azure OpenAI Service Backend'
    url: 'https://${openAiServiceName}.openai.azure.com'
    protocol: 'http'
    circuitBreaker: {
      rules: [
        {
          failureCondition: {
            count: 3
            errorReasons: [
              'Server errors'
            ]
            interval: 'PT5M'
            statusCodeRanges: [
              {
                min: 500
                max: 599
              }
            ]
          }
          name: 'openai-breaker'
          tripDuration: 'PT1M'
        }
      ]
    }
    tls: {
      validateCertificateChain: true
      validateCertificateName: true
    }
  }
}

// Create AI Foundry Project Backend
resource aiFoundryBackend 'Microsoft.ApiManagement/service/backends@2023-09-01-preview' = if (aiFoundryProjectEndpoint != '') {
  parent: apimService
  name: 'ai-foundry-backend'
  properties: {
    description: 'Azure AI Foundry Project Backend'
    url: '${aiFoundryProjectEndpoint}/models'
    protocol: 'http'
    circuitBreaker: {
      rules: [
        {
          failureCondition: {
            count: 3
            errorReasons: [
              'Server errors'
            ]
            interval: 'PT5M'
            statusCodeRanges: [
              {
                min: 500
                max: 599
              }
            ]
          }
          name: 'ai-foundry-breaker'
          tripDuration: 'PT1M'
        }
      ]
    }
    tls: {
      validateCertificateChain: true
      validateCertificateName: true
    }
  }
}

// Create OpenAI API
resource openAiApi 'Microsoft.ApiManagement/service/apis@2023-09-01-preview' = {
  parent: apimService
  name: 'openai-api'
  properties: {
    displayName: 'OpenAI API'
    description: 'Azure OpenAI API for LLM routing'
    path: 'openai'
    protocols: ['https']
    serviceUrl: 'https://${openAiServiceName}.openai.azure.com'
    subscriptionRequired: true
    format: 'openapi+json'
    value: string(loadJsonContent('./specs/azureaifoundryopenai.json'))
  }
}

// Create AI Foundry Project API
resource aiFoundryApi 'Microsoft.ApiManagement/service/apis@2023-09-01-preview' = if (aiFoundryProjectEndpoint != '') {
  parent: apimService
  name: 'ai-foundry-api'
  properties: {
    displayName: 'AI Foundry Project API'
    description: 'Azure AI Foundry Project API for multi-model access'
    path: 'ai-foundry'
    protocols: ['https']
    serviceUrl: '${aiFoundryProjectEndpoint}/models'
    subscriptionRequired: true
    format: 'openapi+json'
    value: string(loadJsonContent('./specs/azureaifoundry.json'))
  }
}

// Create a subscription for OpenAI access
resource openAiSubscription 'Microsoft.ApiManagement/service/subscriptions@2023-09-01-preview' = {
  parent: apimService
  name: 'openai-subscription'
  properties: {
    displayName: 'OpenAI API Subscription'
    scope: '/apis/${openAiApi.name}'
    state: 'active'
  }
}

// Create a subscription for AI Foundry API access
resource aiFoundrySubscription 'Microsoft.ApiManagement/service/subscriptions@2023-09-01-preview' = if (aiFoundryProjectEndpoint != '') {
  parent: apimService
  name: 'ai-foundry-subscription'
  properties: {
    displayName: 'AI Foundry API Subscription'
    scope: '/apis/${aiFoundryApi.name}'
    state: 'active'
  }
}

// Create hybrid router API if endpoint is provided
resource hybridRouterApi 'Microsoft.ApiManagement/service/apis@2023-09-01-preview' = if (modelRouterEndpoint != '') {
  parent: apimService
  name: 'hybrid-router-api'
  properties: {
    displayName: 'Hybrid Router API'
    description: 'Intelligent model routing API'
    path: 'router'
    protocols: ['https']
    serviceUrl: modelRouterEndpoint
    subscriptionRequired: true
    format: 'openapi+json'
    value: '''
    {
      "openapi": "3.0.1",
      "info": {
        "title": "Hybrid Router API",
        "version": "1.0"
      },
      "paths": {
        "/route": {
          "post": {
            "operationId": "routeQuery",
            "requestBody": {
              "content": {
                "application/json": {
                  "schema": {
                    "type": "object",
                    "properties": {
                      "query": {
                        "type": "string"
                      },
                      "strategy": {
                        "type": "string"
                      }
                    }
                  }
                }
              }
            },
            "responses": {
              "200": {
                "description": "Success"
              }
            }
          }
        }
      }
    }
    '''
  }
}

@description('The resource ID of the API Management service.')
output id string = apimService.id

@description('The name of the API Management service.')
output name string = apimService.name

@description('The gateway URL of the API Management service.')
output gatewayUrl string = apimService.properties.gatewayUrl

@description('The management API URL.')
output managementApiUrl string = apimService.properties.managementApiUrl

@description('The developer portal URL.')
output portalUrl string = apimService.properties.portalUrl ?? '${name}.developer.azure-api.net'

@description('The principal ID of the managed identity.')
output principalId string = apimService.identity.principalId

@description('The subscription key for OpenAI API access.')
output openAiSubscriptionId string = openAiSubscription.id

@description('The subscription key for AI Foundry API access.')
output aiFoundrySubscriptionId string = aiFoundryProjectEndpoint != '' ? aiFoundrySubscription.id : ''

@description('The AI Foundry API endpoint through APIM.')
output aiFoundryApiEndpoint string = aiFoundryProjectEndpoint != '' ? '${apimService.properties.gatewayUrl}/ai-foundry' : ''

@description('The OpenAI API endpoint through APIM.')
output openAiApiEndpoint string = '${apimService.properties.gatewayUrl}/openai'
