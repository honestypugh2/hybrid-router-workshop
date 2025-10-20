@description('The name of the Log Analytics workspace.')
param name string

@description('The location where the Log Analytics workspace will be created.')
param location string

@description('Tags to apply to the Log Analytics workspace.')
param tags object = {}

@description('The pricing tier for the Log Analytics workspace.')
@allowed(['PerGB2018', 'Free', 'Standalone', 'PerNode', 'Standard', 'Premium'])
param skuName string = 'PerGB2018'

@description('The data retention period in days.')
@minValue(30)
@maxValue(730)
param retentionInDays int = 30

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      name: skuName
    }
    retentionInDays: retentionInDays
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

@description('The resource ID of the Log Analytics workspace.')
output id string = logAnalyticsWorkspace.id

@description('The name of the Log Analytics workspace.')
output name string = logAnalyticsWorkspace.name

@description('The customer ID of the Log Analytics workspace.')
output customerId string = logAnalyticsWorkspace.properties.customerId

// Note: Workspace key is sensitive and not exposed as output
// Access via: logAnalyticsWorkspace.listKeys().primarySharedKey when needed
