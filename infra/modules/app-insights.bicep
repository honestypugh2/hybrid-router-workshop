@description('The name of the Application Insights instance.')
param name string

@description('The location where Application Insights will be created.')
param location string

@description('Tags to apply to the Application Insights instance.')
param tags object = {}

@description('The resource ID of the Log Analytics workspace.')
param workspaceResourceId string

@description('The application type for Application Insights.')
@allowed(['web', 'other'])
param applicationType string = 'web'

@description('The sampling percentage for Application Insights.')
@minValue(0)
@maxValue(100)
param samplingPercentage int = 100

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: name
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: applicationType
    WorkspaceResourceId: workspaceResourceId
    SamplingPercentage: samplingPercentage
    RetentionInDays: 90
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

@description('The resource ID of the Application Insights instance.')
output id string = applicationInsights.id

@description('The name of the Application Insights instance.')
output name string = applicationInsights.name

@description('The instrumentation key of the Application Insights instance.')
output instrumentationKey string = applicationInsights.properties.InstrumentationKey

@description('The connection string of the Application Insights instance.')
output connectionString string = applicationInsights.properties.ConnectionString

@description('The app ID of the Application Insights instance.')
output appId string = applicationInsights.properties.AppId
