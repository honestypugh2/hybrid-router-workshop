@description('The role definition ID to assign.')
param roleDefinitionId string

@description('The principal ID to assign the role to.')
param principalId string

@description('The resource ID to assign the role on.')
param resourceId string

@description('The type of principal (User, Group, ServicePrincipal).')
@allowed(['User', 'Group', 'ServicePrincipal'])
param principalType string = 'ServicePrincipal'

// Create role assignment
resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceId, principalId, roleDefinitionId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionId)
    principalId: principalId
    principalType: principalType
  }
}

@description('The ID of the role assignment.')
output roleAssignmentId string = roleAssignment.id
