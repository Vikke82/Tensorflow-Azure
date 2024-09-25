from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Initialize ML client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="ddfbea37-0529-4562-ac51-2ebbbe770146",
    resource_group_name="cloudservicess242",
    workspace_name="myWorkspace"
)

# Specify the endpoint and deployment name
endpoint_name = "my-mnist-endpoint-ville"
deployment_name = "mnist-deployment-v1"

# Retrieve the logs
logs = ml_client.online_deployments.get_logs(endpoint_name=endpoint_name, name=deployment_name, lines=10)
print(logs)
