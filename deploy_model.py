from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineDeployment

# Initialize ML client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="ddfbea37-0529-4562-ac51-2ebbbe770146",
    resource_group_name="cloudservicess242",  # Correct resource group
    workspace_name="myWorkspace"  # Correct workspace
)

# Define the deployment
deployment_name = "mnist-deployment-v1"
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name="my-mnist-endpoint-ville",  # The endpoint you already created
    model="my_tensorflow_model:3",  # Ensure the model is registered in your workspace
    environment="tensorflow-env:1",  # Ensure the environment is registered
    instance_type="Standard_DS3_v2",  # Ensure this VM size has available quota
    instance_count=1
)

# Deploy the model to the endpoint
deployment_poller = ml_client.online_deployments.begin_create_or_update(deployment)
deployment_result = deployment_poller.result()  # Wait for deployment to complete
print(f"Deployment '{deployment_name}' created successfully.")

# Set the traffic to this deployment
endpoint = ml_client.online_endpoints.get("my-mnist-endpoint-ville")
endpoint.traffic = {deployment_name: 100}  # Direct 100% traffic to this deployment
ml_client.online_endpoints.create_or_update(endpoint)

# Get the scoring URI
endpoint_details = ml_client.online_endpoints.get("my-mnist-endpoint-ville")
print(f"Scoring URI: {endpoint_details.scoring_uri}")
