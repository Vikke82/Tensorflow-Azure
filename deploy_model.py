from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineDeployment, Environment, CodeConfiguration
import os


ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="ddfbea37-0529-4562-ac51-2ebbbe770146",
    resource_group_name="cloudServicesS242",
    workspace_name="myWorkspace"
)

code_path = os.path.abspath("./")
# Define your deployment
deployment = ManagedOnlineDeployment(
    name="mnist-deployment-v1",
    endpoint_name="myworkspace-ville",
    model="my_tensorflow_model:3",
    #environment="tensorflow-env:1",
    code_configuration=CodeConfiguration(
        code="./", scoring_script="score.py"
        ),
    environment=Environment(
            conda_file="conda.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            ),
    
    instance_type="Standard_DS3_v2",
    instance_count=1
)

# Deploy locally
deployment_poller = ml_client.online_deployments.begin_create_or_update(deployment, local=True)
deployment_result = deployment_poller.result()


# Get the scoring URI
endpoint_details = ml_client.online_endpoints.get("myworkspace-ville")
print(f"Scoring URI: {endpoint_details.scoring_uri}")
