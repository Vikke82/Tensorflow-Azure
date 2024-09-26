from azureml.core.webservice import AciWebservice
from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace.from_config()

models = Model.list(ws)
for model in models:
    print(f"Model name: {model.name}, Model ID: {model.id}")

# Get the deployed service
service = AciWebservice(workspace=ws, name="mnist-deployment-v1")

# Print the logs
logs = service.get_logs()
print(logs)


