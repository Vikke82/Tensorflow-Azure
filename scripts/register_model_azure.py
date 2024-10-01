from azureml.core import Workspace
from azureml.core.model import Model

# Connect to the Azure ML workspace
ws = Workspace.from_config()

# Register the model
model = Model.register(
    workspace=ws,
    model_name="my_tensorflow_model",  # Model name in Azure
    model_path="my_model.keras"  # Local path to the SavedModel directory
)

# List all registered models
models = Model.list(ws)
for model in models:
    print(f"Model name: {model.name}, Version: {model.version}")
