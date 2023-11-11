from mlflow_export_import.export import export_runs

# Specify the source MLflow tracking server URI
src_tracking_uri = "http://34.213.211.223:80"

# Specify the destination directory where the exported data will be stored
output_directory = "./mlflow_exported_data"

# Call the export function
export_mlflow.export(src_tracking_uri, output_directory, export_metadata_tags=True)




