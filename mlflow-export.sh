mlflow_export --export-metadata-tags --src-uri http://34.213.211.223:80 --dst-dir ./mlflow_exported_data

# Set the MLflow tracking URI to the local server
# mlflow.set_tracking_uri("http://34.213.211.223:80")


# C:\Users\sbarr\AppData\Roaming\Python\Python311\site-packages\mlflow_export_import\bulk\export-all.py --export-metadata-tags --src-uri http://34.213.211.223:80 --dst-dir ./mlflow_exported_data
# python C:\Users\sbarr\AppData\Roaming\Python\Python311\site-packages\mlflow_export_import\bulk\export_all.py --tracking-uri http://34.213.211.223:80 --output-dir ./output-ml

C:/Users/sbarr/.conda/envs/mlflow/python C:\Users\sbarr\AppData\Roaming\Python\Python311\site-packages\mlflow_export_import\bulk\export_all.py --output-dir ./output-ml

C:/Users/sbarr/.conda/envs/mlflow/python C:\Users\sbarr\AppData\Roaming\Python\Python311\site-packages\mlflow_export_import\bulk\export_models.py --models ElasticnetWineModel --output-dir ./output-ml2