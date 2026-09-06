"""Create the Northstar app's experiment and volume in your Databricks workspace."""

import re


def prepare(catalog="workspace", schema="northstar_support", experiment_name=None):
    import mlflow
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.catalog import VolumeType

    for identifier in (catalog, schema):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError("Use simple catalog and schema names containing letters, digits, and underscores.")
    client = WorkspaceClient()
    experiment_name = experiment_name or f"/Users/{client.current_user.me().user_name}/northstar-support"
    volume_name = f"{catalog}.{schema}.reports"
    artifact_location = f"dbfs:/Volumes/{catalog}/{schema}/reports/artifacts"
    try:
        client.schemas.get(f"{catalog}.{schema}")
    except NotFound:
        client.schemas.create(name=schema, catalog_name=catalog, comment="Storage for the Northstar workshop app")
    try:
        client.volumes.read(volume_name)
    except NotFound:
        client.volumes.create(catalog_name=catalog, schema_name=schema, name="reports", volume_type=VolumeType.MANAGED)
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    else:
        if experiment.artifact_location.rstrip("/") != artifact_location or experiment.trace_location is not None:
            raise ValueError("This experiment uses different storage. Choose a new experiment_name for the app.")
        experiment_id = experiment.experiment_id
    # Do not change the notebook's active experiment or its model configuration.
    return {"experiment_id": experiment_id, "experiment_name": experiment_name,
            "volume": volume_name, "artifact_location": artifact_location}
