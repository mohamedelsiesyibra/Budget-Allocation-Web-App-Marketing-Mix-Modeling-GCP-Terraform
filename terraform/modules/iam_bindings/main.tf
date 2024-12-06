# BigQuery Data Viewer Role
resource "google_project_iam_member" "mmm_sa_bigquery_access" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${var.mmm_job_sa_email}"
}

# BigQuery Job User Role
resource "google_project_iam_member" "mmm_sa_bigquery_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${var.mmm_job_sa_email}"
}

# Storage Object Admin Role
resource "google_project_iam_member" "mmm_sa_storage_access" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${var.mmm_job_sa_email}"
}

# Cloud Run Invoker Role for Cloud Scheduler
resource "google_project_iam_member" "cloud_scheduler_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${var.cloud_scheduler_sa_email}"
}
