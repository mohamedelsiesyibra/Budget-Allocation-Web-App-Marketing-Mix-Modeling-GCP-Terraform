resource "google_cloud_scheduler_job" "mmm_scheduler_job" {
  name        = var.scheduler_job_name
  description = "Schedules the MMM Cloud Run job"
  schedule    = var.schedule
  time_zone   = var.time_zone

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${var.cloud_run_job_name}:run"

    oauth_token {
      service_account_email = var.cloud_scheduler_sa_email
    }

    headers = {
      "Content-Type" = "application/json"
    }
  }

  attempt_deadline = "320s"
}
