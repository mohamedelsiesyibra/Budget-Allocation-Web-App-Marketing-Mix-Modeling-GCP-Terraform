resource "google_service_account" "mmm_job_sa" {
  account_id   = var.mmm_job_sa_account_id
  display_name = "Service Account for MMM Job"
}

resource "google_service_account" "cloud_scheduler_sa" {
  account_id   = var.cloud_scheduler_sa_account_id
  display_name = "Service Account for Cloud Scheduler"
}
