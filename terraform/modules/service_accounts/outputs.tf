output "mmm_job_sa_email" {
  value = google_service_account.mmm_job_sa.email
}

output "cloud_scheduler_sa_email" {
  value = google_service_account.cloud_scheduler_sa.email
}
