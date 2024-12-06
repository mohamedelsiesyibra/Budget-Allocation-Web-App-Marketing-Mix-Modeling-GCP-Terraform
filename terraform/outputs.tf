output "mmm_job_name" {
  description = "Name of the Cloud Run Job"
  value       = module.cloud_run_job.job_name
}

output "streamlit_service_url" {
  description = "URL of the Streamlit Cloud Run Service"
  value       = module.cloud_run_service.service_url
}

output "mmm_bucket_name" {
  description = "Name of the GCS bucket created"
  value       = module.storage_bucket.bucket_name
}