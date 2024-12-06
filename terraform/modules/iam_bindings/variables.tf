variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "mmm_job_sa_email" {
  description = "Email of the MMM Job service account"
  type        = string
}

variable "cloud_scheduler_sa_email" {
  description = "Email of the Cloud Scheduler service account"
  type        = string
}
