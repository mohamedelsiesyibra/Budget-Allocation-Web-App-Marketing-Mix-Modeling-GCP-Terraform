variable "mmm_job_sa_account_id" {
  description = "Account ID for the MMM Job service account"
  type        = string
  default     = "mmm-job-sa"
}

variable "cloud_scheduler_sa_account_id" {
  description = "Account ID for the Cloud Scheduler service account"
  type        = string
  default     = "cloud-scheduler-sa"
}
