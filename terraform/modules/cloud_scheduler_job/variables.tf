variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region"
  type        = string
}

variable "scheduler_job_name" {
  description = "Name of the Cloud Scheduler job"
  type        = string
  default     = "mmm-scheduler-job"
}

variable "schedule" {
  description = "Cron schedule for the job"
  type        = string
  default     = "0 3 * * *"
}

variable "time_zone" {
  description = "Time zone for the scheduler"
  type        = string
  default     = "UTC"
}

variable "cloud_run_job_name" {
  description = "Name of the Cloud Run job to trigger"
  type        = string
}

variable "cloud_scheduler_sa_email" {
  description = "Email of the Cloud Scheduler service account"
  type        = string
}
