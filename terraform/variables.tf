variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region"
  type        = string
}

variable "mmm_app_directory" {
  description = "Path to the MMM application source directory"
  type        = string
  default     = "../mmm_model"
}

variable "mmm_image_name" {
  description = "Name of the Docker image for MMM Cloud Run Job"
  type        = string
  default     = "mmm-image"
}

variable "mmm_image_tag" {
  description = "Tag of the Docker image for MMM Cloud Run Job"
  type        = string
  default     = "latest"
}

variable "mmm_job_name" {
  description = "Name of the Cloud Run Job"
  type        = string
  default     = "mmm-job"
}

variable "memory_limit" {
  description = "Memory limit for the Cloud Run Job"
  type        = string
  default     = "4Gi"
}

variable "cpu_limit" {
  description = "CPU limit for the Cloud Run Job"
  type        = string
  default     = "2"
}

variable "streamlit_app_directory" {
  description = "Path to the Streamlit application source directory"
  type        = string
  default     = "../streamlit"
}

variable "streamlit_image_name" {
  description = "Name of the Docker image for Streamlit app"
  type        = string
  default     = "streamlit-app"
}

variable "streamlit_image_tag" {
  description = "Tag of the Docker image for Streamlit app"
  type        = string
  default     = "latest"
}

variable "streamlit_service_name" {
  description = "Name of the Cloud Run Service for Streamlit app"
  type        = string
  default     = "streamlit-service"
}

variable "scheduler_job_name" {
  description = "Name of the Cloud Scheduler Job"
  type        = string
  default     = "mmm-scheduler-job"
}

variable "schedule" {
  description = "Schedule for the Cloud Scheduler Job in cron format"
  type        = string
  default     = "0 9 * * *"
}

variable "time_zone" {
  description = "Time zone for the Cloud Scheduler Job"
  type        = string
  default     = "UTC"
}

variable "gcs_bucket_name" {
  description = "Name of the GCS bucket containing the model"
  type        = string
  default     = "lightweight-mmm-pipeline"
}

variable "lifecycle_rule_age" {
  description = "Number of days after which objects in the bucket are deleted"
  type        = number
  default     = 30
}
