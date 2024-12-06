variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region"
  type        = string
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "streamlit-service"
}

variable "image_name" {
  description = "Name of the Docker image"
  type        = string
  default     = "streamlit-app"
}

variable "image_tag" {
  description = "Tag of the Docker image"
  type        = string
  default     = "latest"
}

variable "service_account_id" {
  description = "Account ID for the service account"
  type        = string
  default     = "streamlit-sa"
}

variable "gcs_bucket_name" {
  description = "Name of the GCS bucket containing the model"
  type        = string
  default     = "lightweight-mmm-pipeline"
}

variable "app_directory" {
  description = "Path to the application source directory"
  type        = string
  default     = "../streamlit" 
}


variable "memory_limit" {
  description = "Memory limit for the Cloud Run job"
  type        = string
  default     = "4Gi"
}

variable "cpu_limit" {
  description = "CPU limit for the Cloud Run job"
  type        = string
  default     = "2"
}