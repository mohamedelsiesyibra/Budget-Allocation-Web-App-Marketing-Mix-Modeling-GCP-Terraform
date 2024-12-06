variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region"
  type        = string
}

variable "job_name" {
  description = "Name of the Cloud Run job"
  type        = string
  default     = "mmm-job"
}

variable "image_name" {
  description = "Name of the Docker image"
  type        = string
}

variable "image_tag" {
  description = "Tag of the Docker image"
  type        = string
  default     = "latest"
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

variable "mmm_job_sa_email" {
  description = "Email of the MMM Job service account"
  type        = string
}

variable "app_directory" {
  description = "Path to the application source directory"
  type        = string
  default     = "../mmm_model" 
}
