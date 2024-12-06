variable "bucket_name" {
  description = "Name of the GCS bucket"
  type        = string
  default     = "lightweight-mmm-pipeline"
}

variable "region" {
  description = "Google Cloud region"
  type        = string
}

variable "lifecycle_rule_age" {
  description = "Age in days to delete objects"
  type        = number
  default     = 365
}

variable "mmm_job_sa_email" {
  description = "Email of the MMM Job service account"
  type        = string
}
