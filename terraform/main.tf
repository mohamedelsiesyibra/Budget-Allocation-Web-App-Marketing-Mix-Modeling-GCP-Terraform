terraform {
  required_version = ">= 0.13"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "service_accounts" {
  source = "./modules/service_accounts"
}

module "iam_bindings" {
  source = "./modules/iam_bindings"

  project_id               = var.project_id
  mmm_job_sa_email         = module.service_accounts.mmm_job_sa_email
  cloud_scheduler_sa_email = module.service_accounts.cloud_scheduler_sa_email
}

module "storage_bucket" {
  source = "./modules/storage_bucket"

  region             = var.region
  bucket_name        = var.gcs_bucket_name
  lifecycle_rule_age = var.lifecycle_rule_age
  mmm_job_sa_email   = module.service_accounts.mmm_job_sa_email
}

module "cloud_run_job" {
  source = "./modules/cloud_run_job"

  project_id       = var.project_id
  region           = var.region
  app_directory    = var.mmm_app_directory
  image_name       = var.mmm_image_name
  image_tag        = var.mmm_image_tag
  job_name         = var.mmm_job_name
  memory_limit     = var.memory_limit
  cpu_limit        = var.cpu_limit
  mmm_job_sa_email = module.service_accounts.mmm_job_sa_email
}

module "cloud_scheduler_job" {
  source = "./modules/cloud_scheduler_job"

  project_id               = var.project_id
  region                   = var.region
  cloud_run_job_name       = module.cloud_run_job.job_name
  cloud_scheduler_sa_email = module.service_accounts.cloud_scheduler_sa_email
  scheduler_job_name       = var.scheduler_job_name
  schedule                 = var.schedule
  time_zone                = var.time_zone
}

module "cloud_run_service" {
  source = "./modules/cloud_run_service"

  project_id            = var.project_id
  region                = var.region
  app_directory         = var.streamlit_app_directory
  image_name            = var.streamlit_image_name
  image_tag             = var.streamlit_image_tag
  service_name          = var.streamlit_service_name
}
