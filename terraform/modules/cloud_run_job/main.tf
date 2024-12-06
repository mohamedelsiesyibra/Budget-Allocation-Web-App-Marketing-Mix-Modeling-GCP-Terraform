# Build and push the Docker image using Cloud Build via local-exec
resource "null_resource" "mmm_cloud_build" {
  provisioner "local-exec" {
    command = "gcloud builds submit ${var.app_directory} --tag=gcr.io/${var.project_id}/${var.image_name}:${var.image_tag} --project=${var.project_id}"
  }
}

# Cloud Run Job
resource "google_cloud_run_v2_job" "mmm_job" {
  name     = var.job_name
  location = var.region

  template {
    template {
      containers {
        image = "gcr.io/${var.project_id}/${var.image_name}:${var.image_tag}"

        resources {
          limits = {
            memory = var.memory_limit
            cpu    = var.cpu_limit
          }
        }
      }
      service_account = var.mmm_job_sa_email
    }
  }

  depends_on = [
    null_resource.mmm_cloud_build,
  ]
}
