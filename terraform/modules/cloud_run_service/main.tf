# Build and push the Docker image using Cloud Build via local-exec (Optional)
resource "null_resource" "streamlit_cloud_build" {
  provisioner "local-exec" {
    working_dir = path.root
    command     = "gcloud builds submit ${var.app_directory} --tag=gcr.io/${var.project_id}/${var.image_name}:${var.image_tag} --project=${var.project_id}"
  }

  triggers = {
    image_name = var.image_name
    image_tag  = var.image_tag
  }
}

# Service Account for Cloud Run Service
resource "google_service_account" "streamlit_sa" {
  account_id   = var.service_account_id
  display_name = "Service Account for Streamlit App"
}

# Grant necessary IAM roles to the service account
resource "google_project_iam_member" "streamlit_sa_storage_access" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.streamlit_sa.email}"
}

# Cloud Run Service
resource "google_cloud_run_service" "streamlit_service" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/${var.image_name}:${var.image_tag}"
        
        resources {
          limits = {
            memory = var.memory_limit  # Will use "4Gi"
            cpu    = var.cpu_limit     # Will use "2"
          }
        }

        env {
          name  = "GCS_BUCKET_NAME"
          value = var.gcs_bucket_name
        }
      }
      service_account_name = google_service_account.streamlit_sa.email
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    null_resource.streamlit_cloud_build,
  ]
}

# Allow unauthenticated access (Optional)
resource "google_cloud_run_service_iam_member" "noauth" {
  service  = google_cloud_run_service.streamlit_service.name
  location = google_cloud_run_service.streamlit_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
