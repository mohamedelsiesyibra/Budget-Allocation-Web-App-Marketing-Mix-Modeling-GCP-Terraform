resource "google_storage_bucket" "mmm_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = var.lifecycle_rule_age
    }
  }
}

# Grant the service account access to the bucket
resource "google_storage_bucket_iam_member" "mmm_sa_bucket_access" {
  bucket = google_storage_bucket.mmm_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.mmm_job_sa_email}"
}
