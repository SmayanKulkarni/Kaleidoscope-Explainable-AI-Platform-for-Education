# ─────────────────────────────────────────────────────────────────────────────
# Outputs — values needed by CI/CD and local configuration
# ─────────────────────────────────────────────────────────────────────────────

output "ec2_public_ip" {
  description = "Public IP of the API EC2 instance — add to GitHub Secret EC2_HOST"
  value       = aws_instance.api.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS of the API EC2 instance"
  value       = aws_instance.api.public_dns
}

output "ecr_repo_url" {
  description = "ECR repository URL — add to GitHub Secret ECR_REPO_URL"
  value       = aws_ecr_repository.api.repository_url
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint (host:port)"
  value       = "${aws_db_instance.postgres.address}:${aws_db_instance.postgres.port}"
}

output "database_url" {
  description = "Full PostgreSQL connection string — add to GitHub Secret DATABASE_URL (sensitive)"
  value     = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.postgres.address}:5432/${var.db_name}"
  sensitive = true
}

output "s3_bucket_name" {
  description = "S3 bucket for model artifacts — add to GitHub Secret AWS_S3_BUCKET"
  value       = aws_s3_bucket.models.bucket
}

output "api_url" {
  description = "API base URL"
  value       = "http://${aws_instance.api.public_ip}:8000"
}
