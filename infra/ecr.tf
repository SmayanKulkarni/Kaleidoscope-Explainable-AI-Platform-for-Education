# ─────────────────────────────────────────────────────────────────────────────
# ECR — Elastic Container Registry for Docker images
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"   # "latest" tag always points to newest image

  image_scanning_configuration {
    scan_on_push = true   # free vulnerability scanning
  }
}

# Keep only the 5 most recent images to avoid storage costs
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}
