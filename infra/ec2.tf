# ─────────────────────────────────────────────────────────────────────────────
# EC2 — Application server
# t3.micro (free tier). IAM role grants ECR pull + S3 read/write for models.
# ─────────────────────────────────────────────────────────────────────────────

# ── SSH Key pair ──────────────────────────────────────────────────────────────
resource "aws_key_pair" "api" {
  key_name   = "${var.project_name}-key"
  public_key = file(var.ssh_public_key_path)
}

# ── IAM Role for EC2 (ECR pull + S3 model access) ────────────────────────────
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ec2_ecr_s3" {
  name = "${var.project_name}-ec2-ecr-s3"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
        ]
        Resource = aws_ecr_repository.api.arn
      },
      {
        Sid    = "S3Models"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject",
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*",
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# ── EC2 Instance ──────────────────────────────────────────────────────────────
resource "aws_instance" "api" {
  ami                    = var.ec2_ami_id
  instance_type          = var.ec2_instance_type
  key_name               = aws_key_pair.api.key_name
  vpc_security_group_ids = [aws_security_group.api.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  # 20GB root volume (stays within free tier)
  root_block_device {
    volume_size = 20
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/userdata.sh.tpl", {
    ecr_repo_url  = aws_ecr_repository.api.repository_url
    aws_region    = var.aws_region
    s3_bucket     = aws_s3_bucket.models.bucket
    db_url        = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.postgres.address}:5432/${var.db_name}"
  })

  tags = {
    Name = "${var.project_name}-api"
  }

  # Replace instance on user_data change (rolling deploy)
  lifecycle {
    create_before_destroy = true
  }
}
