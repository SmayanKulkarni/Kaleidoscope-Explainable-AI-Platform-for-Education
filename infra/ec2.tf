# ─────────────────────────────────────────────────────────────────────────────
# EC2 — Application server
# t3.micro (free tier). Runs without an instance profile when IAM role creation
# permissions are not available in the provisioning account.
# ─────────────────────────────────────────────────────────────────────────────

# ── SSH Key pair ──────────────────────────────────────────────────────────────
resource "aws_key_pair" "api" {
  key_name   = "${var.project_name}-key-v3"
  public_key = trimspace(file(pathexpand(var.ssh_public_key_path)))
}

# ── EC2 Instance ──────────────────────────────────────────────────────────────
resource "aws_instance" "api" {
  ami                    = var.ec2_ami_id
  instance_type          = var.ec2_instance_type
  key_name               = aws_key_pair.api.key_name
  vpc_security_group_ids = [aws_security_group.api.id]

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
