# ─────────────────────────────────────────────────────────────────────────────
# RDS — PostgreSQL 16, db.t3.micro (free tier eligible)
# Private within VPC — only accessible from EC2 API security group.
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_db_subnet_group" "postgres" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = data.aws_subnets.public.ids

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

resource "aws_db_instance" "postgres" {
  identifier        = "${var.project_name}-postgres"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  storage_type      = "gp2"
  storage_encrypted = true

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  port     = 5432

  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Free tier: single-AZ, no Multi-AZ
  multi_az               = false
  publicly_accessible    = false
  skip_final_snapshot    = false
  final_snapshot_identifier = "${var.project_name}-final-snapshot"

  # Automated backups — 7-day retention (free tier)
  backup_retention_period = 7
  backup_window           = "02:00-03:00"   # UTC
  maintenance_window      = "Mon:03:00-Mon:04:00"

  # Parameter group — use defaults (PostgreSQL 16 family)
  parameter_group_name = "default.postgres16"

  # Performance Insights — free tier (7 days)
  performance_insights_enabled          = true
  performance_insights_retention_period = 7

  tags = {
    Name = "${var.project_name}-postgres"
  }
}
