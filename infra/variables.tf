variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Short project identifier used in resource names"
  type        = string
  default     = "xai-rec"
}

variable "environment" {
  description = "Deployment environment tag"
  type        = string
  default     = "production"
}

# ── EC2 ───────────────────────────────────────────────────────────────────────

variable "ec2_instance_type" {
  description = "EC2 instance type. t3.micro (1GB) = free tier eligible. WARNING: the full XAI stack (PyTorch + SHAP + LightGBM) uses ~1.5GB peak — if you hit OOM on startup, upgrade to t3.small."
  type        = string
  default     = "t3.micro"
}

variable "ec2_ami_id" {
  description = "Amazon Linux 2023 AMI ID — update for your region"
  type        = string
  default     = "ami-0f58b397bc5c1f2e8"   # Amazon Linux 2023, ap-south-1
}

variable "ssh_public_key_path" {
  description = "Path to the SSH public key for EC2 access"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into EC2 (restrict to your IP in prod)"
  type        = string
  default     = "0.0.0.0/0"
}

# ── RDS ───────────────────────────────────────────────────────────────────────

variable "db_instance_class" {
  description = "RDS instance class (db.t3.micro = free tier)"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "xai_db"
}

variable "db_username" {
  description = "PostgreSQL master username"
  type        = string
  default     = "xai"
}

variable "db_password" {
  description = "PostgreSQL master password — override via TF_VAR_db_password or terraform.tfvars"
  type        = string
  sensitive   = true
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB (20 = free tier max)"
  type        = number
  default     = 20
}

# ── S3 ────────────────────────────────────────────────────────────────────────

variable "s3_bucket_suffix" {
  description = "Unique suffix to make S3 bucket name globally unique"
  type        = string
  default     = "001"
}
