terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state — store in S3 so CI/CD runners share state.
  # Bootstrap: create this bucket manually once before `terraform init`.
  # Comment out this block on first run if the state bucket doesn't exist yet.
  backend "s3" {
    bucket         = "xai-rec-tf-state"
    key            = "production/terraform.tfstate"
    region         = "ap-south-1"
    encrypt        = true
    dynamodb_table = "xai-rec-tf-locks"   # for state locking
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
