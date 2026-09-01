# Sample Knowledge Base Document

## Deployment Workflow

To deploy the data pipeline:

1. Run the CI/CD pipeline from the main branch
2. The pipeline builds Docker images and pushes to ECR
3. ECS tasks are updated via Terraform
4. Health checks run before traffic is switched

## Architecture Overview

- **API**: FastAPI service on port 8000
- **Database**: PostgreSQL with read replicas
- **Cache**: Redis for session and rate limiting
- **Queue**: SQS for async job processing

## Runbook: Incident Response

1. Check CloudWatch dashboards for anomalies
2. Review recent deployments in the last 24h
3. Escalate to on-call if P1/P2
