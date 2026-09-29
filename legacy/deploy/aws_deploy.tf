# ==============================================================================
# AWS ECS Fargate & ALB 배포를 위한 Terraform 스크립트 (면접 대비 주석용)
# ==============================================================================
# [인프라 아키텍처 흐름 요약]
# 1. User -> ALB (Application Load Balancer): 사용자가 도메인이나 ALB 주소로 접속합니다.
# 2. ALB -> ECS Target Group: 로드밸런서가 트래픽을 ECS Fargate 태스크들로 고르게 분산시킵니다.
# 3. ECS Fargate Task (Docker Container): 가상 서버나 관리 인프라 없이 컨테이너 단위로 배포된
#    FastAPI 서버가 요청을 받아 처리합니다. (기본 L7 헬스체크 /health 적용)
# ==============================================================================

provider "aws" {
  region = "us-east-1"
}

# 1. VPC (Virtual Private Cloud) & Subnets 정의
# 네트워크 격리 공간과 퍼블릭 서브넷 2개를 정의합니다 (ALB 이중화 및 가용영역 분리를 위해 최소 2개 필요).
resource "aws_vpc" "main_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = {
    Name = "smartrouter-vpc"
  }
}

resource "aws_subnet" "public_subnet_a" {
  vpc_id            = aws_vpc.main_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-1a"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "public_subnet_b" {
  vpc_id            = aws_vpc.main_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-east-1b"
  map_public_ip_on_launch = true
}

# 2. 보안 그룹 (Security Groups) 설정
# ALB는 인터넷망(0.0.0.0/0)의 80포트를 허용하고, ECS 태스크는 ALB로부터의 8000포트 진입만 허용하도록 격리합니다.
resource "aws_security_group" "alb_sg" {
  name        = "smartrouter-alb-sg"
  description = "Allow HTTP inbound traffic"
  vpc_id      = aws_vpc.main_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs_sg" {
  name        = "smartrouter-ecs-sg"
  description = "Allow inbound traffic from ALB only"
  vpc_id      = aws_vpc.main_vpc.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id] # ALB를 거친 트래픽만 ECS 진입 허용 (보안성 확보)
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 3. ALB (Application Load Balancer) 설정
# 트래픽 분산과 서비스 도메인 진입점을 제공합니다.
resource "aws_lb" "main_alb" {
  name               = "smartrouter-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = [aws_subnet.public_subnet_a.id, aws_subnet.public_subnet_b.id]
}

resource "aws_lb_target_group" "ecs_target_group" {
  name        = "smartrouter-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main_vpc.id
  target_type = "ip"

  # L7 Healthcheck 경로 연동 (FastAPI의 /health 엔드포인트 호출)
  health_check {
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http_listener" {
  load_balancer_arn = aws_lb.main_alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ecs_target_group.arn
  }
}

# 4. ECS Fargate Cluster & Task Definition
# Fargate는 AWS가 EC2 가상 머신 관리를 대행해 주는 서버리스 컨테이너 구동 인프라입니다.
resource "aws_ecs_cluster" "main_cluster" {
  name = "smartrouter-cluster"
}

resource "aws_ecs_task_definition" "app_task" {
  family                   = "smartrouter-task"
  network_mode             = "awsvpc" # Fargate에서는 필수 모드
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"  # 0.5 vCPU
  memory                   = "1024" # 1GB RAM

  # ECR(Elastic Container Registry)에 업로드된 Docker 이미지를 기반으로 태스크 생성
  container_definitions = jsonencode([{
    name      = "smartrouter-container"
    image     = "123456789012.dkr.ecr.us-east-1.amazonaws.com/smartrouter:latest"
    essential = true
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
    }]
    environment = [
      { name = "AWS_DEFAULT_REGION", value = "us-east-1" },
      { name = "ROUTER_THRESHOLD", value = "0.78" }
    ]
  }])
}

# 5. ECS Service 정의
# 선언한 Cluster 내에서 Task 정의를 기반으로 항상 2개의 컨테이너(태스크)가 살아있도록 유지보수합니다.
resource "aws_ecs_service" "main_service" {
  name            = "smartrouter-service"
  cluster         = aws_ecs_cluster.main_cluster.id
  task_definition = aws_ecs_task_definition.app_task.arn
  desired_count   = 2 # 이중화(고가용성) 설정
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.public_subnet_a.id, aws_subnet.public_subnet_b.id]
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ecs_target_group.arn
    container_name   = "smartrouter-container"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http_listener]
}
