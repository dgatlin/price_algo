# Deployment Guide

## 🚀 Production Deployment

This guide covers deploying the Pricing Random Forest API to production environments.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.11+
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ disk space
- **CPU**: 2+ cores

### Dependencies
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.20+ (optional)
- **Nginx**: 1.18+ (optional)

## 🐳 Docker Deployment

### 1. Build Docker Image

```bash
# Build the image
docker build -t pricing-rf-api:latest .

# Tag for registry
docker tag pricing-rf-api:latest your-registry/pricing-rf-api:v1.0.0
```

### 2. Run Container

```bash
# Run with environment variables
docker run -d \
  --name pricing-rf-api \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e MODEL_NAME=pricing-rf \
  pricing-rf-api:latest
```

### 3. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  mlflow:
    image: mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow
    command: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow --host 0.0.0.0 --port 5000

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=pricing-rf
      - MODEL_VERSION=latest
    restart: unless-stopped

volumes:
  mlflow_data:
```

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

## ☸️ Kubernetes Deployment

### 1. Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pricing-rf
```

### 2. Deploy MLflow

```yaml
# mlflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: pricing-rf
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: mlflow/mlflow:latest
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_BACKEND_STORE_URI
          value: "sqlite:///mlflow.db"
        - name: MLFLOW_DEFAULT_ARTIFACT_ROOT
          value: "/mlflow"
        command: ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow", "--host", "0.0.0.0", "--port", "5000"]
        volumeMounts:
        - name: mlflow-storage
          mountPath: /mlflow
      volumes:
      - name: mlflow-storage
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: pricing-rf
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
```

### 3. Deploy API

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pricing-rf-api
  namespace: pricing-rf
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pricing-rf-api
  template:
    metadata:
      labels:
        app: pricing-rf-api
    spec:
      containers:
      - name: api
        image: pricing-rf-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: MODEL_NAME
          value: "pricing-rf"
        - name: MODEL_VERSION
          value: "latest"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: pricing-rf-api-service
  namespace: pricing-rf
spec:
  selector:
    app: pricing-rf-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 4. Apply Deployments

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f mlflow-deployment.yaml
kubectl apply -f api-deployment.yaml

# Check status
kubectl get pods -n pricing-rf
kubectl get services -n pricing-rf
```

## 🌐 Nginx Configuration

### 1. Install Nginx

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
```

### 2. Configure Nginx

```nginx
# /etc/nginx/sites-available/pricing-rf
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
```

### 3. Enable Site

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/pricing-rf /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

## 🔒 Security Configuration

### 1. Environment Variables

```bash
# Create environment file
cat > .env << EOF
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_NAME=pricing-rf
MODEL_VERSION=latest
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
EOF
```

### 2. SSL/TLS Configuration

```nginx
# SSL configuration
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

## 📊 Monitoring and Logging

### 1. Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check MLflow health
curl http://localhost:5000/health

# Check monitoring status
curl http://localhost:8000/monitoring/status
```

### 2. Data Drift Monitoring

The API includes built-in data drift detection that monitors incoming predictions against the training data distribution.

#### Monitoring Configuration
```bash
# Check current monitoring configuration
curl http://localhost:8000/monitoring/status

# Update monitoring settings
curl -X POST http://localhost:8000/monitoring/config/update \
     -H "Content-Type: application/json" \
     -d '{
       "drift_threshold": 0.05,
       "monitoring_frequency": 100,
       "alert_on_drift": true
     }'
```

#### Drift Detection Methods
- **Kolmogorov-Smirnov (KS) Test**: Detects distribution shifts
- **Maximum Mean Discrepancy (MMD)**: Detects differences in feature space
- **Feature-level Monitoring**: Individual feature drift detection

#### Monitoring Endpoints
```bash
# Get drift history
curl http://localhost:8000/monitoring/drift/history

# Manual drift check
curl -X POST http://localhost:8000/monitoring/drift/check \
     -H "Content-Type: application/json" \
     -d '{
       "feature1": 1.5,
       "feature2": 2.3,
       "feature3": 0.8,
       "category_feature": "category_a"
     }'

# Enable/disable monitoring
curl -X POST http://localhost:8000/monitoring/enable
curl -X POST http://localhost:8000/monitoring/disable
```

### 3. Logging Configuration

```python
# logging.conf
[loggers]
keys=root,pricing_rf

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_pricing_rf]
level=INFO
handlers=consoleHandler,fileHandler
qualname=pricing_rf
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=INFO
formatter=simpleFormatter
args=('pricing_rf.log',)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### 3. Prometheus Metrics

```python
# Add to FastAPI app
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 🔄 CI/CD Pipeline

### 1. GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build Docker image
      run: docker build -t pricing-rf-api:${{ github.sha }} .
    - name: Deploy to production
      run: |
        # Deploy to your production environment
        kubectl apply -f k8s/
```

### 2. Docker Registry

```bash
# Push to registry
docker tag pricing-rf-api:latest your-registry/pricing-rf-api:v1.0.0
docker push your-registry/pricing-rf-api:v1.0.0
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Check MLflow connection
   curl http://localhost:5000/health
   
   # Check model registry
   mlflow models list
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase memory limits
   docker run --memory=2g pricing-rf-api
   ```

3. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different port
   docker run -p 8001:8000 pricing-rf-api
   ```

### Debugging Commands

```bash
# Check container logs
docker logs pricing-rf-api

# Check container status
docker ps -a

# Check resource usage
docker stats pricing-rf-api

# Access container shell
docker exec -it pricing-rf-api /bin/bash
```

## 📈 Performance Optimization

### 1. Resource Limits

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### 2. Horizontal Scaling

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pricing-rf-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pricing-rf-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 3. Caching

```python
# Add Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

## 🔄 Backup and Recovery

### 1. MLflow Backup

```bash
# Backup MLflow database
cp mlflow.db mlflow_backup_$(date +%Y%m%d).db

# Backup artifacts
tar -czf mlruns_backup_$(date +%Y%m%d).tar.gz mlruns/
```

### 2. Model Backup

```bash
# Export model
mlflow models export -m "pricing-rf" -v latest --output-dir ./model_backup

# Import model
mlflow models import -m "pricing-rf" --input-dir ./model_backup
```

### 3. Disaster Recovery

```bash
# Restore from backup
cp mlflow_backup_20240101.db mlflow.db
tar -xzf mlruns_backup_20240101.tar.gz

# Restart services
docker-compose restart
```

## 📞 Support

### Monitoring Tools
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **ELK Stack**: Log aggregation
- **Jaeger**: Distributed tracing

### Alerting
- **Slack**: Real-time notifications
- **Email**: Critical alerts
- **PagerDuty**: On-call management

### Documentation
- **API Docs**: http://your-domain.com/docs
- **Health Check**: http://your-domain.com/health
- **Metrics**: http://your-domain.com/metrics
