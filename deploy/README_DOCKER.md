# Docker部署指南

使用Docker可以快速部署应用，无需手动配置环境。

## 📋 前置要求

- Docker 20.10+
- Docker Compose 2.0+

## 🚀 快速开始

### 1. 构建镜像

```bash
cd deploy
docker-compose build
```

### 2. 启动服务

```bash
docker-compose up -d
```

### 3. 查看日志

```bash
# 应用日志
docker-compose logs -f mlapp

# Nginx日志
docker-compose logs -f nginx
```

### 4. 停止服务

```bash
docker-compose down
```

## 🔧 配置说明

### 修改端口

编辑 `docker-compose.yml`:

```yaml
ports:
  - "8080:5000"  # 将8080改为您想要的端口
```

### 修改资源限制

编辑 `docker-compose.yml` 中的 `deploy.resources` 部分。

### 环境变量

在 `docker-compose.yml` 的 `environment` 部分添加或修改环境变量。

## 📊 监控

```bash
# 查看容器状态
docker-compose ps

# 查看资源使用
docker stats

# 进入容器
docker-compose exec mlapp bash
```

## 🔄 更新部署

```bash
# 1. 拉取最新代码
git pull

# 2. 重新构建
docker-compose build

# 3. 重启服务
docker-compose up -d
```

## 🐛 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker-compose logs mlapp

# 检查配置
docker-compose config
```

### 端口冲突

```bash
# 检查端口占用
netstat -tlnp | grep 5000

# 修改docker-compose.yml中的端口映射
```

### 内存不足

```bash
# 减少workers数量
# 编辑Dockerfile中的CMD，减少--workers参数
```


