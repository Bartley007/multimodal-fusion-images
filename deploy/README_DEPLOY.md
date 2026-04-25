# 服务器部署指南

本指南将帮助您将多模态融合预测系统部署到生产服务器上。

## 📋 部署前准备

### 系统要求
- Ubuntu 20.04+ / CentOS 7+ / Debian 10+
- Python 3.8+
- 8GB+ RAM
- GPU (可选，用于加速)
- 10GB+ 磁盘空间

### 端口要求
- 5000: Flask应用端口（内部）
- 80/443: HTTP/HTTPS端口（Nginx）

## 🚀 部署步骤

### 1. 服务器环境准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python和pip
sudo apt install python3 python3-pip python3-venv -y

# 安装Nginx
sudo apt install nginx -y

# 安装Git（如果需要从仓库克隆）
sudo apt install git -y

# 安装GPU驱动（如果有GPU）
# NVIDIA驱动安装请参考: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
```

### 2. 创建应用用户

```bash
# 创建专用用户
sudo useradd -m -s /bin/bash mlapp
sudo mkdir -p /opt/mlapp
sudo chown mlapp:mlapp /opt/mlapp
```

### 3. 上传项目文件

```bash
# 方式1: 使用SCP上传
scp -r . mlapp@your-server:/opt/mlapp/

# 方式2: 使用Git克隆
sudo -u mlapp git clone <your-repo-url> /opt/mlapp
```

### 4. 安装Python依赖

```bash
cd /opt/mlapp
sudo -u mlapp python3 -m venv venv
sudo -u mlapp ./venv/bin/pip install --upgrade pip
sudo -u mlapp ./venv/bin/pip install -r requirements.txt
sudo -u mlapp ./venv/bin/pip install gunicorn  # 生产环境WSGI服务器
```

### 5. 配置环境变量

```bash
sudo -u mlapp nano /opt/mlapp/.env
```

添加以下内容：
```
FLASK_ENV=production
FLASK_APP=api_server.py
SECRET_KEY=your-secret-key-here-change-this
HOST=127.0.0.1
PORT=5000
WORKERS=4
```

### 6. 创建必要的目录

```bash
sudo -u mlapp mkdir -p /opt/mlapp/{uploads,results,models,logs}
sudo chmod 755 /opt/mlapp/{uploads,results,models,logs}
```

## 🔧 配置Gunicorn

### 创建Gunicorn配置文件

```bash
sudo -u mlapp nano /opt/mlapp/gunicorn_config.py
```

参考 `deploy/gunicorn_config.py` 文件

### 创建Systemd服务

```bash
sudo nano /etc/systemd/system/mlapp.service
```

参考 `deploy/mlapp.service` 文件

```bash
# 重新加载systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start mlapp

# 设置开机自启
sudo systemctl enable mlapp

# 查看状态
sudo systemctl status mlapp
```

## 🌐 配置Nginx反向代理

```bash
sudo nano /etc/nginx/sites-available/mlapp
```

参考 `deploy/nginx.conf` 文件

```bash
# 创建符号链接
sudo ln -s /etc/nginx/sites-available/mlapp /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启Nginx
sudo systemctl restart nginx
```

## 🔒 配置SSL证书（HTTPS）

### 使用Let's Encrypt

```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx -y

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期测试
sudo certbot renew --dry-run
```

## 📊 监控和日志

### 查看应用日志

```bash
# Gunicorn日志
sudo journalctl -u mlapp -f

# Nginx日志
sudo tail -f /var/log/nginx/mlapp_access.log
sudo tail -f /var/log/nginx/mlapp_error.log

# 应用日志
tail -f /opt/mlapp/logs/app.log
```

### 设置日志轮转

```bash
sudo nano /etc/logrotate.d/mlapp
```

参考 `deploy/logrotate.conf` 文件

## 🔄 更新部署

```bash
# 1. 备份当前版本
sudo -u mlapp cp -r /opt/mlapp /opt/mlapp.backup.$(date +%Y%m%d)

# 2. 更新代码
cd /opt/mlapp
sudo -u mlapp git pull  # 或上传新文件

# 3. 更新依赖
sudo -u mlapp ./venv/bin/pip install -r requirements.txt

# 4. 重启服务
sudo systemctl restart mlapp
```

## 🛡️ 安全建议

1. **防火墙配置**
```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

2. **文件权限**
```bash
# 确保敏感文件权限正确
chmod 600 /opt/mlapp/.env
chmod 755 /opt/mlapp
```

3. **定期更新**
```bash
sudo apt update && sudo apt upgrade -y
```

## 🐛 故障排查

### 服务无法启动
```bash
# 查看详细错误
sudo journalctl -u mlapp -n 50

# 检查端口占用
sudo netstat -tlnp | grep 5000

# 检查Python环境
sudo -u mlapp /opt/mlapp/venv/bin/python --version
```

### Nginx 502错误
```bash
# 检查Gunicorn是否运行
sudo systemctl status mlapp

# 检查socket文件权限
ls -l /opt/mlapp/mlapp.sock
```

### 内存不足
```bash
# 减少worker数量
# 编辑 gunicorn_config.py，减少 workers
```

## 📈 性能优化

1. **增加Gunicorn workers**
   - CPU核心数 × 2 + 1

2. **启用Nginx缓存**
   - 参考nginx配置中的缓存设置

3. **使用Redis缓存**（可选）
   - 缓存模型预测结果

4. **CDN加速**（可选）
   - 静态资源使用CDN

## 📞 支持

如遇问题，请检查：
1. 日志文件
2. 系统资源使用情况
3. 网络连接
4. 文件权限


