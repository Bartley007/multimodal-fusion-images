#!/bin/bash
# 快速部署脚本（适用于已有服务器环境）
# 使用方法: bash deploy/quick_deploy.sh

set -e

APP_DIR="/opt/mlapp"
APP_USER="mlapp"

echo "快速部署多模态融合预测系统..."
echo ""

# 检查是否为root
if [ "$EUID" -ne 0 ]; then 
    echo "错误: 请使用root权限运行"
    exit 1
fi

# 1. 创建用户和目录
if ! id "$APP_USER" &>/dev/null; then
    useradd -m -s /bin/bash $APP_USER
fi
mkdir -p $APP_DIR
chown $APP_USER:$APP_USER $APP_DIR

# 2. 复制文件到部署目录
echo "复制文件..."
cp -r . $APP_DIR/
chown -R $APP_USER:$APP_USER $APP_DIR

# 3. 安装依赖
echo "安装Python依赖..."
cd $APP_DIR
sudo -u $APP_USER python3 -m venv venv
sudo -u $APP_USER ./venv/bin/pip install --upgrade pip
sudo -u $APP_USER ./venv/bin/pip install -r requirements.txt
sudo -u $APP_USER ./venv/bin/pip install gunicorn

# 4. 创建目录
sudo -u $APP_USER mkdir -p uploads results models logs

# 5. 配置Gunicorn
if [ -f "deploy/gunicorn_config.py" ]; then
    cp deploy/gunicorn_config.py $APP_DIR/
fi

# 6. 配置Systemd
if [ -f "deploy/mlapp.service" ]; then
    cp deploy/mlapp.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable mlapp
    echo "启动服务..."
    systemctl start mlapp
fi

echo ""
echo "部署完成！"
echo "查看状态: systemctl status mlapp"
echo "查看日志: journalctl -u mlapp -f"


