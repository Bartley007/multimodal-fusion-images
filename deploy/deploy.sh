#!/bin/bash
# 自动化部署脚本
# 使用方法: bash deploy/deploy.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "多模态融合预测系统 - 自动化部署"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置变量
APP_USER="mlapp"
APP_DIR="/opt/mlapp"
APP_NAME="mlapp"
DOMAIN="your-domain.com"  # 修改为您的域名

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}错误: 请使用root权限运行此脚本${NC}"
    exit 1
fi

echo -e "${YELLOW}[1/8] 创建应用用户...${NC}"
if ! id "$APP_USER" &>/dev/null; then
    useradd -m -s /bin/bash $APP_USER
    echo -e "${GREEN}✓ 用户创建成功${NC}"
else
    echo -e "${GREEN}✓ 用户已存在${NC}"
fi

echo -e "${YELLOW}[2/8] 创建应用目录...${NC}"
mkdir -p $APP_DIR
chown $APP_USER:$APP_USER $APP_DIR

echo -e "${YELLOW}[3/8] 安装系统依赖...${NC}"
apt update
apt install -y python3 python3-pip python3-venv nginx git

echo -e "${YELLOW}[4/8] 设置Python虚拟环境...${NC}"
cd $APP_DIR
sudo -u $APP_USER python3 -m venv venv
sudo -u $APP_USER ./venv/bin/pip install --upgrade pip
sudo -u $APP_USER ./venv/bin/pip install -r requirements.txt
sudo -u $APP_USER ./venv/bin/pip install gunicorn

echo -e "${YELLOW}[5/8] 创建必要目录...${NC}"
sudo -u $APP_USER mkdir -p $APP_DIR/{uploads,results,models,logs}
chmod 755 $APP_DIR/{uploads,results,models,logs}

echo -e "${YELLOW}[6/8] 配置Gunicorn...${NC}"
# 复制配置文件
cp deploy/gunicorn_config.py $APP_DIR/
chown $APP_USER:$APP_USER $APP_DIR/gunicorn_config.py

# 修改gunicorn配置中的socket路径
sed -i "s|unix:/opt/mlapp/mlapp.sock|unix:$APP_DIR/mlapp.sock|g" $APP_DIR/gunicorn_config.py

echo -e "${YELLOW}[7/8] 配置Systemd服务...${NC}"
# 复制service文件并修改路径
cp deploy/mlapp.service /etc/systemd/system/$APP_NAME.service
sed -i "s|/opt/mlapp|$APP_DIR|g" /etc/systemd/system/$APP_NAME.service
sed -i "s|mlapp|$APP_USER|g" /etc/systemd/system/$APP_NAME.service

systemctl daemon-reload
systemctl enable $APP_NAME
echo -e "${GREEN}✓ Systemd服务配置完成${NC}"

echo -e "${YELLOW}[8/8] 配置Nginx...${NC}"
# 复制nginx配置
cp deploy/nginx.conf /etc/nginx/sites-available/$APP_NAME
sed -i "s|your-domain.com|$DOMAIN|g" /etc/nginx/sites-available/$APP_NAME
sed -i "s|/opt/mlapp|$APP_DIR|g" /etc/nginx/sites-available/$APP_NAME

# 创建符号链接
ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/

# 测试nginx配置
if nginx -t; then
    echo -e "${GREEN}✓ Nginx配置正确${NC}"
else
    echo -e "${RED}✗ Nginx配置错误，请检查${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "部署完成！"
echo "==========================================${NC}"
echo ""
echo "下一步操作："
echo "1. 将项目文件复制到 $APP_DIR"
echo "2. 修改 $APP_DIR/gunicorn_config.py 中的配置"
echo "3. 修改 /etc/nginx/sites-available/$APP_NAME 中的域名"
echo "4. 启动服务: systemctl start $APP_NAME"
echo "5. 重启Nginx: systemctl restart nginx"
echo "6. 配置SSL证书: certbot --nginx -d $DOMAIN"
echo ""
echo "查看日志: journalctl -u $APP_NAME -f"


