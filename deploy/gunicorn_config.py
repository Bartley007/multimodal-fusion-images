# -*- coding: utf-8 -*-
"""
Gunicorn配置文件 - 生产环境
"""

import multiprocessing
import os

# 服务器socket
bind = "127.0.0.1:5000"
# 或使用Unix socket（性能更好）
# bind = "unix:/opt/mlapp/mlapp.sock"

# Worker进程数
# 推荐: (2 × CPU核心数) + 1
workers = multiprocessing.cpu_count() * 2 + 1

# Worker类型
worker_class = "sync"  # 或 "gevent" (需要安装gevent)

# 每个worker的线程数（如果使用threads）
threads = 2

# Worker超时时间（秒）
timeout = 120

# 保持连接
keepalive = 5

# 最大请求数（防止内存泄漏）
max_requests = 1000
max_requests_jitter = 50

# 日志配置
accesslog = "/opt/mlapp/logs/gunicorn_access.log"
errorlog = "/opt/mlapp/logs/gunicorn_error.log"
loglevel = "info"

# 进程名称
proc_name = "mlapp"

# 用户和组（需要root权限启动）
# user = "mlapp"
# group = "mlapp"

# 工作目录
chdir = "/opt/mlapp"

# Python路径
pythonpath = "/opt/mlapp"

# 预加载应用（节省内存，但可能导致worker间不共享）
preload_app = False

# 守护进程（由systemd管理，设为False）
daemon = False

# PID文件
pidfile = "/opt/mlapp/gunicorn.pid"

# 临时目录
tmp_upload_dir = "/opt/mlapp/uploads"

# 环境变量
raw_env = [
    'FLASK_ENV=production',
]

def when_ready(server):
    """服务器启动时调用"""
    server.log.info("Server is ready. Spawning workers")

def on_starting(server):
    """服务器启动前调用"""
    server.log.info("Starting server")

def on_reload(server):
    """重载时调用"""
    server.log.info("Reloading server")

def worker_int(worker):
    """Worker中断时调用"""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Fork worker前调用"""
    pass

def post_fork(server, worker):
    """Fork worker后调用"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    """Exec新进程前调用"""
    server.log.info("Forked child, re-executing")

def when_ready(server):
    """服务器就绪时调用"""
    server.log.info("Server is ready. Spawning workers")

def worker_abort(worker):
    """Worker异常退出时调用"""
    worker.log.info("Worker received SIGABRT signal")


