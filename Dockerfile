# 基础镜像
FROM python:3.6

# 工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装依赖
# RUN pip install -r requirements.txt

# 指定项目启动命令
# CMD ["python", "main.py"]
