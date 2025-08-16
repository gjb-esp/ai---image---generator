from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import uuid
import json
from datetime import datetime
from PIL import Image, ImageEnhance
import random
import io
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np  # 添加入口处的numpy导入

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 用于会话管理

# 配置
UPLOAD_FOLDER = 'generated_images'
FEEDBACK_FOLDER = 'feedback'
STATS_FOLDER = 'visitor_stats'  # 新增：访问统计文件夹
ADMIN_CODE = 'admin_activation_code'  # 管理员激活码
RANDOM_PROMPTS = [
    "一只会飞的紫色大象在月球上",
    "未来城市的空中交通",
    "海底的神秘宫殿",
    "蒸汽朋克风格的森林",
    "漂浮在云端的城堡",
    "外星人访问古代埃及",
    "赛博朋克风格的猫咪",
    "下雪天的东京街道"
]

# 创建必要的文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)
os.makedirs(STATS_FOLDER, exist_ok=True)  # 新增：创建访问统计文件夹

# 图像生成模型配置
device = torch.device("cpu")  # 确保在CPU上运行，适应更多环境
noise_dim = 64
image_size = 24

# 生成器模型定义
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = image_size // 4  # 24→6×6初始特征图
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim, 64 * self.init_size **2),
            nn.BatchNorm1d(64 * self.init_size** 2),
            nn.ReLU(inplace=True)
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.l1(x)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        return self.conv_blocks(out)

# 加载预训练模型
def load_generator_model():
    generator = Generator().to(device)
    try:
        # 尝试加载已训练的模型
        if os.path.exists("generator_best.pth"):
            generator.load_state_dict(torch.load("generator_best.pth", map_location=device))
            print("已加载预训练模型")
        else:
            print("未找到预训练模型，使用随机初始化模型")
    except Exception as e:
        print(f"模型加载失败: {e}，使用随机初始化模型")
    generator.eval()
    return generator

# 初始化生成器
generator = load_generator_model()

# 新增：记录访问统计
def log_visit():
    """记录网站访问情况"""
    # 获取访问信息
    visitor_ip = request.remote_addr
    user_agent = request.user_agent.string
    current_time = datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")  # 按日期存储
    
    # 构建访问记录
    visit_record = {
        'id': str(uuid.uuid4()),
        'timestamp': current_time.isoformat(),
        'ip': visitor_ip,
        'user_agent': user_agent,
        'path': request.path
    }
    
    # 按日期保存访问记录
    stats_file = os.path.join(STATS_FOLDER, f"{date_str}.json")
    
    # 读取现有记录
    visits = []
    if os.path.exists(stats_file):
        with open(stats_file, 'r', encoding='utf-8') as f:
            try:
                visits = json.load(f)
            except json.JSONDecodeError:
                visits = []
    
    # 添加新记录
    visits.append(visit_record)
    
    # 保存更新后的记录
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(visits, f, ensure_ascii=False, indent=2)

# 新增：获取访问统计数据
def get_visit_stats():
    """获取网站访问统计数据"""
    stats = {
        'total_visits': 0,
        'daily_visits': {},
        'popular_pages': {},
        'recent_visits': []
    }
    
    # 遍历所有统计文件
    for filename in os.listdir(STATS_FOLDER):
        if filename.endswith('.json'):
            date = filename.replace('.json', '')
            filepath = os.path.join(STATS_FOLDER, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    visits = json.load(f)
                    stats['daily_visits'][date] = len(visits)
                    stats['total_visits'] += len(visits)
                    
                    # 收集最近的访问记录（限制数量）
                    for visit in visits[-5:]:  # 取每天最后5条记录
                        stats['recent_visits'].append(visit)
                    
                    # 统计页面访问量
                    for visit in visits:
                        path = visit['path']
                        if path in stats['popular_pages']:
                            stats['popular_pages'][path] += 1
                        else:
                            stats['popular_pages'][path] = 1
                except json.JSONDecodeError:
                    continue
    
    # 对最近访问排序
    stats['recent_visits'].sort(key=lambda x: x['timestamp'], reverse=True)
    stats['recent_visits'] = stats['recent_visits'][:20]  # 只保留最近20条
    
    # 对页面访问量排序
    stats['popular_pages'] = dict(sorted(
        stats['popular_pages'].items(), 
        key=lambda item: item[1], 
        reverse=True
    ))
    
    return stats

# AI图像生成函数
def generate_image(prompt=None):
    """使用PyTorch模型生成图像"""
    with torch.no_grad():
        noise = torch.randn(1, noise_dim, device=device)
        generated_img = generator(noise).squeeze(0).cpu()
        generated_img = generated_img.permute(1, 2, 0).numpy()
        generated_img = (generated_img * 0.5 + 0.5) * 255  # 反归一化
        generated_img = np.clip(generated_img, 0, 255).astype(np.uint8)

    # 图像后处理，提升视觉效果
    pil_img = Image.fromarray(generated_img)
    pil_img = pil_img.resize((512, 512), Image.LANCZOS)  # 高质量放大
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)  # 提升对比度
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.5)  # 增强锐度

    # 保存图像到内存
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # 同时保存到文件系统
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    pil_img.save(filepath)
    
    return img_byte_arr, filename

# 保存用户反馈
def save_feedback(username, email, message, rating):
    feedback_id = str(uuid.uuid4())
    feedback = {
        'id': feedback_id,
        'username': username,
        'email': email,
        'message': message,
        'rating': rating,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(FEEDBACK_FOLDER, f"{feedback_id}.json"), 'w', encoding='utf-8') as f:
        json.dump(feedback, f, ensure_ascii=False, indent=4)
    
    return feedback_id

# 获取所有用户反馈
def get_all_feedback():
    feedback_list = []
    for filename in os.listdir(FEEDBACK_FOLDER):
        if filename.endswith('.json'):
            with open(os.path.join(FEEDBACK_FOLDER, filename), 'r', encoding='utf-8') as f:
                feedback = json.load(f)
                feedback_list.append(feedback)
    
    # 按时间排序，最新的在前
    feedback_list.sort(key=lambda x: x['timestamp'], reverse=True)
    return feedback_list

@app.route('/')
def index():
    # 新增：记录访问
    log_visit()
    
    is_admin = session.get('is_admin', False)
    return render_template('index.html', is_admin=is_admin)

@app.route('/generate', methods=['POST'])
def generate():
    # 新增：记录访问
    log_visit()
    
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': '请输入提示词'}), 400
    
    try:
        # 生成图像
        img_byte_arr, filename = generate_image(prompt)
        
        # 返回图像URL
        return jsonify({
            'success': True,
            'image_url': f'/generated/{filename}'
        })
    except Exception as e:
        return jsonify({'error': f'生成图像时出错: {str(e)}'}), 500

@app.route('/generate-random')
def generate_random():
    # 新增：记录访问
    log_visit()
    
    try:
        # 随机选择一个提示词
        prompt = random.choice(RANDOM_PROMPTS)
        
        # 生成图像
        img_byte_arr, filename = generate_image(prompt)
        
        # 返回图像URL和使用的提示词
        return jsonify({
            'success': True,
            'image_url': f'/generated/{filename}',
            'prompt': prompt
        })
    except Exception as e:
        return jsonify({'error': f'生成图像时出错: {str(e)}'}), 500

@app.route('/generated/<filename>')
def get_generated_image(filename):
    # 新增：记录访问
    log_visit()
    
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='image/png')

@app.route('/save-image/<filename>')
def save_image(filename):
    # 新增：记录访问
    log_visit()
    
    # 提供图片下载
    return send_file(
        os.path.join(UPLOAD_FOLDER, filename),
        mimetype='image/png',
        as_attachment=True,
        download_name=filename
    )

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    # 新增：记录访问
    log_visit()
    
    data = request.json
    username = data.get('username', '匿名用户')
    email = data.get('email', '')
    message = data.get('message', '')
    rating = data.get('rating', 3)
    
    if not message:
        return jsonify({'error': '反馈内容不能为空'}), 400
    
    feedback_id = save_feedback(username, email, message, rating)
    return jsonify({
        'success': True,
        'message': '反馈提交成功，感谢您的建议！',
        'feedback_id': feedback_id
    })

@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    code = data.get('code', '')
    
    if code == ADMIN_CODE:
        session['is_admin'] = True
        return jsonify({
            'success': True,
            'message': '管理员模式已激活'
        })
    else:
        return jsonify({
            'success': False,
            'message': '验证码不正确'
        }), 401

@app.route('/admin/feedback')
def admin_feedback():
    if not session.get('is_admin', False):
        return jsonify({'error': '未授权访问'}), 403
    
    feedback_list = get_all_feedback()
    return jsonify({
        'success': True,
        'feedback': feedback_list
    })

# 新增：获取访问统计的API
@app.route('/admin/stats')
def admin_stats():
    if not session.get('is_admin', False):
        return jsonify({'error': '未授权访问'}), 403
    
    stats = get_visit_stats()
    return jsonify({
        'success': True,
        'stats': stats
    })

@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return jsonify({
        'success': True,
        'message': '已退出管理员模式'
    })

if __name__ == '__main__':
    app.run(debug=True)
