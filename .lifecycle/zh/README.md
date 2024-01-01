
# Team Unmute 手语识别项目的安装与部署指南

## 简介
本 README 指南将帮助您设置并运行 Team Unmute 的实时手语识别项目。这个 Python 应用程序整合了 OpenCV、Keras、MediaPipe、gTTS 和 Pygame，用于实时手势识别和音频反馈。

## 先决条件
- Python（版本 3.6 或更高）
- Pip（Python 包管理器）

## 安装步骤

### 1. 克隆仓库
从 GitHub 克隆 Team Unmute 仓库。
```bash
git clone https://github.com/khemagarwal/Team-Unmute
cd Team-Unmute/Milestone\ 4\ Model
```

### 2. 设置虚拟环境（可选但推荐）
为了依赖管理，创建并激活一个虚拟环境。
```bash
python -m venv venv
source venv/bin/activate  # 在 Windows 上使用 `venv\Scripts\activate`
```

### 3. 安装所需库
使用 pip 安装所需的 Python 库。
```bash
pip install numpy opencv-python keras mediapipe pygame gTTS
```

### 4. 加载预训练模型
确保预训练模型（`actionsFinalFinal.h5`）被放置在 `Team-Unmute/Milestone 4 Model` 目录中。

### 5. 验证网络摄像头访问权限
检查您的开发环境是否有权限访问网络摄像头进行视频输入。

## 运行应用

1. 通过运行主 Python 脚本启动应用程序。
   ```bash
   python Team-Unmute/Milestone\ 4\ Model/main.py
   ```

2. 应用程序将启动网络摄像头并开始进行手势识别处理。

3. 在网络摄像头前进行手语手势，以查看识别结果和相应的音频反馈。

4. 要退出，当应用程序窗口处于激活状态时按 'q' 键。

## 应用特性和用户指南

### 手语识别
- 应用程序能够实时检测特定的手语手势。

### 音频播报
- 识别的手势会触发对应手语的音频播报。

### 计时和冷却
- 每次手势识别后都会有 2 秒的冷却期，在此期间不识别新的手势。

### 注意点
- 屏幕上的视觉指示器会显示已识别的手势和冷却倒计时。

### 使用最佳实践
- 确保手部清晰可见并有良好的光线条件。
- 在进行下一个手势之前，请等待冷却时间结束。

## 故障排除
- **网络摄像头访问**：确保网络摄像头已正确连接并被应用程序授权使用。
- **库依赖问题**：如果遇到库的问题，请参考它们的官方文档。
- **模型兼容性**：确认 Keras 模型与您的 Keras 版本兼容。

## 支持
如需帮助或反馈，请联系 hgcai@cmu.edu。
