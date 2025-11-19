import os
import torch
from PIL import Image
from pipeline.kiss3d_wrapper import init_wrapper_from_config

def main():
    # 1. 初始化 wrapper
    # 确保配置文件路径正确
    config_path = './pipeline/pipeline_config/default.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    print("Initializing Kiss3D wrapper...")
    k3d_wrapper = init_wrapper_from_config(config_path)

    # 2. 设置输入参数
    input_image_path = "examples/1.png"  # 替换为你想要编辑的图片路径
    if not os.path.exists(input_image_path):
        print(f"Input image not found at {input_image_path}")
        # 如果找不到示例图片，创建一个简单的测试图片
        print("Creating a dummy test image...")
        Image.new('RGB', (512, 512), color='red').save("test_input.png")
        input_image_path = "test_input.png"

    prompt = "Change the color to blue"  # 你的编辑提示词
    output_path = "edited_output.png"
    
    # SDE-Edit 参数
    strength = 0.75       # 0.0 到 1.0，越大变化越大
    steps = 20            # 推理步数
    guidance_scale = 3.5  # 提示词引导系数
    seed = 42             # 随机种子

    print(f"Processing image: {input_image_path}")
    print(f"Prompt: '{prompt}'")
    
    # 3. 加载图片
    input_image = Image.open(input_image_path).convert("RGB")

    # 4. 调用 Img2Img 编辑功能
    result_image = k3d_wrapper.run_flux_img2img(
        prompt=prompt,
        image=input_image,
        strength=strength,
        num_inference_steps=steps,
        seed=seed,
        guidance_scale=guidance_scale
    )

    # 5. 保存结果
    if result_image:
        result_image.save(output_path)
        print(f"Successfully saved edited image to {output_path}")
    else:
        print("Failed to generate image.")

if __name__ == "__main__":
    main()

