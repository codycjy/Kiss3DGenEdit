from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_edit_3d_bundle, init_minimum_wrapper_from_config
import os
from pipeline.utils import logger, TMP_DIR, OUT_DIR
import time

if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_DIR, 'text_to_3d'), exist_ok=True)
    k3d_wrapper = init_minimum_wrapper_from_config('./pipeline/pipeline_config/default.yaml')

    src_prompt = 'A charming 3D doll of a young girl styled as a Hogwarts student, ' \
                'with a whimsical, magical flair. Front view features a radiant smile, ' \
                'large, twinkling eyes, and a Gryffindor scarf with a cozy, fluffy texture; ' \
                'left side view showcases a miniature wand tucked behind her ear, ' \
                'with a tiny golden snitch hanging from a string; ' \
                'rear view presents a backpack adorned with house badges and a small owl figurine; ' \
                'right side view displays a playful broomstick with feathers fluttering gently, ' \
                'alongside a tiny house elf mascot. No background. ' \
                'Arranged in a 2x4 grid with RGB images on top and normal maps below.'

    tgt_prompt = 'A charming 3D doll of a young girl styled as a Hogwarts student, ' \
                'with a whimsical, magical flair. Front view features a sad expression, ' \
                'large, twinkling eyes, and a Gryffindor scarf with a cozy, fluffy texture; ' \
                'left side view showcases a miniature wand tucked behind her ear, ' \
                'with a tiny golden snitch hanging from a string; ' \
                'rear view presents a backpack adorned with house badges and a small owl figurine; ' \
                'right side view displays a playful broomstick with feathers fluttering gently, ' \
                'alongside a tiny house elf mascot. No background.' \
                'Arranged in a 2x4 grid with RGB images on top and normal maps below.'

    name = "doll_girl"
    os.system(f'rm -rf {TMP_DIR}/*')
    end = time.time()
    p2p_tau = 0.2

    # ========== 使用 CLIPSeg Mask 进行编辑 ==========
    # CLIPSeg 会根据 clipseg_mask_prompt 识别图像中的区域
    # mask_invert=False: prompt 匹配的区域保持不变
    # mask_invert=True:  prompt 匹配的区域允许编辑

    result = run_edit_3d_bundle(
        k3d_wrapper,
        prompt_src=src_prompt,
        prompt_tgt=tgt_prompt,
        p2p_tau=p2p_tau,
        # T2I Mask 参数（可选，用于自动生成 mask）
        use_t2i_mask=False,
        return_mask=True,
        # CLIPSeg Spatial Mask 参数
        clipseg_mask_prompt="scarf",  # 保持围巾区域不变，只编辑表情
        mask_threshold=0.5,           # CLIPSeg 二值化阈值
        mask_invert=False,            # False: scarf 区域 mask=1（保持）
    )

    # 解析返回值
    if len(result) == 6:
        src_tensor, tgt_tensor, src_save_path, tgt_save_path, t2i_mask, mask_save_path = result
        print(f"Mask saved to: {mask_save_path}")
        print(f"Mask shape: {t2i_mask.shape if t2i_mask is not None else 'None'}")
    else:
        src_tensor, tgt_tensor, src_save_path, tgt_save_path = result

    print(f"edit_3d_bundle time: {time.time() - end}")

    # 复制结果到输出目录
    timestamp = int(time.time())
    os.system(f'cp -f {src_save_path} {OUT_DIR}/text_to_3d/{name}_tau{p2p_tau}_src_3d_bundle_{timestamp}.png')
    os.system(f'cp -f {tgt_save_path} {OUT_DIR}/text_to_3d/{name}_tau{p2p_tau}_tgt_3d_bundle_{timestamp}.png')
    if len(result) == 6:
        os.system(f'cp -f {mask_save_path} {OUT_DIR}/text_to_3d/{name}_tau{p2p_tau}_mask_{timestamp}.png')

    print("\n=== CLIPSeg Mask 使用说明 ===")
    print("clipseg_mask_prompt: 指定要识别的区域（如 'face', 'hair', 'background'）")
    print("mask_invert=False: prompt 区域保持不变")
    print("mask_invert=True:  prompt 区域允许编辑")
    

