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
    #
    # 支持传入列表来组合多个区域的 mask:
    #   clipseg_mask_prompt=["hair", "clothes", "scarf"]
    #
    # mask_combine_mode:
    #   - "union": 并集（OR），任意 prompt 匹配即为 1
    #   - "intersection": 交集（AND），所有 prompt 都匹配才为 1
    #
    # mask_invert:
    #   - False: prompt 区域 mask=1（保持不变）
    #   - True:  prompt 区域 mask=0（允许编辑）

    result = run_edit_3d_bundle(
        k3d_wrapper,
        prompt_src=src_prompt,
        prompt_tgt=tgt_prompt,
        p2p_tau=p2p_tau,
        # T2I Mask 参数（可选）
        use_t2i_mask=False,
        return_mask=True,
        # CLIPSeg Spatial Mask 参数
        clipseg_mask_prompt=["scarf", "clothes"],  # 保持围巾和衣服区域
        mask_threshold=0.5,
        mask_invert=False,            # False: 这些区域 mask=1（保持）
        mask_combine_mode="union",    # OR: 任意匹配即保持
    )

    # 解析返回值
    # return_mask=True 时返回 8 个值:
    # (bundle_src, bundle_tgt, save_path_src, save_path_tgt,
    #  t2i_mask, save_path_t2i_mask, clipseg_mask, save_path_clipseg_mask)
    if len(result) == 8:
        (src_tensor, tgt_tensor, src_save_path, tgt_save_path,
         t2i_mask, t2i_mask_path,
         clipseg_mask, clipseg_mask_path) = result

        print(f"\n=== Mask 信息 ===")
        if t2i_mask is not None:
            print(f"T2I attention mask: {t2i_mask.shape} -> {t2i_mask_path}")
        if clipseg_mask is not None:
            print(f"CLIPSeg mask: {clipseg_mask.shape} -> {clipseg_mask_path}")
    else:
        src_tensor, tgt_tensor, src_save_path, tgt_save_path = result
        t2i_mask_path = None
        clipseg_mask_path = None

    print(f"\nedit_3d_bundle time: {time.time() - end}")

    # 复制结果到输出目录
    timestamp = int(time.time())
    os.system(f'cp -f {src_save_path} {OUT_DIR}/text_to_3d/{name}_tau{p2p_tau}_src_3d_bundle_{timestamp}.png')
    os.system(f'cp -f {tgt_save_path} {OUT_DIR}/text_to_3d/{name}_tau{p2p_tau}_tgt_3d_bundle_{timestamp}.png')

    # 复制 mask 文件
    if len(result) == 8:
        if t2i_mask_path:
            os.system(f'cp -f {t2i_mask_path} {OUT_DIR}/text_to_3d/{name}_tau{p2p_tau}_t2i_mask_{timestamp}.png')
        if clipseg_mask_path:
            os.system(f'cp -f {clipseg_mask_path} {OUT_DIR}/text_to_3d/{name}_tau{p2p_tau}_clipseg_mask_{timestamp}.png')

    print("\n=== CLIPSeg Mask 使用说明 ===")
    print("clipseg_mask_prompt: 指定要识别的区域")
    print("  - 单个: 'hair'")
    print("  - 多个: ['hair', 'clothes', 'scarf']")
    print("")
    print("mask_combine_mode:")
    print("  - 'union': 并集（OR），任意 prompt 匹配即保持")
    print("  - 'intersection': 交集（AND），所有 prompt 都匹配才保持")
    print("")
    print("mask_invert:")
    print("  - False: prompt 区域 mask=1（保持不变）")
    print("  - True:  prompt 区域 mask=0（允许编辑）")
    

