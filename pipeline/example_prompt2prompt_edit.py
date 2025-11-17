"""
Prompt-to-Prompt 3D Editing Example

This script demonstrates how to use prompt-to-prompt editing to modify 3D assets
while preserving their structure through attention control.

Usage:
    python pipeline/example_prompt2prompt_edit.py
"""

from pipeline.kiss3d_wrapper import init_wrapper_from_config
from pipeline.utils import logger, TMP_DIR, OUT_DIR, preprocess_input_image
from PIL import Image
import os
import torch
import torchvision
import time


def run_prompt2prompt_edit(
    k3d_wrapper,
    input_image_path,
    source_prompt,
    target_prompt,
    p2p_replace_steps=0.5,
    p2p_blend_ratio=0.8
):
    """
    Run prompt-to-prompt editing for 3D generation.

    This uses pure Prompt2Prompt attention control (no ControlNet) to edit the 3D asset
    while preserving structure through cross-attention manipulation.

    Args:
        k3d_wrapper: Kiss3D wrapper instance
        input_image_path: Path to input image
        source_prompt: Original prompt describing the input
        target_prompt: Target prompt for editing
        p2p_replace_steps: Timestep threshold for switching prompts (0-1)
        p2p_blend_ratio: Attention blending ratio (0-1)

    Returns:
        gen_save_path: Path to generated 3D bundle image
        recon_mesh_path: Path to reconstructed 3D mesh

    Note:
        For ControlNet-based editing, use generate_3d_bundle_image_controlnet() instead.
    """
    # Renew UUID for this generation
    k3d_wrapper.renew_uuid()

    # Load and preprocess input image
    input_image = preprocess_input_image(Image.open(input_image_path))
    input_image.save(os.path.join(TMP_DIR, f'{k3d_wrapper.uuid}_input_image.png'))

    # Step 1: Generate reference 3D bundle from input image using Zero123++
    logger.info("Generating reference 3D bundle with Zero123++...")
    reference_3d_bundle_image, reference_save_path = k3d_wrapper.generate_reference_3D_bundle_image_zero123(
        input_image,
        use_mv_rgb=True
    )

    # Step 2: Apply prompt-to-prompt editing (pure P2P, no ControlNet)
    logger.info(f"Applying prompt-to-prompt editing...")
    logger.info(f"  Source prompt: {source_prompt}")
    logger.info(f"  Target prompt: {target_prompt}")
    logger.info(f"  Replace threshold: {p2p_replace_steps}")
    logger.info(f"  Blend ratio: {p2p_blend_ratio}")

    # Generate with prompt-to-prompt only
    gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_prompt2prompt(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        image=reference_3d_bundle_image.unsqueeze(0),
        strength=0.95,
        p2p_replace_steps=p2p_replace_steps,
        p2p_blend_ratio=p2p_blend_ratio,
        lora_scale=1.0,
    )

    # Step 3: 3D reconstruction
    logger.info("Reconstructing 3D mesh from edited bundle...")
    vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo = \
        k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image)

    # Step 4: Mesh optimization (optional)
    logger.info("Optimizing mesh with Isomer...")
    recon_mesh_path = k3d_wrapper.optimize_mesh(
        vertices,
        faces,
        lrm_multi_view_rgb,
        lrm_multi_view_albedo,
        lrm_multi_view_normals
    )

    logger.info(f"✓ Generated 3D bundle saved to: {gen_save_path}")
    logger.info(f"✓ Reconstructed mesh saved to: {recon_mesh_path}")

    return gen_save_path, recon_mesh_path


if __name__ == "__main__":
    # Initialize wrapper
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/default.yaml')

    # Clean temporary directory
    os.system(f'rm -rf {TMP_DIR}/*')
    os.makedirs(os.path.join(OUT_DIR, 'prompt2prompt_edit'), exist_ok=True)

    # ===========================
    # Example 1: Color Change
    # ===========================
    print("\n" + "="*60)
    print("Example 1: Changing car color from red to blue")
    print("="*60)

    start_time = time.time()
    gen_save_path, recon_mesh_path = run_prompt2prompt_edit(
        k3d_wrapper,
        input_image_path='./examples/car.png',  # Replace with your image
        source_prompt="a red sports car",
        target_prompt="a blue sports car",
        p2p_replace_steps=0.5,  # Switch prompts at 50% of denoising
        p2p_blend_ratio=0.8     # Use 80% of stored attention
    )
    print(f"Time elapsed: {time.time() - start_time:.2f}s")

    # Copy results
    os.system(f'cp -f {gen_save_path} {OUT_DIR}/prompt2prompt_edit/car_red_to_blue_bundle.png')
    os.system(f'cp -f {recon_mesh_path} {OUT_DIR}/prompt2prompt_edit/car_red_to_blue.glb')

    # ===========================
    # Example 2: Style Change
    # ===========================
    print("\n" + "="*60)
    print("Example 2: Changing style from modern to vintage")
    print("="*60)

    start_time = time.time()
    gen_save_path, recon_mesh_path = run_prompt2prompt_edit(
        k3d_wrapper,
        input_image_path='./examples/chair.png',  # Replace with your image
        source_prompt="a modern minimalist chair",
        target_prompt="a vintage wooden chair",
        p2p_replace_steps=0.6,  # Later switch for more structural change
        p2p_blend_ratio=0.7     # Lower ratio for more freedom
    )
    print(f"Time elapsed: {time.time() - start_time:.2f}s")

    # Copy results
    os.system(f'cp -f {gen_save_path} {OUT_DIR}/prompt2prompt_edit/chair_modern_to_vintage_bundle.png')
    os.system(f'cp -f {recon_mesh_path} {OUT_DIR}/prompt2prompt_edit/chair_modern_to_vintage.glb')

    # ===========================
    # Example 3: Material Change
    # ===========================
    print("\n" + "="*60)
    print("Example 3: Changing material from plastic to metal")
    print("="*60)

    start_time = time.time()
    gen_save_path, recon_mesh_path = run_prompt2prompt_edit(
        k3d_wrapper,
        input_image_path='./examples/toy.png',  # Replace with your image
        source_prompt="a plastic toy robot",
        target_prompt="a metal toy robot",
        p2p_replace_steps=0.5,
        p2p_blend_ratio=0.85    # Higher ratio to preserve shape
    )
    print(f"Time elapsed: {time.time() - start_time:.2f}s")

    # Copy results
    os.system(f'cp -f {gen_save_path} {OUT_DIR}/prompt2prompt_edit/toy_plastic_to_metal_bundle.png')
    os.system(f'cp -f {recon_mesh_path} {OUT_DIR}/prompt2prompt_edit/toy_plastic_to_metal.glb')

    print("\n" + "="*60)
    print("✓ All prompt-to-prompt editing examples completed!")
    print(f"✓ Results saved to: {OUT_DIR}/prompt2prompt_edit/")
    print("="*60)

    # ===========================
    # Parameter Guidance
    # ===========================
    print("\n📌 Parameter Tuning Guide:")
    print("  • p2p_replace_steps: When to switch from source to target prompt")
    print("    - 0.3-0.4: More structural changes allowed")
    print("    - 0.5-0.6: Balanced (recommended)")
    print("    - 0.7-0.8: Preserve more structure")
    print()
    print("  • p2p_blend_ratio: How much stored attention to use")
    print("    - 0.6-0.7: More creative freedom")
    print("    - 0.8-0.9: Better structure preservation (recommended)")
    print("    - 0.9-1.0: Maximum structure preservation")
    print()
    print("Note: This example uses pure Prompt2Prompt (no ControlNet).")
    print("      For ControlNet-based editing, use generate_3d_bundle_image_controlnet().")
