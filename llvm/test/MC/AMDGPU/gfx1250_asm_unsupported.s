; RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

;; Ray Tracing: DS_BVH_STACK ops

ds_bvh_stack_rtn_b32 v255, v254, v253, v[249:252]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_bvh_stack_push4_pop1_rtn_b32 v1, v0, v1, v[2:5]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_bvh_stack_push8_pop1_rtn_b32 v1, v0, v1, v[2:9]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_bvh_stack_push8_pop2_rtn_b64 v[254:255], v253, v252, v[244:251]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

;; S_WAIT_*CNT instructions.

s_wait_samplecnt 0x1234
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_bvhcnt 0x1234
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

;; S_WAITCNT instruction.

s_waitcnt 0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

;; All "TBUFFER" ops, and BUFFER_LOAD/STORE_FORMAT ops.

tbuffer_load_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_8_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_32_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_SSCALED] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[8:11], s3 format:[BUF_FMT_11_11_10_FLOAT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[8:11], s3 format:[BUF_FMT_32_32_32_32_UINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[8:11], s3 format:[BUF_FMT_32_FLOAT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xy v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xy v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

;; Image load/store/atomic, Sample & Gather ops, image_bvh.

image_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_mip v[252:255], [v0, v1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_pck_sgn v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_mip_pck v5, [v0, v1], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store_mip v[252:255], [v0, v1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store_mip_pck v5, [v0, v1], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_swap v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_cmpswap v[0:1], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_add_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_sub_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_min_int v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_min_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_max_int v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_max_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_or v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_xor v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_inc_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_dec_uint v1, [v2, v3], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_pk_add_f16 v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_pk_add_bf16 v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_min_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_max_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh_intersect_ray v[4:7], [v9, v10, v[11:13], v[14:16], v[17:19]], s[4:7]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh64_intersect_ray v[4:7], [v[9:10], v11, v[12:14], v[15:17], v[18:20]], s[4:7]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh_dual_intersect_ray v[0:9], [v[0:1], v[11:12], v[3:5], v[6:8], v[9:10]], s[0:3]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh8_intersect_ray v[0:9], [v[0:1], v[11:12], v[3:5], v[6:8], v9], s[0:3]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample v64, v32, s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_l v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_lz v64, v32, s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_l v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_lz v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_o v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_l_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_lz_o v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_o v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_l_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_lz_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4 v[64:67], [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_l v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_lz v[64:67], [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_lz v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_o v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_lz_o v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_lz_o v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_get_lod v64, v32, s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_g16 v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_g16 v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_o_g16 v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_o_g16 v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cl v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b_cl v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cl v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b_cl v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cl_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl_o v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b_cl_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cl_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl_o v64, [v32, v33, v34, v[35:37]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b_cl_o v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl_g16 v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl_o_g16 v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl_o_g16 v64, [v32, v33, v34, v[35:37]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl_g16 v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_cl v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b_cl v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_cl v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_l v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b_cl v[64:67], [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4h v[64:67], [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_msaa_load v[1:4], [v5, v6, v7], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
