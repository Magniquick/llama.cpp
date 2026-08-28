# Restoring openvino-test/.venv to its pre-export state

transformers was pinned to 5.2.* (optimum-intel requires exactly 5.2.*; 5.15.0.dev0 removed `Qwen3_5DynamicCache`).
installed 5.15.0.dev0 removed `Qwen3_5DynamicCache`, which the qwen3_5 export patcher
imports). To restore the exact original:

    uv pip install --python ./openvino-test/.venv/bin/python --force-reinstall --no-deps \
      "git+https://github.com/huggingface/transformers.git@9ed46fb37cf4c7f885677ad194d2797265e89186"

Other pins in this venv (unchanged):
  optimum-intel 1.27.0.dev0+609b072  git+https://github.com/openvino-agent/optimum-intel.git@609b072a6fc8f87a854292c072c360d80bed95dc  (branch support_mtp)
  optimum       2.2.0.dev0           git+https://github.com/huggingface/optimum.git@a6c775e11118d62712057bd3a8c5649898a5312d
