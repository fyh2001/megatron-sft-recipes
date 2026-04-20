# 把 VLM 新架构砍成纯文本 LLM 跑 Megatron 的操作手册

**场景**：你拿到一个 ms-swift / mcore_bridge **不原生支持**的新模型（常见是刚发布的 VLM，比如 Ministral-3、Qwen3-VL、Gemma-VL），但你**只需要训练它的文本能力**（比如用 RP 对话、多轮 chat、工具调用数据做 SFT）。

**核心思路**：与其花 2-5 人周给 mcore_bridge 写一份新架构的适配代码，不如写一个 **200 行的 Python 转换脚本**：

1. 砍掉 vision tower 和 projector 的权重
2. 把 `config.json` 的 architecture 改成 mcore_bridge **已经支持**的等价纯文本架构（Llama / Qwen2 / Qwen3 / DeepSeek …）
3. 保留 tokenizer 不动
4. 直接丢给现有的 Megatron pipeline 训练

本手册基于 2026-04 适配 `mistralai/Ministral-3-3B-Instruct-2512-BF16` 的实战经验。

---

## 一、什么情况下能用这套方案

### ✅ 适用

- 你**只训文本能力**，不需要 vision tower 参与梯度
- 目标模型的 **text backbone 和某个 mcore_bridge 已支持架构大致同构**（decoder-only transformer、GQA、RMSNorm、RoPE、SwiGLU 这套标配）
- 你已经能跑通 HF 后端的 `swift sft`，想把吞吐提上去

### ❌ 不适用

- 你要训 vision encoder 或 multimodal projector（那必须真的适配 mcore_bridge + megatron-core 的 VLM 框架）
- 目标模型有 **架构级**的非标准结构：
  - Sliding Window Attention（Mistral 早期、Gemma 早期）
  - Scaled RoPE 非主流变种（某些 Llama 4 的 `llama_4_scaling_beta` 等）
  - 非 RMSNorm（比如 LayerNorm with bias）
  - QKV 融合方式和已支持架构不同
- mcore_bridge 版本太旧，连 Llama / Qwen2 都不支持（这时候该先升级 mcore_bridge）

### ⚠ 边界情况（可以做，但要小心）

- **RoPE 用了 YaRN 做长上下文外推**：如果 SFT `max_length ≤ original_max_position_embeddings`，可以直接关 YaRN 训短序列；若要训长上下文，得确认 mcore_bridge 的 Llama 路径支持 YaRN
- **tie\_word\_embeddings=True**：权重里没有独立 `lm_head.weight`，要确认 mcore_bridge 支持 tied LM（Qwen2.5 小模型和 Ministral 都是 tied）
- **vocab 特别大**（≥128K）：CE fused kernel 的 fp32 buffer 会吃 4-10 GB 显存，TP=1 容易 OOM，准备 TP=2

---

## 二、前置检查清单

按顺序做这几步，如果任何一步答案是"否"，就**停下来想想要不要继续**。

### 1. mcore\_bridge 支持哪些 text-only model\_type？

```bash
head -20 /usr/local/lib/python3.12/dist-packages/mcore_bridge/model/gpts/llm.py
```

会看到类似：

```python
register_model(
    ModelMeta(
        ModelType.gpt,
        [
            'qwen2', 'llama', 'qwen3', 'qwen2_moe', 'qwen3_moe',
            'internlm3', 'mimo', 'deepseek', 'deepseek_v2',
            'deepseek_v3', 'deepseek_v32', 'kimi_k2', 'dots1',
            'ernie4_5', 'ernie4_5_moe', 'glm4_moe', 'glm4_moe_lite',
            'glm_moe_dsa', 'gpt_oss'
        ],
    ))
```

**目标是把 VLM 的 text backbone 伪装成这个列表里的某一个**。通常 `llama` 是最通用的"托底"选择（GQA + RoPE + RMSNorm + SwiGLU + 无 SWA 的都能套）。

### 2. 对比关键字段

读源模型的 `config.json`（如果是 VLM，字段在 `text_config` 子对象里），核对这些字段是否和目标 mcore 架构兼容：

| 字段 | Llama 期望值 | 不兼容时的症状 |
|---|---|---|
| `model_type` | `llama` | 必改 |
| `hidden_act` | `silu` | 若为 `gelu` / `swish` 可能要改或跳过 |
| `sliding_window` | `null` 或缺省 | 若有 SWA，mcore_bridge 的 Llama 路径不支持 |
| `attention_bias` | `false` | 若为 `true`，Llama 路径会报 shape 不匹配 |
| `rope_scaling` / `rope_parameters` | 标准 RoPE 或 `linear`/`dynamic` | YaRN / Llama4-scaling 要谨慎（见上一节边界情况）|
| `tie_word_embeddings` | `true` 或 `false` 都支持 | 检查权重里是否有独立 `lm_head.weight` |
| `head_dim` | `hidden_size / num_attention_heads` | 不标准的 head\_dim 有时候 mcore 会报错 |
| `rms_norm_eps` | 存在 | Llama 必需 |
| **无** cross-attention 层 | 必须 | 有 cross-attn 就不能伪装成 Llama |

### 3. 权重结构探测

```bash
python -c "
import json
idx = json.load(open('PATH/TO/model.safetensors.index.json'))
keys = list(idx['weight_map'].keys())
prefixes = {}
for k in keys:
    p = '.'.join(k.split('.')[:2])
    prefixes[p] = prefixes.get(p, 0) + 1
for p, n in sorted(prefixes.items()):
    print(f'{p:50s} {n}')
"
```

你会看到类似（以 Ministral-3 为例）：

```
language_model.model.embed_tokens           1
language_model.model.layers                 234
language_model.model.norm                   1
multi_modal_projector.linear_1.weight       1
multi_modal_projector.linear_2.weight       1
multi_modal_projector.norm.weight           1
multi_modal_projector.patch_merger          1
vision_tower.ln_pre.weight                  1
vision_tower.patch_conv.weight              1
vision_tower.transformer.layers             216
```

**关注三件事**：

- **LM 前缀**：通常是 `language_model.model.*` / `model.language_model.*` / `text_model.*` 之一
- **Vision 前缀**：通常是 `vision_tower.*` / `vision_model.*` / `visual.*`
- **Projector 前缀**：通常是 `multi_modal_projector.*` / `mm_projector.*` / `connector.*`

转换脚本里要做的：
- 剥掉 LM 前缀（`language_model.model.X` → `model.X`）
- 丢掉 Vision 和 Projector 前缀

### 4. tokenizer 能不能用 `AutoTokenizer` 加载

```bash
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('PATH'); print(type(tok).__name__, tok.vocab_size)"
```

只要目标目录有 `tokenizer.json`（HF Fast tokenizer 格式），Mistral 的 tekken、Qwen 的 BPE、Llama 的 SentencePiece 全都能被 HF 统一识别。**transformers 会以 `PreTrainedTokenizerFast` 或 `TokenizersBackend` 加载，不依赖原厂 SDK**（如 `mistral-common`）。

---

## 三、转换脚本（三步）

完整参考实现见 [`scripts/convert_ministral3_to_llama.py`](../scripts/convert_ministral3_to_llama.py)。

### 步骤 1：重写 `config.json`

核心逻辑：把 `text_config` 的字段全部拍平到顶层，替换 `architectures` 和 `model_type`，删掉 `vision_config` 和 VLM 专属字段。

```python
def build_new_config(src_cfg: dict) -> dict:
    tc = src_cfg["text_config"]
    new = {
        "architectures": ["LlamaForCausalLM"],   # 偷梁换柱的核心
        "model_type": "llama",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "hidden_size": tc["hidden_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "intermediate_size": tc["intermediate_size"],
        "head_dim": tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
        "vocab_size": tc["vocab_size"],

        # YaRN 外推仅在 max_len 超过 original_max_position_embeddings 时触发，
        # SFT 的 4K/8K 序列永远走不到 → 关掉最省事。
        "max_position_embeddings": 16384,
        "rope_theta": float(tc.get("rope_parameters", {}).get("rope_theta", 1_000_000.0)),
        "rope_scaling": None,

        "sliding_window": None,
        "tie_word_embeddings": tc.get("tie_word_embeddings", False),
        "hidden_act": tc.get("hidden_act", "silu"),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-5),
        "attention_bias": False,
        "mlp_bias": False,
        "initializer_range": tc.get("initializer_range", 0.02),
    }
    return new
```

**注意**：

- 不要保留 `image_token_index` / `vision_feature_layer` / `projector_hidden_act` 这些 VLM 专属字段，transformers 对未知字段宽容，但 mcore_bridge 的 parser 可能会报 unknown key
- `dtype` / `torch_dtype` 两个都写上，不同 transformers 版本要求不一样
- 不写 `bos_token_id` / `eos_token_id` / `pad_token_id`，让 transformers 从 `tokenizer_config.json` 自动补

### 步骤 2：过滤 + rename 权重

```python
LM_PREFIX = "language_model."
STRIPPED_PREFIXES = ("vision_tower.", "multi_modal_projector.")

def rename_weight_key(old: str) -> str | None:
    """Return new key, or None to drop."""
    for bad in STRIPPED_PREFIXES:
        if old.startswith(bad):
            return None
    if old.startswith(LM_PREFIX):
        return old[len(LM_PREFIX):]   # language_model.model.X → model.X
    if old.startswith(("model.", "lm_head.")):
        return old                    # 已经是 Llama 布局
    return None                        # 保险起见，不认识就丢
```

遍历原 `model-*.safetensors` shard，按新 key 重新分 shard 写出，同时更新 `model.safetensors.index.json`：

```python
from safetensors import safe_open
from safetensors.torch import save_file

for shard_idx, shard_name in enumerate(src_shards):
    new_shard_name = f"model-{shard_idx+1:05d}-of-{len(src_shards):05d}.safetensors"
    tensors = {}
    with safe_open(os.path.join(src_dir, shard_name), framework="pt") as f:
        for old_key in f.keys():
            new_key = rename_weight_key(old_key)
            if new_key is None:
                continue
            tensors[new_key] = f.get_tensor(old_key)
            new_weight_map[new_key] = new_shard_name
    if tensors:
        save_file(tensors, os.path.join(dst_dir, new_shard_name),
                  metadata={"format": "pt"})
```

**踩坑提示**：

- `safetensors.save_file` 必须传 `metadata={"format": "pt"}`，否则 transformers 加载时会抱怨
- 原模型如果有 `consolidated.safetensors`（Mistral 旧格式单文件完整权重），**不要**处理它（和分片 shard 是同一批权重的重复拷贝），直接忽略

### 步骤 3：tokenizer / chat\_template 原样复制

```python
for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "chat_template.jinja", "tekken.json",  # Mistral 专属
             "generation_config.json", "SYSTEM_PROMPT.txt"]:
    src = os.path.join(src_dir, name)
    if os.path.exists(src):
        shutil.copy2(src, dst_dir)
```

tokenizer 全盘保留不改。训练时对话格式靠 `chat_template.jinja` 或 ms-swift 的 template 注册，**不依赖 tokenizer 内部 class 名**。

---

## 四、训练脚本要点

基于 `scripts/08_sft_ministral3_text.sh`，关键差异和普通 Qwen 脚本比：

```bash
megatron sft \
    --model /home/ubuntu/perf_opt/models/ministral3-3b-text-llama \
    --model_type llama \           # 本地路径不在 MODEL_MAPPING 里，必须显式传
    --template llama \             # Mistral [INST]...[/INST] 在 swift 里叫 llama template
    --dataset ... --val_dataset ... \
    --tensor_model_parallel_size 2 \     # 大 vocab 防 CE OOM
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 4 --global_batch_size 64 \
    --packing true --max_length 4096 \
    --lr 1e-5 --min_lr 1e-6 --lr_decay_style cosine \
    --num_train_epochs 2 --finetune true \
    --cross_entropy_loss_fusion true \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true --overlap_param_gather true \
    --no_save_optim true --no_save_rng true
```

**关键参数说明**：

- `--model_type` 和 `--template`：本地路径的模型 ms-swift 从 `config.json` 反推不到它的 MODEL\_MAPPING entry（因为 architectures 已被改成 `LlamaForCausalLM`），**必须手动指定**
- `--template llama`：Mistral/Ministral 的 `[INST] … [/INST]` 格式在 swift 里用 `llama` template 对接（`mistral` key 注册的 template 也是 `llama`）
- `--packing true`：经此方案改造后的模型走 `llama` template，原生支持 packing，**不会触发 HF 后端那种 "template mistral\_2512 does not support padding free or packing" 的限制**
- `--tensor_model_parallel_size 2`：Ministral vocab=131072，MBS=4 seq=4096 下 CE fp32 buffer ≈ 4.3 GB（TP=1 时 8.6 GB，在 8 卡 H100 上会顶满 80GB）

### 显存估算公式（复用）

```
单卡静态显存 ≈
  模型权重 / TP                  (3.4 GB / 2 = 1.7 GB)
+ 梯度 / TP                     (1.7 GB)
+ 优化器 Adam / (DP × TP)       (Params×8 / DP×TP = 3.4 GB)
+ activation (取决于 MBS×MAX_LEN)  (约 15 GB @ MBS=4)
+ CE fp32 logits buffer / TP    (MBS × MAX_LEN × vocab × 4 bytes / TP = 4.3 GB)
+ NCCL buffer / cuBLAS workspace  (3-5 GB)
————————————————————————————————
≈ 30-35 GB/卡
```

Ministral-3 3B 实测 42 GB/卡（比公式略高，因 overlap\_param\_gather 的临时 all-gather buffer 2-3 GB）。

---

## 五、典型错误 & 诊断

### 错误 1：mcore\_bridge 加载时 "unknown model\_type: xxxx"

**原因**：`config.json` 的 `model_type` 还是原 VLM 的名字（`mistral3` / `qwen2_vl` / `llava` 等），没改成 `llama` / `qwen2` 之类支持列表里的值。

**排查**：

```bash
python -c "import json; print(json.load(open('PATH/config.json'))['model_type'])"
```

应该是 `llama`（或你目标的 mcore 支持名称）。

### 错误 2：权重 shape 不匹配 / `KeyError: language_model.model.*`

**原因**：权重 rename 没处理干净，LM 前缀没剥掉。

**排查**：

```bash
python -c "
import json
idx = json.load(open('PATH/model.safetensors.index.json'))
for k in list(idx['weight_map'])[:10]:
    print(k)
"
```

应该全部以 `model.` 或 `lm_head.` 开头，**不应有** `language_model.` / `vision_tower.` / `multi_modal_projector.` 前缀。

### 错误 3：`tie_word_embeddings=True` 但 mcore\_bridge 抱怨缺 `lm_head.weight`

**原因**：Llama 路径在 mcore\_bridge 里有两种模式——tied 和 untied。如果 config 写 `tie_word_embeddings=true` 但 bridge 还是去找 `lm_head.weight`，说明 bridge 版本较老不支持 tied。

**绕过方案**：在权重里**复制一份** `model.embed_tokens.weight` 到 `lm_head.weight`（浪费一点显存但能跑）：

```python
# 在 convert 脚本里
if new_cfg["tie_word_embeddings"]:
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
```

### 错误 4：`CUDA out of memory` 在 fused\_cross\_entropy

**原因**：vocab 太大，CE fp32 buffer 爆。典型特征：trace 定位在 `megatron/core/fusions/fused_cross_entropy.py` 的 `calculate_predicted_logits`，分配一个 `(B*S, V)` × fp32 的临时 tensor。

**解法优先级**：

1. `TP=2`（最简单，vocab 切两卡，CE buffer 减半）
2. `MBS` 减半
3. `MAX_LEN` 减半（seq 小了 CE buffer 也小）
4. 升级到 TE ≥ 2.8 + mcore ≥ 0.16.1，ms-swift 会自动切到 `cross_entropy_fusion_impl=te`，不再需要 fp32 buffer

### 错误 5：训练 loss 从第一步就是 NaN

**原因大概率**：

- RoPE `rope_theta` 或 `max_position_embeddings` 写错了 → 位置编码数值失控
- `tie_word_embeddings` 写反了 → embed 和 lm\_head 数值不一致

**排查**：

```bash
python -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('CONVERTED_PATH', dtype=torch.bfloat16, device_map='cuda:0')
# 随便 forward 一个短序列看 loss 是不是合理（应在 2-10 之间）
ids = torch.arange(1, 20, device='cuda:0').unsqueeze(0)
out = m(ids, labels=ids)
print('loss:', out.loss.item())
"
```

如果 transformers 那边 loss 就是 NaN/Inf，是转换本身坏了；如果 transformers 正常但 megatron 爆炸，是 mcore\_bridge 对某个字段解读不同。

### 错误 6：推理时 tokenizer 输出乱码 / 对话格式错位

**原因**：ms-swift 用 `--template llama` 渲染，但源模型的 `chat_template.jinja` 是 Mistral 自有的更新版格式。两者 `<s>[INST]…[/INST]</s>` 细节可能差异（比如 system prompt 的包装、工具调用标记）。

**解法**：训练时用 swift 内置 template（和数据对齐），推理时用模型自带的 `chat_template.jinja`（和部署端对齐）。两边一致即可。

---

## 六、实测对比数据

**模型**：`mistralai/Ministral-3-3B-Instruct-2512-BF16`
**硬件**：8 × H100 80GB，NVLink 全互联
**数据**：19K 条 RP 对话（英文为主，约 21M tokens），2 epoch
**Checkpoint 存于**：`/home/ubuntu/perf_opt/megatron_output/ministral3_3b_text/`

| 指标 | HF 后端（`swift sft`） | **Megatron 后端（伪装 Llama 后）** |
|---|---|---|
| 训练时间 | 5m 41s | **5m 40s**（init/save 占比）|
| iter 总数 | 368 | **162** |
| 单 iter 时间（稳态）| 0.93 s/it | 2.10 s/it |
| 单 iter tokens | 64k | **262k** |
| **有效吞吐** | 69k tok/s | **122k tok/s (+77%)** |
| 显存/卡（稳态）| 49.5 GB | **42 GB (-15%)** |
| 数据利用率 | **丢 32% 长对话**（template 不支持 packing）| **100%（packing 生效）**|
| 工程量 | 开箱即用 | **1 个 Python 脚本 + 1 个 shell 脚本 ≈ 1 人天** |

定性效果：SFT 后角色知识迁移成功（正确描绘了训练数据里反复出现的 Muichiro 角色的外貌、动作模式、对话风格），中文通用能力无退化（训练数据几乎全英文）。

---

## 七、换别的 VLM 怎么改

手册的精髓在于**「判断架构等价性 → 选目标 model\_type → 三步转换」**这套流程。下面是几个常见候选的提示：

### Qwen3-VL（72B / 8B）

- text backbone 是 Qwen3 架构
- 目标 `model_type: "qwen3"`（mcore\_bridge 已支持）
- LM 前缀常见是 `model.text_model.*` 或 `text_model.*`，需探测
- Vision 前缀 `visual.*` + `merger.*`
- **特别注意**：Qwen3-VL 的 `rope_scaling` 如果是 `mrope`（多模态位置编码），text-only 推理下可以改成 standard RoPE，但会小幅损精（mrope 对纯文本等价于 standard，但字段名不同会被 bridge 拒绝）

### Gemma-3-VL

- text backbone 是 Gemma 3
- 目标 `model_type: "gemma3"`（**检查 mcore\_bridge 是否已加入**，2026-04 时还是 TODO）
- Gemma 3 有 interleaved sliding window attention（每 N 层一个 SWA），**这个比较难伪装**——Llama 路径不支持 SWA，Qwen2/3 也不支持
- 如果 mcore\_bridge 已适配 Gemma 3，走 Gemma 3 路径；否则本方案不适用

### Llama-3.2 Vision（11B / 90B）

- text backbone 是 Llama 3.1
- 目标 `model_type: "llama"`（最直接）
- 有 cross-attention 层（text 和 vision token 在中间交互），**本方案不适用**——Llama 3.2 Vision 的 text-only 推理需要掩掉所有 cross-attn 层，而 mcore\_bridge 的 Llama 路径不认识 cross-attn
- 换用 Llama 3.1 Instruct（纯文本）反而更合适

### LLaVA / InternVL 系列

- text backbone 多数是 Vicuna / Qwen / Mistral，分别对应 `llama` / `qwen2` 系列
- Projector 前缀 `mm_projector.*`
- 架构上相对干净，方案通用性最好

---

## 八、TL;DR 一张表

| 想要 | 推荐做法 |
|---|---|
| 训新 VLM 的 vision 能力 | 老老实实用 HF 后端 / NeMo / 自己扩 mcore\_bridge |
| 训新 VLM 的 **文本部分** + Megatron 吞吐 | **本手册方案 D**（砍 ViT + 伪装 Llama / Qwen2） |
| 训纯文本 LLM 但 mcore\_bridge 不认识 | 同上思路，看架构等价性选目标 `model_type` |
| 只想跑推理 / LoRA 轻量微调 | HF 后端 / vLLM，不值得折腾 Megatron |

**判断能不能做只需要 5 分钟**（执行"前置检查清单"）。能做就做，1 人天收 +77% 吞吐是划算的。

---

## 参考文件

- 转换脚本：[`scripts/convert_ministral3_to_llama.py`](../scripts/convert_ministral3_to_llama.py)
- Megatron 训练脚本：[`scripts/08_sft_ministral3_text.sh`](../scripts/08_sft_ministral3_text.sh)
- HF 后端对照脚本（未改造版）：[`scripts/07_sft_ministral3_3b.sh`](../scripts/07_sft_ministral3_3b.sh)
