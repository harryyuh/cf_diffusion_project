# 实验代码与结果审计（2026-10-02）

## 范围与结论

检查 DeltaAI 当前代码、已保存配置、逐样本预测和队列，重点为 CelebA batch2/batch40 learning curve、已完成主表 full-test、待运行 matched50k full-test、Pendulum full-test 和同卡 profiling。

这是诊断，不是修复：本轮没有修改远端实验代码、取消任务或重新提交。本文不构成对历史所有实验的全面认证。

远端项目：`/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba`。下述源码路径相对此目录。

**确认存在会影响协议一致性和结论的问题；但当前 CelebA 全量主表没有漏样本或二值目标错配的证据。修复不保证 Label-only 会超过 CA。**

## 1. 高优先级：FP32 配置实际变成 FP16

**2026-10-02 16:04 CDT 修复更新：** 已在 DeltaAI 项目的两个旧 evaluator 中将 `bf16/fp16/fp32/no` 显式映射到对应 `torch.dtype`，并在 LoRA 加载后断言 UNet 实际 dtype 与配置一致。远端 `py_compile` 与 `git diff --check` 通过。修复时任务 `3284415_[0-163]` 和 `3283291_[0-57]` 均仍 PENDING；保留原 Slurm 提交号，分片启动时会从项目路径读取修正后的模块。此修复不追溯更改已经保存的旧曲线或旧全量图像。

- `evaluation/evaluate_latent_prompt_lora_celeba_n2048.py:330` 和 `evaluation/evaluate_latent_prompt_lora_celeba.py:328` 使用 `bf16 if config == bf16 else fp16`，所以 `fp32` 和 `no` 都被映射为 FP16。
- `evaluation/evaluate_label200k_smoke.py:330` 已有正确的显式 dtype 映射，但不能据此认为其他入口也修好。
- 已完成 CelebA batch40 曲线调用旧入口，配置写 FP32，实际 Label-only 生成 FP16；CA 路径 FP32。
- 排队中的 CelebA matched50k full-test `3284415_[0-163]` 调用同一个旧入口。
- 排队中的 Pendulum full-test `3283291_[0-57]` 通过 `evaluation/evaluate_pendulum.py:119` 间接导入另一个旧入口，也受影响。
- 本轮最后查询：上述两个数组仍为 PENDING / Priority。
- batch2 曲线走修好的入口，但 Label-only 为 bf16、CA 推理 FP32；依然不能称完全相同推理精度。

建议在这些任务开始前统一入口并增加运行时 dtype 断言。改变精度可能需要重新生成图像，不能靠重算 F1 修复。

## 2. Learning curve 的“512 张”不是全测试集随机 512 张

四个 do 各取128，来自计划 `(0,20,40,61)` 的列表前缀。索引范围如下：

| do | n | 测试集索引范围 |
|---|---:|---:|
| Young | 128 | 0–463 |
| Male | 128 | 12–471 |
| No_Beard | 128 | 8–491 |
| Bald | 128 | 32–667 |

所有源图位于19962张测试集的前668个索引中。它可以比较这组固定图片上不同 checkpoint 的表现，但不能据此判断完整测试分布上已收敛或200k训练无意义。

主表旧 Label-only20k 与新曲线20k在相同源样本上的目标F1分别为：

| checkpoint | Young | Male | No_Beard | Bald | 平均 |
|---|---:|---:|---:|---:|---:|
| 主表旧20k | .539 | .987 | .167 | .428 | .530 |
| 曲线新20k | .539 | .980 | .200 | .428 | .537 |

因此，主表全量均值 .585 与曲线 .537 的差距，不能直接归因于新模型退化。

## 3. No_Beard 低 F1 混合了普通编辑与女性 OOD 长胡子

`control_heads.py` 的 `_celeA_ood_hook` 对女性 do(No_Beard) 输出 -1；Label-only 的 `2*out-1` 转换后为 -3。作者 CelebA notebook 明确说明 -1 是女性长胡子的 OOD 外推，不是普通缺失值。

所以不能删除这部分样本并将男性子集得分替代主表。此前把它当无效标签的初步解释已撤回。

当前全量主表 do(No_Beard) 共5016张，其中男性1986、女性3030：

- Label-only 全组F1：.268119。
- 男性子集：Label-only .793829，CA .602752。
- Label-only 女性子集目标均为 No_Beard=0，但2796/3030仍被分类器判为 No_Beard=1，成为大量假阳性。

这支持“女性 OOD 长胡子编辑不足是该分数的主要来源”，不支持笼统的“普通去胡子能力只有0.2”。建议保留原协议全组分数，同时分性别、编辑方向报告诊断。

## 4. CLD：修好分组仍未解决 KL 实现错误

`external/Causal-Adapter/counterfactual-benchmark/counterfactual_benchmark/evaluation/metrics/minimality.py` 中 `kl_divergence` 将第一项同时当均值和logscale，完全忽略第二项：

```python
q_loc, q_logscale = embedding_1[:, 0], embedding_1[:, 0]
p_loc, p_logscale = embedding_2[:, 0], embedding_2[:, 0]
```

VAE编码器实际返回 mu、logvar；正确使用还需区分 logvar 和 logscale（后者为前者的一半）。

CPU数值测试：N(0,1) 与 N(0,4) 的真实 KL 为 .318147，当前实现给0。当前 `evaluation/evaluate_celeba_cld_saved.py` 调用此实现。已有修正分组后的 CA .300720、Label-only .309216 仍不是经过数学验证的 Gaussian-KL CLD。

VAE输入范围另行核对为[0,1]，与该benchmark数据加载一致；本次没有发现这里的归一化错误。

建议保留“bundled-code-compatible”旧结果，另行验证并统一重算正确KL版本，不能悄悄修公式后与论文旧数值直接比较。该问题确认于当前捆绑源码，不足以反推作者发表结果当时使用了哪个版本。

## 5. CA 与 Label-only 图像处理不完全相同

| 路径 | CA | Label-only |
|---|---|---|
| factual 64px | crop150 → PIL bilinear256 → PIL bicubic64 | crop150 → PIL bilinear64 |
| generated 64px | 解码后PIL量化 → PIL bicubic64 | 浮点张量 → torch bicubic64，无显式antialias |

逐样本对齐后，同一源图的保存 real tensor 平均绝对像素差为 .0041129（[0,1]标度）。这会混入 classifier、FID、LPIPS及CLD比较。

主表 FID CA9.264 与 Label-only9.294 仅差约 .03，不能据此认定哪方生成分布更好。统一预处理前应视为近似比较。

负向prompt另行检查一致，均为 `ugly, blurry, low res, unrealistic`；DDIM调度与核心 inversion 公式也未在本次检查中发现不一致。

## 6. F1 的旧阈值函数有 bug，但不是当前曲线低分的解释

旧 `_n2048` evaluator 会根据整个预测数组的min/max猜测 logits 或 probabilities。分类器返回logits，应固定阈值0，而不是某些批次变成0.5。

已用保存的原始logits、固定阈值0核算近期 batch2/batch40曲线：报告数值未改变，原因是当前样本组刚好触发了正确分支。因此不能声称修这个bug就能提高这些分数。

当前 full-test merger 将所有样本拼接后、按logit>0计算全局F1，不是平均 shard F1。

## 7. 当前 CelebA 全量主表通过的完整性检查

数据根目录：`/work/nvme/bhje/hyu29/evaluations/celeba/ca_official_fulltest_ddim100_semantics_v2`。

对双方全部 `plan_*/tensors/batch_*.pt` 检查：

- CA：19962个样本，重复0，完整覆盖测试集。
- Label-only：19962个样本，重复0，完整覆盖测试集。
- 干预索引不一致：0。
- 二值反事实目标不一致：0。
- factual标签不一致：0。

历史漏分片问题不能再直接套用到这份已完成结果。但二值目标相同不等于连续条件编码数值相同，且精度/预处理问题依然存在。

## 8. 训练配置不能概括为“除了结构都相同”

主表旧 Label-only20k 和 batch2曲线的峰值LR相同（LoRA5e-5、mapper1e-3），但前者cosine终点20k、后者50k，所以20k时实际LR不同。采样权重算法和随机水平翻转也不同。

batch2曲线两方法另有差异：CA LR1e-5 constant、weight decay .01、CTC .0005、flip0；Label-only 使用双LR cosine、weight decay0、CTC0、flip开启。

可以称配对batch/steps下的方法训练配方比较，不能称其他超参数完全相同，也不能仅凭它将差异归因于LoRA架构。

官方checkpoint的200k步数不能自动证明其训练batch40；缺少对应训练日志时，不能用论文Table6的batch直接推算作者权重训练成本。

## 9. 最新同卡 profiling：约1.35倍，不是1.96倍

任务3294900已完成。同一GH200 GPU UUID，双方FP32、batch40，预热30步、测200步、独立进程3次。

| 方法 | 秒/step，mean ± sample SD |
|---|---:|
| CA | .588201 ± .000578 |
| Label-only | .435842 ± .000478 |

吞吐量比1.34957，单步时间约减少25.90%。测得峰值allocated memory减少约37.1%。

仍存在loader/augmentation细节差异：CA pin_memory=False、LO=True；CA flip0、LO flip=True；drop_last不同（本短测未到尾batch）。不能标成完全相同data-loader协议。计时为稳态step间隔，不含模型加载等一次性成本，不是完整训练总耗时。

原始汇总：`/work/nvme/bhje/hyu29/evaluations/celeba/efficiency_profile_same_gpu/3294900/summary.json`。

## 10. Pendulum 与剩余任务边界

**2026-10-03 CA full-test interim check:** CA50k/batch40 的29个分片已完成；1826张测试图恰好覆盖完整本地测试集，无重复。用四个 CA benchmark 回归器按样本数加权汇总，四个目标对角 MAE 为 Pendulum 0.5085、Light 0.4863、Shadow length 0.4247、Shadow position 0.4464，平均0.4665。Label-only 分片尚未完成，因此这不是成对主表。相同 CA50k checkpoint 的旧 n256 结果在同回归器下平均0.4719，接近全量值，表明高 MAE 不是分片合并产生。相同 n256 源图/干预值下，作者发布的 CA20k checkpoint 在同回归器下平均0.2042；若改用我们自训的回归器，CA50k n256平均0.2097、作者 CA20k平均约0.0574，这些数值不能跨回归器直接比较。四个官方回归器在1826张真实测试图的平均 MAE 约0.01365，说明其对真实图本身并非失效。Grid 定性检查显示自训 CA50k 对若干 do(Pendulum) 样本改动很小，而作者 CA20k 对同样本的角度改变更明显；较高误差与编辑不足相符，但自训模型表现差的训练根因仍待查。

旧n256按文件名排序的偏斜子集、ours先降64再放大的问题，不能被当作full-test结论。新full-test路径改善了原生输出分辨率，但仍间接导入旧dtype入口，见第1节。

本地完整test为1826，论文描述2000；即使跑完本地full-test，也不能未经数据核对就称严格论文同测试集。

null-edit相关性任务曾因 `models.classifiers` 导入冲突失败；rank64已有旧回归器结果不等于已完成官方回归器重评。尚未完成的项目不能作为论文已有证据。

## 建议处理顺序（尚未执行）

1. 优先修复两条待运行full-test调用链的dtype映射并加入断言；先讨论如何保留队列，再决定部署/重提。
2. 统一两方法图像预处理；从尚存原生分辨率输出重算，缺失时才重新生成。
3. 用固定manifest覆盖全测试分布评估learning curve；所有checkpoints共享样本与干预值。
4. 分层解释No_Beard，保留作者OOD协议主分数，不筛掉困难样本。
5. CLD增加Gaussian KL单元测试，区分原代码兼容值与修正值。
6. profiling统一loader及augmentation细节；现阶段1.35倍是这次实际短测结果，不能继续用1.96倍作为同协议证据。

在这些核查完成前，既不能声称“我们少10倍步骤达到同样性能”，也不能从当前偏斜子集反推“CA20k足够、200k完全浪费”。

## 2026-10-05 更新：CelebA null-edit composition LPIPS

已完成的 `merged_metrics_fid_f1_lpips.json` 中 `lpips_identity` 是 factual 与**实际干预后**图像的距离，不是 CA 论文 Table 2/13 的 null-intervention reconstruction LPIPS。全测试集没有已保存的 null-edit 输出，不能只从现有 tensor 重算该列。

已在 DeltaAI 新增 `evaluation/evaluate_celeba_null_lpips_fulltest.py`、`evaluation/merge_celeba_null_lpips_fulltest.py` 和 `slurm/evaluation/celeba_null_lpips_fulltest.slurm`。CA 使用作者 200k checkpoint、无干预 `ddim_editing` pass-through；Label-only 使用原 20k checkpoint，source 与 target 均为 factual labels。双方按现有 full-test 的 82 个计划覆盖同一 19,962 张 test 图、DDIM 100、inversion CFG 1、sampling CFG 3；比较图像统一为 CenterCrop150→bilinear256→bicubic64。LPIPS 使用 benchmark 代码当前实际选择的 AlexNet 网络，另外记录 `[0,1]` 标度的像素 MAE。注意 CA 论文文字描述 VGG-16，所检仓库代码则实例化 `net_type='alex'`；最终报告须标明遵循的代码版本。

Smoke array `3315300_[0,82]` 已提交，全量 array `3315302_[0-163%8]` 依赖 smoke 全部成功；合并任务 `3315309` 依赖全量成功，输出 `.../evaluations/celeba/null_composition_author_ca_vs_label20k/merged.json`，合并器检查每个方法 19,962 个唯一来源及逐计划配对。10 月 5 日 14:39 美东时间三者仍在排队/等待依赖，**暂无 null-edit LPIPS 数值**。监控已恢复并包括该任务；首批输出需要检查实际运行、图像形状和无干预条件。
