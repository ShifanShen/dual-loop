# ICLR 图像英文化与一致性修改指南

本目录中的 3 张图片已完成英文替换，并由 `pics/` 中的 SVG 图源导出为供 ICLR 稿件使用的 PNG。后续修改应继续从 Draw.io/Visio/SVG 图源导出，并以相同文件名覆盖 PNG；不要在 LaTeX 中拉伸图片改变宽高比。

## Figure 1: `pics/framework_cn.png`

建议英文图题：

> Overview of the semantic-contract-based dual-loop framework.

### 已核对的逻辑

图中已删除“比较种子规约与局部变体”及 bounded specification selection / `SpecSearch` 逻辑。后续编辑应继续保持以下流程：

1. 第 $t$ 轮从当前规约 $s^{(t)}$ 生成局部补丁 $\Delta^{(t)}$。
2. 合并后得到候选规约 $\hat{s}^{(t+1)}$。
3. 候选通过 acceptance gate 时，直接令 $s^{(t+1)}=\hat{s}^{(t+1)}$；否则保留 $s^{(t)}$ 并停止 SAL。
4. SAL 终止时，直接把当前规约冻结为 $s^\star$。图中不得再出现 seed specification、local variants、bounded selector、`SpecSearch` 或 `SelectContract`。
5. IRL 应画成 $\mathcal{C}^{(0)}\rightarrow C^{(0)}$；随后每轮只验证和修复当前 $C^{(k)}$，生成本轮候选集 $\mathcal{C}^{(k+1)}$，再选出 $C^{(k+1)}$。

### 标签替换

| 中文标签 | 英文标签 |
|:---|:---|
| 1) 规约归纳 | 1) Specification Induction |
| 自然语言问题 | Natural-language Problem |
| 规约归纳器 | Specification Inducer |
| 初始规约 | Initial Specification |
| 输入 | Input |
| 规约评价器 | Specification Judge |
| 覆盖性 | Coverage |
| 忠实性 | Faithfulness |
| 精确性 | Precision |
| 提出局部修订 | Propose Local Revision |
| 合并 | Merge |
| 接纳门控 | Acceptance Gate |
| 候选规约 | Candidate Specification |
| 停止修订 | Stop Refinement |
| 冻结当前规约 | Freeze Current Specification |
| 2) 代码生成 | 2) Code Generation |
| 冻结语义契约 | Frozen Semantic Contract |
| 代码生成 | Code Generation |
| 候选程序集 | Candidate Program Set |
| 属性子句编译 | Property-Clause Compilation |
| 可执行属性集合 | Executable Property Set |
| 当前程序 | Current Program |
| 验证器 | Feedback Verifier |
| 公开测试反馈 | Public-Test Feedback |
| 属性子句反馈 | Property-Clause Feedback |
| 实现修复候选生成 | Repair Candidate Generation |
| 验证器感知选择 | Verifier-Aware Selection |
| 下一程序 | Next Program |
| 迁移轨迹 | Transition Trace |
| 语义状态、实现状态与反馈证据 | Semantic State, Implementation State, and Feedback Evidence |
| 最终程序 | Final Program |
| 数据流 / 控制流 | Data / Control Flow |
| 接受 | Accept |
| 修订 | Revise |

图中的数学符号统一使用 $s^{(0)}$、$s^{(t)}$、$\hat{s}^{(t+1)}$、$s^\star$、$\mathcal{C}^{(0)}$、$C^{(k)}$、$\mathcal{C}^{(k+1)}$、$C^{(k+1)}$、$F_v^{(k)}$、$F_p^{(k)}$ 和 $P^\star$。不要出现 $\bar{s}$ 或额外的 selected contract 状态。

## Figure 2: `pics/trace_evidence_cn.png`

建议英文图题：

> Final-aligned trace evidence on 1,055 LiveCodeBench problems.

### 标签与数字

| 中文标签 | 英文标签 |
|:---|:---|
| 1055 个问题 | 1,055 Problems |
| 无需循环即通过 288 | Pass without Loop-Associated Change 288 |
| 仅 SAL 65 | SAL-Associated Path 65 |
| 仅 IRL 38 | IRL-Associated Path 38 |
| SAL + IRL 10 | SAL + IRL Path 10 |
| 解决 / 循环相关轨迹 | Solved / Loop-Associated Paths |
| 规约侧残余证据 579 | Specification-Side Residual Evidence 579 |
| 实现侧残余证据 4 | Implementation-Side Residual Evidence 4 |
| 未知残余 71 | Unknown Residual 71 |
| 实现侧 + 未知残余 75 | Implementation-Side + Unknown Residuals 75 |
| 可观察内容 | Inspectable Evidence |
| 解决路径 | Solved Paths |
| 经循环解决的案例 | Loop-Associated Solved Cases |
| 规约 / 实现侧残余证据 | Specification- / Implementation-Side Residual Evidence |

数字必须保持 $288+65+38+10=401$ 和 $401+579+4+71=1055$。图注及图内措辞均使用 `evidence` 或 `trace label`，不要使用 `root cause`、`causal attribution` 或暗示因果证明的表述。

## Figure 3: `pics/case_study_cn.png`

建议英文图题：

> Case-study trace for LiveCodeBench problem 2883.

### 内容一致性

- 候选规约 $\hat{s}^{(1)}$ 因 `rejected_no_gain` 被拒绝。
- 保留 $s^{(0)}$，随后直接冻结为 $s^\star$。
- 冻结后的 SAS 仍为 80，不得写成 84，也不得增加局部变体选择步骤。
- $P^\star$ 包含 3 个 `numeric_output` 子句，$F_p^{(0)}=\varnothing$。
- 实际修复由公开验证器反馈 $F_v^{(0)}$ 驱动，最终程序为 $C^{(3)}$。

### 主要标签替换

| 中文标签 | 英文标签 |
|:---|:---|
| 规约修订 | Specification Revision |
| 自然语言问题 | Natural-language Problem |
| 初始规约 | Initial Specification |
| 无依据假设 | Unsupported Assumption |
| SAS 评价器 | SAS Judge |
| 诊断与候选修订 | Diagnosis and Candidate Revision |
| 门控结果 | Gate Result |
| 冻结语义契约 | Frozen Semantic Contract |
| 契约—代码比较 | Contract--Implementation Comparison |
| 契约要求 | Contract Obligation |
| 代码实现 | Implementation |
| 实现修复 | Implementation Repair |
| 初始代码 | Initial Program |
| 属性子句检查 | Property-Clause Check |
| 公开反馈验证器 | Public-Feedback Verifier |
| 程序未改变 | Program Unchanged |
| 输出 3 | Output 3 |
| 重写实现 | Rewrite Implementation |
| 最终程序 | Final Program |
| 最终结果：通过完整测试 | Final Result: Complete-Suite Pass |
| 轨迹证据 | Trace Evidence |
| 契约—实现失配可见 | Contract--Implementation Mismatch Is Visible |
| 验证器反例驱动修复 | Verifier Counterexample Drives Repair |
| IRL 重写后通过 | Pass after IRL Rewrite |

保留代码标识符和状态值的英文原样，例如 `powers_of_5`、`bin()`、`numeric_output`、`wrong_answer`、`rejected_no_gain` 和 `Solved`。

## 导出要求

- 保持白色背景和当前宽高比。
- 英文字体统一为 Arial、Helvetica 或 Times New Roman，最小字号应在 ICLR 单栏宽度下仍可辨认。
- 数学符号使用 LaTeX 风格斜体，函数名 `SAS`、`SAL`、`IRL` 和代码标识符使用正体。
- 推荐同时导出 PNG（至少 300 dpi）和矢量 PDF；当前 `main.tex` 默认读取上述三个 PNG 文件名。
