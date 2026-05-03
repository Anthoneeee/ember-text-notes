# Deliverables Workspace

该目录用于统一管理 CIS 4190/5190 Final Project 的最终提交产物、报告素材与复现清单。

## 目录约定

- `dataset/`：最终提交数据集与数据说明
- `external_headlines/`：额外收集的真实 headline 数据与原始抓取文件
- `report/`：报告表格、笔记与待整合内容
- `figures/`：报告图表（指标曲线、数据统计图）
- `manifests/`：提交清单、版本映射、校验记录

## 使用原则

1. Hugging Face 上传以项目根目录的 `model.py`、`preprocess.py`、`model.pt` 为准。
2. `submission_hf_urlfetch/` 和 `experiments/submission_hf_best_8742_cur_nb_sgdcal_t5525/` 保留同一最佳版本的备份。
3. 数据、图表、报告素材保留在 `deliverables/` 下。
4. 最终提交前只打包必要代码、数据、模型与报告，不包含缓存、虚拟环境或旧实验日志。

## 当前状态

- Workspace initialized: 2026-04-22
- Owner: Anthoneeee / CIS5190 team
- Current best HF accuracy: 0.8741666666666666
