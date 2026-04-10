# 本地测试说明（中文）

这个 README 面向队友：拿到代码后，能在自己电脑上快速跑通抓取、训练、评测流程。

## 1. 环境准备

推荐使用 `conda`，Python 版本建议 `3.10`。

### 1.1 创建并激活环境（如果你还没有）
```bash
conda create -n cis5450 python=3.10 -y
conda activate cis5450
```

### 1.2 安装依赖
```bash
pip install pandas numpy scikit-learn requests beautifulsoup4 joblib torch
```

## 2. 获取代码

```bash
git clone https://github.com/Anthoneeee/ember-text-notes.git
cd ember-text-notes
```

后面命令默认都在仓库根目录执行（即包含 `model.py` 的目录）。

## 3. 目录说明（核心文件）

- `Newsheadlines/scrape_headlines.py`：从 URL 抓标题（含多级回退策略）
- `Newsheadlines/url_only_data.csv`：原始 URL 列表
- `Newsheadlines/scraped_headlines_clean.csv`：抓取后清洗数据（训练用）
- `train_news_b_v1.py`：训练脚本
- `model.py` / `preprocess.py`：评测入口
- `Newsheadlines/eval_project_b.py`：本地评测脚本

## 4. 推荐运行流程（队友本地）

## 4.1 第一步：抓取标题数据
```bash
python -u Newsheadlines/scrape_headlines.py \
  --input-csv Newsheadlines/url_only_data.csv \
  --output-raw Newsheadlines/scraped_headlines_raw.csv \
  --output-clean Newsheadlines/scraped_headlines_clean.csv \
  --max-workers 4 \
  --timeout 10 \
  --retries 1 \
  --min-delay 0.15 \
  --max-delay 0.55 \
  --allow-url-fallback
```

跑完后关注最后的 `summary`，特别是：
- `headline_method_counts`（看真实抓取比例）
- `saved_raw` 和 `saved_clean`（确认文件落盘）

## 4.2 第二步：训练第一版模型
```bash
python train_news_b_v1.py \
  --input-csv Newsheadlines/scraped_headlines_clean.csv \
  --output-model Newsheadlines/artifacts/news_b_tfidf_lr.joblib
```

## 4.3 第三步：本地评测
```bash
python Newsheadlines/eval_project_b.py \
  --model model.py \
  --preprocess preprocess.py \
  --csv Newsheadlines/scraped_headlines_clean.csv \
  --batch-size 64
```

## 5. 快速验证（不重训）

如果仓库里已经有 `Newsheadlines/artifacts/news_b_tfidf_lr.joblib`，可以直接跑评测：
```bash
python Newsheadlines/eval_project_b.py \
  --model model.py \
  --preprocess preprocess.py \
  --csv Newsheadlines/scraped_headlines_clean.csv \
  --batch-size 64
```

## 6. 你最可能需要改的“路径位置”

如果你的本地目录不一样，只要改这几处参数即可：

1. `--input-csv`
2. `--output-raw`
3. `--output-clean`
4. `--output-model`
5. `--csv`（评测时输入数据）

建议用“相对路径”（像上面的示例），这样队友不需要改成绝对路径。

## 7. 常见问题

### Q1: `No module named torch`
在当前环境安装：
```bash
pip install torch
```

### Q2: 评测报错缺少参数
`eval_project_b.py` 必须传 3 个参数：
- `--model`
- `--preprocess`
- `--csv`

### Q3: 抓取时出现 403/406
这是网站反爬常见现象。当前脚本已做：
- 降并发
- 随机延时
- URL 变体重试
- Wayback / Jina 回退
- 最后 URL slug 兜底

如果你要更稳，可以再降低并发，例如 `--max-workers 2`。

---

如果你只想“复现一次完整流程”，按 4.1 -> 4.2 -> 4.3 顺序跑就行。
