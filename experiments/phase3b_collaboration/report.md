# DARI-MAS Phase 3B

阶段状态：协作机制对比及两项补充实验已完成。高难度 Benchmark 的 80 题推理与评分已完成。

高难度 Benchmark 已固定 80 题，包含 30 道 HumanEval+、20 道 Level 4/5 MATH-500、20 道 LiveCodeBench 和 10 道高难桥接 HotpotQA。GPQA Diamond 按用户要求移出本轮。样本清单、数据版本指纹、运行日志及结果保存在 `challenging_benchmarks/`。编程题使用官方检查器在无网络 Bubblewrap 环境内评分。

已导出 122 个历史 Help 与同前缀内部对照组。各时期分别按原始评分统计。

## refinement_r2 · gsm8k:1241 · help_submit

Rose bought five dozens of eggs for $2.40 a dozen. She will sell it for $1 for 3 eggs. How much will be Rose's profit?

参考答案 `8`

完整输入和原始证据见 historical_cases.jsonl。

共享前缀与动态请求

```json
{}
```

### continue

终局奖励 0.0，最终答案 `None`。

anchor

```text
{"step": "The total cost is calculated by multiplying the number of dozens by the cost per dozen: 5 * $2.40 = $12.00.", "refs": ["s1"], "goal": "Calculate Rose's profit.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Next, calculate the total revenue from selling the eggs. Rose sells each egg for $1 for 3 eggs, which is equivalent to $0.333... per egg.", "refs": ["s2"]}
```

anchor

```text
{"step": "Next, calculate the total revenue from selling the eggs. Rose sells each egg for $1 for 3 eggs, which is equivalent to $1/3 per egg.", "refs": ["s2"]}
```

### self_repair

终局奖励 0.0，最终答案 `None`。

anchor

```text
{"step": "The total cost is calculated by multiplying the number of dozens by the cost per dozen: 5 * $2.40 = $12.00.", "refs": ["s1"], "goal": "Calculate Rose's profit.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Next, calculate the total revenue from selling the eggs. Rose sells each egg for $1 for 3 eggs, which is equivalent to $0.333... per egg.", "refs": ["s2"]}
```

anchor

```text
{"step": "Next, calculate the total revenue from selling the eggs. Rose sells each egg for $1 for 3 eggs, which is equivalent to $1/3 per egg.", "refs": ["s2"]}
```

### self_repair_extended

终局奖励 0.0，最终答案 `None`。

anchor

```text
{"step": "The total cost is calculated by multiplying the number of dozens by the cost per dozen: 5 * $2.40 = $12.00.", "refs": ["s1"], "goal": "Calculate Rose's profit.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Next, calculate the total revenue from selling the eggs. Rose sells each egg for $1 for 3 eggs, which is equivalent to $1/3 per egg.", "refs": ["s2"]}
```

anchor

```text
{"step": "Next, calculate the total revenue from selling the eggs. Rose sells each egg for $1 for 3 eggs, which is equivalent to $1/3 per egg.", "refs": ["s2"]}
```

### help_submit

终局奖励 1.0，最终答案 `$8.00`。

specialist

```text
{
  "work": "The Anchor's reasoning has identified the initial step of calculating the total cost of the eggs, which is correct. However, the reasoning is incomplete as it does not proceed to calculate the selling price, determine the profit, or identify any potential errors or omissions. To fully address the question, we need to calculate both the total cost and the total revenue from selling the eggs, then subtract the cost from the revenue to determine the profit. Additionally, we should check for any possible miscalculations or assumptions that might affect the final result.",
  "draft": "To calculate Rose's profit, we first need to determine the total cost of the eggs and then the total revenue from selling them. The total cost is calculated as follows: Rose bought five dozens of eggs at $2.40 per dozen, so the total cost is 5 * $2.40 = $12.00. Next, we calculate the total revenue. Rose sells each egg for $1 for 3 eggs, which means the price per egg is $1/3 = $0.333... (approximately $0.33). She has 5 dozens of eggs, which is 60 eggs, so the total revenue is 60 * $0.33 = $19.80. The profit is then the total revenue minus the total cost: $19.80 - $12.00 = $7.80. Therefore, Rose's profit is $7.80."
}
```

anchor

```text
{"step": "Calculate the total cost of the eggs. Rose bought five dozens of eggs at $2.40 per dozen, so the total cost is 5 * $2.40 = $12.00.", "refs": [], "goal": "Calculate Rose's profit.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Next, calculate the total revenue from selling the eggs. Rose sells each egg for $1 for 3 eggs, which is equivalent to $1/3 per egg.", "refs": ["s2"], "goal": "Calculate Rose's profit.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Calculate the total revenue from selling the eggs. Rose has 5 dozen eggs, which is 5 * 12 = 60 eggs. She sells each egg for $1/3, so the total revenue is 60 * ($1/3) = $20.00.", "refs": ["s2"], "goal": "Calculate Rose's profit.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Calculate Rose's profit by subtracting the total cost from the total revenue. Total revenue is $20.00 and total cost is $12.00, so the profit is $20.00 - $12.00 = $8.00.", "refs": ["s4"], "goal": null, "answer": "$8.00", "help_request": null, "invalidate_answer": false}
```

## refinement_r3 · gsm8k:1165 · help_submit

John collects garbage from 3 different apartment complexes.  The first two have 200 apartments each and the last one is 60% bigger than the other two combined.  He collects garbage 3 times a week from each place and he gets paid $.40 per collection for each apartment.  How much money does he make in a week?

参考答案 `1248`

完整输入和原始证据见 historical_cases.jsonl。

共享前缀与动态请求

```json
{
  "steps": [
    {
      "step_id": "s1",
      "record": {
        "step": "First, calculate the number of apartments in the last complex. The last complex is 60% bigger than the other two combined. The first two complexes have 200 apartments each, so combined they have 200 + 200 = 400 apartments. The last complex has 60% more than 400, which is 400 * 0.6 = 240. Therefore, the last complex has 400 + 240 = 640 apartments.",
        "refs": [],
        "goal": "Calculate how much money John makes in a week.",
        "answer": null,
        "help_request": null,
        "invalidate_answer": false
      }
    }
  ],
  "candidate": null,
  "request": {
    "instruction": "Identify potential errors, omissions, or unresolved issues in the current reasoning and provide useful assistance.",
    "target": "s1",
    "required_refs": [],
    "fallback": true,
    "reason": "anchor_help_request_null_or_missing"
  }
}
```

### continue

终局奖励 0.0，最终答案 `$374.40`。

anchor

```text
{"step": "Next, calculate the total number of apartments across all three complexes. The first two complexes have 200 apartments each, and the last complex has 640 apartments. Therefore, the total number of apartments is 200 + 200 + 640 = 1040 apartments.", "refs": ["s1"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Now, calculate the total number of garbage collections John makes in a week. He collects garbage 3 times a week from each complex. There are 3 complexes, so the total number of collections is 3 * 3 = 9 collections per week.", "refs": ["s2"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Finally, calculate John's weekly earnings. He gets paid $.40 per collection for each apartment. The total number of collections is 9, and the total number of apartments is 1040. Therefore, his weekly earnings are 9 * 1040 * 0.40 = $374.40.", "refs": ["s3"], "goal": null, "answer": "$374.40", "help_request": null, "invalidate_answer": false}
```

### self_repair

终局奖励 0.0，最终答案 `$374.40`。

anchor

```text
{"step": "Next, calculate the total number of apartments across all three complexes. The first two complexes have 200 apartments each, totaling 400 apartments. The last complex has 640 apartments. Therefore, the total number of apartments is 400 + 640 = 1040 apartments.", "refs": ["s1"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Now, calculate the total number of garbage collections per week. John collects garbage 3 times a week from each complex. There are 3 complexes, so the total number of collections is 3 * 3 = 9 collections per week.", "refs": ["s2"]}, {"id": "s3", "step": "Finally, calculate the total earnings per week. John gets paid $.40 per collection for each apartment. The total number of apartments is 1040, and the total number of collections is 9. Therefore, the total earnings are 1040 * 9 * 0.40 = 3744.", "refs": ["s2"]}, {"id": "s4", "step": "The total amount of money John makes in a week is $3744.", "refs": ["s3"], "answer": "3744", "invalidate_answer": false}
```

anchor

```text
{"step": "Now, calculate the total number of garbage collections John makes in a week. He collects garbage 3 times a week from each complex. There are 3 complexes, so the total number of collections is 3 collections/complex * 3 complexes = 9 collections per week.", "refs": ["s2"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Finally, calculate John's weekly earnings. He gets paid $.40 per collection for each apartment. The total number of apartments is 1040, and he makes 9 collections per week. Therefore, his weekly earnings are 1040 apartments * 9 collections/week * $0.40 per collection = $374.40.", "refs": ["s3"], "goal": null, "answer": "$374.40", "help_request": null, "invalidate_answer": false}
```

### self_repair_extended

终局奖励 0.0，最终答案 `$374.40`。

anchor

```text
{"step": "Next, calculate the total number of apartments across all three complexes. The first two complexes have 200 apartments each, totaling 400 apartments. The last complex has 640 apartments. Therefore, the total number of apartments is 400 + 640 = 1040 apartments.", "refs": ["s1"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Now, calculate the total number of garbage collections per week. John collects garbage 3 times a week from each complex. There are 3 complexes, so the total number of collections is 3 collections/complex * 3 complexes = 9 collections per week.", "refs": ["s2"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Finally, calculate John's weekly earnings. He gets paid $.40 per collection for each apartment. The total number of apartments is 1040, and he collects garbage 9 times a week. Therefore, the total earnings are 1040 apartments * 9 collections/week * $0.40/collection = $374.40 per week.", "refs": ["s3"], "goal": null, "answer": "$374.40", "help_request": null, "invalidate_answer": false}
```

### help_submit

终局奖励 1.0，最终答案 `1248`。

specialist

```text
{
  "work": "The Anchor's reasoning correctly calculates the number of apartments in the last complex. The first two complexes have 200 apartments each, totaling 400. The last complex is 60% bigger than the other two combined, so it has 400 * 1.6 = 640 apartments. The total number of apartments is 200 + 200 + 640 = 1040. John collects garbage 3 times a week from each place, so the total number of collections is 1040 * 3 = 3120. He gets paid $0.40 per collection, so his weekly earnings are 3120 * 0.40 = $1248. This calculation is correct and complete, with no errors or omissions identified.",
  "draft": "The calculation for the last complex is correct: 200 + 200 = 400 apartments, and 400 * 1.6 = 640 apartments. Total apartments: 200 + 200 + 640 = 1040. Total collections: 1040 * 3 = 3120. Weekly earnings: 3120 * 0.40 = $1248. No errors or omissions found in the reasoning."
}
```

anchor

```text
{"step": "Calculate the total number of apartments across all three complexes. The first two complexes have 200 apartments each, and the last complex has 640 apartments. Total apartments = 200 + 200 + 640 = 1040.", "refs": ["s1"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Calculate the total number of garbage collections John makes in a week. He collects garbage 3 times a week from each complex. Total collections = 3 * 1040 = 3120.", "refs": ["s2"], "goal": "Calculate John's weekly earnings.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Calculate John's weekly earnings. He gets paid $.40 per collection for each apartment. Total earnings = 3120 * 0.40 = 1248.", "refs": ["s3"], "goal": null, "answer": "1248", "help_request": null, "invalidate_answer": false}
```

## phase2b · math500:test/precalculus/697.json · help_base

Simplify $\tan 100^\circ + 4 \sin 100^\circ.$

参考答案 `-\sqrt{3}`

完整输入和原始证据见 historical_cases.jsonl。

共享前缀与动态请求

```json
{
  "steps": [
    {
      "step_id": "s1",
      "record": {
        "step": "To simplify $\\tan 100^\\circ + 4 \\sin 100^\\circ$, we can use trigonometric identities and numerical approximations. First, note that $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ}$. This allows us to express the entire expression in terms of sine and cosine. Next, we can compute the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ to evaluate the expression.",
        "refs": [],
        "goal": "Simplify $\\tan 100^\\circ + 4 \\sin 100^\\circ$",
        "answer": null,
        "help_request": {
          "instruction": "Calculate the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ using a calculator or known approximations.",
          "target": "s1",
          "required_refs": [],
          "fallback": false,
          "reason": null
        },
        "invalidate_answer": false
      }
    }
  ],
  "candidate": null,
  "request": {
    "instruction": "Calculate the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ using a calculator or known approximations.",
    "target": "s1",
    "required_refs": [],
    "fallback": false,
    "reason": null
  }
}
```

### continue

终局奖励 0.0，最终答案 `-1.732`。

anchor

```text
{"step": "Using a calculator, we find that $\\sin 100^\\circ \\approx 0.9848$ and $\\cos 100^\\circ \\approx -0.1736$. Therefore, $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ} \\approx \\frac{0.9848}{-0.1736} \\approx -5.671$. Adding $4 \\sin 100^\\circ$ gives $-5.671 + 4(0.9848) \\approx -5.671 + 3.9392 \\approx -1.7318$.", "refs": [], "goal": null, "answer": "-1.732", "help_request": null, "invalidate_answer": false}
```

### self_repair

终局奖励 0.0，最终答案 `-1.7318`。

anchor

```text
{"step": "Using a calculator, we find that $\\sin 100^\\circ \\approx 0.9848$ and $\\cos 100^\\circ \\approx -0.1736$. Therefore, $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ} \\approx \\frac{0.9848}{-0.1736} \\approx -5.671$. Substituting these values into the expression, we get $\\tan 100^\\circ + 4 \\sin 100^\\circ \\approx -5.671 + 4(0.9848) \\approx -5.671 + 3.9392 \\approx -1.7318$.", "refs": [], "goal": null, "answer": "-1.7318", "help_request": null, "invalidate_answer": false}
```

### self_repair_extended

终局奖励 0.0，最终答案 `-1.7318`。

anchor

```text
{"step": "Using a calculator, we find that $\\sin 100^\\circ \\approx 0.9848$ and $\\cos 100^\\circ \\approx -0.1736$. Therefore, $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ} \\approx \\frac{0.9848}{-0.1736} \\approx -5.671$. Substituting these values into the expression, we get $\\tan 100^\\circ + 4 \\sin 100^\\circ \\approx -5.671 + 4(0.9848) \\approx -5.671 + 3.9392 \\approx -1.7318$.", "refs": [], "goal": null, "answer": "-1.7318", "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "The numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ have been calculated as approximately 0.9848 and -0.1736, respectively. Using these values, we computed $\\tan 100^\\circ \\approx -5.671$ and substituted into the expression $\\tan 100^\\circ + 4 \\sin 100^\\circ$ to obtain approximately -1.7318. This completes the simplification of the given expression.", "refs": ["s1", "s2"], "goal": null, "answer": "-1.7318", "help_request": null, "invalidate_answer": false}
```

### help_base

终局奖励 1.0，最终答案 `-\sqrt{3}`。

specialist

```text
{
  "work": "To calculate the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$, I will use a calculator or known approximations. The angle $100^\\circ$ is in the second quadrant, where sine is positive and cosine is negative. Using a calculator, $\\sin 100^\\circ$ is approximately 0.9848, and $\\cos 100^\\circ$ is approximately -0.1736. These values can then be used to simplify $\\tan 100^\\circ + 4 \\sin 100^\\circ$ by substituting $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ}$, resulting in $\\frac{0.9848}{-0.1736} + 4(0.9848)$. This simplifies to approximately -5.671 + 3.939 = -1.732.",
  "draft": "The numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ are approximately 0.9848 and -0.1736, respectively. Using these values, we can compute $\\tan 100^\\circ + 4 \\sin 100^\\circ$ as $\\frac{0.9848}{-0.1736} + 4(0.9848)$, which simplifies to approximately -1.732."
}
```

anchor

```text
{"step": "Using the numerical values provided, we calculate $\\tan 100^\\circ + 4 \\sin 100^\\circ$ as follows: $\\frac{0.9848}{-0.1736} + 4(0.9848) = -5.671 + 3.9392 = -1.7318$. This value is approximately equal to $-\\sqrt{3}$, which is approximately -1.732. Therefore, the simplified form of the expression is $-\\sqrt{3}$.", "refs": ["h1"], "goal": null, "answer": "-\\sqrt{3}", "help_request": null, "invalidate_answer": false}
```

## phase2b · math500:test/precalculus/697.json · help_sft

Simplify $\tan 100^\circ + 4 \sin 100^\circ.$

参考答案 `-\sqrt{3}`

完整输入和原始证据见 historical_cases.jsonl。

共享前缀与动态请求

```json
{
  "steps": [
    {
      "step_id": "s1",
      "record": {
        "step": "To simplify $\\tan 100^\\circ + 4 \\sin 100^\\circ$, we can use trigonometric identities and numerical approximations. First, note that $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ}$. This allows us to express the entire expression in terms of sine and cosine. Next, we can compute the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ to evaluate the expression.",
        "refs": [],
        "goal": "Simplify $\\tan 100^\\circ + 4 \\sin 100^\\circ$",
        "answer": null,
        "help_request": {
          "instruction": "Calculate the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ using a calculator or known approximations.",
          "target": "s1",
          "required_refs": [],
          "fallback": false,
          "reason": null
        },
        "invalidate_answer": false
      }
    }
  ],
  "candidate": null,
  "request": {
    "instruction": "Calculate the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ using a calculator or known approximations.",
    "target": "s1",
    "required_refs": [],
    "fallback": false,
    "reason": null
  }
}
```

### continue

终局奖励 0.0，最终答案 `-1.732`。

anchor

```text
{"step": "Using a calculator, we find that $\\sin 100^\\circ \\approx 0.9848$ and $\\cos 100^\\circ \\approx -0.1736$. Therefore, $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ} \\approx \\frac{0.9848}{-0.1736} \\approx -5.671$. Adding $4 \\sin 100^\\circ$ gives $-5.671 + 4(0.9848) \\approx -5.671 + 3.9392 \\approx -1.7318$.", "refs": [], "goal": null, "answer": "-1.732", "help_request": null, "invalidate_answer": false}
```

### self_repair

终局奖励 0.0，最终答案 `-1.7318`。

anchor

```text
{"step": "Using a calculator, we find that $\\sin 100^\\circ \\approx 0.9848$ and $\\cos 100^\\circ \\approx -0.1736$. Therefore, $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ} \\approx \\frac{0.9848}{-0.1736} \\approx -5.671$. Substituting these values into the expression, we get $\\tan 100^\\circ + 4 \\sin 100^\\circ \\approx -5.671 + 4(0.9848) \\approx -5.671 + 3.9392 \\approx -1.7318$.", "refs": [], "goal": null, "answer": "-1.7318", "help_request": null, "invalidate_answer": false}
```

### self_repair_extended

终局奖励 0.0，最终答案 `-1.7318`。

anchor

```text
{"step": "Using a calculator, we find that $\\sin 100^\\circ \\approx 0.9848$ and $\\cos 100^\\circ \\approx -0.1736$. Therefore, $\\tan 100^\\circ = \\frac{\\sin 100^\\circ}{\\cos 100^\\circ} \\approx \\frac{0.9848}{-0.1736} \\approx -5.671$. Substituting these values into the expression, we get $\\tan 100^\\circ + 4 \\sin 100^\\circ \\approx -5.671 + 4(0.9848) \\approx -5.671 + 3.9392 \\approx -1.7318$.", "refs": [], "goal": null, "answer": "-1.7318", "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "The numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$ have been calculated as approximately 0.9848 and -0.1736, respectively. Using these values, we computed $\\tan 100^\\circ \\approx -5.671$ and substituted into the expression $\\tan 100^\\circ + 4 \\sin 100^\\circ$ to obtain approximately -1.7318. This completes the simplification of the given expression.", "refs": ["s1", "s2"], "goal": null, "answer": "-1.7318", "help_request": null, "invalidate_answer": false}
```

### help_sft

终局奖励 1.0，最终答案 `-\sqrt{3}`。

specialist

```text
{"work": "Compute the numerical values of $\\sin 100^\\circ$ and $\\cos 100^\\circ$. Then, use these values to compute $\\tan 100^\\circ + 4 \\sin 100^\\circ$.", "draft": "Using a calculator, $\\sin 100^\\circ\\approx0.984807753012208$ and $\\cos 100^\\circ\\approx-0.1736481776669303$. Then, $\\tan 100^\\circ=\\frac{\\sin 100^\\circ}{\\cos 100^\\circ}\\approx\\frac{0.984807753012208}{-0.1736481776669303}\\approx-5.67128182960175$. Finally, $\\tan 100^\\circ + 4 \\sin 100^\\circ\\approx-5.67128182960175 + 4\\cdot0.984807753012208\\approx-5.67128182960175 + 3.939231012048832\\approx-1.732050817552918$. Thus, the value of $\\tan 100^\\circ + 4 \\sin 100^\\circ\\approx-1.732050817552918$."}
```

anchor

```text
{"step": "The numerical approximation of $\\tan 100^\\circ + 4 \\sin 100^\\circ$ is approximately $-1.732050817552918$. This value is very close to $-\sqrt{3}$, which is approximately $-1.7320508075688772$. Therefore, the expression simplifies to $-\sqrt{3}$.", "refs": ["h1"], "goal": null, "answer": "-\\sqrt{3}", "help_request": null, "invalidate_answer": false}
```

## phase2b · gsm8k:992 · help_base

Buford writes many checks every year.  Once per month he writes a check to pay the electric bill.  He also writes a check every month for the gas bill.  Twice per month he writes a check to the church.  And quarterly, he writes a check to the pest and lawn service.  How many checks does Buford write per year?

参考答案 `52`

完整输入和原始证据见 historical_cases.jsonl。

共享前缀与动态请求

```json
{
  "steps": [
    {
      "step_id": "s1",
      "record": {
        "step": "Buford writes checks for various monthly and quarterly expenses. To find the total number of checks he writes per year, we need to calculate the number of checks for each category and sum them up.",
        "refs": [],
        "goal": "Calculate the total number of checks Buford writes per year.",
        "answer": null,
        "help_request": {
          "instruction": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.",
          "target": "s1",
          "required_refs": [],
          "fallback": false,
          "reason": null
        },
        "invalidate_answer": false
      }
    }
  ],
  "candidate": null,
  "request": {
    "instruction": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.",
    "target": "s1",
    "required_refs": [],
    "fallback": false,
    "reason": null
  }
}
```

### continue

终局奖励 0.0，最终答案 `{"electric_bill": 12, "gas_bill": 12, "church": 24, "pest_and_lawn_service": 4, "total_checks": 52}`。

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, 1 check per month for the gas bill, 2 checks per month for the church, and 1 check every 3 months for the pest and lawn service. To find the total number of checks per year, we calculate each category separately and then sum them up.", "refs": ["s1"], "goal": "Calculate the total number of checks Buford writes per year.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, which amounts to 12 checks per year. Similarly, he writes 1 check per month for the gas bill, also totaling 12 checks per year. For the church, he writes 2 checks per month, which results in 24 checks per year. For the pest and lawn service, he writes 1 check every 3 months, which is equivalent to 4 checks per year.", "refs": ["s2"]}, {"step": "Adding up all the checks: 12 (electric) + 12 (gas) + 24 (church) + 4 (pest and lawn service) = 52 checks per year.", "refs": ["s2"]}, {"step": "Buford writes a total of 52 checks per year.", "refs": ["s2"], "answer": "52", "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, which amounts to 12 checks per year. Similarly, he writes 1 check per month for the gas bill, also totaling 12 checks per year. For the church, he writes 2 checks per month, which results in 24 checks per year. For the pest and lawn service, he writes 1 check every 3 months, which is equivalent to 4 checks per year.", "refs": ["s2"], "goal": null, "answer": "{\"electric_bill\": 12, \"gas_bill\": 12, \"church\": 24, \"pest_and_lawn_service\": 4, \"total_checks\": 52}", "help_request": null, "invalidate_answer": false}
```

### self_repair

终局奖励 0.0，最终答案 `{"total_checks": "4 * 12 + 4 = 52"}`。

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. Additionally, he writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. He writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "How many checks does Buford write per year?", "answer": "{\"total_checks\": \"4 * 12 + 4 = 52\"}", "help_request": null, "invalidate_answer": false}
```

### self_repair_extended

终局奖励 0.0，最终答案 `{"monthly_checks": 48, "quarterly_checks": 4, "total_checks": 52}`。

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. Additionally, he writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. Additionally, he writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.", "answer": "{\"monthly_checks\": 48, \"quarterly_checks\": 4, \"total_checks\": 52}", "help_request": null, "invalidate_answer": false}
```

### help_base

终局奖励 1.0，最终答案 `52`。

specialist

```text
{
  "work": "To calculate the total number of checks Buford writes per year, we need to consider each type of check and how frequently it occurs. First, Buford writes a check for the electric bill once per month. There are 12 months in a year, so he writes 12 checks for the electric bill. Second, he writes a check for the gas bill once per month as well, which also amounts to 12 checks per year. Third, he writes two checks per month to the church, which results in 2 checks per month multiplied by 12 months, totaling 24 checks per year. Lastly, he writes a check to the pest and lawn service quarterly, meaning four times a year. Adding these together: 12 (electric) + 12 (gas) + 24 (church) + 4 (pest and lawn) = 52 checks per year.",
  "draft": "Buford writes checks for various expenses. Calculating the checks per year: 12 (electric bill) + 12 (gas bill) + 24 (church) + 4 (pest and lawn service) = 52 checks per year."
}
```

anchor

```text
{"step": "Buford writes 1 check for the electric bill each month, which amounts to 12 checks per year. Similarly, he writes 1 check for the gas bill each month, totaling 12 checks per year. For the church, he writes 2 checks per month, resulting in 24 checks per year. Lastly, he writes 1 check for the pest and lawn service every quarter, which is 4 checks per year. Adding these together: 12 (electric) + 12 (gas) + 24 (church) + 4 (pest and lawn service) = 52 checks per year.", "refs": ["s1"], "goal": null, "answer": "52", "help_request": null, "invalidate_answer": false}
```

## phase2b · gsm8k:992 · help_sft

Buford writes many checks every year.  Once per month he writes a check to pay the electric bill.  He also writes a check every month for the gas bill.  Twice per month he writes a check to the church.  And quarterly, he writes a check to the pest and lawn service.  How many checks does Buford write per year?

参考答案 `52`

完整输入和原始证据见 historical_cases.jsonl。

共享前缀与动态请求

```json
{
  "steps": [
    {
      "step_id": "s1",
      "record": {
        "step": "Buford writes checks for various monthly and quarterly expenses. To find the total number of checks he writes per year, we need to calculate the number of checks for each category and sum them up.",
        "refs": [],
        "goal": "Calculate the total number of checks Buford writes per year.",
        "answer": null,
        "help_request": {
          "instruction": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.",
          "target": "s1",
          "required_refs": [],
          "fallback": false,
          "reason": null
        },
        "invalidate_answer": false
      }
    }
  ],
  "candidate": null,
  "request": {
    "instruction": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.",
    "target": "s1",
    "required_refs": [],
    "fallback": false,
    "reason": null
  }
}
```

### continue

终局奖励 0.0，最终答案 `{"electric_bill": 12, "gas_bill": 12, "church": 24, "pest_and_lawn_service": 4, "total_checks": 52}`。

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, 1 check per month for the gas bill, 2 checks per month for the church, and 1 check every 3 months for the pest and lawn service. To find the total number of checks per year, we calculate each category separately and then sum them up.", "refs": ["s1"], "goal": "Calculate the total number of checks Buford writes per year.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, which amounts to 12 checks per year. Similarly, he writes 1 check per month for the gas bill, also totaling 12 checks per year. For the church, he writes 2 checks per month, which results in 24 checks per year. For the pest and lawn service, he writes 1 check every 3 months, which is equivalent to 4 checks per year.", "refs": ["s2"]}, {"step": "Adding up all the checks: 12 (electric) + 12 (gas) + 24 (church) + 4 (pest and lawn service) = 52 checks per year.", "refs": ["s2"]}, {"step": "Buford writes a total of 52 checks per year.", "refs": ["s2"], "answer": "52", "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, which amounts to 12 checks per year. Similarly, he writes 1 check per month for the gas bill, also totaling 12 checks per year. For the church, he writes 2 checks per month, which results in 24 checks per year. For the pest and lawn service, he writes 1 check every 3 months, which is equivalent to 4 checks per year.", "refs": ["s2"], "goal": null, "answer": "{\"electric_bill\": 12, \"gas_bill\": 12, \"church\": 24, \"pest_and_lawn_service\": 4, \"total_checks\": 52}", "help_request": null, "invalidate_answer": false}
```

### self_repair

终局奖励 0.0，最终答案 `{"total_checks": "4 * 12 + 4 = 52"}`。

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. Additionally, he writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. He writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "How many checks does Buford write per year?", "answer": "{\"total_checks\": \"4 * 12 + 4 = 52\"}", "help_request": null, "invalidate_answer": false}
```

### self_repair_extended

终局奖励 0.0，最终答案 `{"monthly_checks": 48, "quarterly_checks": 4, "total_checks": 52}`。

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. Additionally, he writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 2 checks per month for the electric and gas bills, totaling 2 checks per month. Additionally, he writes 2 checks per month to the church, adding 2 checks per month. This results in 4 checks per month from these categories. Quarterly, he writes 1 check to the pest and lawn service, which occurs 4 times a year. To find the total number of checks per year, we calculate the monthly checks and add the quarterly checks.", "refs": [], "goal": "Calculate the total number of checks Buford writes per year by summing up the monthly and quarterly checks.", "answer": "{\"monthly_checks\": 48, \"quarterly_checks\": 4, \"total_checks\": 52}", "help_request": null, "invalidate_answer": false}
```

### help_sft

终局奖励 1.0，最终答案 `52`。

specialist

```text
{"work": "The total number of checks is calculated as 12 months*2 checks for electric and gas bill=24, 24 months*2 checks for church=48, and 4 quarters*1 check for pest and lawn service=4. The total is 24+48+4=76 checks.", "draft": "The total number of checks is 76. 12 months*2 checks for electric and gas bill=24, 24 months*2 checks for church=48, and 4 quarters*1 check for pest and lawn service=4. 24+48+4=76."}
```

anchor

```text
{"step": "Buford writes checks for various monthly and quarterly expenses. To find the total number of checks he writes per year, we need to calculate the number of checks for each category and sum them up.", "refs": [], "goal": "Calculate the total number of checks Buford writes per year.", "answer": null, "help_request": {"instruction": "Calculate the total number of checks Buford writes per year by summing up the checks for each category: monthly electric and gas bills, monthly church donations, and quarterly pest and lawn service.", "target": "s1", "required_refs": []}, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, 1 check per month for the gas bill, 2 checks per month for the church, and 1 check every 3 months for the pest and lawn service. To find the total number of checks per year, we calculate each category separately and then sum them up.", "refs": ["s1", "s2"], "goal": "Calculate the total number of checks Buford writes per year.", "answer": null, "help_request": null, "invalidate_answer": false}
```

anchor

```text
{"step": "Buford writes 1 check per month for the electric bill, which amounts to 12 checks per year. Similarly, he writes 1 check per month for the gas bill, also totaling 12 checks per year. For the church, he writes 2 checks per month, which results in 24 checks per year. For the pest and lawn service, he writes 1 check every 3 months, which is equivalent to 4 checks per year. Adding these together, the total number of checks Buford writes per year is 12 + 12 + 24 + 4 = 52 checks.", "refs": ["s3"], "goal": null, "answer": "52", "help_request": null, "invalidate_answer": false}
```


## 真实配对实验

耗时为共享服务器上的实际串行执行时间。独立上下文仍保留动态任务原文。

### screen

| 条件 | 样本 | 平均奖励 | 平均毫秒 | 格式失败 |
|---|---:|---:|---:|---:|
| autonomous | 18 | 0.667 | 42703.7 | 5 |
| autonomous_independent | 18 | 0.722 | 47929.5 | 4 |
| check | 18 | 0.722 | 47802.9 | 6 |
| continue | 18 | 0.667 | 23919.4 | 2 |
| directed | 18 | 0.778 | 43368.7 | 5 |
| extended | 18 | 0.778 | 29174.3 | 4 |
| independent | 18 | 0.722 | 49821.7 | 5 |
| independent_compare | 18 | 0.667 | 49924.9 | 14 |
| integration_abort | 16 | 0.688 | 17470.7 | 0 |
| integration_comparison | 16 | 0.812 | 13751.0 | 1 |
| integration_critical | 16 | 0.812 | 12610.5 | 0 |
| integration_direct | 16 | 0.812 | 14135.8 | 1 |
| recompute | 18 | 0.722 | 48053.7 | 5 |
| repair | 18 | 0.722 | 20311.8 | 5 |
| solver | 18 | 0.722 | 50213.7 | 6 |

### integration_history

| 条件 | 样本 | 平均奖励 | 平均毫秒 | 格式失败 |
|---|---:|---:|---:|---:|
| integration_abort | 12 | 0.917 | 7858.0 | 0 |
| integration_comparison | 12 | 0.917 | 9240.5 | 0 |
| integration_critical | 12 | 0.917 | 8892.9 | 0 |
| integration_direct | 12 | 0.917 | 8637.6 | 0 |

### fresh_validation

| 条件 | 样本 | 平均奖励 | 平均毫秒 | 格式失败 |
|---|---:|---:|---:|---:|
| autonomous_independent | 8 | 0.750 | 30618.6 | 1 |
| continue | 8 | 0.625 | 16486.2 | 0 |
| directed | 8 | 0.750 | 42953.9 | 1 |
| extended | 8 | 0.750 | 26465.7 | 2 |
| repair | 8 | 0.750 | 18623.6 | 0 |

### matched_sampling

| 条件 | 样本 | 平均奖励 | 平均毫秒 | 格式失败 |
|---|---:|---:|---:|---:|
| autonomous_independent | 18 | 0.556 | 54514.1 | 3 |
| continue | 18 | 0.667 | 33142.4 | 2 |
| directed | 18 | 0.722 | 53087.7 | 5 |
| extended | 18 | 0.778 | 38755.2 | 4 |
| repair | 18 | 0.722 | 27597.7 | 5 |

### combined_validation

| 条件 | 样本 | 平均奖励 | 平均毫秒 | 格式失败 |
|---|---:|---:|---:|---:|
| autonomous_independent | 12 | 0.750 | 40526.5 | 4 |
| autonomous_independent_critical | 12 | 0.750 | 40514.3 | 4 |
| continue | 12 | 0.750 | 20861.8 | 4 |
| directed | 12 | 0.750 | 35979.9 | 2 |
| extended | 12 | 0.750 | 25995.7 | 3 |
| repair | 12 | 0.833 | 22508.9 | 5 |


## 高难度编程题官方评分

| 条件 | HumanEval | HumanEval+ | LiveCodeBench |
|---|---:|---:|---:|
| standard | 26/30 | 25/30 | 6/20 |
| anchor_only | 6/30 | 6/30 | 1/20 |
| self_repair | 6/30 | 6/30 | 1/20 |
| self_repair_extended | 6/30 | 6/30 | 2/20 |
| internal_restart | 8/30 | 8/30 | 1/20 |
| help | 12/30 | 12/30 | 1/20 |

使用 EvalPlus 0.3.1 与 HumanEval+ v0.1.10。LiveCodeBench 使用固定提交 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 的官方检查器和固定 test6 分片。缺失候选作为失败计入。Docker 套接字不可访问，隔离由 Bubblewrap 提供。
