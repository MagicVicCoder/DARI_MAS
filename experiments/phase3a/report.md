# DARI-MAS Phase 3A

## C0 资源成本

独立验证平均相对误差 3.38%。成本确认状态为 True。
成本单位为串行执行毫秒，覆盖模型调用和同步的策略执行耗时，模型加载在计时外。Anchor 与 Specialist 分别拟合，实际标签使用同步墙钟时间并扣除共享历史。GPU 共享负载可能改变测量值，独占 GPU 时间和跨设备资源用量需要另行测量。每 10 秒归一化为 1。
- A 验证 MAE 691.26 ms。
- S 验证 MAE 74.24 ms。

## C1–C2 实测数据

A/S 均使用冻结的原始 Qwen3-8B。评价和语法修复沿用 Phase 2B。分支采用 Narrow Short Help，两阶段上限，固定续推优先 Finish，否则 Continue，Specialist 优先提交合法 Draft，否则 Abort。未覆盖 Wide 和第三阶段。单次 rollout 标签是 Monte Carlo 观察值。

- train 实际覆盖 36 题，{'anchor': 55, 'specialist': 43}，333 次 rollout。
- anchor/continue 共 55 项，奖励均值 0.745，未来耗时均值 30825.4 ms。
- anchor/finish 共 30 项，奖励均值 0.800，未来耗时均值 0.0 ms。
- anchor/help 共 47 项，奖励均值 0.755，未来耗时均值 53601.4 ms。
- anchor/self_repair 共 55 项，奖励均值 0.745，未来耗时均值 28811.5 ms。
- specialist/abort 共 43 项，奖励均值 0.767，未来耗时均值 21908.5 ms。
- specialist/expand 共 43 项，奖励均值 0.814，未来耗时均值 41322.7 ms。
- specialist/submit 共 43 项，奖励均值 0.814，未来耗时均值 19943.0 ms。
- abort/expand 同前缀胜/平/负 1/39/3。
- abort/submit 同前缀胜/平/负 1/39/3。
- continue/finish 同前缀胜/平/负 1/29/0。
- continue/help 同前缀胜/平/负 3/41/3。
- continue/self_repair 同前缀胜/平/负 1/53/1。
- expand/submit 同前缀胜/平/负 0/43/0。
- finish/help 同前缀胜/平/负 0/24/1。
- finish/self_repair 同前缀胜/平/负 0/29/1。
- help/self_repair 同前缀胜/平/负 3/41/3。

- dev 实际覆盖 12 题，{'anchor': 16, 'specialist': 15}，104 次 rollout。
- anchor/continue 共 16 项，奖励均值 0.875，未来耗时均值 8525.2 ms。
- anchor/finish 共 12 项，奖励均值 0.833，未来耗时均值 0.0 ms。
- anchor/help 共 15 项，奖励均值 0.867，未来耗时均值 30089.2 ms。
- anchor/self_repair 共 16 项，奖励均值 0.875，未来耗时均值 8169.7 ms。
- specialist/abort 共 15 项，奖励均值 0.867，未来耗时均值 8649.7 ms。
- specialist/expand 共 15 项，奖励均值 0.867，未来耗时均值 28758.6 ms。
- specialist/submit 共 15 项，奖励均值 0.867，未来耗时均值 9488.5 ms。
- abort/expand 同前缀胜/平/负 0/15/0。
- abort/submit 同前缀胜/平/负 0/15/0。
- continue/finish 同前缀胜/平/负 1/11/0。
- continue/help 同前缀胜/平/负 0/15/0。
- continue/self_repair 同前缀胜/平/负 0/16/0。
- expand/submit 同前缀胜/平/负 0/15/0。
- finish/help 同前缀胜/平/负 0/10/1。
- finish/self_repair 同前缀胜/平/负 0/11/1。
- help/self_repair 同前缀胜/平/负 0/15/0。

## C3–C4 训练与开发集评价

- full 最佳开发集步数 5，Reward MAE 0.282，MSE 0.118。
- full anchor/full 奖励 0.875，效用 0.790，regret 0.058。
- full anchor/fixed 奖励 0.812，效用 0.793，regret 0.055。
- full anchor/random 奖励 0.875，效用 0.777，regret 0.071。
- full anchor/oracle 奖励 0.875，效用 0.848，regret 0.000。
- full anchor/restricted_oracle 奖励 0.875，效用 0.848，regret 0.000。
- full anchor/reward_oracle 奖励 0.875，效用 0.846，regret 0.002。
- full specialist/full 奖励 0.867，效用 0.780，regret 0.008。
- full specialist/fixed 奖励 0.867，效用 0.772，regret 0.016。
- full specialist/random 奖励 0.867，效用 0.697，regret 0.091。
- full specialist/oracle 奖励 0.867，效用 0.788，regret 0.000。
- full specialist/restricted_oracle 奖励 0.867，效用 0.772，regret 0.016。
- full specialist/reward_oracle 奖励 0.867，效用 0.772，regret 0.016。
- numeric 最佳开发集步数 65，Reward MAE 0.286，MSE 0.122。
- numeric anchor/numeric 奖励 0.812，效用 0.794，regret 0.053。
- numeric anchor/fixed 奖励 0.812，效用 0.793，regret 0.055。
- numeric anchor/random 奖励 0.875，效用 0.777，regret 0.071。
- numeric anchor/oracle 奖励 0.875，效用 0.848，regret 0.000。
- numeric anchor/restricted_oracle 奖励 0.875，效用 0.848，regret 0.000。
- numeric anchor/reward_oracle 奖励 0.875，效用 0.846，regret 0.002。
- numeric specialist/numeric 奖励 0.867，效用 0.772，regret 0.016。
- numeric specialist/fixed 奖励 0.867，效用 0.772，regret 0.016。
- numeric specialist/random 奖励 0.867，效用 0.697，regret 0.091。
- numeric specialist/oracle 奖励 0.867，效用 0.788，regret 0.000。
- numeric specialist/restricted_oracle 奖励 0.867，效用 0.772，regret 0.016。
- numeric specialist/reward_oracle 奖励 0.867，效用 0.772，regret 0.016。

## C5 独立闭环

- anchor 12 题，平均奖励 0.667，平均耗时 20209.1 ms，动作分布 {'finish': 12, 'continue': 5}。
- bootstrap 12 题，平均奖励 0.667，平均耗时 20729.2 ms，动作分布 {'finish': 12, 'continue': 5}。
- extended 12 题，平均奖励 0.583，平均耗时 26867.2 ms，动作分布 {'finish': 11, 'self_repair': 6, 'continue': 5}。
- help 12 题，平均奖励 0.750，平均耗时 31474.2 ms，动作分布 {'finish': 12, 'help': 4, 'submit': 4, 'continue': 3}。
- learned 12 题，平均奖励 0.833，平均耗时 78976.5 ms，动作分布 {'continue': 84}。
- repair 12 题，平均奖励 0.583，平均耗时 30607.3 ms，动作分布 {'finish': 11, 'self_repair': 4, 'continue': 7}。

## 证据与限制

逐分支 JSON 保存完整生成轨迹、候选、终止原因与资源明细。训练目录保存按动作的成本误差、奖励校准和事后 Oracle 数据。Oracle 基于当前有限采样，固定续推下的离线价值与学习策略部署时的价值可能不同。
回归检查记录见 logs/regression.log。历史 Phase 结果及权重独立保存。
