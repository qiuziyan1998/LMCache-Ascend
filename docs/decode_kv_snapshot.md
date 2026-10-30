# D 端完整 KV 接收快照

用于对比同一会话相邻两轮：D 收到的旧前缀有没有发生变化，变化位于哪个 TP rank、KV group、层、token 和字节。
不计算 hash，不抽样，不增加诊断后台线程或队列。文件保留全部有效 token 的原始字节，包括 latent 的 nope/rope、indexer key 和 C8 scale；不保存物理槽位中的无效填充。

## 开启

只在两台 D 机器的服务启动环境中加入：

```bash
export LMCACHE_DECODE_KV_SNAPSHOT_DIR=/workspace/fsi_lab/sqh/vllm-ascend/decode-kv-snapshots
```

P 不设置该变量。为了复现原始问题，P/D 均关闭旧的 `LMCACHE_PREFILL_STORE_DEBUG`、`LMCACHE_PREFILL_FENCE_DEBUG` 和 `LMCACHE_PREFILL_AUDIT`，保留正常的图与生产参数。目录应当位于持久磁盘，不要放到 `/dev/shm`；容器中使用的目录需与收集脚本能访问的目录对应，或收集时使用 `--container`。

该变量在 worker 初始化时读取。取消变量并重启 D 服务即关闭；P 即使继承此变量也不会启用。关闭时不导入保存模块、不创建文件、不增加 NPU 操作或同步，只有 D cold-load 完成位置的布尔判断。

## 保存边界与范围

接在当前双 KV group 的 D cold-load 完成位置：包括普通 compact、resident 和进入同一完成路径的 live-source 加载。原有加载返回并完成接收 fence 后，在请求可被首次 decode 调度之前，同步保存一次：

- Group 0：读取请求持有的完整 CPU latent source。
- Group 1：读取已经载入 NPU 的完整 indexer 历史，按照实际 slot mapping/C8 block mapping 转回逻辑 token 顺序。每次最多读回 4096 个 token，不另建完整历史的 NPU 副本。
- 全部注册层均保存，包括注册在 group 内的 MTP 层。每个 TP rank 单独保存。
- 范围为 D 实际接收的 `[0, token_count)`，不是把当前新算的 token 或后续 decode 输出混进历史。D 需要重算的末尾 token 可能不在接收范围内，文件明确记录实际长度。

不拦截模型层的 Python forward，因而不要求关闭图。未经过此 cold-load 完成边界的请求不会生成快照，收集/比较会明确报缺失，不能当作一致。

保存会延迟 D 请求就绪和 TTFT，并可能通过背压改变并发节奏；不会在 P 插入同步。仍不能保证开启保存后原问题必然复现。完整 latent 500k × 79 × 576 × 2 字节约 45.5 GB/TP rank，另外还有 indexer。多个 rank、请求、轮次按份数累加。不要把诊断运行当作性能测试。

## 文件与完成标记

```text
decode-kv-snapshots/<公共 request UUID>/<host>-dp<DP>-tp<TP>-p<PID>/
    manifest.json
    tokens.json
    g0-l0-c0.bin
    g0-l0-c1.bin
    ...
    g1-l21-c0.bin
```

`g` 是 KV group，`l` 是该 group 的层序号，真实层名在 `manifest.json` 的 `layer_names` 中。`c` 是组件序号，dtype 和每 token 字节数也在 manifest 中。数据文件按逻辑 token 连续排列，无压缩、无精度转换。

日志每个请求/rank 仅输出 `begin`、`latent_done`、`complete`，出错输出 `[D_KV_SNAPSHOT] ... failed`。只有全部文件关闭并检查各层长度后才写 `status=complete`。出错保留不完整目录并记录错误，不触发生产加载的重试/回退。

## 比较与结论

使用 vllm-ascend 的 `tools/decode_kv_compare.py`，传两轮**同一 session** 的 response UUID。脚本按 TP rank 对齐，允许两轮路由到不同 D 机器；按 `tokens.json` 计算实际公共 token 前缀，不按字符数或物理槽号对齐。

`received_history_identical` 表示已接收的公共历史所有字节相同；它不证明后续 latent H2D、稀疏映射或模型运算正确。`history_differs` 给出差异位置；重算、模型/参数变化也可能产生数值差异，不能只凭不同就认定内存被覆盖。新增区间没有上轮对应值，单独提供字节数、零字节数和浮点非有限值数量，不作精度判定。文件缺失、rank 不全或仍在写入时报 `incomplete`。

本地已验证 500k/16k 分块、完整字节保留、C8 映射与 scale、完成事件前为错误数据/事件后才写正确数据、缺失和篡改检测。NPU 实机接收与读回仍需服务端验证。
