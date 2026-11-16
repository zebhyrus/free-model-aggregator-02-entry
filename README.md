# free-model-aggregator-02-entry

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/zebhyrus/free-model-aggregator-02-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/zebhyrus/free-model-aggregator-02-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that aggregates the best free models behind a single API: it answers OpenAI Chat Completions and Anthropic Messages (JSON and streaming) and every completion comes from a real free model dialled through the grader's gateway. Routing is cause-aware — only models meeting the prompt's needs (tools, JSON mode, images, context) are eligible, best house rating first — with a retry and failover to the next-best model, reported in the `x-makefun-upstream-failures` trail header.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9srzvfjv&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
