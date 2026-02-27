# task-external

Claude Code の Task ツールと同じスキーマを持つ MCP サーバー。Task の呼び出しを OpenAI Codex SDK に委譲し、外部エージェントとしてタスクを実行します。

## セットアップ

```bash
npm install
npm run build
```

## Claude Code での設定

`~/.claude/settings.json` の `mcpServers` に追加:

```json
{
  "mcpServers": {
    "task-external": {
      "command": "node",
      "args": ["/path/to/task-external/build/index.js"]
    }
  }
}
```

## 環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `TASK_EXTERNAL_DEFAULT_MODEL` | `gpt-5.3-codex` | デフォルトモデル |
| `TASK_EXTERNAL_MODEL_OPUS` | `gpt-5.3-codex` | opus 指定時のモデル |
| `TASK_EXTERNAL_MODEL_SONNET` | `gpt-5.3-codex` | sonnet 指定時のモデル |
| `TASK_EXTERNAL_MODEL_HAIKU` | `gpt-5.3-codex-spark` | haiku 指定時のモデル |

## エージェントタイプ

- **general-purpose** — 汎用エージェント（読み書き可、承認あり）
- **Explore** — コードベース探索用（読み取り専用）
- **Plan** — 設計・計画用（読み取り専用）

## ライセンス

MIT
