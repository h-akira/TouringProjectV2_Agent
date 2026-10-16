# TouringProjectV2_Agent

TouringProjectV2 のエージェント（Bedrock AgentCore Runtime＋Strands・`us-east-1`）。
Backend から質問（プロンプトと数値の現在地）を受け、Web 検索と周辺検索を使って、読み上げ向けの回答を返す。
プロジェクト全体の要件・方針・契約は [docs-parent/](docs-parent/README.md)（親リポジトリの写し。編集しない）。

## 構成

| 場所 | 中身 |
|---|---|
| `agentcore/agentcore.json` | Runtime（`agentcore_trv2_dev_ask`）と Gateway（`gwTrv2DevMain`・Web 検索のコネクタ）。環境変数とアイドルの時間切れ（3600秒） |
| `agentcore/cdk/` | CLI が生成した CDK。`lib/cdk-stack.ts` で Runtime に Location と SSM の権限を足している（生成し直したら足し直す） |
| `app/agentcore_trv2_dev_ask/` | エージェント本体。`main.py`（組み立て）・`handler.py`（1回の呼び出しの流れ）・`history.py`・`hooks.py`・`stream.py`・`model_config.py`・`tools/` |
| `scripts/smoke.py` | デプロイ後に決まった質問を投げて確かめる |

## 振る舞い

- 会話の文脈は `runtimeSessionId`（Backend が渡す会話の識別子）ごとに、Runtime のプロセスの中に持つ。回答のあと、ツールの呼び出しと結果を取り除き、質問と回答の文だけを直近10メッセージまで残す。
- モデル ID は SSM `/trv2/dev/agent-model-id`（us-east-1）から読む（60秒キャッシュ）。読めないときは失敗を返す。
- ツールは1回の質問につき4回まで。ツールの結果は合計3000文字までに切り詰める。
- 応答は SSE で `text`・`tool`・`done`・`error` のイベントを返す（親の `docs/07_units_contracts.md` UC-8）。

## テスト

```sh
cd app/agentcore_trv2_dev_ask && uv run pytest
```

## デプロイ

前提: `agentcore/aws-targets-sample.json` のアカウント ID を埋めて `agentcore/aws-targets.json` に置く（追跡しない）。

```sh
AWS_PROFILE=touring AWS_REGION=us-east-1 npx -y @aws/agentcore@0.31.1 deploy -y
ARN=$(AWS_PROFILE=touring aws bedrock-agentcore-control list-agent-runtimes --region us-east-1 \
  --query "agentRuntimes[?agentRuntimeName=='trv2Agent_agentcore_trv2_dev_ask'].agentRuntimeArn" --output text)
AWS_PROFILE=touring uv run --project app/agentcore_trv2_dev_ask python scripts/smoke.py "$ARN"
AWS_PROFILE=touring aws ssm put-parameter --region ap-northeast-1 --name /trv2/dev/agent-runtime-arn --type String --overwrite --value "$ARN"
```

ARN が変わったら Backend もデプロイし直す。

## モデルの切り替え

最初のモデルは `us.anthropic.claude-sonnet-4-6`（構築時にこのアカウントで使えた Sonnet 系の最新。5 系の推論プロファイルは一覧に出るが、呼ぶと「not available for this account」になった）。

```sh
AWS_PROFILE=touring aws ssm put-parameter --region us-east-1 --name /trv2/dev/agent-model-id --type String --overwrite --value <推論プロファイルの ID>
```

60秒以内に反映される。切り替えたら `scripts/smoke.py` で通ることを確かめる。
