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

- 会話の文脈は `runtimeSessionId`（Backend が渡すセッションの識別子。ユーザー名と会話の識別子から作る値。親の `docs/07_units_contracts.md` UC-8）ごとに、Runtime のプロセスの中に持つ。回答のあと、ツールの呼び出しと結果を取り除き、質問と回答の文だけを直近10メッセージまで残す。
- モデル ID は SSM `/trv2/dev/agent-model-id`（us-east-1）から読む（60秒キャッシュ）。読めないときは失敗を返す。
- モデル ID の読み込みとエージェントの組み立て（MCP の接続）は、イベントループを止めないよう別のスレッドで行う。
- ツールは1回の質問につき4回まで。ツールの結果は合計3000文字までに切り詰める。周回の上限（8）に達して回答の文が無いときは、「調べきれませんでした。」を回答として返す。モデルの出力のトークンの上限（`BEDROCK_MAX_TOKENS`）で止まったときは、最後のツールの呼び出しより後ろの途中までの文を回答とし、文が無ければ（ツールの呼び出しの途中で止まったときを含む。その前置きは回答にしない）同じ決まった文を返す。エージェントの組み立ては、MCP の接続の開始がぶつからないよう1つずつ行う。
- Web 検索の MCP の接続は、質問ごとに作り直すエージェントをまたいで使い回す。つなげなかったとき・つないだ接続が生きていないとき・前の質問で Web 検索が失敗したとき・`AGENT_MCP_MAX_IDLE_SECONDS`（既定600秒）より長く使わなかったときは、次の質問の前につなぎ直す。接続は `tools/web_search.WebSearchConnection` がプロセスの中で持ち続ける（Strands は、質問ごとのエージェントが捨てられると接続を止めるため）。つなぎ直しの判断は Strands の内部の名前（`_connection_failed`・`_tool_provider_started`・`_is_session_active`）を読むので、Strands の版を上げたら `tests/test_web_search.py` の本物の `MCPClient` のテストで確かめる。`AGENT_MCP_MAX_IDLE_SECONDS=0` にすると、質問ごとにつなぎ直す。
- 呼び出しは、Backend が渡す `deadlineSeconds`（無い・変なときは `AGENT_DEFAULT_DEADLINE_SECONDS`、既定100秒）で打ち切り、`timeout` のエラーを返して履歴を残さない。同じセッションへの呼び出しは、錠（`history.SessionLocks`）で1つずつ処理し、錠の待ちも締め切りに数える（[adr/0014](../adr/0014-ask-time-budget-from-acceptance.md)）。
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

いまのモデルは `us.anthropic.claude-sonnet-4-6`。切り替えるときは設定値を変える。

```sh
AWS_PROFILE=touring aws ssm put-parameter --region us-east-1 --name /trv2/dev/agent-model-id --type String --overwrite --value <推論プロファイルの ID>
```

60秒以内に反映される。切り替えたら `scripts/smoke.py` で通ることを確かめる。

## 課題

### Claude 5 系のモデルを使えない（2026-10-16 時点）

Sonnet 5.5 は Sonnet 4.6 より約3割安く（us-east-1 の `us.` で 100万トークンあたり入力 2.2・出力 11 米ドル。4.6 は 3.3・16.5）、Haiku 5.5 は Haiku 4.5 の 1/10 の料金なので、使えるようになったら切り替えたい。

- 症状: Sonnet 5.5・Haiku 5.5・Sonnet 5・Opus 5・Fable 5.1 を呼ぶと、`us.`・`global.` の推論プロファイルとも、CLI とコンソールのプレイグラウンドのどちらからでも `AccessDeniedException: ... is not available for this account. ... contact AWS Sales` になる。Sonnet 4.6・Haiku 4.5 は同じユーザー・同じリージョンで使える。
- 状態: `aws bedrock get-foundation-model-availability` で、5 系は `agreementAvailability` だけが `NOT_AVAILABLE`（`authorizationStatus` は `AUTHORIZED`、利用権と提供地域は `AVAILABLE`）。呼び出したユーザーには Marketplace の購読の権限があるが、最初の呼び出しでの自動の契約は結ばれなかった。
- 原因（推定）: IAM や SCP ではなく、AWS 側のアカウント単位の提供の制限。AWS のドキュメントは「一部のモデルはアカウント単位の追加のアクセスが必要で、条件を満たしても拒否されるなら営業窓口へ」としている。コミュニティの報告では、AWS の営業窓口から「Claude 5 系は利用実績と支出額に応じて段階的に開放している。申請と承認を経て使えるようになる」という回答があった（公式の説明ではない）。
- 対応: 当面は Sonnet 4.6 で運用する。5 系を使うには、エラー文にある AWS の営業窓口（https://aws.amazon.com/contact-us/sales-support/ ）に申請する（回答まで数週間かかった事例がある）。開放されたかは次で確かめる。

  ```sh
  AWS_PROFILE=touring aws bedrock get-foundation-model-availability --region us-east-1 \
    --model-id anthropic.claude-sonnet-5-5 --query agreementAvailability.status
  ```

  `AVAILABLE` になったら、上の手順で設定値を `us.anthropic.claude-sonnet-5-5` に変え、`scripts/smoke.py` で確かめる。
