# fmeta

OpenMM をエンジンとする funnel metadynamics（fun-metaD）フレームワークです。**タンパク質の PDB と、結合ポーズに置かれたリガンドの SDF の 2 つ**を渡すと、ファネルの定義・系の構築（水溶液または脂質二重膜）・平衡化・本計算・絶対結合自由エネルギーの算出までを既定設定で実行します。系の構築も計算も OpenMM だけで行い、バイアスと拘束は OpenMM の Custom Force なので GPU プラットフォーム（CUDA / OpenCL）上でそのまま動きます。

```bash
# 可溶性タンパク質
fmeta auto --protein protein.pdb --ligand ligand.sdf -o run1

# 膜タンパク質（GPCR など）: OpenMM の Modeller.addMembrane で POPC 膜に埋め込む
fmeta auto --protein gpcr_oriented.pdb --ligand ligand.sdf --membrane POPC -o run1
```

2 つのファイルに対して、次の前処理を自動で行います（行った内容は `fmeta funnel` のレポートに出ます）。

- **タンパク質の PDB**: アミノ酸残基以外（水、イオン、ほかのリガンド、補因子）を取り除く。PDBFixer があれば非標準残基を標準残基に置き換え、欠けた重原子を補う。水素は力場に合わせて補完する（名前が合わない水素は付け直す）。PDB からダウンロードしたままのファイルでも構いません。
- **リガンドの SDF**: 座標はタンパク質と同じ座標系で、結合ポケットに入っていることが前提です。タンパク質と接していなければ構築前に止めます。水素がなければ RDKit で付加します（SDF に描かれた形式電荷どおり。プロトン化状態は確認してください）。力場は OpenFF で自動生成します。

## インストール

公式の実行環境は Docker イメージです。テストと GPU での検証は、すべてこのイメージの中で行っています。必要なものは Docker と、GPU を使うための NVIDIA ドライバおよび NVIDIA Container Toolkit だけです。

```bash
git clone https://github.com/ytworks/fmeta && cd fmeta
docker build -t fmeta .       # 約 6 GB。OpenMM（CUDA）、OpenFF / GAFF、RDKit、PDBFixer を含む

# 入力ファイルのあるディレクトリで実行する。そこが /work にマウントされ、出力は自分の所有になる
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD":/work fmeta \
    auto --protein protein.pdb --ligand ligand.sdf -o run1
```

- イメージのエントリポイントは `fmeta` です。この README に出てくる `fmeta ...` のコマンドは、`docker run ... fmeta` の後ろに続けて書きます。`doctor` を付けると、使えるプラットフォームと同梱パッケージを確認できます。
- 環境は `conda-linux-64.lock` で固定してあります。`environment.yml` から解いた明示的なロックで、イメージは常にこのロックからビルドします。検証した環境と配布する環境が同じになります。
- 使う GPU は `--gpus '"device=2,3"'` のように絞れます。コンテナの中では、渡した GPU が 0 から番号づけされます。
- 長い計算は `--rm` の代わりに `-d --name run1` を付けて切り離します。ログは `docker logs -f run1`、停止は `docker stop run1` です。`docker rm run1` のあと同じコマンドを実行すると、チェックポイントから再開します。

### Docker を使わない場合

conda でも環境を作れます。ただし、検証は公式イメージでしか行っていません。

```bash
conda env create -f environment.yml      # openmm (CUDA), openmmforcefields, pdbfixer, rdkit, matplotlib
conda activate fmeta
pip install -e .
```

コア部分（ファネル、エンジン、解析）だけなら `pip install -e .` で足ります（依存は openmm / numpy / scipy / pyyaml）。CI の `cpu` ジョブがこの構成でテストしています。任意の依存は次の機能に使います。

| パッケージ | 使う機能 |
|---|---|
| `openmmforcefields` + `openff-toolkit` + `ambertools`（conda-forge） | SDF からのリガンド力場の自動生成（OpenFF / GAFF）。**PDB と SDF だけで実行するには必要**です。ない場合はリガンドの ffxml を `--ligand-xml` で渡します |
| `pdbfixer` | タンパク質の欠損原子・非標準残基の修復（入っていれば自動で使う） |
| `rdkit` | 水素のない SDF への水素付加、`fmeta split` |
| `matplotlib` | 図の出力（`fmeta analyze --plot`） |

## GPU マシンでの確認

GPU に依存する部分とリガンド力場の自動生成は、次の 1 コマンドでまとめて確認できます。新しいマシンや新しいドライバでは、最初にこれを実行してください。

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD":/work fmeta check --gpus 0,1     # 並列モードの項目は GPU 2 枚を使う。1 枚なら --gpus を省く
```

OpenMM 同梱の 35 残基タンパク質にベンゼンを置いた小さな系を自分で組み立て、項目ごとに PASS / FAIL / SKIP を表示します（結果は `fmeta_check/check_report.json`、失敗時のトレースバックも同じ場所）。

| 項目 | 確認内容 |
|---|---|
| `platform` | GPU プラットフォームが使えること |
| `forces` | ファネルの壁とバイアスの力・エネルギー・集団変数が Reference プラットフォームと一致（6 配置） |
| `toy_restart` | トイ系での実行とチェックポイントからの再開 |
| `toy_free_energy` | 結合定数が厳密に求まるトイ系で、10 ns の fun-metaD が厳密な ΔG° を 2 kJ/mol 以内で再現 |
| `pdb_sdf_openff`, `pdb_sdf_gaff` | PDB と SDF だけの既定経路: OpenFF / GAFF でリガンド力場を自動生成して系を構築（未インストールなら SKIP） |
| `prepare` | 溶媒和とファネルの自動配置・検証 |
| `equilibrate` | 段階的平衡化と、ファネルが結合状態に仕事をしていないこと |
| `periodic_images` | 溶媒和した実系で、GPU が複合体を別の周期セルへ移しても、またファネルの先端が箱の最も狭い方向を向いても、集団変数と壁のエネルギーが座標から計算した値と一致する |
| `production` | レプリカ 2 本の本計算、チェックポイントからの継続、集団変数に飛びがないこと |
| `walkers` | 2 ウォーカーでのバイアス共有 |
| `replicas_sequential`, `replicas_parallel` | 実行モード: 子プロセスとして起動したレプリカが、1 枚の GPU で順に、2 枚の GPU で同時に走ること。割り当ての記録 `schedule.jsonl`、各レプリカの `device.json`、割り当てた GPU でのメモリ使用の増加を確認（GPU がなければ SKIP、`replicas_parallel` は GPU 2 枚が必要） |
| `analysis` | 解析と図の出力 |
| `benchmark` | バイアスあり／なしの ns/day、チェックポイント 1 回の時間、500 ns のレプリカにかかる日数の見積もり（参考値で、合否には使わない） |
| `membrane_build`, `membrane_run` | `Modeller.addMembrane` による POPC 膜の構築と、半等方バロスタットでの平衡化・本計算 |

所要時間は、RTX PRO 6000 Blackwell 2 枚のマシンで約 4 分でした（17 項目の合計 234 秒）。`fmeta check --quick` は同じ手順を最短の長さで流す動作確認です。追加で `pytest -m gpu` も用意してあります。

結果を証跡としてリポジトリに残すときは `scripts/in_docker.sh --gpus 6,7 python scripts/gpu_gates.py --gpus 0,1` を使います。GPU テストと `fmeta check` の全項目を実行し、すべて通ると `bench/results/<日付>-<コミット>-gpu-gates.json` と `index.json` を書き出します。記録にはイメージの識別子、ロックのハッシュ、GPU の UUID が入り、公式イメージの外で実行した場合は記録しません（手順は `docs/gpu-runbook.md`）。

このあと実在の系（`examples/`）で本計算を流せば検証は完了です。

## 使い方

段階ごとに実行することもできます。各段階は前段の出力を読むので、途中からやり直せます。

```bash
fmeta init fmeta.yaml              # 設定ファイルのひな形
fmeta prepare fmeta.yaml -o run1   # 系の構築とファネルの自動配置
fmeta funnel run1                  # ファネルの検証レポート（funnel.pdb を構造と重ねて目視確認）
fmeta equilibrate run1
fmeta run run1 --mode parallel     # 見えている GPU に 1 本ずつ割り振って同時に実行。再実行すると続きから再開
fmeta status run1                  # レプリカごとの状態、進んだ時間、使った GPU
fmeta analyze run1 --plot          # ΔG°、収束診断、図
```

### 入力

必要なのはタンパク質の PDB（mmCIF も可）とリガンドの SDF（MOL、MOL2 も可）だけです。設定ファイルなしで `fmeta auto --protein … --ligand …` と実行でき、細かく指定したいときだけ `fmeta init` のひな形を使います。

- 取り除きたくない非タンパク質成分（金属イオンなど）は `system.keep_hetero: [ZN]` のように残基名で指定し、それを扱える力場ファイルを `system.forcefields` に加えます。結晶水を残すなら `system.keep_water: true`。
- リガンドの力場は `system.ligand_method` で選びます: `openff`（既定）、`gaff`、`xml`（自分の ffxml を `ligand_xml` に指定）。
- リガンドの SDF が手元になく PDB エントリの中にあるだけなら、`fmeta split entry.pdb --resname BEN --smiles "NC(=[NH2+])c1ccccc1"` で `protein.pdb` と `ligand.sdf` に切り出せます（SMILES に書いたプロトン化状態どおりに水素を付けます。重原子座標はそのまま）。`input: {complex:, ligand_resname:, ligand_smiles:}` と書けば構築時に同じ処理をします。
- すでに OpenMM の `System` と `Topology` として組み上がった系には、Python から `fmeta.system.builder.prepare_from_system()` でファネルだけを付けられます。他の MD プログラム（GROMACS、AMBER、CHARMM）の入力形式は扱いません。

### 膜タンパク質

`membrane: {lipid: POPC}`（または `--membrane POPC`）で、`Modeller.addMembrane` により膜・水・イオンを付けます。内蔵の脂質は POPC / POPE / DLPC / DLPE / DMPC / DOPC / DPPC で、`patch` に平衡化済みのパッチを渡すこともできます。

`addMembrane` は膜を xy 平面に置くだけなので、タンパク質の向きは入力側で決まっている必要があります。`membrane.orient` で扱いを選びます。

- `check`（既定）: Cα と残基の疎水性から膜面を推定し、入力と食い違えば警告する。
- `auto`: 推定した膜面に合わせて複合体を回転・並進する（疎水帯がはっきり見つからなければ止まる）。
- `input`: 入力をそのまま信用する。

膜面の推定は粗い方法（傾きで 10° 程度）なので、OPM データベースなどの配向済み構造があるならそれを使ってください。

### レプリカとウォーカー

- `run.replicas`（既定 5）: 独立した計算。レプリカ間のばらつきから 95% 信頼区間を出します。
- `run.walkers`（既定 1）: 1 つのレプリカの中でバイアスを共有するウォーカーの数。各ウォーカーは他のウォーカーの `HILLS` を読み込みます。

### 実行モード

レプリカを複数の GPU で同時に走らせるか、1 枚の GPU で順に走らせるかを選べます。

| モード | 動作 |
|---|---|
| `parallel` | `gpus` の各 GPU にレプリカを 1 本ずつ割り当てて同時に実行する。`gpus` がなければ見えている GPU をすべて使う。レプリカ数が GPU 数を超えたら、空いた GPU から次を始める |
| `sequential` | 1 枚の GPU で、レプリカを 0 から順に実行する。GPU は `--gpus 3` のように 1 つだけ指定できる |
| `auto`（既定） | `gpus` があれば `parallel`、なければ `sequential` |

- 設定ファイルでは `run.mode` と `run.gpus`（番号のリストか `all`）で指定します。コマンドラインの `--mode` と `--gpus` はその実行だけに効き、保存済みの設定は書き換えません。
- ウォーカーを使うレプリカは、ウォーカー数ぶんの GPU を取ります。
- レプリカは子プロセスとして起動します。`Ctrl-C` や `docker stop` で止めると、子プロセスも一緒に終わります。同じコマンドをもう一度実行すると、最後のチェックポイントから続きます。
- どのレプリカがどの GPU でいつ走ったかは `schedule.jsonl` に、各レプリカが実際に使った GPU は `device.json` に残ります。`fmeta status run1` で進み具合を確認できます。
- Docker では、コンテナに渡した GPU が「見えている GPU」です。`scripts/in_docker.sh --gpus 0,1,2,3 … fmeta run run1 --mode parallel` は、ホストの GPU 0〜3 だけを使います。

### Python から

```python
from fmeta.config import Config
from fmeta import workflow

config = Config.load("fmeta.yaml")
workflow.prepare_stage(config, "run1")
workflow.equilibrate_stage(config, "run1")
workflow.run_replicas(config, "run1")
print(workflow.format_summary(workflow.analyze_stage("run1", plot=True)))
```

## 仕組み

### ファネル

リガンド重心の、軸（p0→p1）への射影 `s` と軸からの距離 `ρ` を集団変数とし、半径 `R(s) = width / (1 + exp(steepness·(s − inflection))) + buffer` の壁（調和ポテンシャル）と `s` の下限・上限壁で囲みます。p0・p1 はタンパク質の Cα 原子群の重み付き重心で、ファネルはタンパク質と一緒に動きます。

自動配置（`fmeta/funnel/autodetect.py`）は次の順に行います。

1. **出口方向**: リガンド重心から約 1000 方向に直線を伸ばし、「タンパク質（と膜）をどれだけ横切るか」「どれだけ早くバルクに出るか」で採点して最良の方向を選びます。膜系では、構築前は膜を厚さ 4.4 nm のスラブとして、構築後は脂質原子そのものを障害物として扱うので、ファネルは水側を向きます。脂質側に抜けるリガンドなどは `funnel.exit_direction` で方向を指定します。
2. **アンカー**: 結合部位と、そこから出口と逆向きに 1.2 nm 入った点のまわりの Cα を選び、重心がちょうど軸上に来るように重みを決めます（二次構造をとる残基、膜系では膜貫通領域を優先）。
3. **長さ**: 軸に沿ってタンパク質表面と「リガンドが溶質から 1 nm 離れる位置」を測り、`inflection`・`upper`・`lower`・`width` を決めます。電荷をもつリガンドでは 1.5 nm 離します。
4. **検証**: ポーズが壁の内側にあるか、非結合領域がバルクにあるか、膜表面から離れているか、ファネル内に脂質がないか、周期ボックスが十分かを確認します。エラーがあれば構築を止めます。

BioSimSpace の `makeFunnel` と同じ考え方の格子平均法は `funnel: {method: bss}` で使えます。

### 周期ボックス

集団変数、壁、バイアスの力は周期境界を使いません。座標をそのまま使って重心間の距離を計算します。これで正しく動くのは、重心を使う力がリガンドとタンパク質を OpenMM の中で 1 つの分子にまとめるからです。GPU プラットフォームは分子単位で周期セルを移すので、複合体は丸ごと移動し、リガンドとタンパク質が別々の周期像に分かれることはありません。`fmeta check` の `periodic_images` がこれを確認します。

最初の版は最小像規約で計算していました。しかし OpenMM の最小像はボックスベクトルごとの折り返しで、菱形十二面体では z 方向に一辺の 0.354 倍までしか正しくありません。GPU での最初の実行で、リガンドが上限壁の手前で別の像として読まれ、計算が壊れました。そのため周期境界を使わない方式に改めました（`docs/design/engine.md` の FM-D2）。

ボックスの大きさは、ファネルの中のリガンドが、自分のタンパク質より近い周期像に出会わないように決めます。fmeta はファネルを溶媒和の**前**に決め、その幾何からボックスを決めます。

- 可溶性: タンパク質は回転するので、向きによらない条件で菱形十二面体の大きさを決めます。
- 膜: XY は `Modeller.addMembrane` の膜パッチ、Z はファネルの長さから決めます。

### 本計算

well-tempered metadynamics を `(s, ρ)` の 2 次元で行います。バイアスは格子上のテーブルで、hill を置くたびに更新します。`openmm.app.Metadynamics` と同じ仕組みですが、次の点が異なります。

- テーブル参照を格子内にクランプする（OpenMM のテーブルは範囲外で 0 になるため、そのままでは集団変数が格子を越えた瞬間にバイアスが消える）。
- `ρ = 0` で hill を鏡映し、軸上でバイアスの勾配を 0 にする。
- すべての hill を PLUMED 形式の `HILLS` に記録する（`COLVAR` も同様）。
- チェックポイント（バイナリ + State XML + バイアス格子 + カウンタを 1 ファイルにまとめて原子的に置換）から再開でき、最後のチェックポイント以降に書かれた行は再開時に巻き戻す。
- 集団変数やエネルギーが有限でなくなったら、その値を `HILLS` や `COLVAR` に書く前にエラーで止まる。ファイルは読める状態で残り、同じコマンドの再実行で最後のチェックポイントから続く。

既定値は hill 高さ 1.5 kJ/mol、bias factor 10、σ = 0.025 / 0.05 nm、hill 間隔 1 ps、水素質量再配分（1.5 amu）+ 4 fs、独立レプリカ 5 本です。

### 解析

軸方向の自由エネルギープロファイル `W(s)` から、標準結合自由エネルギーを Limongelli らの式で求めます。

```
K_b = ∫_site ds exp(−W(s)/kT)  /  ⟨ exp(−W(s_u)/kT) / A_eff(s_u) ⟩_unbound
ΔG° = −kT ln(K_b / 1.6605 nm³)
```

`A_eff` はリガンド重心が実際に使える断面積で、調和壁の「しみ出し」を含みます（既定の壁の強さでは幾何学的な πR² より 2 割ほど大きく、無視すると約 0.1 kcal/mol ずれます。壁が柔らかいほど差は大きくなります）。

`fmeta analyze` はレプリカごとに次を出し、レプリカ平均と 95% 信頼区間をまとめます。

- `dG_plateau`: ΔG°(t) の後半平均（主値）。
- `dG_reweighted`: 軌跡そのものを時間依存の重み `exp((V − c(t))/kT)` で再重み付けした推定（独立な経路での照合）。
- `dG_simple`: 「井戸の深さ + 断面積補正」の簡易式（参考値。上の式と同じ量ではありません）。
- `dG_empty_funnel`: 何も結合しない場合にこの解析が返す値。測定値がこれと 1 kT 以内なら「結合なし」と表示します。
- 再結合回数、ΔG°(t) のドリフト、非結合領域の平坦さ。基準を満たさなければ `NOT CONVERGED` と理由を表示します。

`--plot` を付けると、レプリカごとの `analysis.png`（W(s)、ΔG°(t)、2 次元自由エネルギー面、s(t)）と `summary.png` を書きます。

## 出力

```
run1/config.yaml
run1/inputs/     水素を付加した SDF、切り出した protein.pdb / ligand.sdf（該当するとき）
run1/prepared/   system.xml  topology.cif  funnel.yaml  funnel.pdb  report.json
run1/equil/      state.xml  equil.json  equilibrated.cif
run1/schedule.jsonl  rep0.log ...   レプリカの割り当ての記録と、子プロセスごとのログ
run1/rep0/ ...   run.yaml  HILLS  COLVAR  checkpoint.zip  traj.dcd  device.json  result.json  fes.npz  analysis.png
run1/rep0/w0 ... （ウォーカーを使うとき。HILLS などはウォーカーごと、result.json はレプリカに 1 つ）
run1/summary.json  summary.png
```

## 検証状況

検証は 2 段です。CPU のテストは厳密な答えのある系で正しさを確かめ、GPU ゲートは同じ内容を本計算に使うプラットフォームで確かめます。どちらも公式イメージの中で実行しています（OpenMM 8.6.1）。

**テストで確認済み**（`pytest`、`pytest -m slow`）

2026-11-16 以降、CPU プラットフォームでの動作は保証の対象外です（オーナーの判断、FM-D12）。`pytest -m slow` と `fmeta check --platform CPU` は 2c4ad4e のソースで実行したのが最後で、その後は再実行していません。既定の `pytest` は論理と数値の回帰テストとして、コードを変えるたびに実行しています。

- 実行モード: レプリカの割り当て順、同時に走る本数が GPU の数を超えないこと、失敗の伝わり方、停止要求のあとは何も始まらないことを、起動部分を差し替えたテストで確認（実行時間に依存しない）。
- 集団変数と壁: numpy 参照値・数値微分と一致、周期ボックスの形や位置に依存しない、軸上でも力が有限。
- バイアス: 同じ hill 列で `openmm.app.Metadynamics` と同じテーブルになる。同じ乱数種なら同じ軌跡をたどる。
- 再開: 2 回に分けた実行と 1 回の実行で HILLS / COLVAR / 軌跡が完全に一致。チェックポイント後に「クラッシュ」させても巻き戻して一致。
- 自由エネルギー: 解析的に結合定数が求まるトイ系（ガウス型井戸、深さ 14 kJ/mol）で、fun-metaD の結果が厳密値を再現。40 ns × 4 本で `dG_plateau` の誤差は −0.15〜+0.50 kJ/mol、20 ns × 4 本で `dG_reweighted` の誤差は −0.12〜−0.27 kJ/mol。井戸をなくしたトイ系では空のファネルの値を返し、「結合なし」と表示する。hill の鏡映あり／なしの平均誤差は +0.17 ± 0.15 ／ −0.33 ± 0.21 kJ/mol で、この系でははっきりした優劣は出ていません（既定は鏡映あり）。
- 複数ウォーカー: 各ウォーカーの格子が全 HILLS の和に一致、再開後も一致、2 ウォーカー × 10 ns でトイ系の厳密値を再現。実系でもサブプロセス 2 本で実行。
- ファネル自動配置: 出口の分かっている合成ポケット（直線チャネル、埋没部位、膜スラブつき）で方向誤差 10° 未満、膜があると水側を選ぶ。
- 入力: ダウンロードしたままの PDB を模したファイル（水・イオン・別のリガンドつき、水素あり／なし）と SDF だけからの構築、名前の合わない水素の付け直し、PDBFixer による欠損原子の修復、水素のない SDF への水素付加、タンパク質から離れたリガンドの拒否、`fmeta split`。
- 実系の通し実行（CPU、35 残基タンパク質 + ベンゼン）: 構築 → 平衡化 → 本計算 → 再開 → 解析 → 図。
- 膜: `Modeller.addMembrane` による POPC 膜の構築、ファネルから決めた Z ボックス、脂質の認識、膜表面からの距離とファネル内脂質の検証、半等方バロスタットでの平衡化と短い本計算。
- 膜面の推定: 合成の膜貫通ヘリックスバンドルで傾き誤差 12° 未満・位置誤差 0.3 nm 未満、可溶性タンパク質では「疎水帯なし」と判定。

**GPU で確認済み**（`scripts/gpu_gates.py`、CUDA、混合精度、RTX PRO 6000 Blackwell 2 枚）

最新の証跡は `bench/results/2026-11-15-8d1f10a89-gpu-gates.json` です（日付は UTC）。GPU テストと `fmeta check` の 17 項目がすべて通りました。主な実測値は次のとおりです。

| 項目 | 実測 | 基準 |
|---|---|---|
| 力とエネルギー（Reference との差、6 配置） | 集団変数 3.8e-07 nm、エネルギー 7.2e-06、力 5.6e-06（相対） | 2e-5 nm、1e-4、2e-3 |
| 周期セルをまたぐ移動 | 集団変数 8.4e-07 nm、壁のエネルギー 1.3e-04 kJ/mol | 2e-3 nm、0.1 kJ/mol |
| トイ系 10 ns の ΔG°（厳密値 −4.95 kJ/mol） | 再重み付け −2.99、プラトー −3.46 kJ/mol（誤差 1.96、1.49） | 2.0、3.0 kJ/mol |
| 逐次モード（レプリカ 2 本、GPU 1 枚） | 順に実行。割り当てた GPU のメモリ使用が 1 本あたり 596 MiB 増加 | 記録と割り当てが一致 |
| 並列モード（レプリカ 3 本、GPU 2 枚） | 2 本が同時に走り、3 本目は空いた GPU で開始 | 記録と割り当てが一致 |
| 速度（24062 原子、4 fs） | バイアスなし 1371 ns/day、あり 1312 ns/day（差 4.5%） | 参考値 |
| 速度（POPC 膜、35483 原子、4 fs） | 812 ns/day | 参考値 |

- OpenFF と GAFF によるリガンド力場の自動生成、溶媒和、平衡化、本計算、再開、2 ウォーカー、膜の構築と実行を、実物のパッケージと GPU で通しました。
- トイ系の再重み付けの誤差 1.96 kJ/mol は、基準 2.0 kJ/mol に対して余裕がほとんどありません。同じ 10 ns の計算を別の乱数種 48 個で繰り返すと、誤差の平均と標準偏差は CUDA 混合精度で +0.48 ± 0.83、CUDA 倍精度で +0.51 ± 0.83、Reference で +0.46 ± 0.83 kJ/mol でした。プラットフォームによる差はなく、基準を超えたのは 144 本のうち 5 本です。40 ns にすると誤差は +0.12 ± 0.33 kJ/mol（36 本）に縮みます。10 ns は短く、結合を弱めに見積もる傾向があります。ゲートの乱数種はその分布の端にあります。基準は変えていません（`docs/design/engine.md` の FM-D12）。

**実在の系での確認**（ゲートではなく報告、`examples/trypsin_benzamidine/README.md`）

- PDB エントリ 3PTB（トリプシンとベンザミジン）を入力に、分割、構築、ファネルの自動配置、平衡化が修正なしで通りました（72434 原子、ジスルフィド結合 6 本）。
- 並列モードで 5 レプリカを 5 枚の GPU で同時に実行し、1 本あたり 660〜700 ns/day でした。
- 並列実行を `docker stop` で止めると、1 秒以内に終わり、GPU にプロセスは残りませんでした。同じコマンドで、5 本ともチェックポイントから続きました（`HILLS` に重複も欠落もなし）。

**まだ確認していないもの**

- 実在の結合系での ΔG°。トリプシンとベンザミジンの 5 レプリカ × 500 ns を実行中です（2026-11-16 開始、約 18 時間）。
- キナーゼと膜タンパク質の実エントリを入力にした実行、実在の膜タンパク質での膜面推定。膜系はいまは合成データで確認しています。
- `HILLS` を PLUMED 本体で読むこと（書式は PLUMED に合わせていますが、PLUMED は `ρ = 0` の鏡映を知らないので、`metad: {reflect: false}` の実行でのみ意味が合います）。
- CI の `docker` ジョブと公開ワークフロー（まだ一度も実行されていません）。

## 注意点

- hill を置くたびに、バイアスのスプライン係数を計算し直してテーブル全体を GPU に送ります。コストは格子点数に比例します。24062 原子の系（格子 19198 点、hill 間隔 1 ps、4 fs）では 1 回 約 1.8 ms で、速度はバイアスなしの 96〜97% でした。自分の系での値は `fmeta check` の `benchmark` で分かります。
- 強く結合するリガンド（nM 以下）は μs 規模の計算と大きめの bias factor が必要です。
- 電荷をもつリガンドは非結合領域が平らになりにくいので、ファネルを自動で長くし、レポートに警告を出します。解析で `W(s)` が平らでなければ `funnel_overrides: {upper: ...}` でさらに延ばしてください。
- 膜系では、脱離したリガンドが脂質頭部に分配しないよう、非結合領域を膜表面から 1 nm 以上離しています（検証で確認）。
- タンパク質ファイルの非タンパク質成分は既定ですべて取り除きます。金属イオンや補因子が結合に関わる系では `system.keep_hetero` と対応する力場ファイルを指定してください。未解像のループは作りません。

## 開発

開発ルールはリポジトリに置いてあります（[kairyu](https://github.com/ytworks/kairyu) / [pypic](https://github.com/ytworks/pypic) と同じ構成）。

- `CLAUDE.md`（`AGENTS.md` はこれを参照）: テスト方針、ファイルの置き場所、開発コマンド。
- `.claude/rules/`: Docker での実行規則、進捗ログの書き方、自由エネルギーとファネルに関する不変条件。
- `PROGRESS.md`: 設計変更・マイルストーン・検証状況・ブロッカーの記録（セッションをまたぐメモ）。大きさは `scripts/check_progress_size.py` がセッション開始時と CI で検査します。
- `docs/`: 設計判断（`design/`、FM-D 番号）、ロードマップ、計画、GPU 手順書。
- CI（`.github/workflows/ci.yml`）: `cpu` ジョブは pip で入れたコアだけで lint、PROGRESS.md の検査、GPU 証跡の整合性検査、CPU テストを行います。`docker` ジョブは公式イメージをビルドし、その中で lint と CPU テストを行います。GPU を使う検証は CI では行わず、`scripts/gpu_gates.py` の記録を CI が検証します。
- `.github/workflows/image.yml`: 公式イメージを `ghcr.io/ytworks/fmeta` に公開します。バージョンタグか手動実行でだけ動きます。

実行とテストは、CI を除いてすべて公式イメージのコンテナの中で行います（`.claude/rules/docker-execution.md`）。ホストに Python 環境は作りません。`scripts/in_docker.sh` はリポジトリを `/work` にマウントし、そのソースを使って、呼び出したユーザーとしてコマンドを実行します。イメージがなければ最初にビルドします。

```bash
scripts/in_docker.sh ruff check .
scripts/in_docker.sh python scripts/check_progress_size.py
scripts/in_docker.sh python scripts/verify_bench_results.py
scripts/in_docker.sh pytest
scripts/in_docker.sh pytest -m slow                                       # 時間のかかる CPU テスト
scripts/in_docker.sh --gpus 6,7 python scripts/gpu_gates.py --gpus 0,1    # GPU ゲート。ホストの GPU 6 と 7 を使う
```

パッケージを変えるときは `environment.yml` を編集し、`scripts/update_lock.sh` でロックを作り直します。そのあとイメージを作り直し（`scripts/in_docker.sh --build ...`）、テストと GPU ゲートをやり直します。

## 参考文献

- V. Limongelli, M. Bonomi, M. Parrinello, *PNAS* **110**, 6358 (2013) — funnel metadynamics。
- S. Raniolo, V. Limongelli, *Nat. Protoc.* **15**, 2837 (2020)。
- BioSimSpace の funnel metadynamics チュートリアル（OpenBioSim `biosimspace_tutorials/02_funnel_metad`）— ファネル形状と自動配置の考え方の出発点。コードは流用していません。
- P. Tiwary, M. Parrinello, *J. Phys. Chem. B* **119**, 736 (2015) — 時間依存の再重み付け。
- G. Postic, Y. Ghouzam, V. Guiraud, J.-C. Gelly, *Protein Eng. Des. Sel.* **29**, 87 (2016) — スラブ探索による膜面推定（ANVIL）の考え方。
