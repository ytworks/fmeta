# fmeta

OpenMM をエンジンとする funnel metadynamics（fun-metaD）フレームワークです。タンパク質と、結合ポーズの決まったリガンドを渡すと、ファネルの定義・系の構築（水溶液または脂質二重膜）・平衡化・本計算・絶対結合自由エネルギーの算出までを既定設定で実行します。バイアスも拘束も OpenMM の Custom Force として定義しているので、GPU プラットフォーム（CUDA / OpenCL）上でそのまま動きます。

```bash
# 可溶性タンパク質
fmeta auto --protein protein.pdb --ligand ligand.sdf -o run1

# PDB の複合体からそのまま（リガンドを切り出し、SMILES から水素を付加）
fmeta auto --complex 3ptb.pdb --ligand-resname BEN --ligand-smiles "NC(=[NH2+])c1ccccc1" -o run1

# 膜タンパク質（GPCR など）: OpenMM の Modeller.addMembrane で POPC 膜に埋め込む
fmeta auto --protein gpcr_oriented.pdb --ligand ligand.sdf --membrane POPC -o run1
```

## インストール

```bash
conda env create -f environment.yml      # openmm (CUDA), openmmforcefields, pdbfixer, rdkit, matplotlib
conda activate fmeta
pip install -e .
```

コア部分（ファネル、エンジン、解析）だけなら `pip install -e .` で足ります（依存は openmm / numpy / scipy / pyyaml）。任意の依存は次の機能に使います。

| パッケージ | 使う機能 |
|---|---|
| `openmmforcefields` + `openff-toolkit`（conda-forge） | OpenFF / GAFF によるリガンドの自動パラメータ化 |
| `rdkit` | 複合体 PDB のリガンドに SMILES から水素と結合次数を付ける |
| `pdbfixer` | タンパク質の欠損原子・非標準残基の修復（`system.fix_protein: true`） |
| `matplotlib` | 図の出力（`fmeta analyze --plot`） |

## GPU マシンでの確認

開発は GPU のない環境で行いました。GPU に依存する部分と conda 専用パッケージに依存する部分は、次の 1 コマンドでまとめて確認できます。

```bash
fmeta check                # 複数 GPU のランチャも試すなら --gpus 0,1
```

OpenMM 同梱の 35 残基タンパク質にベンゼンを置いた小さな系を自分で組み立て、項目ごとに PASS / FAIL / SKIP を表示します（結果は `fmeta_check/check_report.json`、失敗時のトレースバックも同じ場所）。

| 項目 | 確認内容 |
|---|---|
| `platform` | GPU プラットフォームが使えること |
| `forces` | ファネルの壁とバイアスの力・エネルギー・集団変数が Reference プラットフォームと一致（周期像を含む 7 配置） |
| `toy_restart` | トイ系での実行とチェックポイントからの再開 |
| `toy_free_energy` | 結合定数が厳密に求まるトイ系で、10 ns の fun-metaD が厳密な ΔG° を 2 kJ/mol 以内で再現 |
| `ligand_openff`, `ligand_gaff` | OpenFF / GAFF によるリガンドのパラメータ化（未インストールなら SKIP） |
| `prepare` | 溶媒和とファネルの自動配置・検証 |
| `periodic_images` | 溶媒和した実系で、系全体やリガンドを別の周期像に置いても集団変数が変わらない |
| `equilibrate` | 段階的平衡化と、ファネルが結合状態に仕事をしていないこと |
| `production` | レプリカ 2 本の本計算、チェックポイントからの継続、集団変数に飛びがないこと |
| `walkers` | 2 ウォーカーでのバイアス共有 |
| `analysis` | 解析と図の出力 |
| `benchmark` | バイアスあり／なしの ns/day と、500 ns のレプリカにかかる日数の見積もり |
| `membrane_build`, `membrane_run` | `Modeller.addMembrane` による POPC 膜の構築と、半等方バロスタットでの平衡化・本計算 |

所要時間は GPU で 10〜20 分程度の見込みです（未計測）。`fmeta check --platform CPU --quick` は同じ手順を最短の長さで流す動作確認で、開発環境ではこれが全項目通ることを確認しています。追加で `pytest -m gpu` も用意してあります。

このあと実在の系（`examples/`）で本計算を流せば検証は完了です。

## 使い方

段階ごとに実行することもできます。各段階は前段の出力を読むので、途中からやり直せます。

```bash
fmeta init fmeta.yaml              # 設定ファイルのひな形
fmeta prepare fmeta.yaml -o run1   # 系の構築とファネルの自動配置
fmeta funnel run1                  # ファネルの検証レポート（funnel.pdb を構造と重ねて目視確認）
fmeta equilibrate run1
fmeta run run1 --gpus 0,1          # レプリカを GPU に割り振って実行。再実行すると続きから再開
fmeta analyze run1 --plot          # ΔG°、収束診断、図
```

### 入力

| 入力 | 設定 |
|---|---|
| タンパク質 (PDB / mmCIF) + ポーズ済みリガンド (SDF / MOL / MOL2、水素つき、タンパク質と同じ座標系) | `input: {protein:, ligand:}` |
| 複合体 (PDB / mmCIF) + リガンドの残基名 + SMILES | `input: {complex:, ligand_resname:, ligand_smiles:}`（`fmeta split` で切り出しだけもできる） |
| AMBER | `input: {amber: {prmtop:, coordinates:}}` |
| GROMACS | `input: {gromacs: {top:, gro:}}` |
| CHARMM / CHARMM-GUI | `input: {charmm: {psf:, coordinates:, parameters: [...]}}` |

- 複合体入力では、SMILES に書いたプロトン化状態どおりにリガンドの水素を付けます（結晶構造の重原子座標はそのまま）。
- パラメータ化済み入力では系の構築を省き、ファネルの配置と検証だけを行います。座標がボックス内に折り返されていても、読み込み時に分子をつなぎ直します。脂質が含まれていれば膜系として扱います。リガンドは「非ポリマー・非脂質・非溶媒で重原子 6 個以上の残基が 1 つだけ」のとき自動で見つけ、そうでなければ `ligand_selection` で指定します。
- リガンドの力場は `system.ligand_method` で選びます: `openff`（既定）、`gaff`、`xml`（自分の ffxml を `ligand_xml` に指定）。

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
- `fmeta run --gpus 0,1,2,3` で GPU に割り振ります。レプリカはウォーカー数ぶんの GPU を取り、空きがあれば並行して走ります。

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

GPU は分子を内部でボックス内に折り返すので、集団変数は最小像規約で計算します。そのためボックスは「上限壁までの距離が最短像距離の半分より小さい」必要があり、これを満たさないとリガンドが上限壁の手前で像を乗り換えてしまいます。fmeta はファネルを溶媒和の**前**に決め、その幾何からボックスを決めます。

- 可溶性: タンパク質は回転するので、向きによらない条件で菱形十二面体の大きさを決めます。
- 膜: XY は `Modeller.addMembrane` の膜パッチ、Z はファネルの長さから決めます。

### 本計算

well-tempered metadynamics を `(s, ρ)` の 2 次元で行います。バイアスは格子上のテーブルで、hill を置くたびに更新します。`openmm.app.Metadynamics` と同じ仕組みですが、次の点が異なります。

- テーブル参照を格子内にクランプする（OpenMM のテーブルは範囲外で 0 になるため、そのままでは集団変数が格子を越えた瞬間にバイアスが消える）。
- `ρ = 0` で hill を鏡映し、軸上でバイアスの勾配を 0 にする。
- すべての hill を PLUMED 形式の `HILLS` に記録する（`COLVAR` も同様）。
- チェックポイント（バイナリ + State XML + バイアス格子 + カウンタを 1 ファイルにまとめて原子的に置換）から再開でき、最後のチェックポイント以降に書かれた行は再開時に巻き戻す。

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
run1/inputs/     protein.pdb  ligand.sdf        （複合体入力のとき）
run1/prepared/   system.xml  topology.cif  funnel.yaml  funnel.pdb  report.json
run1/equil/      state.xml  equil.json  equilibrated.cif
run1/rep0/ ...   run.yaml  HILLS  COLVAR  checkpoint.zip  traj.dcd  result.json  fes.npz  analysis.png
run1/rep0/w0 ... （ウォーカーを使うとき。HILLS などはウォーカーごと、result.json はレプリカに 1 つ）
run1/summary.json  summary.png
```

## 検証状況

GPU のない環境（OpenMM 8.6.1、Reference / CPU プラットフォーム）で開発・検証しました。

**確認済み**（`pytest`、`pytest -m slow`、`fmeta check --platform CPU --quick`）

- 集団変数と壁: numpy 参照値・数値微分と一致、周期像の移動で不変、軸上でも力が有限。
- バイアス: 同じ hill 列で `openmm.app.Metadynamics` と同じテーブルになる。同じ乱数種なら同じ軌跡をたどる。
- 再開: 2 回に分けた実行と 1 回の実行で HILLS / COLVAR / 軌跡が完全に一致。チェックポイント後に「クラッシュ」させても巻き戻して一致。
- 自由エネルギー: 解析的に結合定数が求まるトイ系（ガウス型井戸、深さ 14 kJ/mol）で、fun-metaD の結果が厳密値を再現。40 ns × 4 本で `dG_plateau` の誤差は −0.15〜+0.50 kJ/mol、20 ns × 4 本で `dG_reweighted` の誤差は −0.12〜−0.27 kJ/mol。井戸をなくしたトイ系では空のファネルの値を返し、「結合なし」と表示する。hill の鏡映あり／なしの平均誤差は +0.17 ± 0.15 ／ −0.33 ± 0.21 kJ/mol で、この系でははっきりした優劣は出ていません（既定は鏡映あり）。
- 複数ウォーカー: 各ウォーカーの格子が全 HILLS の和に一致、再開後も一致、2 ウォーカー × 10 ns でトイ系の厳密値を再現。実系でもサブプロセス 2 本で実行。
- ファネル自動配置: 出口の分かっている合成ポケット（直線チャネル、埋没部位、膜スラブつき）で方向誤差 10° 未満、膜があると水側を選ぶ。
- 入力: 複合体 PDB + SMILES、PDBFixer による修復、折り返された座標のつなぎ直し、AMBER / GROMACS / CHARMM 形式（ParmEd で書き出したファイル）を、それぞれ構築段階まで通して確認。
- 実系の通し実行（CPU、35 残基タンパク質 + ベンゼン）: 構築 → 平衡化 → 本計算 → 再開 → 解析 → 図。
- 膜: `Modeller.addMembrane` による POPC 膜の構築、ファネルから決めた Z ボックス、脂質の認識、膜表面からの距離とファネル内脂質の検証、半等方バロスタットでの平衡化と短い本計算。
- 膜面の推定: 合成の膜貫通ヘリックスバンドルで傾き誤差 12° 未満・位置誤差 0.3 nm 未満、可溶性タンパク質では「疎水帯なし」と判定。

**GPU マシンで確認する項目**（`fmeta check` が実行する）

- GPU プラットフォームでの力の一致、周期像の扱い、トイ系の ΔG°、通し実行、スループット。
- OpenFF / GAFF によるリガンドのパラメータ化。開発環境には conda 専用の `openmmforcefields` を入れられなかったため、実物では未実行です。fmeta 側の呼び出しは代役のモジュールでテストしてあります。

**この環境では確認できなかったもの**（外部データが必要）

- 実在の結合系での ΔG°（数百 ns × 複数レプリカの GPU 計算が必要）。`examples/` に設定があります。
- PDB の実エントリでの複合体入力、実在の膜タンパク質での膜面推定。構造データベースに接続できなかったため、OpenMM 同梱のタンパク質と合成データで確認しています。
- `HILLS` を PLUMED 本体で読むこと（書式は PLUMED に合わせていますが、PLUMED は `ρ = 0` の鏡映を知らないので、`metad: {reflect: false}` の実行でのみ意味が合います）。

## 注意点

- hill を置くたびに OpenMM がテーブルのスプラインを CPU で作り直します。コストは格子点数に比例し（開発環境で 1 点あたり約 0.5 マイクロ秒）、実際のオーバーヘッドは `fmeta check` の `benchmark` で分かります。
- 強く結合するリガンド（nM 以下）は μs 規模の計算と大きめの bias factor が必要です。
- 電荷をもつリガンドは非結合領域が平らになりにくいので、ファネルを自動で長くし、レポートに警告を出します。解析で `W(s)` が平らでなければ `funnel_overrides: {upper: ...}` でさらに延ばしてください。
- 膜系では、脱離したリガンドが脂質頭部に分配しないよう、非結合領域を膜表面から 1 nm 以上離しています（検証で確認）。
- `system.fix_protein: true` は水以外のヘテロ原子（金属イオンや補因子）を取り除きます。必要なものは自分で用意した構造と力場ファイルで与えてください。未解像のループは作りません。

## 参考文献

- V. Limongelli, M. Bonomi, M. Parrinello, *PNAS* **110**, 6358 (2013) — funnel metadynamics。
- S. Raniolo, V. Limongelli, *Nat. Protoc.* **15**, 2837 (2020)。
- BioSimSpace の funnel metadynamics チュートリアル（OpenBioSim `biosimspace_tutorials/02_funnel_metad`）— ファネル形状と自動配置の考え方の出発点。コードは流用していません。
- P. Tiwary, M. Parrinello, *J. Phys. Chem. B* **119**, 736 (2015) — 時間依存の再重み付け。
- G. Postic, Y. Ghouzam, V. Guiraud, J.-C. Gelly, *Protein Eng. Des. Sel.* **29**, 87 (2016) — スラブ探索による膜面推定（ANVIL）の考え方。
