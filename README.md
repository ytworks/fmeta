# fmeta

OpenMM をエンジンとする funnel metadynamics（fun-metaD）フレームワークです。タンパク質と、結合ポーズの決まったリガンドを渡すと、ファネルの定義・系の構築（水溶液または脂質二重膜）・平衡化・本計算・絶対結合自由エネルギーの算出までを既定設定で実行します。バイアスも拘束も OpenMM の Custom Force として定義しているので、GPU プラットフォーム（CUDA / OpenCL）上でそのまま動きます。

```bash
# 可溶性タンパク質
fmeta auto --protein protein.pdb --ligand ligand.sdf -o run1

# 膜タンパク質（GPCR など）: OpenMM の Modeller.addMembrane で POPC 膜に埋め込む
fmeta auto --protein gpcr_oriented.pdb --ligand ligand.sdf --membrane POPC -o run1
```

## インストール

本番用（GPU と自動リガンドパラメータ化）は conda 環境を使います。

```bash
conda env create -f environment.yml
conda activate fmeta
pip install -e .
fmeta doctor        # プラットフォームの検出、GPU と Reference の一致、テーブル更新コストを表示
```

コア部分（ファネル、エンジン、解析）だけなら `pip install -e .` で足ります（依存は openmm / numpy / scipy / pyyaml）。OpenFF・GAFF によるリガンドの自動パラメータ化には conda-forge の `openmmforcefields` が必要です。入れない場合は、リガンドの ffxml を自分で渡すか、パラメータ化済みの系（AMBER / GROMACS / CHARMM-GUI）を入力にします。

## 使い方

段階ごとに実行することもできます。各段階は前段の出力を読むので、途中からやり直せます。

```bash
fmeta init fmeta.yaml              # 設定ファイルのひな形
fmeta prepare fmeta.yaml -o run1   # 系の構築とファネルの自動配置
fmeta funnel run1                  # ファネルの検証レポート（funnel.pdb を構造と重ねて目視確認）
fmeta equilibrate run1
fmeta run run1 --gpus 0,1          # レプリカを GPU に割り振って実行。再実行すると続きから再開
fmeta analyze run1
```

入力は次のいずれかです。

| 入力 | 設定 |
|---|---|
| タンパク質 (PDB / mmCIF) + ポーズ済みリガンド (SDF / MOL / MOL2、水素つき、タンパク質と同じ座標系) | `input: {protein:, ligand:}` |
| AMBER | `input: {amber: {prmtop:, coordinates:}}` |
| GROMACS | `input: {gromacs: {top:, gro:}}` |
| CHARMM / CHARMM-GUI | `input: {charmm: {psf:, coordinates:, parameters: [...]}}` |

パラメータ化済み入力では系の構築を省き、ファネルの配置と検証だけを行います。脂質が含まれていれば膜系として扱います。リガンドは「非ポリマー・非脂質・非溶媒で重原子 6 個以上の残基が 1 つだけ」のとき自動で見つけ、そうでなければ `ligand_selection` で指定します。

Python からは同じ段階を関数として呼べます。

```python
from fmeta.config import Config
from fmeta import workflow

config = Config.load("fmeta.yaml")
workflow.prepare_stage(config, "run1")
workflow.equilibrate_stage(config, "run1")
workflow.run_replicas(config, "run1")
print(workflow.format_summary(workflow.analyze_stage("run1")))
```

## 仕組み

### ファネル

リガンド重心の、軸（p0→p1）への射影 `s` と軸からの距離 `ρ` を集団変数とし、半径 `R(s) = width / (1 + exp(steepness·(s − inflection))) + buffer` の壁（調和ポテンシャル）と `s` の下限・上限壁で囲みます。p0・p1 はタンパク質の Cα 原子群の重み付き重心で、ファネルはタンパク質と一緒に動きます。

自動配置（`fmeta/funnel/autodetect.py`）は次の順に行います。

1. **出口方向**: リガンド重心から約 1000 方向に直線を伸ばし、「タンパク質（と膜）をどれだけ横切るか」「どれだけ早くバルクに出るか」で採点して最良の方向を選びます。膜系では、構築前は膜を厚さ 4.4 nm のスラブとして、構築後は脂質原子そのものを障害物として扱うので、ファネルは水側を向きます。脂質側に抜けるリガンドなどは `funnel.exit_direction` で方向を指定します。
2. **アンカー**: 結合部位と、そこから出口と逆向きに 1.2 nm 入った点のまわりの Cα を選び、重心がちょうど軸上に来るように重みを決めます（二次構造をとる残基、膜系では膜貫通領域を優先）。
3. **長さ**: 軸に沿ってタンパク質表面と「リガンドが溶質から 1 nm 離れる位置」を測り、`inflection`・`upper`・`lower`・`width` を決めます。
4. **検証**: ポーズが壁の内側にあるか、非結合領域がバルクにあるか、膜表面から離れているか、周期ボックスが十分かを確認します。エラーがあれば構築を止めます。

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
- 再結合回数、ΔG°(t) のドリフト、非結合領域の平坦さ。基準を満たさなければ `NOT CONVERGED` と理由を表示します。

## 出力

```
run1/config.yaml
run1/prepared/   system.xml  topology.cif  funnel.yaml  funnel.pdb  report.json
run1/equil/      state.xml  equil.json  equilibrated.cif
run1/rep0/ ...   run.yaml  HILLS  COLVAR  checkpoint.zip  traj.dcd  result.json  fes.npz
run1/summary.json
```

## 検証状況

GPU のない環境（OpenMM 8.6.1、Reference / CPU プラットフォーム）で開発・検証しました。

**確認済み**（`pytest`、`pytest -m slow`）

- 集団変数と壁: numpy 参照値・数値微分と一致、周期像の移動で不変、軸上でも力が有限。
- バイアス: 同じ hill 列で `openmm.app.Metadynamics` と同じテーブルになる。同じ乱数種なら同じ軌跡をたどる。
- 再開: 2 回に分けた実行と 1 回の実行で HILLS / COLVAR / 軌跡が完全に一致。チェックポイント後に「クラッシュ」させても巻き戻して一致。
- 自由エネルギー: 解析的に結合定数が求まるトイ系（ガウス型井戸、深さ 14 kJ/mol）で、fun-metaD の結果が厳密値を再現。40 ns × 4 本で `dG_plateau` の誤差は −0.15〜+0.50 kJ/mol、20 ns × 4 本で `dG_reweighted` の誤差は −0.12〜−0.27 kJ/mol。hill の鏡映あり／なしの平均誤差は +0.17 ± 0.15 ／ −0.33 ± 0.21 kJ/mol で、この系でははっきりした優劣は出ていません（既定は鏡映あり）。
- ファネル自動配置: 出口の分かっている合成ポケット（直線チャネル、埋没部位、膜スラブつき）で方向誤差 10° 未満、膜があると水側を選ぶ。
- 実系の通し実行（CPU、35 残基タンパク質 + ベンゼン）: 構築 → 平衡化 → 本計算 → 再開 → 解析。AMBER 入力（ParmEd で作成）も同じ経路で確認。
- 膜: `Modeller.addMembrane` による POPC 膜の構築、ファネルから決めた Z ボックス、脂質の認識、膜表面からの距離の検証、半等方バロスタットでの平衡化と短い本計算。

**未確認**（この環境では実行できなかったもの）

- GPU プラットフォーム。`pytest -m gpu` に Reference との一致と再開のテストを用意してあります。まず `fmeta doctor` と合わせて実行してください。
- OpenFF / GAFF によるリガンドのパラメータ化（conda 専用の依存のため）。
- GROMACS / CHARMM 入力の読み込み（OpenMM のローダを呼ぶだけですが、テスト入力がありません）。
- 実在の結合系での ΔG°（数百 ns × 複数レプリカの GPU 計算が必要）。`examples/` に設定例があります。
- `HILLS` を PLUMED 本体で読むこと（書式は PLUMED に合わせていますが、PLUMED は `ρ = 0` の鏡映を知らないので、`metad: {reflect: false}` の実行でのみ意味が合います）。

**未実装**

- 複数ウォーカー（バイアス共有）。並列化は独立レプリカのみです。
- 膜タンパク質の自動配向。入力は膜法線が z 軸、膜中心が `membrane.center_z` になるよう配向済み（OPM データベースの構造など）である必要があります。
- 図の出力。数値は `fes.npz` と `result.json` にあります。

## 注意点

- hill を置くたびに OpenMM がテーブルのスプラインを CPU で作り直します。コストは格子点数に比例し（開発環境で 1 点あたり約 0.5 マイクロ秒）、既定の格子・1 ps 間隔なら数 % 程度の見込みです。実際の値は `fmeta doctor` で確認できます。
- 強く結合するリガンド（nM 以下）は μs 規模の計算と大きめの bias factor が必要です。
- 電荷をもつリガンドは非結合領域が平らになりにくいので、`funnel_overrides: {upper: ...}` で上限壁を延ばしてください。
- 膜系では、脱離したリガンドが脂質頭部に分配しないよう、非結合領域を膜表面から 1 nm 以上離しています（検証で確認）。

## 参考文献

- V. Limongelli, M. Bonomi, M. Parrinello, *PNAS* **110**, 6358 (2013) — funnel metadynamics。
- S. Raniolo, V. Limongelli, *Nat. Protoc.* **15**, 2837 (2020)。
- BioSimSpace の funnel metadynamics チュートリアル（OpenBioSim `biosimspace_tutorials/02_funnel_metad`）— ファネル形状と自動配置の考え方の出発点。コードは流用していません。
- P. Tiwary, M. Parrinello, *J. Phys. Chem. B* **119**, 736 (2015) — 時間依存の再重み付け。
