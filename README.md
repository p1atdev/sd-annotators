# sd-annotators

個人的にStable Diffusion の学習素材の事前処理に使っているスクリプトなど。

大体は並列処理できるようにしてあるので、WebUI のやつよりは多分数倍速いと思います。

中身

- blip_captioner.py: [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) でキャプションをつける (並列処理可能)
- cafe_aesthetic.py: [cafe_aesthetic](https://huggingface.co/spaces/cafeai/cafe_aesthetic_demo) で分類する (並列処理可能)
- caption_filter.py: `.txt` のタグファイルが入ったフォルダを指定して、特定のタグが入ったキャプションと画像をフィルターする
- cut_up_down.py: 画像の上下の一部をカットする (主にロゴの削除向け)
- merge_captions.py: 二種類のテキストファイルの拡張子を指定して、どちらか一方にまとめる

## blip captioner

BLIP2 でキャプションを生成しますが、VRAM 8GB だと多分1スレッドが限界なので特段速くなるわけではないです。

引数は多くないので `argparse` のとこを見てもらえれば使い方はわかると思います。Copilotが一部勝手に生成してるので不自然でも無視してください。

lavis のライブラリが必要です。(謎に時間かかる)

```bash
pip install salesforce-lavis
```

1スレッドで `.caption` の拡張子をつけてキャプションを生成する場合。

```bash
python blip_captioner.py path/to/caption --threads 1 --ext caption 
```

- `-o` か `--captions_dir` を指定すると、キャプションファイルをそこに作成します
- `--model` は指定してもなんにもなりません (VRAM 8GB だと多分 `coco` しか動かないので)
- `--overwite` を指定すると、既に同じ名前のキャプションファイルがある場合に上書きするようになります。デフォルトの場合は、既にキャプションがある場合はスキップされます。

## cafe aesthetic

Cafe さんの画像分類機を使って画像を分類できます。

```bash
pip install transformers
```

美的評価のモデルを 10 スレッド (VRAM 8GB なら多分これくらいが上限) 使って分類する場合:

```bash
python cafe_aesthetic.py path/to/classify -o path/to/output -b 10 --model_type aesthetic
```

- `-o` か `--output_path` を指定しなかった場合は、入力フォルダに新しくフォルダが作成されてそこに突っ込まれます。
- `-t` か `--threshold` を指定した場合、その数値の割合を超えるか超えないかで分類されます。指定しなかったら最も当てはまるものに分類されます。
- `--model_type` には `aesthetic`、`style`、`waifu` が指定できます

## caption filter

タグファイルに特定のタグが含まれるかどうかで画像とキャプションをフィルタリングできます。

`monochrome, greyscale` を含むものを `path/to/output` に移動させる場合:

```bash
python caption_filter.py --captions_dir path/to/captions --images_dir path/to/images --caption_output_dir path/to/output --filter_tags "monochrome" "greyscale"
```

- `--captions_dir` は判定に使うキャプションファイルが入ったフォルダを指定します
- `--images_dir` キャプションのついでに移動させる画像が入ったフォルダを指定しますが、指定されなかったら `--captions_dir` と同じものが使われます。
- `--caption_output_dir` と `--image_output_dir` は、フィルターに当てはまったものがそれぞれ移動する先のパスで、 `--image_output_dir` が指定されなかったらキャプションと同じところに行きます。
- `--filter_tags` フィルターに使うタグを指定します。

## cut up down

画像の(上)下の数%をカットします。学習画像にやたらと透かしなどが含まれている場合に無理矢理除去するのに使います。

TODO: 今は下しかカットしない上に、カット割合がハードコートされてるので後で直す

## merge captions

二つの種類の拡張子のキャプションを、片方にまとめることができます。`.caption` と `.txt` のキャプションを一つにまとめたいときなど。

キャプションファイルは同じフォルダに入っている前提です。

`.caption` のキャプションのうしろに `.txt` のキャプションの内容をくっつけて、処理後に `.txt` を削除する場合。  

```bash
python merge_captions.py path/to/captions -p caption -s txt --delete --threads 200
```
- `-p` `--prefix` 頭になる方のキャプションの拡張子 (こちら側に統合される)
- `-s` `--suffix` 後ろになる方のキャプションの拡張子 
- `--delete` をつけると `--suffix` で指定された方のファイルは処理後に削除される
- `--threads` スレッド数。これは GPU つかわないのでめちゃくちゃ上げてもヨシ。

<details>
<summary>統合の例</summary>

`-p` に BLIP2 のキャプションの `.caption` ファイル、 `-s` に Tagger のタグの `.txt` を指定して上のコマンドを実行した場合は、

`.caption` のキャプションファイルは

```
an anime illustration of a girl, 1girl, black hair, short hair....
```

みたいな感じになり、`.txt` ファイルは削除される。

</details>

## TODO

- [ ] cut のやつを整える
- [ ] WD14Tagger
- [ ] イラスト→スケッチ変換
