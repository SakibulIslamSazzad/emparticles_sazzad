// Model & Report Part I. Every number below is read from results/results.json, every figure
// from figures/: nothing is typed in by hand, so a new run of `just` rewrites the whole report.
#let r = json("/results/results.json")
#let f(x, d: 3) = {
  let s = str(calc.round(x, digits: d))
  let parts = s.split(".")
  if parts.len() == 1 { s + "." + "0" * d } else { s + "0" * (d - parts.at(1).len()) }
}
#let t = r.test
#let m = t.model.overall
#let neg = t.dummy.overall
#let perm = t.permuted.overall
#let pos = t.planted.overall
#let ok(c) = if c { text(fill: rgb("#1a7f37"))[*passed*] } else { text(fill: rgb("#cf222e"))[*FAILED*] }
#let src-names = ("emps", "hrtem", "co3o4")

#set page(paper: "us-letter", margin: 1.8cm, numbering: "1")
#set text(size: 10pt)
#set heading(numbering: "1.")
#show figure.caption: set text(size: 8.5pt)

#align(center)[
  #text(size: 17pt, weight: "bold")[Segmenting nanoparticles in EM images: model and report, part I] \
  #text(size: 10pt)[EMA 5937 · Project A · automatically generated from `results/results.json`]
]

= Summary

A small U-Net (#r.model.parameters parameters) was trained to label every pixel of a 256 × 256 EM tile as
"particle" or "background". On the held-out test set (#r.data.test_tiles tiles from #r.data.test_groups groups
that never appear in training) it reaches a pixel ROC AUC of *#f(m.auc)*, an average precision of
#f(m.ap) and an IoU of #f(m.iou) at probability 0.5. The negative controls score
AUC #f(neg.auc) (constant score) and #f(perm.auc) (shuffled masks); the positive control, which is
handed the mask, scores #f(pos.auc). This part is about the framework, not the score: the model is small and
trained for #r.model.epochs epochs only.

= Data

The tensors come from the database (`data/dataset.db`) through `emparticles.tensors`. All pictures of a group
(EMPS: paper, HRTEM: session, Co3O4: single image) are on one side of the split, so no group is in both train and test.

#figure(
  table(columns: (auto, auto, auto), align: (left, right, right),
    table.header[*source*][*training tiles*][*test tiles*],
    ..src-names.map(s => (s, str(r.data.train_tiles_by_source.at(s)), str(r.data.test_tiles_by_source.at(s)))).flatten(),
    [*all*], [#r.data.train_tiles], [#r.data.test_tiles],
  ),
  caption: [Tiles per source. Training tiles exclude cross-validation fold #r.data.val_fold (#r.data.val_tiles used for validation after thinning).],
)

Foreground (particle) fraction: #f(r.data.train_foreground_rate * 100, d: 1) % of training pixels and
#f(r.data.test_foreground_rate * 100, d: 1) % of test pixels, so a model that predicts "background" everywhere is
#f((1 - r.data.test_foreground_rate) * 100, d: 1) % accurate and has an IoU of 0.

= Model and training

U-Net with #r.model.depth down-sampling steps and #r.model.base channels at full size, loss = #r.model.loss, AdamW
(learning rate #r.model.lr), batches of #r.model.batch_size random #r.model.crop × #r.model.crop crops with
flips. Cross-validation fold #r.data.val_fold (whole groups) is the validation set; the epoch with the best
validation AUC (epoch #r.model.best_epoch) is kept. The test set is not used for any choice.

#figure(image("/figures/training_curve.png", width: 100%),
  caption: [Training curve of the model and of the positive control: training loss, validation loss, validation AUC.])

= Held-out test results

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right),
    table.header[*model*][*AUC*][*AP*][*IoU*][*Dice*][*recall*],
    [trained U-Net], f(m.auc), f(m.ap), f(m.iou), f(m.dice), f(m.recall),
    [negative: constant score (dummy)], f(neg.auc), f(neg.ap), f(neg.iou), f(neg.dice), f(neg.recall),
    [negative: shuffled test masks], f(perm.auc), f(perm.ap), f(perm.iou), f(perm.dice), f(perm.recall),
    [positive: mask given as input], f(pos.auc), f(pos.ap), f(pos.iou), f(pos.dice), f(pos.recall),
  ),
  caption: [Pixel-wise metrics on the held-out test set. IoU, Dice and recall use probability 0.5.],
)

#grid(columns: (1fr, 1.5fr), gutter: 8pt,
  figure(image("/figures/roc.png", width: 100%), caption: [ROC curves on the test pixels.]),
  figure(image("/figures/by_source.png", width: 100%), caption: [AUC and IoU per source (test).]),
)

#let bs = t.model.by_source
#let ranked = bs.pairs().sorted(key: p => p.at(1).auc)
#let worst = ranked.first()
#let best = ranked.last()

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right),
    table.header[*source (test)*][*AUC*][*AP*][*IoU*][*recall*][*foreground*],
    ..bs.pairs().map(p => (p.at(0), f(p.at(1).auc), f(p.at(1).ap), f(p.at(1).iou), f(p.at(1).recall),
      f(p.at(1).foreground_rate * 100, d: 1) + " %")).flatten(),
  ),
  caption: [The trained model on each source separately (same test tiles as above).],
)

The model separates particles best on *#best.at(0)* (AUC #f(best.at(1).auc)) and worst on
*#worst.at(0)* (AUC #f(worst.at(1).auc)); the gap of #f(best.at(1).auc - worst.at(1).auc) AUC is the clearest
place to improve in Part II.

#figure(image("/figures/examples.png", width: 70%),
  caption: [One test tile per source: picture, true mask and predicted probability.])

= Controls

The controls say whether the numbers above can be trusted.

- Negative control, constant score: no information, so AUC must be 0.5 (#f(neg.auc)):
  #ok(calc.abs(neg.auc - 0.5) < 0.02).
- Negative control, shuffled masks: the trained model scored against masks of other tiles, AUC must be near 0.5
  (#f(perm.auc)): #ok(calc.abs(perm.auc - 0.5) < 0.05).
- Positive control: a second U-Net that gets the true mask as an extra input channel must score near 1
  (#f(pos.auc)): #ok(pos.auc > 0.95). If this failed, training or scoring would be broken.
- The trained model beats the negative controls by #f(m.auc - neg.auc) AUC: #ok(m.auc > neg.auc + 0.05).

= Limits and next steps

- The model is deliberately small (#r.model.parameters parameters, #r.model.epochs epochs, #r.model.crop × #r.model.crop
  crops); #if r.model.best_epoch == r.model.epochs [validation AUC was still improving at the last epoch (best epoch
  #r.model.best_epoch of #r.model.epochs), so a longer run is the first thing to try.] else [validation AUC peaked at
  epoch #r.model.best_epoch of #r.model.epochs, so more epochs alone will not help.]
- At probability 0.5 the model reaches an IoU of #f(m.iou) and a recall of #f(m.recall): a better threshold or
  a loss weighted towards particles may help, and the AUC does not depend on the threshold.
- All sources are mixed in one model. HRTEM and Co3O4 have binary masks only, so touching particles cannot be
  told apart; the metrics are pixel-wise, not per particle.
- The test set has #r.data.test_tiles tiles from #r.data.test_groups groups. A confidence interval over groups
  (not only over pixels) would show how stable the numbers are.

= How this report is produced

`just` runs the whole chain: `Justfile` → `Containerfile` → `Makefile` → `data/dataset.db` →
`data/inputs/{train,eval}.safetensors` → `results/results.json` → `report/report.typ` → `results/report.pdf`.
The tensor files used for this report have the checksums (sha256)
#raw(r.data.file_sha256.train.slice(0, 16)) (train) and #raw(r.data.file_sha256.eval.slice(0, 16)) (eval).
Changing the configuration or the data changes these checksums, and `make` re-runs training, the figures and
this report. Seed #r.seed; training took #r.timing.seconds_total s with torch #r.software.torch.
