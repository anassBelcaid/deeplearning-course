# GAN chapter assets

`build_notes_assets.py` rebuilds the original text-free SVG cover, the blind gallery,
18 standalone visual panels, and native code excerpts from the approved GAN lecture.
Run it after rendering `lectures/12-generative-adversarial-networks/index.qmd`.
It requires BeautifulSoup but performs no training or inference.

Panels use the lecture's shared/local styles and recorded PNG/JSON assets through a
relative base URL. Each panel retains its actual source slide ID for existing control
scripts. Hidden dependency panels supply the other DOM fields required by section-wide
scripts. All visible fragments are expanded for reading; chapter exercises wrap solutions
in native details elements. Panel heights are reported to the notes page using a
same-origin, source-window-checked message.

Coverage: source identification, target/difference experiment, frozen-VAE judge,
adversarial feedback graph, VAE/GAN module comparison, batch routing, G/D tensor
contracts and executable code, source-loss arithmetic, opposing losses, logit-gradient
curves, update routing and code, disconnected-gradient exercise, dense/convolutional
comparison, paired spatial architecture, transposed-convolution overlap, exact G/D
layers, fixed-code checkpoints, fresh batches/repetition control, loss history,
nearest neighbors, and final five-part synthesis.

The cover embeds one observed image and one actual epoch-60 output. Blue bars depict
G's expansion and coral bars D's contraction; the dashed coral return is training
feedback. Geometry is schematic, not measured activation data. All evidence remains
single-sourced in the lecture assets with its original protocol and limitations.

`_chapter.qmd` is the extensive HTML chapter. `_review.qmd` is a separate four-page
PDF technical review, with no embedded simulations. Render `index.qmd` for both.
