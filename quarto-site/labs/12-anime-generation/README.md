# Anime faces: design, train, defend

Final open-ended coding homework, after the VAE and GAN chapters.

Open anime_faces_homework.ipynb from the ZIP in Jupyter, Colab or Kaggle. The notebook
embeds all utility code and preview figures; support.py is also supplied for editing.
Attach/download splcher/animefacedataset and set DATA_ROOT. No API keys, dataset,
trained generator, model architecture solution or loss/training implementation are included.

Dependencies: torch, torchvision, Pillow, NumPy, Matplotlib; kagglehub only for optional
download. CPU is supported for reduced experiments; time and quality depend on hardware.

Student decisions: VAE/GAN family, architecture, objective implementation, optimizer
updates, training/validation procedure, controlled variant, checkpoint selection,
interpretation and report. Provided: RGB preprocessing, exact processed-duplicate audit,
seeded split, loaders, scalar logs, visualization, simple pixel diagnostics, checkpoint
serialization and self-contained HTML report formatting. Test data is reserved until
validation-based selection. Pixel diagnostics are not perceptual quality scores.

The notebook's report renderer needs no Quarto/LaTeX. Alternatively fill report_template.qmd:
  quarto render report_template.qmd --to html
  quarto render report_template.qmd --to pdf
The PDF route requires LaTeX. Notebook appendix:
  jupyter nbconvert --to html --embed-images anime_faces_homework.ipynb

Authoring: build_evidence.py /path/to/12-preview-jpegs builds measured data/diagnostic
previews; build_notebook.py creates both the downloadable notebook and reading.qmd from
one source plus the ZIP. Rendering Quarto never downloads data or trains a model.
verify.py checks infrastructure, notebook syntax, data integrity, diagnostic formulas,
report portability and serialization without supplying a student model solution.
