# Lecture-note template
Every chapter placed below `notes/` inherits the shared settings in
`_metadata.yml`. This produces a web page and a companion A4 PDF with the same
course identity.

## Starting a chapter

Create a directory such as `notes/02-neural-networks/` with an `index.qmd`:

```yaml
---
title: "Chapter title"
subtitle: "Chapter 2 · Neural Networks from Scratch"
description: "One-sentence description"
author: "Dr. A. Belcaid"
toc: true
toc-depth: 3
number-sections: true
---
```

Write content once using Quarto Markdown. Standard headings, mathematics,
tables, figures, citations, code blocks, cross-references, and callouts work in
both formats.

## Rendering

Render the complete website and all companion formats:

```bash
quarto render quarto-site
```

Render only one chapter as HTML:

```bash
quarto render quarto-site/notes/01-what-is-learning/index.qmd --to html
```

Render only its PDF:

```bash
quarto render quarto-site/notes/01-what-is-learning/index.qmd --to pdf
```

The PDF design is centralized in `pdf-preamble.tex`; do not copy formatting
into individual chapters. Chapter-specific LaTeX should be a last resort.

## Authoring rule

Prefer native Quarto Markdown and fenced divs over raw HTML. When a visual is
meaningful only on the website, mark it with `.content-visible` and
`when-format="html"`, then provide a concise print alternative.
