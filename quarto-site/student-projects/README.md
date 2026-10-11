# Student project gallery

The page uses Quarto's native YAML listing with `_cards.ejs.md`. There is no
training, browser-side YAML loading, or additional build dependency.

## Student submission

Students download `project-proposal.yml`, fill it in, and send it plus an optional
thumbnail to the instructor through the usual course submission channel.
The template is a single YAML mapping; `projects.yml` is a list of public cards.

## Publish an approved proposal

1. Review submissions outside this public repository. Copy only `title`,
   `students`, `description`, and the three thumbnail fields into a new list item
   in `projects.yml`. Do not copy the `proposal` review block.
2. Put the image in `images/` and set `thumbnail: "images/your-project.webp"`.
   Use a landscape image, ideally 1200 × 750, with meaningful `thumbnail_alt`
   and attribution in `thumbnail_credit`. Omit or leave `thumbnail` empty for
   the supplied placeholder. Both SVGs shipped here are original course artwork.
3. Set `status` to `Proposal approved`, `In progress`, or `Completed`.
   Remove the fictional example when adding the first real project; omit
   `example` for real projects. Entries appear in YAML order.
4. Optionally add `report`, `code`, and `demo` as full HTTPS URLs when available.
5. Run `quarto render quarto-site/student-projects/index.qmd` from the repo root.
   Check the page, then publish through the normal site workflow when ready.
   The gallery updates automatically on every site build.

One project gets a spacious horizontal card; multiple projects use three
columns on wide screens, two on tablets, and one on phones. Card text is escaped;
names must be a nonempty YAML list. The renderer checks status, thumbnail path,
alt text, and link schemes. An empty collection can be represented by `[]`.

The old root `_data/projects.yml` belongs to the previous site and is untouched.
Reference: https://quarto.org/docs/websites/website-listings-custom.html
