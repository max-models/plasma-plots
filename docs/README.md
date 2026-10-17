# plasma-plots docs

Astro + Starlight documentation site for
[plasma-plots](https://github.com/max-models/plasma-plots), deployed to
GitHub Pages at https://max-models.github.io/plasma-plots on every push to
`devel` (see `.github/workflows/docs.yml`).

The setup shared by the max-models sites (palette, header, base path, API
reference, link check) comes from
[@max-models/starlight-theme](https://github.com/max-models/starlight-theme);
`astro.config.mjs` holds only what is particular to plasma-plots (the sidebar,
NumPy docstrings and the griffe extension, the llms.txt files), and
`src/styles/custom.css` the few rules of its own. Update the theme by changing
the tag in `package.json` (`#vX.Y.Z`) and running `npm install`.

## Commands

Run from this directory:

| Command           | Action                                       |
| :----------------- | :-------------------------------------------- |
| `npm install`      | Install dependencies                          |
| `npm run dev`      | Start local dev server at `localhost:4321`    |
| `npm run build`    | Build the production site to `./dist/` (fails on broken internal links) |
| `npm run preview`  | Preview the build locally                     |

`DOCS_VALIDATE_LINKS=false` skips the link check (`make docs-dev` sets it).
Pages live under `src/content/docs/`; each `.md`/`.mdx` file there is a
route. The figures they import are generated: run `make figures` in the repo
root first.
