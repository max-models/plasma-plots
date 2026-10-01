# plasma-plots docs

Astro + Starlight documentation site for
[plasma-plots](https://github.com/max-models/plasma-plots), deployed to
GitHub Pages at https://max-models.github.io/plasma-plots on every push to
`devel` (see `.github/workflows/docs.yml`).

## Commands

Run from this directory:

| Command           | Action                                       |
| :----------------- | :-------------------------------------------- |
| `npm install`      | Install dependencies                          |
| `npm run dev`      | Start local dev server at `localhost:4321`    |
| `npm run build`    | Build the production site to `./dist/`        |
| `npm run preview`  | Preview the build locally                     |

Pages live under `src/content/docs/`; each `.md`/`.mdx` file there is a
route.
