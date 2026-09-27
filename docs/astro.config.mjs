// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';

// https://astro.build/config
export default defineConfig({
	site: 'https://struphy-hub.github.io',
	base: '/struphy-plots',
	integrations: [
		starlight({
			title: 'struphy-plots',
			description: 'Optional plotting and diagnostics layer for Struphy output',
			customCss: ['./src/styles/custom.css'],
			// /llms.txt, /llms-full.txt and /llms-small.txt: the docs for language models
			plugins: [
				// the API reference: one page per module under /api/struphy_plots/, from the docstrings
				starlightPydocs({
					packages: [
						{
							name: 'struphy_plots',
							label: 'All modules',
							search: ['../src'],
							docstringStyle: 'numpy',
							// accessor methods get the parameter docs of the functions they wrap
							extensions: ['../scripts/griffe_extension.py'],
							sourceLink: { host: 'github', repo: 'struphy-hub/struphy-plots', ref: 'devel', root: '..' },
							sidebar: { collapsed: true },
						},
					],
					inventories: [
						'python',
						{ url: 'https://numpy.org/doc/stable/objects.inv' },
						{ url: 'https://docs.xarray.dev/en/stable/objects.inv' },
						{ url: 'https://matplotlib.org/stable/objects.inv' },
						{ url: 'https://docs.pyvista.org/objects.inv' },
					],
				}),
				starlightLlmsTxt({
					projectName: 'struphy-plots',
					description:
						'struphy-plots is the plotting and diagnostics package for the output of Struphy, a Python code for plasma physics simulations. It works on labeled xarray data.',
					details: [
						'How to use it:',
						'',
						'- It adds accessors rather than functions to call: `out.plot` and `out.analysis` on a Struphy `Output`, `.struphy.plot`, `.struphy.analysis` and `.struphy.data` on every `xarray.DataArray` product (e.g. `out.evaluate("em_fields/phi")`), and the same on marker Datasets such as orbits.',
						'- Creating a Struphy `Output` loads it, so Struphy output needs no import. For xarray data from elsewhere, run `import struphy_plots` first, or `.struphy` raises an AttributeError.',
						'- Select every dimension a plot does not draw by keyword: an integer is a position (`t=-1` the last), a float the nearest coordinate value.',
						'- In Python, `import struphy_plots; help(struphy_plots)` (or `python -m struphy_plots`) gives an overview, printing an accessor (e.g. `print(phi.struphy.plot)`) lists its methods, and `help()` on a method shows every parameter.',
						'- `struphy_plots.theory` has analytic results to compare runs with: kinetic, fluid, MHD and cold-plasma dispersion relations and growth rates (complex ω), plasma parameters and Struphy units, orbits, exact solutions, and the errors of the numerical schemes. Plain numpy; its functions work directly as `branches=`, `reference=` and `theory=` of the plots.',
						'- The Reference pages list every accessor method and function with all parameters (also as plain text at /api/struphy_plots/llms.txt); the Guides show them with figures.',
					].join('\n'),
					promote: ['guides/getting-started', 'guides/data', 'reference', 'reference/**'],
					customSets: [
						{
							label: 'API reference',
							description: 'every accessor method and function, with all parameters',
							paths: ['reference', 'reference/**'],
						},
					],
				}),
			],
			components: {
				Header: './src/components/Header.astro',
			},
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/struphy-hub/struphy-plots' },
			],
			sidebar: [
				{
					label: 'Guides',
					items: [
						{ label: 'Getting started', slug: 'guides/getting-started' },
						{ label: 'Selecting data', slug: 'guides/data' },
						{ label: 'Field plots', slug: 'guides/field-plots' },
						{ label: '3-D views', slug: 'guides/3d-views' },
						{ label: 'Time series & comparisons', slug: 'guides/timeseries' },
						{ label: 'Diagnostics', slug: 'guides/analysis' },
						{ label: 'Spectral analysis', slug: 'guides/spectral' },
						{ label: 'Particles & distributions', slug: 'guides/particles' },
						{ label: 'Whole-run plots', slug: 'guides/output-plots' },
						{ label: 'Profiling', slug: 'guides/profiling' },
						{ label: 'Recipes', slug: 'guides/recipes' },
						{ label: 'MHD slab waves', slug: 'guides/real-example' },
						{ label: 'Theory toolbox', slug: 'guides/theory' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Overview', slug: 'reference' },
						{ label: 'array.struphy.plot', slug: 'reference/plot' },
						{ label: 'array.struphy.analysis', slug: 'reference/analysis' },
						{ label: 'array.struphy.data', slug: 'reference/data' },
						{ label: 'dataset.struphy', slug: 'reference/dataset' },
						{ label: 'out.plot, out.analysis', slug: 'reference/output' },
						pydocsSidebarGroup,
					],
				},
			],
		}),
	],
});
