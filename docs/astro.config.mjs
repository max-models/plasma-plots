// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';

// https://astro.build/config
export default defineConfig({
	site: 'https://struphy-hub.github.io',
	base: '/plasma-plots',
	integrations: [
		starlight({
			title: 'plasma-plots',
			description: 'Plots and diagnostics of labeled xarray output from plasma simulations, such as Struphy',
			customCss: ['./src/styles/custom.css'],
			// /llms.txt, /llms-full.txt and /llms-small.txt: the docs for language models
			plugins: [
				// the API reference: one page per module under /api/plasma_plots/, from the docstrings
				starlightPydocs({
					packages: [
						{
							name: 'plasma_plots',
							label: 'All modules',
							search: ['../src'],
							docstringStyle: 'numpy',
							// accessor methods get the parameter docs of the functions they wrap
							extensions: ['../scripts/griffe_extension.py'],
							sourceLink: { host: 'github', repo: 'struphy-hub/plasma-plots', ref: 'devel', root: '..' },
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
					projectName: 'plasma-plots',
					description:
						'plasma-plots is the plotting and diagnostics package for the output of Struphy, a Python code for plasma physics simulations. It works on labeled xarray data.',
					details: [
						'How to use it:',
						'',
						'- It adds accessors rather than functions to call: `out.plot` and `out.analysis` on a Struphy `Output`, `.plasma.plot`, `.plasma.analysis` and `.plasma.data` on every `xarray.DataArray` product (e.g. `out.evaluate("em_fields/phi")`), and the same on marker Datasets such as orbits.',
						'- Creating a Struphy `Output` loads it, so Struphy output needs no import. For xarray data from elsewhere, run `import plasma_plots` first, or `.plasma` raises an AttributeError.',
						'- Select every dimension a plot does not draw by keyword: an integer is a position (`t=-1` the last), a float the nearest coordinate value.',
						'- In Python, `import plasma_plots; help(plasma_plots)` (or `python -m plasma_plots`) gives an overview, printing an accessor (e.g. `print(phi.plasma.plot)`) lists its methods, and `help()` on a method shows every parameter.',
						'- `plasma_plots.theory` has analytic results to compare runs with: kinetic, fluid, MHD and cold-plasma dispersion relations and growth rates (complex ω), plasma parameters and Struphy units, orbits, exact solutions, and the errors of the numerical schemes. Plain numpy; its functions work directly as `branches=`, `reference=` and `theory=` of the plots.',
						'- Start with the API index (/llms-api.txt, the same as `python -m plasma_plots api`): the conventions every method shares, then every accessor method and function with its signature and one-line summary, in about 40 kB.',
						'- The Reference pages list every accessor method and function with all parameters (also as plain text at /api/plasma_plots/llms.txt); the Guides show them with figures.',
					].join('\n'),
					promote: ['guides/getting-started', 'guides/data', 'reference', 'reference/**'],
					optionalLinks: [
						{
							label: 'API index',
							url: 'https://struphy-hub.github.io/plasma-plots/llms-api.txt',
							description: 'the conventions, then every accessor method and function with its signature and summary, one line each (about 40 kB)',
						},
						{
							label: 'API reference',
							url: 'https://struphy-hub.github.io/plasma-plots/api/plasma_plots/llms.txt',
							description: 'every module, class and function of plasma_plots with all parameters, as plain text',
						},
					],
				}),
			],
			components: {
				Header: './src/components/Header.astro',
			},
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/struphy-hub/plasma-plots' },
			],
			sidebar: [
				{
					label: 'Guides',
					items: [
						{ label: 'Getting started', slug: 'guides/getting-started' },
						{ label: 'Selecting data', slug: 'guides/data' },
						{ label: 'Field plots', slug: 'guides/field-plots' },
						{ label: 'Interactive plots (Plotly)', slug: 'guides/plotly' },
						{ label: '3-D views', slug: 'guides/3d-views' },
						{ label: 'Time series & comparisons', slug: 'guides/timeseries' },
						{ label: 'Diagnostics', slug: 'guides/analysis' },
						{ label: 'Spectral analysis', slug: 'guides/spectral' },
						{ label: 'Particles & distributions', slug: 'guides/particles' },
						{ label: 'Whole-run plots', slug: 'guides/output-plots' },
						{ label: 'Profiling', slug: 'guides/profiling' },
						{ label: 'Recipes', slug: 'guides/recipes' },
						{ label: 'MHD slab waves', slug: 'guides/real-example' },
						{ label: 'GVEC equilibria', slug: 'guides/gvec' },
						{ label: 'DESC equilibria', slug: 'guides/desc' },
						{ label: 'Theory toolbox', slug: 'guides/theory' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Overview', slug: 'reference' },
						{ label: 'array.plasma.plot', slug: 'reference/plot' },
						{ label: 'array.plasma.analysis', slug: 'reference/analysis' },
						{ label: 'array.plasma.data', slug: 'reference/data' },
						{ label: 'dataset.plasma', slug: 'reference/dataset' },
						{ label: 'out.plot, out.analysis', slug: 'reference/output' },
						pydocsSidebarGroup,
					],
				},
			],
		}),
	],
});
