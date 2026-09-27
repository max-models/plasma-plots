// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';

// https://astro.build/config
export default defineConfig({
	site: 'https://struphy-hub.github.io',
	base: '/struphy-plots',
	integrations: [
		starlight({
			title: 'struphy-plots',
			description: 'Optional plotting and diagnostics layer for Struphy output',
			customCss: ['./src/styles/custom.css', './src/styles/api.css'],
			// /llms.txt, /llms-full.txt and /llms-small.txt: the docs for language models
			plugins: [
				starlightLlmsTxt({
					projectName: 'struphy-plots',
					description:
						'struphy-plots is the plotting and diagnostics package for the output of Struphy, a Python code for plasma physics simulations. It works on labeled xarray data.',
					details: [
						'How to use it:',
						'',
						'- It adds accessors rather than functions to call: `out.plot` and `out.analysis` on a Struphy `Output`, `.struphy.plot`, `.struphy.analysis` and `.struphy.data` on every `xarray.DataArray` product (e.g. `out.evaluate("em_fields/phi")`), and the same on marker Datasets such as orbits.',
						'- Recent Struphy versions load it when an `Output` is created; otherwise run `import struphy_plots` first, or `.struphy` raises an AttributeError.',
						'- Select every dimension a plot does not draw by keyword: an integer is a position (`t=-1` the last), a float the nearest coordinate value.',
						'- In Python, `help(struphy_plots)` gives an overview, printing an accessor (e.g. `print(phi.struphy.plot)`) lists its methods, and `help()` on a method shows every parameter.',
						'- The Reference pages list every accessor method and function with all parameters; the Guides show them with figures.',
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
						{
							label: 'Functions',
							collapsed: true,
							items: [{ autogenerate: { directory: 'reference/functions' } }],
						},
					],
				},
			],
		}),
	],
});
