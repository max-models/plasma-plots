// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://struphy-hub.github.io',
	base: '/struphy-plots',
	integrations: [
		starlight({
			title: 'struphy-plots',
			description: 'Optional plotting and diagnostics layer for Struphy output',
			customCss: ['./src/styles/custom.css'],
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
					items: [{ autogenerate: { directory: 'reference' } }],
				},
			],
		}),
	],
});
