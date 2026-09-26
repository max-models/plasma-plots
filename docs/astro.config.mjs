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
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/struphy-hub/struphy-plots' },
			],
			sidebar: [
				{
					label: 'Guides',
					items: [
						{ label: 'Getting started', slug: 'guides/getting-started' },
						{ label: 'Plotting', slug: 'guides/plotting' },
						{ label: 'Analysis', slug: 'guides/analysis' },
						{ label: 'Whole-run plots', slug: 'guides/output-plots' },
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
