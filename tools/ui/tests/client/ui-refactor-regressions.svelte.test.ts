import ChatFormActionAddDropdownWrapper from './components/ChatFormActionAddDropdownWrapper.svelte';
import DialogContentWrapper from './components/DialogContentWrapper.svelte';
import { ATTACHMENT_TOOLTIP_TEXT } from '$lib/constants';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from 'vitest-browser-svelte';

describe('b10642 UI refactor regressions', () => {
	afterEach(cleanup);

	it('keeps desktop reasoning and the complete MCP submenu mounted', async () => {
		const screen = await render(ChatFormActionAddDropdownWrapper);

		await screen.getByRole('button', { name: ATTACHMENT_TOOLTIP_TEXT }).click();

		await expect.element(screen.getByText('Reasoning', { exact: true })).toBeVisible();

		const mcpTrigger = screen.getByText('MCP', { exact: true });

		await expect.element(mcpTrigger).toBeVisible();
		await mcpTrigger.hover();

		await expect.element(screen.getByText('Servers', { exact: true })).toBeVisible();
		await expect.element(screen.getByText('Prompts', { exact: true })).toBeVisible();
		await expect.element(screen.getByText('Resources', { exact: true })).toBeVisible();
	});

	it('gives shared dialogs a responsive width instead of shrinking to their contents', async () => {
		const screen = await render(DialogContentWrapper);
		const dialog = screen.getByRole('dialog');

		await expect.element(dialog).toBeVisible();

		const element = await dialog.element();
		const rect = element.getBoundingClientRect();

		expect(rect.width).toBeGreaterThan(400);
		expect(rect.width).toBeLessThanOrEqual(window.innerWidth - 32);
	});
});
