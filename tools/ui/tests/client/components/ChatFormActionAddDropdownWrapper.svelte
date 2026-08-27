<script lang="ts">
	import ChatFormActionAddDropdown from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionAdd/ChatFormActionAddDropdown.svelte';
	import { setChatFormActionsContext } from '$lib/contexts';
	import { ServerRole } from '$lib/enums';
	import { serverStore } from '$lib/stores';

	serverStore.role = ServerRole.MODEL;
	serverStore.props = {
		chat_template: '{% if enable_thinking %}<think>{% endif %}',
		default_generation_settings: { n_ctx: 2048 }
	} as ApiLlamaCppServerProps;

	setChatFormActionsContext({
		disabled: false,
		hasAudioModality: false,
		hasMcpPromptsSupport: true,
		hasMcpResourcesSupport: true,
		hasVideoModality: false,
		hasVisionModality: false,
		onFileUpload: () => {},
		onMcpPromptClick: () => {},
		onMcpResourcesClick: () => {},
		onMcpSettingsClick: () => {},
		onSystemPromptClick: () => {}
	});
</script>

<ChatFormActionAddDropdown />
