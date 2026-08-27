<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';

	let open = $state(true);
</script>

<Dialog.Root bind:open>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Responsive dialog</Dialog.Title>
		</Dialog.Header>

		<p>Body</p>
	</Dialog.Content>
</Dialog.Root>
