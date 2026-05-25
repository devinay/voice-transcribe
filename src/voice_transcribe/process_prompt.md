Create a structured markdown document from the source notes below. Return only the markdown document, with no extra commentary or code fences. Use exactly these sections:

# [Title extracted from source notes]

## Summary
[2-3 sentence summary]

## Source Notes
[original source notes, unchanged, inside a closed `<details>` block with a `<summary>` label]

## Processed Transcript
[correct errors, apply punctuation, break into paragraphs where the topic changes]

## Actions / Follow-ups
[markdown task list using checkboxes (`- [ ]`); leave blank if purely informational]

Source notes:
{transcript}
